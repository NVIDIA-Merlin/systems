#
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import importlib.resources
import os
import pathlib
from shutil import copyfile
from typing import List, Tuple

import tritonclient.grpc.model_config_pb2 as model_config
from google.protobuf import text_format

from merlin.core.protocols import Transformable
from merlin.dag import Graph, postorder_iter_nodes
from merlin.systems.dag.ops import compute_dims
from merlin.systems.dag.ops.compat import (
    cuml_ensemble,
    lightgbm,
    sklearn_ensemble,
    treelite_sklearn,
    xgboost,
)
from merlin.systems.dag.ops.operator import add_model_param
from merlin.systems.dag.ops.workflow import TransformWorkflow
from merlin.systems.dag.runtimes import Runtime
from merlin.systems.dag.runtimes.triton.ops.workflow import TransformWorkflowTriton

tensorflow = None
try:
    from nvtabular.loader.tf_utils import configure_tensorflow

    # everything tensorflow related must be imported after this.
    configure_tensorflow()
    import tensorflow
except ImportError:
    ...

torch = None
try:
    import torch
except ImportError:
    ...


TRITON_OP_TABLE = {}
TRITON_OP_TABLE[TransformWorkflow] = TransformWorkflowTriton

if cuml_ensemble or lightgbm or sklearn_ensemble or treelite_sklearn or xgboost:
    from merlin.systems.dag.ops.fil import PredictForest
    from merlin.systems.dag.runtimes.triton.ops.fil import PredictForestTriton

    TRITON_OP_TABLE[PredictForest] = PredictForestTriton

if tensorflow:
    from merlin.systems.dag.ops.tensorflow import PredictTensorflow
    from merlin.systems.dag.runtimes.triton.ops.tensorflow import PredictTensorflowTriton

    TRITON_OP_TABLE[PredictTensorflow] = PredictTensorflowTriton

if torch:
    from merlin.systems.dag.ops.pytorch import PredictPyTorch
    from merlin.systems.dag.runtimes.triton.ops.pytorch import PredictPyTorchTriton

    TRITON_OP_TABLE[PredictPyTorch] = PredictPyTorchTriton


class TritonEnsembleRuntime(Runtime):
    """Runtime for Triton. Runs each operator in DAG as a separate model in a Triton Ensemble."""

    def __init__(self):
        super().__init__()
        self.op_table = TRITON_OP_TABLE

    def transform(self, graph: Graph, transformable: Transformable):
        raise NotImplementedError("Transform handled by Triton")

    def export(
        self, ensemble, path: str, version: int = 1, name: str = None
    ) -> Tuple[model_config.ModelConfig, List[model_config.ModelConfig]]:
        """Exports an 'Ensemble' as a triton model repository.

        Every operator is represented as a separate model,
        loaded individually in Triton.

        The entry point is the ensemble model with the name `name`, by default "ensemble_model"

        Parameters
        ----------
        ensemble : merlin.systems.dag.Ensemble
            Systems ensemble to export
        path : str
            Path to directory where Triton model repository will be created.
        version : int, optional
            Version for Triton models created, by default 1
        name : str, optional
            The name of the ensemble triton model, by default "ensemble_model"

        Returns
        -------
        Tuple[model_config.ModelConfig, List[model_config.ModelConfig]]
            Tuple of ensemble config and list of non-python backend model configs
        """
        name = name or "ensemble_model"
        # Build node id lookup table

        nodes = list(postorder_iter_nodes(ensemble.graph.output_node))

        for node in nodes:
            if type(node.op) in self.op_table:
                node.op = self.op_table[type(node.op)](node.op)

        node_id_table, num_nodes = _create_node_table(nodes, "ensemble")

        nodes = nodes or []
        node_id_table = node_id_table or {}

        # Create ensemble config
        ensemble_config = model_config.ModelConfig(
            name=name,
            platform="ensemble",
            # max_batch_size=configs[0].max_batch_size
        )

        for _, col_schema in ensemble.graph.input_schema.column_schemas.items():
            add_model_param(
                ensemble_config.input,
                model_config.ModelInput,
                col_schema,
                compute_dims(col_schema),
            )

        for _, col_schema in ensemble.graph.output_schema.column_schemas.items():
            add_model_param(
                ensemble_config.output,
                model_config.ModelOutput,
                col_schema,
                compute_dims(col_schema),
            )

        node_configs = []
        for node in nodes:
            if node.exportable("ensemble"):
                node_id = node_id_table.get(node, None)
                node_name = f"{node_id}_{node.export_name}"

                found = False
                for step in ensemble_config.ensemble_scheduling.step:
                    if step.model_name == node_name:
                        found = True
                if found:
                    continue

                node_config = node.export(
                    path, node_id=node_id, version=version, backend="ensemble"
                )
                if node_config is not None:
                    node_configs.append(node_config)

                config_step = model_config.ModelEnsembling.Step(
                    model_name=node_name, model_version=-1
                )

                for input_col_name, input_col_schema in node.input_schema.column_schemas.items():
                    source = self._find_column_source(
                        node.parents_with_dependencies, input_col_name, "ensemble"
                    )
                    source_id = node_id_table.get(source, None)
                    in_suffix = f"_{source_id}" if source_id is not None else ""

                    if input_col_schema.is_list and input_col_schema.is_ragged:
                        config_step.input_map[input_col_name + "__values"] = (
                            input_col_name + "__values" + in_suffix
                        )
                        config_step.input_map[input_col_name + "__lengths"] = (
                            input_col_name + "__lengths" + in_suffix
                        )
                    else:
                        config_step.input_map[input_col_name] = input_col_name + in_suffix

                for output_col_name, output_col_schema in node.output_schema.column_schemas.items():
                    out_suffix = (
                        f"_{node_id}" if node_id is not None and node_id < num_nodes - 1 else ""
                    )

                    if output_col_schema.is_list and output_col_schema.is_ragged:
                        config_step.output_map[output_col_name + "__values"] = (
                            output_col_name + "__values" + out_suffix
                        )
                        config_step.output_map[output_col_name + "__lengths"] = (
                            output_col_name + "__lengths" + out_suffix
                        )
                    else:
                        config_step.output_map[output_col_name] = output_col_name + out_suffix

                ensemble_config.ensemble_scheduling.step.append(config_step)

        # Write the ensemble config file
        ensemble_path = os.path.join(path, name)
        os.makedirs(ensemble_path, exist_ok=True)
        os.makedirs(os.path.join(ensemble_path, str(version)), exist_ok=True)

        config_path = os.path.join(ensemble_path, "config.pbtxt")
        with open(config_path, "w", encoding="utf-8") as o:
            text_format.PrintMessage(ensemble_config, o)

        return (ensemble_config, node_configs)

    def _find_column_source(self, upstream_nodes, column_name, backend):
        source_node = None
        for upstream_node in upstream_nodes:
            if column_name in upstream_node.output_columns.names:
                source_node = upstream_node
                break

        if source_node and not source_node.exportable(backend):
            return self._find_column_source(
                source_node.parents_with_dependencies, column_name, backend
            )
        else:
            return source_node


class TritonExecutorRuntime(Runtime):
    """Runtime for Triton.
    This will run the DAG in a single Triton model and call out to other
    Triton models for nodes that use any non-python backends.
    """

    def __init__(self):
        super().__init__()
        self.op_table = TRITON_OP_TABLE

    def export(
        self, ensemble, path: str, version: int = 1, name: str = None
    ) -> Tuple[model_config.ModelConfig, List[model_config.ModelConfig]]:
        """Exports an 'Ensemble' as a Triton model repository.

        All operators in the ensemble will run in a single python backend model except
        those that have a non-python Triton backend.

        The entrypoint is the model with name `name`, by default "executor_model"

        Parameters
        ----------
        ensemble : merlin.systems.dag.Ensemble
            Systems ensemble to export
        path : str
            Path to directory where Triton model repository will be created.
        version : int, optional
            Version for Triton models created, by default 1
        name : str, optional
            The name of the ensemble triton model, by default "executor_model"

        Returns
        -------
        Tuple[model_config.ModelConfig, List[model_config.ModelConfig]]
            Tuple of ensemble config and list of non-python backend model configs
        """
        name = name or "executor_model"

        nodes = list(postorder_iter_nodes(ensemble.graph.output_node))

        for node in nodes:
            if type(node.op) in self.op_table:
                node.op = self.op_table[type(node.op)](node.op)

        node_id_table, _ = _create_node_table(nodes, "executor")

        node_configs = []
        for node in nodes:
            if node.exportable("executor"):
                node_id = node_id_table.get(node, None)

                node_config = node.export(
                    path, node_id=node_id, version=version, backend="executor"
                )
                if node_config is not None:
                    node_configs.append(node_config)

        executor_config = self._executor_model_export(path, name, ensemble)

        return (executor_config, node_configs)

    def _executor_model_export(
        self,
        path: str,
        export_name: str,
        ensemble,
        params: dict = None,
        node_id: int = None,
        version: int = 1,
    ) -> model_config.ModelConfig:
        """Export the ensemble and all related files to the  path.

        Parameters
        ----------
        path : str
            Artifact export path
        export_name : str
            The name for the Triton model to export to.
        ensemble : merlin.systems.dag.Ensemble
            The ensemble to export.
        params : dict, optional
            Parameters dictionary of key, value pairs stored in exported config, by default None.
        node_id : int, optional
            The placement of the node in the graph (starts at 1), by default None.
        version : int, optional
            The version of the model, by default 1.

        Returns
        -------
        model_config.ModelConfig
            The config for the ensemble.
        """

        params = params or {}

        node_name = f"{node_id}_{export_name}" if node_id is not None else export_name

        node_export_path = pathlib.Path(path) / node_name
        node_export_path.mkdir(parents=True, exist_ok=True)

        config = model_config.ModelConfig(
            name=node_name, backend="python", platform="merlin_executor"
        )

        input_schema = ensemble.input_schema
        output_schema = ensemble.output_schema

        for col_schema in input_schema.column_schemas.values():
            col_dims = compute_dims(col_schema)
            add_model_param(config.input, model_config.ModelInput, col_schema, col_dims)

        for col_schema in output_schema.column_schemas.values():
            col_dims = compute_dims(col_schema)
            add_model_param(config.output, model_config.ModelOutput, col_schema, col_dims)

        with open(os.path.join(node_export_path, "config.pbtxt"), "w", encoding="utf-8") as o:
            text_format.PrintMessage(config, o)

        os.makedirs(node_export_path, exist_ok=True)
        os.makedirs(os.path.join(node_export_path, str(version)), exist_ok=True)
        with importlib.resources.path(
            "merlin.systems.triton.models", "executor_model.py"
        ) as executor_model:
            copyfile(
                executor_model,
                os.path.join(node_export_path, str(version), "model.py"),
            )

        ensemble.save(os.path.join(node_export_path, str(version), "ensemble"))

        return config


def _create_node_table(nodes, backend):
    exportable_node_idx = 0
    node_id_lookup = {}
    for node in nodes:
        if node.exportable(backend):
            node_id_lookup[node] = exportable_node_idx
            exportable_node_idx += 1

    return node_id_lookup, exportable_node_idx
