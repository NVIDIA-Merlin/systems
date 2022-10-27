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
import os
import pathlib
from shutil import copyfile, copytree
from typing import List, Tuple

from merlin.dag import postorder_iter_nodes

# this needs to be before any modules that import protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tritonclient.grpc.model_config_pb2 as model_config  # noqa
from google.protobuf import text_format  # noqa

from merlin.core.protocols import Transformable  # noqa
from merlin.dag import ColumnSelector, Graph  # noqa
from merlin.schema import Schema  # noqa
from merlin.systems.dag.ops import compute_dims  # noqa
from merlin.systems.dag.ops.compat import pb_utils  # noqa
from merlin.systems.dag.ops.operator import add_model_param  # noqa
from merlin.systems.dag.runtimes import Runtime  # noqa

tensorflow = None
try:
    import tensorflow
except ImportError:
    ...


class TritonOperator:
    def __init__(self, base_op):
        self.op = base_op
        self.input_schema = self.op.input_schema
        self.output_schema = self.op.output_schema

    @property
    def export_name(self):
        """
        Provides a clear common english identifier for this operator.

        Returns
        -------
        String
            Name of the current class as spelled in module.
        """
        return self.__class__.__name__.lower()

    @property
    def exportable_backends(self) -> List[str]:
        """Returns list of supported backends.

        Returns
        -------
        List[str]
            List of supported backends
        """
        return ["ensemble"]


class PredictTensorflowTriton(TritonOperator):
    """TensorFlow Model Prediction Operator for running inside Triton."""

    def __init__(self, base_op):
        super().__init__(base_op)

        self.path = self.op.path
        self.model = self.op.model
        self.scalar_shape = self.op.scalar_shape

    def transform(self, col_selector: ColumnSelector, transformable: Transformable):
        """Run transform of operator callling TensorFlow model with a Triton InferenceRequest.

        Returns
        -------
        Transformable
            TensorFlow Model Outputs
        """
        # TODO: Validate that the inputs match the schema
        # TODO: Should we coerce the dtypes to match the schema here?
        input_tensors = []
        for col_name in self.input_schema.column_schemas.keys():
            input_tensors.append(pb_utils.Tensor(col_name, transformable[col_name]))

        inference_request = pb_utils.InferenceRequest(
            model_name=self.tf_model_name,
            requested_output_names=self.output_schema.column_names,
            inputs=input_tensors,
        )
        inference_response = inference_request.exec()

        # TODO: Validate that the outputs match the schema
        outputs_dict = {}
        for out_col_name in self.output_schema.column_schemas.keys():
            output_val = pb_utils.get_output_tensor_by_name(
                inference_response, out_col_name
            ).as_numpy()
            outputs_dict[out_col_name] = output_val

        return type(transformable)(outputs_dict)

    def export(
        self,
        path: str,
        input_schema: Schema,
        output_schema: Schema,
        params: dict = None,
        node_id: int = None,
        version: int = 1,
        backend: str = "ensemble",
    ):
        """Create a directory inside supplied path based on our export name"""
        # Export Triton TF back-end directory and config etc
        export_name = self.__class__.__name__.lower()
        node_name = f"{node_id}_{export_name}" if node_id is not None else export_name

        node_export_path = pathlib.Path(path) / node_name
        node_export_path.mkdir(exist_ok=True)

        tf_model_path = pathlib.Path(node_export_path) / str(version) / "model.savedmodel"

        if self.path:
            copytree(
                str(self.path),
                tf_model_path,
                dirs_exist_ok=True,
            )
        else:
            self.model.save(tf_model_path, include_optimizer=False)

        self.set_tf_model_name(node_name)
        backend_model_config = self._export_model_config(node_name, node_export_path)
        return backend_model_config

    def _export_model_config(self, name, output_path):
        """Exports a TensorFlow model for serving with Triton

        Parameters
        ----------
        model:
            The tensorflow model that should be served
        name:
            The name of the triton model to export
        output_path:
            The path to write the exported model to
        """
        config = model_config.ModelConfig(
            name=name, backend="tensorflow", platform="tensorflow_savedmodel"
        )

        config.parameters["TF_GRAPH_TAG"].string_value = "serve"
        config.parameters["TF_SIGNATURE_DEF"].string_value = "serving_default"

        for _, col_schema in self.input_schema.column_schemas.items():
            add_model_param(
                config.input,
                model_config.ModelInput,
                col_schema,
                compute_dims(col_schema, self.scalar_shape),
            )

        for _, col_schema in self.output_schema.column_schemas.items():
            add_model_param(
                config.output,
                model_config.ModelOutput,
                col_schema,
                compute_dims(col_schema, self.scalar_shape),
            )

        with open(os.path.join(output_path, "config.pbtxt"), "w", encoding="utf-8") as o:
            text_format.PrintMessage(config, o)
        return config

    @property
    def tf_model_name(self):
        return self._tf_model_name

    def set_tf_model_name(self, tf_model_name: str):
        """
        Set the name of the Triton model to use

        Parameters
        ----------
        tf_model_name : str
            Triton model directory name
        """
        self._tf_model_name = tf_model_name


class TritonEnsembleRuntime(Runtime):
    """Runtime for Triton. Runs each operator in DAG as a separate model in a Triton Ensemble."""

    def __init__(self):
        super().__init__()
        self.op_table = {}
        if tensorflow:
            from merlin.systems.dag.ops.tensorflow import PredictTensorflow

            self.op_table[PredictTensorflow] = PredictTensorflowTriton

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
                col_schema.properties.get("shape", None) or compute_dims(col_schema),
            )

        for _, col_schema in ensemble.graph.output_schema.column_schemas.items():
            add_model_param(
                ensemble_config.output,
                model_config.ModelOutput,
                col_schema,
                col_schema.properties.get("shape", None) or compute_dims(col_schema),
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
                        config_step.input_map[input_col_name + "__nnzs"] = (
                            input_col_name + "__nnzs" + in_suffix
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
                        config_step.output_map[output_col_name + "__nnzs"] = (
                            output_col_name + "__nnzs" + out_suffix
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
        self.op_table = {}
        if tensorflow:
            from merlin.systems.dag.ops.tensorflow import PredictTensorflow

            self.op_table[PredictTensorflow] = PredictTensorflowTriton

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
            col_dims = col_schema.properties.get("shape", None) or compute_dims(col_schema)
            add_model_param(config.input, model_config.ModelInput, col_schema, col_dims)

        for col_schema in output_schema.column_schemas.values():
            col_dims = col_schema.properties.get("shape", None) or compute_dims(col_schema)
            add_model_param(config.output, model_config.ModelOutput, col_schema, col_dims)

        with open(os.path.join(node_export_path, "config.pbtxt"), "w", encoding="utf-8") as o:
            text_format.PrintMessage(config, o)

        os.makedirs(node_export_path, exist_ok=True)
        os.makedirs(os.path.join(node_export_path, str(version)), exist_ok=True)
        copyfile(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "triton",
                "models",
                "executor_model.py",
            ),
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
