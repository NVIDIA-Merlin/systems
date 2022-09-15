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
import json
import os
import pathlib
import sys
import time
import warnings
from shutil import copyfile

import cloudpickle
import fsspec

from merlin.dag import postorder_iter_nodes

# this needs to be before any modules that import protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tritonclient.grpc.model_config_pb2 as model_config  # noqa
from google.protobuf import text_format  # noqa

from merlin.dag import Graph  # noqa
from merlin.systems.dag.ops import compute_dims  # noqa
from merlin.systems.dag.ops.operator import add_model_param  # noqa


class Ensemble:
    """
    Class that represents an entire ensemble consisting of multiple models that
    run sequentially in tritonserver initiated by an inference request.
    """

    def __init__(self, ops, schema, name="ensemble_model", label_columns=None):
        """_summary_

        Parameters
        ----------
        ops : InferenceNode
            An inference node that represents the chain of operators for the ensemble.
        schema : Schema
            The schema of the input data.
        name : str, optional
            Name of the ensemble, by default "ensemble_model"
        label_columns : List[str], optional
            List of strings representing label columns, by default None
        """
        self.graph = Graph(ops)
        self.graph.construct_schema(schema)
        self.name = name
        self.label_columns = label_columns or []

    @property
    def input_schema(self):
        return self.graph.input_schema

    @property
    def output_schema(self):
        return self.graph.output_schema

    def save(self, path):
        """Save this ensemble to disk

        Parameters
        ----------
        path: str
            The path to save the ensemble to
        """
        fs = fsspec.get_fs_token_paths(path)[0]
        fs.makedirs(path, exist_ok=True)

        # TODO: Include the systems version in the metadata file below

        # generate a file of all versions used to generate this bundle
        with fs.open(fs.sep.join([path, "metadata.json"]), "w") as o:
            json.dump(
                {
                    "versions": {
                        "python": sys.version,
                    },
                    "generated_timestamp": int(time.time()),
                },
                o,
            )

        # dump out the full workflow (graph/stats/operators etc) using cloudpickle
        with fs.open(fs.sep.join([path, "ensemble.pkl"]), "wb") as o:
            cloudpickle.dump(self, o)

    @classmethod
    def load(cls, path) -> "Ensemble":
        """Load up a saved ensemble object from disk

        Parameters
        ----------
        path: str
            The path to load the ensemble from

        Returns
        -------
        Ensemble
            The ensemble loaded from disk
        """
        fs = fsspec.get_fs_token_paths(path)[0]

        # check version information from the metadata blob, and warn if we have a mismatch
        meta = json.load(fs.open(fs.sep.join([path, "metadata.json"])))

        def parse_version(version):
            return version.split(".")[:2]

        def check_version(stored, current, name):
            if parse_version(stored) != parse_version(current):
                warnings.warn(
                    f"Loading workflow generated with {name} version {stored} "
                    f"- but we are running {name} {current}. This might cause issues"
                )

        # make sure we don't have any major/minor version conflicts between the stored worklflow
        # and the current environment
        versions = meta["versions"]
        check_version(versions["python"], sys.version, "python")

        ensemble = cloudpickle.load(fs.open(fs.sep.join([path, "ensemble.pkl"]), "rb"))

        return ensemble

    def export(self, export_path, version=1, backend="ensemble"):
        """
        Write out an ensemble model configuration directory. The exported
        ensemble is designed for use with Triton Inference Server.
        """
        if backend == "ensemble":
            return _ensemble_export(self, export_path, version)
        elif backend == "executor":
            return _executor_export(self, export_path, version)
        else:
            raise ValueError(f"Unknown backend provided to `Ensemble.export()`: {backend}")


def _create_node_table(nodes, backend):
    exportable_node_idx = 0
    node_id_lookup = {}
    for node in nodes:
        if node.exportable(backend):
            node_id_lookup[node] = exportable_node_idx
            exportable_node_idx += 1

    return node_id_lookup, exportable_node_idx


def _ensemble_export(
    ensemble: Ensemble,
    path: str,
    version: int = 1,
):
    # Build node id lookup table
    nodes = list(postorder_iter_nodes(ensemble.graph.output_node))
    node_id_table, num_nodes = _create_node_table(nodes, "ensemble")

    nodes = nodes or []
    node_id_table = node_id_table or {}

    # Create ensemble config
    ensemble_config = model_config.ModelConfig(
        name=ensemble.name,
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

            node_config = node.export(path, node_id=node_id, version=version, backend="ensemble")
            if node_config is not None:
                node_configs.append(node_config)

            config_step = model_config.ModelEnsembling.Step(model_name=node_name, model_version=-1)

            for input_col_name, input_col_schema in node.input_schema.column_schemas.items():
                source = _find_column_source(
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
    ensemble_path = os.path.join(path, ensemble.name)
    os.makedirs(ensemble_path, exist_ok=True)
    os.makedirs(os.path.join(ensemble_path, str(version)), exist_ok=True)

    config_path = os.path.join(ensemble_path, "config.pbtxt")
    with open(config_path, "w", encoding="utf-8") as o:
        text_format.PrintMessage(ensemble_config, o)

    return (ensemble_config, node_configs)


def _executor_export(
    ensemble: Ensemble,
    path: str,
    version: int = 1,
):

    nodes = list(postorder_iter_nodes(ensemble.graph.output_node))
    node_id_table, _ = _create_node_table(nodes, "executor")

    node_configs = []
    for node in nodes:
        if node.exportable("executor"):
            node_id = node_id_table.get(node, None)

            node_config = node.export(path, node_id=node_id, version=version, backend="executor")
            if node_config is not None:
                node_configs.append(node_config)

    executor_config = _executor_model_export(path, "executor_model", ensemble)

    return (executor_config, node_configs)


def _executor_model_export(
    path: str,
    export_name: str,
    ensemble: Ensemble,
    params: dict = None,
    node_id: int = None,
    version: int = 1,
    backend: str = "python",
):
    """
    Export the class object as a config and all related files to the user-defined path.

    Parameters
    ----------
    path : str
        Artifact export path
    input_schema : Schema
        A schema with information about the inputs to this operator.
    output_schema : Schema
        A schema with information about the outputs of this operator.
    params : dict, optional
        Parameters dictionary of key, value pairs stored in exported config, by default None.
    node_id : int, optional
        The placement of the node in the graph (starts at 1), by default None.
    version : int, optional
        The version of the model, by default 1.

    Returns
    -------
    Ensemble_config: dict
        The config for the entire ensemble.
    Node_configs: list
        A list of individual configs for each step (operator) in graph.
    """

    params = params or {}

    node_name = f"{node_id}_{export_name}" if node_id is not None else export_name

    node_export_path = pathlib.Path(path) / node_name
    node_export_path.mkdir(parents=True, exist_ok=True)

    config = model_config.ModelConfig(name=node_name, backend=backend, platform="merlin_executor")

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
            "triton",
            "models",
            "executor_model.py",
        ),
        os.path.join(node_export_path, str(version), "model.py"),
    )

    ensemble.save(os.path.join(node_export_path, str(version), "ensemble"))

    return config


def _find_column_source(upstream_nodes, column_name, backend):
    source_node = None
    for upstream_node in upstream_nodes:
        if column_name in upstream_node.output_columns.names:
            source_node = upstream_node
            break

    if source_node and not source_node.exportable(backend):
        return _find_column_source(source_node.parents_with_dependencies, column_name, backend)
    else:
        return source_node
