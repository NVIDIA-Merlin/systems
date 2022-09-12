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

    def export(self, export_path, version=1):
        """
        Write out an ensemble model configuration directory. The exported
        ensemble is designed for use with Triton Inference Server.
        """
        # Create ensemble config
        ensemble_config = model_config.ModelConfig(
            name=self.name,
            platform="ensemble",
            # max_batch_size=configs[0].max_batch_size
        )

        for _, col_schema in self.graph.input_schema.column_schemas.items():
            add_model_param(
                ensemble_config.input,
                model_config.ModelInput,
                col_schema,
                col_schema.properties.get("shape", None) or compute_dims(col_schema),
            )

        for _, col_schema in self.graph.output_schema.column_schemas.items():
            add_model_param(
                ensemble_config.output,
                model_config.ModelOutput,
                col_schema,
                col_schema.properties.get("shape", None) or compute_dims(col_schema),
            )

        # Build node id lookup table
        postorder_nodes = list(postorder_iter_nodes(self.graph.output_node))

        node_idx = 0
        node_id_lookup = {}
        for node in postorder_nodes:
            if node.exportable:
                node_id_lookup[node] = node_idx
                node_idx += 1

        node_configs = []
        # Export node configs and add ensemble steps
        for node in postorder_nodes:
            if node.exportable:
                node_id = node_id_lookup.get(node, None)
                node_name = f"{node_id}_{node.export_name}"

                found = False
                for step in ensemble_config.ensemble_scheduling.step:
                    if step.model_name == node_name:
                        found = True
                if found:
                    continue

                node_config = node.export(export_path, node_id=node_id, version=version)

                config_step = model_config.ModelEnsembling.Step(
                    model_name=node_name, model_version=-1
                )

                for input_col_name, input_col_schema in node.input_schema.column_schemas.items():
                    source = _find_column_source(node.parents_with_dependencies, input_col_name)
                    source_id = node_id_lookup.get(source, None)
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
                        f"_{node_id}" if node_id is not None and node_id < node_idx - 1 else ""
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
                node_configs.append(node_config)

        # Write the ensemble config file
        ensemble_path = os.path.join(export_path, self.name)
        os.makedirs(ensemble_path, exist_ok=True)
        os.makedirs(os.path.join(ensemble_path, str(version)), exist_ok=True)

        config_path = os.path.join(ensemble_path, "config.pbtxt")
        with open(config_path, "w", encoding="utf-8") as o:
            text_format.PrintMessage(ensemble_config, o)

        return (ensemble_config, node_configs)


def _find_column_source(upstream_nodes, column_name):
    source_node = None
    for upstream_node in upstream_nodes:
        if column_name in upstream_node.output_columns.names:
            source_node = upstream_node
            break

    if source_node and not source_node.exportable:
        return _find_column_source(source_node.parents_with_dependencies, column_name)
    else:
        return source_node
