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
import json
import os
import pathlib
from shutil import copyfile
from typing import List

import tritonclient.grpc.model_config_pb2 as model_config  # noqa
from google.protobuf import text_format  # noqa

from merlin.schema import Schema
from merlin.systems.dag.ops import compute_dims  # noqa
from merlin.systems.dag.ops.operator import InferenceOperator
from merlin.systems.triton.export import _convert_dtype  # noqa


class TritonOperator:
    """Base class for Triton operators."""

    def __init__(self, base_op: InferenceOperator):
        """Construct TritonOperator from a base operator.

        Parameters
        ----------
        base_op : merlin.systems.dag.ops.operator.InfereneOperator
            Base operator used to construct this Triton op.
        """
        self.op = base_op

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
        return ["ensemble", "executor"]

    def export(
        self,
        path: str,
        input_schema: Schema,
        output_schema: Schema,
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

        node_name = f"{node_id}_{self.export_name}" if node_id is not None else self.export_name

        node_export_path = pathlib.Path(path) / node_name
        node_export_path.mkdir(parents=True, exist_ok=True)

        config = model_config.ModelConfig(name=node_name, backend=backend, platform="op_runner")

        config.parameters["operator_names"].string_value = json.dumps([node_name])

        config.parameters[node_name].string_value = json.dumps(
            {
                "module_name": self.__class__.__module__,
                "class_name": self.__class__.__name__,
                "input_dict": json.dumps(_schema_to_dict(input_schema)),
                "output_dict": json.dumps(_schema_to_dict(output_schema)),
                "params": json.dumps(params),
            }
        )

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
            "merlin.systems.triton.models", "oprunner_model.py"
        ) as oprunner_model:
            copyfile(
                oprunner_model,
                os.path.join(node_export_path, str(version), "model.py"),
            )

        return config


def _schema_to_dict(schema: Schema) -> dict:
    # TODO: Write the conversion
    schema_dict = {}
    for col_name, col_schema in schema.column_schemas.items():
        schema_dict[col_name] = {
            "dtype": col_schema.dtype.name,
            "is_list": col_schema.is_list,
            "is_ragged": col_schema.is_ragged,
        }

    return schema_dict


def add_model_param(params, paramclass, col_schema, dims=None):
    if col_schema.is_list and col_schema.is_ragged:
        params.append(
            paramclass(
                name=col_schema.name + "__values",
                data_type=_convert_dtype(col_schema.dtype),
                dims=dims,
            )
        )
        params.append(
            paramclass(
                name=col_schema.name + "__nnzs", data_type=model_config.TYPE_INT32, dims=dims
            )
        )
    else:
        params.append(
            paramclass(name=col_schema.name, data_type=_convert_dtype(col_schema.dtype), dims=dims)
        )
