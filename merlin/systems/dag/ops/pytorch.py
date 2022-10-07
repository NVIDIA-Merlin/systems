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
from shutil import copyfile

# this needs to be before any modules that import protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch  # noqa
import tritonclient.grpc.model_config_pb2 as model_config  # noqa
from google.protobuf import text_format  # noqa

from merlin.dag import ColumnSelector  # noqa
from merlin.schema import Schema  # noqa
from merlin.systems.dag.ops import compute_dims  # noqa
from merlin.systems.dag.ops.operator import InferenceOperator, add_model_param  # noqa


class PredictPyTorch(InferenceOperator):
    """
    This operator takes a pytorch model and packages it correctly for tritonserver
    to run, on the pytorch backend.
    """

    def __init__(
        self,
        model_or_path,
        input_schema: Schema,
        output_schema: Schema,
    ):
        """
        Instantiate a PredictPyTorch inference operator.

        Parameters
        ----------
        model_or_path : PyTorch model or string
            This can be a pytorch model or a path to a pytorch model.
        input_schema : Schema
            Input schema for the pytorch model. This could be the output schema of the NVTabular
            workflow that produced your training data.
        output_schema : Schema
            Output schema for the pytorch model.
        """

        if isinstance(model_or_path, (str, os.PathLike)):
            self.path = model_or_path
            self.model = torch.load(self.path)
        else:
            self.path = None
            self.model = model_or_path

        # TODO: figure out if we can infer input / output schemas from the pytorch model. Now we
        # just make them parameters.
        self.input_schema = input_schema
        self.output_schema = output_schema

        # This is a hack to let us store the shapes for the ensemble to use
        for col_name, col_schema in self.input_schema.column_schemas.items():
            self.input_schema[col_name] = col_schema.with_properties(
                {"shape": compute_dims(col_schema, self.scalar_shape)}
            )

        for col_name, col_schema in self.output_schema.column_schemas.items():
            self.output_schema[col_name] = col_schema.with_properties(
                {"shape": compute_dims(col_schema, self.scalar_shape)}
            )

        super().__init__()

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:
        """
        Use the input schema supplied during object creation.
        """
        return self.input_schema

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        """
        Use the output schema supplied during object creation.
        """
        return self.output_schema

    def export(self, path, input_schema, output_schema, node_id=None, version=1):
        """Create a directory inside supplied path based on our export name"""
        node_name = f"{node_id}_{self.export_name}" if node_id is not None else self.export_name

        node_export_path = pathlib.Path(path) / node_name
        node_export_path.mkdir(exist_ok=True)

        export_model_path = pathlib.Path(node_export_path) / str(version)
        export_model_path.mkdir(exist_ok=True)

        if self.path:
            copyfile(
                str(self.path),
                export_model_path / "model.pt",
            )
        else:
            self.model.save(export_model_path / "model.pt")

        return self._export_model_config(node_name, node_export_path)

    def _export_model_config(self, name, output_path):
        """Exports a PyTorch model for serving with Triton

        Parameters
        ----------
        name:
            The name of the triton model to export
        output_path:
            The path to write the exported model to
        """
        config = self._export_torchscript_config(name, output_path)
        # TODO: Add support for Python back-end configs here

        return config

    def _export_torchscript_config(self, name, output_path):
        """Exports a PyTorch model for serving with Triton

        Parameters
        ----------
        name:
            The name of the triton model to export
        output_path:
            The path to write the exported model to
        """
        config = model_config.ModelConfig(name=name)

        config.backend = "pytorch"
        config.platform = "pytorch_libtorch"
        config.parameters["INFERENCE_MODE"].string_value = "true"

        for _, col_schema in self.input_schema.column_schemas.items():
            add_model_param(
                config.input,
                model_config.ModelInput,
                col_schema,
                col_schema.properties["shape"],
            )

        for _, col_schema in self.output_schema.column_schemas.items():
            add_model_param(
                config.output,
                model_config.ModelOutput,
                col_schema,
                col_schema.properties["shape"],
            )

        with open(os.path.join(output_path, "config.pbtxt"), "w", encoding="utf-8") as o:
            text_format.PrintMessage(config, o)
        return config

    @property
    def scalar_shape(self):
        return []
