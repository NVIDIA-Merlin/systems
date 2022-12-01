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

import tritonclient.grpc.model_config_pb2 as model_config  # noqa
from google.protobuf import text_format  # noqa

from merlin.core.protocols import Transformable  # noqa
from merlin.dag import ColumnSelector  # noqa
from merlin.schema import Schema  # noqa
from merlin.systems.dag.ops import compute_dims  # noqa
from merlin.systems.dag.ops.operator import add_model_param  # noqa
from merlin.systems.dag.runtimes.triton.ops.operator import TritonOperator
from merlin.systems.triton.conversions import (  # noqa
    dict_array_to_triton_request,
    triton_response_to_dict_array,
)


class PredictPyTorchTriton(TritonOperator):
    """
    This operator takes a pytorch model and packages it correctly for tritonserver
    to run, on the pytorch backend.
    """

    def __init__(self, op):
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

        super().__init__(op)
        self._torch_model_name = None
        self.input_schema = op.input_schema
        self.output_schema = op.output_schema

    @property
    def torch_model_name(self):
        return self._torch_model_name

    def set_torch_model_name(self, torch_model_name):
        """
        Set the name of the Triton model to use

        Parameters
        ----------
        torch_model_name : str
            Triton model directory name
        """
        self._torch_model_name = torch_model_name

    def transform(self, col_selector: ColumnSelector, transformable: Transformable):
        inference_request = dict_array_to_triton_request(
            self.torch_model_name,
            transformable,
            self.input_schema.column_names,
            self.output_schema.column_names,
        )

        inference_response = inference_request.exec()

        return triton_response_to_dict_array(
            inference_response, type(transformable), self.output_schema.column_names
        )

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
        export_name = self.__class__.__name__.lower()
        node_name = f"{node_id}_{export_name}" if node_id is not None else export_name

        node_export_path = pathlib.Path(path) / node_name
        node_export_path.mkdir(exist_ok=True)

        export_model_path = pathlib.Path(node_export_path) / str(version)
        export_model_path.mkdir(exist_ok=True)

        if self.op.path:
            copyfile(
                str(self.op.path),
                export_model_path / "model.pt",
            )
        else:
            self.op.model.save(export_model_path / "model.pt")

        self.set_torch_model_name(node_name)
        backend_model_config = self._export_model_config(node_name, node_export_path)
        return backend_model_config

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
                compute_dims(col_schema),
            )

        for _, col_schema in self.output_schema.column_schemas.items():
            add_model_param(
                config.output,
                model_config.ModelOutput,
                col_schema,
                compute_dims(col_schema),
            )

        with open(os.path.join(output_path, "config.pbtxt"), "w", encoding="utf-8") as o:
            text_format.PrintMessage(config, o)
        return config

    @property
    def scalar_shape(self):
        return []
