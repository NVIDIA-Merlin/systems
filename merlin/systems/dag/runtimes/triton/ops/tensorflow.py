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
from shutil import copytree

# this needs to be before any modules that import protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tritonclient.grpc.model_config_pb2 as model_config  # noqa
from google.protobuf import text_format  # noqa

from merlin.core.protocols import Transformable  # noqa
from merlin.dag import ColumnSelector  # noqa
from merlin.schema import Schema  # noqa
from merlin.systems.dag.ops import compute_dims  # noqa
from merlin.systems.dag.ops.operator import add_model_param  # noqa
from merlin.systems.dag.runtimes.triton.ops.operator import TritonOperator  # noqa
from merlin.systems.triton.conversions import (  # noqa
    dict_array_to_triton_request,
    triton_response_to_dict_array,
)


class PredictTensorflowTriton(TritonOperator):
    """TensorFlow Model Prediction Operator for running inside Triton."""

    def __init__(self, op):
        super().__init__(op)

        self.input_schema = op.input_schema
        self.output_schema = op.output_schema
        self.path = op.path
        self.model = op.model

        self._tf_model_name = None

    def transform(self, col_selector: ColumnSelector, transformable: Transformable):
        """Run transform of operator callling TensorFlow model with a Triton InferenceRequest.

        Returns
        -------
        Transformable
            TensorFlow Model Outputs
        """
        # TODO: Validate that the inputs match the schema
        # TODO: Should we coerce the dtypes to match the schema here?
        inference_request = dict_array_to_triton_request(
            self.tf_model_name,
            transformable,
            self.input_schema.column_names,
            self.output_schema.column_names,
        )
        inference_response = inference_request.exec()

        # TODO: Validate that the outputs match the schema
        return triton_response_to_dict_array(
            inference_response, type(transformable), self.output_schema.column_names
        )

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
