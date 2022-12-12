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

import numpy as np
import torch  # noqa
import torch.utils.dlpack  # noqa

from merlin.core.protocols import Transformable  # noqa
from merlin.dag import ColumnSelector  # noqa
from merlin.schema import Schema  # noqa
from merlin.systems.dag.dictarray import DictArray
from merlin.systems.dag.ops.operator import PipelineableInferenceOperator  # noqa


class PredictPyTorch(PipelineableInferenceOperator):
    """
    This operator takes a pytorch model and packages it correctly for tritonserver
    to run, on the pytorch backend.
    """

    def __init__(self, model_or_path, input_schema: Schema, output_schema: Schema, backend="torch"):
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

        super().__init__()
        self._torch_model_name = None
        self.input_schema = input_schema
        self.output_schema = output_schema

        if model_or_path is not None:
            if isinstance(model_or_path, (str, os.PathLike)):
                self.path = model_or_path
                self.model = torch.load(self.path)
            else:
                self.path = None
                self.model = model_or_path

            # This is a hack to enable the ensemble to use the same shape as this
            # these lines mutate the input and output schema with the additional property
            # `triton_scalar_shape` which represents the expected shape for this feature
            for col_name, col_schema in self.input_schema.column_schemas.items():
                self.input_schema[col_name] = col_schema.with_properties(
                    {"triton_scalar_shape": self.scalar_shape}
                )

            for col_name, col_schema in self.output_schema.column_schemas.items():
                self.output_schema[col_name] = col_schema.with_properties(
                    {"triton_scalar_shape": self.scalar_shape}
                )

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "model"}

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

    def transform(self, col_selector: ColumnSelector, transformable: Transformable):
        tensor_dict = {}
        for column in transformable.columns:
            tensor_dict[column] = torch.from_numpy(np.squeeze(transformable[column].values))

        result = self.model(tensor_dict)
        output = {}
        for idx, col in enumerate(self.output_schema.column_names):
            output[col] = result[:, idx].detach().numpy()
        return DictArray(output)

    @property
    def scalar_shape(self):
        return []
