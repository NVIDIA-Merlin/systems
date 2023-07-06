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
from typing import Dict

import numpy as np
import pytest

from merlin.dag.base_runtime import Runtime
from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ensemble import Ensemble
from merlin.table import TensorTable

torch = pytest.importorskip("torch")
ptorch_op = pytest.importorskip("merlin.systems.dag.ops.pytorch")


class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, input_dict: Dict[str, torch.Tensor]):
        linear_out = self.linear(input_dict["input"].to(self.linear.weight.device))

        return linear_out


model = CustomModel()
model_scripted = torch.jit.script(model)

model_input_schema = Schema(
    [
        ColumnSchema(
            "input",
            properties={"value_count": {"min": 3, "max": 3}},
            dtype=np.float32,
            is_list=True,
            is_ragged=False,
        )
    ]
)
model_output_schema = Schema([ColumnSchema("OUTPUT__0", dtype=np.float32)])


@pytest.mark.parametrize("torchscript", [True])
@pytest.mark.parametrize("use_path", [True, False])
@pytest.mark.parametrize("runtime", [(Runtime())])
def test_pytorch_op_serving(tmpdir, use_path, torchscript, runtime):
    model_path = str(tmpdir / "model.pt")

    model_to_use = model_scripted if torchscript else model
    model_or_path = model_path if use_path else model_to_use

    if use_path:
        try:
            # jit-compiled version of a model
            model_to_use.save(model_path)
        except AttributeError:
            # non-jit-compiled version of a model
            torch.save(model_to_use, model_path)

    predictions = ["input"] >> ptorch_op.PredictPyTorch(
        model_or_path, model_input_schema, model_output_schema
    )
    ensemble = Ensemble(predictions, model_input_schema)

    input_data = {"input": np.array([[2.0, 3.0, 4.0], [4.0, 8.0, 1.0]]).astype(np.float32)}

    inputs = TensorTable(input_data)
    response = ensemble.transform(inputs, runtime=runtime)

    assert response["OUTPUT__0"].values.shape[0] == input_data["input"].shape[0]


@pytest.mark.parametrize("torchscript", [True])
@pytest.mark.parametrize("use_path", [True, False])
@pytest.mark.parametrize("runtime", [(Runtime())])
def test_pytorch_op_serving_python(tmpdir, use_path, torchscript, runtime):
    model_path = str(tmpdir / "model.pt")

    model_to_use = model_scripted if torchscript else model
    model_or_path = model_path if use_path else model_to_use

    if use_path:
        try:
            # jit-compiled version of a model
            model_to_use.save(model_path)
        except AttributeError:
            # non-jit-compiled version of a model
            torch.save(model_to_use, model_path)

    predictions = ["input"] >> ptorch_op.PredictPyTorch(
        model_or_path, model_input_schema, model_output_schema
    )
    ensemble = Ensemble(predictions, model_input_schema)

    input_data = {"input": np.array([[2.0, 3.0, 4.0], [4.0, 8.0, 1.0]]).astype(np.float32)}

    inputs = TensorTable(input_data)
    response = ensemble.transform(inputs, runtime=runtime)
    assert response["OUTPUT__0"].values.shape[0] == input_data["input"].shape[0]
