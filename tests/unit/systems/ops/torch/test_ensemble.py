#
# Copyright (c) 2023, NVIDIA CORPORATION.
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

import shutil

import numpy as np
import pandas as pd
import pytest
import tritonclient.utils

from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.ops.pytorch import PredictPyTorch
from merlin.systems.triton.utils import run_ensemble_on_tritonserver

torch = pytest.importorskip("torch")

TRITON_SERVER_PATH = shutil.which("tritonserver")


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
def test_model_in_ensemble(tmpdir):
    class MyModel(torch.nn.Module):
        def forward(self, x):
            v = torch.stack(list(x.values())).sum(axis=0)
            return v

    model = MyModel()

    traced_model = torch.jit.trace(model, {"a": torch.tensor(1), "b": torch.tensor(2)}, strict=True)

    model_input_schema = Schema(
        [ColumnSchema("a", dtype="int64"), ColumnSchema("b", dtype="int64")]
    )
    model_output_schema = Schema([ColumnSchema("output", dtype="int64")])

    model_node = model_input_schema.column_names >> PredictPyTorch(
        traced_model, model_input_schema, model_output_schema
    )

    ensemble = Ensemble(model_node, model_input_schema)

    ensemble_config, _ = ensemble.export(str(tmpdir))

    df = pd.DataFrame({"a": [1], "b": [2]})

    response = run_ensemble_on_tritonserver(
        str(tmpdir), model_input_schema, df, ["output"], ensemble_config.name
    )
    np.testing.assert_array_equal(response["output"], np.array([3]))


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
def test_model_error(tmpdir):
    class MyModel(torch.nn.Module):
        def forward(self, x):
            v = torch.stack(list(x.values())).sum()
            return v

    model = MyModel()

    traced_model = torch.jit.trace(model, {"a": torch.tensor(1), "b": torch.tensor(2)}, strict=True)

    model_input_schema = Schema([ColumnSchema("a", dtype="int64")])
    model_output_schema = Schema([ColumnSchema("output", dtype="int64")])

    model_node = model_input_schema.column_names >> PredictPyTorch(
        traced_model, model_input_schema, model_output_schema
    )

    ensemble = Ensemble(model_node, model_input_schema)

    ensemble_config, _ = ensemble.export(str(tmpdir))

    # run inference with missing input (that was present when model was compiled)
    # we're expecting a KeyError at runtime.
    df = pd.DataFrame({"a": [1]})

    with pytest.raises(tritonclient.utils.InferenceServerException) as exc_info:
        run_ensemble_on_tritonserver(
            str(tmpdir), model_input_schema, df, ["output"], ensemble_config.name
        )
    assert "The following operation failed in the TorchScript interpreter" in str(exc_info.value)
    assert "RuntimeError: KeyError: b" in str(exc_info.value)
