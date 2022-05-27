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
from copy import deepcopy

import pytest
import xgboost as xgb

import merlin.systems.dag.ops.fil as fil_op
from merlin.dag import ColumnSelector, Graph
from merlin.schema import Schema

# this needs to be before any modules that import protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from google.protobuf import text_format  # noqa

model_config = pytest.importorskip("tritonclient.grpc.model_config_pb2")


def test_fil_op_exports_own_config(tmpdir):
    params = {"objective": "binary:logistic"}
    X = [[1, 2, 3]]
    y = [0, 1, 0]
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data)

    # Triton
    triton_op = fil_op.FILPredict(model)
    triton_op.export(tmpdir, None, None)

    # Export creates directory
    export_path = pathlib.Path(tmpdir) / triton_op.export_name
    assert export_path.exists()

    # Export creates the config file
    config_path = export_path / "config.pbtxt"
    assert config_path.exists()

    # Read the config file back in from proto
    with open(config_path, "rb") as f:
        config = model_config.ModelConfig()
        raw_config = f.read()
        parsed = text_format.Parse(raw_config, config)

        # The config file contents are correct
        assert parsed.name == triton_op.export_name
        assert parsed.backend == "fil"


def test_fil_op_compute_schema():
    params = {"objective": "binary:logistic"}
    X = [[1, 2, 3]]
    y = [0, 1, 0]
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data)

    # Triton
    triton_op = fil_op.FILPredict(model)

    out_schema = triton_op.compute_output_schema(
        Schema(["input__0"]), ColumnSelector(["input__0"]), None
    )
    assert out_schema.column_names == ["output__0"]


def test_fil_schema_validation():
    params = {"objective": "binary:logistic"}
    X = [[1, 2, 3]]
    y = [0, 1, 0]
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data)

    # Triton
    fil_node = [] >> fil_op.FILPredict(model)
    fil_graph = Graph(fil_node)

    with pytest.raises(ValueError) as exception_info:
        deepcopy(fil_graph).construct_schema(Schema([]))
    assert "Missing column 'input__0'" in str(exception_info.value)

    with pytest.raises(ValueError) as exception_info:
        deepcopy(fil_graph).construct_schema(Schema(["not_input"]))
    assert "Missing column 'input__0'" in str(exception_info.value)

    with pytest.raises(ValueError) as exception_info:
        deepcopy(fil_graph).construct_schema(Schema(["input__0", "not_input"]))
    assert "Mismatched dtypes for column 'input__0'" in str(exception_info.value)
