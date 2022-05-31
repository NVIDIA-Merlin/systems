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

import merlin.systems.dag.ops.fil as fil_op
from merlin.dag import ColumnSelector, Graph
from merlin.schema import Schema

# this needs to be before any modules that import protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from google.protobuf import text_format  # noqa

model_config = pytest.importorskip("tritonclient.grpc.model_config_pb2")


def export_op(export_dir, triton_op) -> model_config.ModelConfig:
    triton_op.export(export_dir, None, None)

    # Export creates directory
    export_path = pathlib.Path(export_dir) / triton_op.export_name
    assert export_path.exists()

    # Export creates the config file
    config_path = export_path / "config.pbtxt"
    assert config_path.exists()

    # Read the config file back in from proto
    with open(config_path, "rb") as f:
        config = model_config.ModelConfig()
        raw_config = f.read()
        parsed = text_format.Parse(raw_config, config)
        return parsed


def test_xgboost_multiclass(tmpdir, xgboost_mutli_classifier):
    triton_op = fil_op.FIL(xgboost_mutli_classifier, threshold=0.75)
    config = export_op(tmpdir, triton_op)
    assert config.parameters["model_type"].string_value == "xgboost_json"
    assert config.parameters["output_class"].string_value == "true"
    assert config.parameters["predict_proba"].string_value == "false"
    assert config.parameters["threshold"].string_value == "0.7500"


def test_proba(tmpdir, xgboost_mutli_classifier):
    triton_op = fil_op.FIL(xgboost_mutli_classifier, predict_proba=True)
    config = export_op(tmpdir, triton_op)
    assert config.parameters["model_type"].string_value == "xgboost_json"
    assert config.parameters["output_class"].string_value == "true"
    assert config.parameters["predict_proba"].string_value == "true"


def test_fil_op_exports_own_config(tmpdir, xgboost_binary_classifier):
    model = xgboost_binary_classifier

    # Triton
    triton_op = fil_op.FIL(model)
    config = export_op(tmpdir, triton_op)

    assert config.name == triton_op.export_name
    assert config.backend == "fil"
    assert config.max_batch_size == 8192
    assert config.input[0].name == "input__0"
    assert config.output[0].name == "output__0"
    assert config.output[0].dims == [1]


def test_fil_op_compute_schema(xgboost_binary_classifier):
    model = xgboost_binary_classifier

    # Triton
    triton_op = fil_op.FIL(model)

    out_schema = triton_op.compute_output_schema(
        Schema(["input__0"]), ColumnSelector(["input__0"]), None
    )
    assert out_schema.column_names == ["output__0"]


def test_fil_schema_validation(xgboost_binary_classifier):
    model = xgboost_binary_classifier

    # Triton
    fil_node = [] >> fil_op.FIL(model)
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
