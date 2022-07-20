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
from inspect import signature

import pytest

from merlin.systems.model_registry import ModelRegistry

# this needs to be before any modules that import protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from unittest.mock import MagicMock, patch  # noqa

from google.protobuf import text_format  # noqa

from merlin.schema import Schema  # noqa
from merlin.systems.dag.ops.operator import InferenceOperator  # noqa
from nvtabular import Workflow  # noqa
from nvtabular import ops as wf_ops  # noqa

ensemble = pytest.importorskip("merlin.systems.dag.ensemble")
model_config = pytest.importorskip("tritonclient.grpc.model_config_pb2")
workflow_op = pytest.importorskip("merlin.systems.dag.ops.workflow")


@pytest.mark.parametrize("engine", ["parquet"])
def test_workflow_op_validates_schemas(dataset, engine):
    input_columns = ["x", "y", "id"]
    request_schema = Schema(input_columns)

    # NVT
    workflow_ops = input_columns >> wf_ops.Rename(postfix="_nvt")
    workflow = Workflow(workflow_ops)
    workflow.fit(dataset)

    # Triton
    triton_ops = ["a", "b", "c"] >> workflow_op.TransformWorkflow(workflow)

    with pytest.raises(ValueError) as exc_info:
        ensemble.Ensemble(triton_ops, request_schema)
        assert "Missing column" in str(exc_info.value)


@pytest.mark.parametrize("engine", ["parquet"])
def test_workflow_op_exports_own_config(tmpdir, dataset, engine):
    input_columns = ["x", "y", "id"]

    # NVT
    workflow_ops = input_columns >> wf_ops.Rename(postfix="_nvt")
    workflow = Workflow(workflow_ops)
    workflow.fit(dataset)

    # Triton
    triton_op = workflow_op.TransformWorkflow(workflow)
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
        assert parsed.backend == "python"


@patch("requests.get")
def test_from_model_registry__forClassesAcceptingModelPaths_works(mock_req, tmpdir):

    registry = ModelRegistry()
    registry.get_artifact_uri = MagicMock(return_value=tmpdir)

    # Make a new subclass of InferenceOperator with the right __init__ signature
    class InferenceOperatorWithModelPath(InferenceOperator):
        def __init__(self, model_or_path):
            pass

    # This is a bit janky, but because we are checking that `model_or_path` is a parameter
    # of __init__ (to ensure the subclass of InferenceOperator accepts a model path), we need
    # to mock the __init__ function but first copy its function signature, then re-assign that
    # signature to the mock.
    init_sig = signature(InferenceOperatorWithModelPath.__init__)
    InferenceOperatorWithModelPath.__init__ = MagicMock(return_value=None)
    InferenceOperatorWithModelPath.__init__.__signature__ = init_sig

    # Now we can call from_model_registry and assert that __init__ was called with the proper
    # model path.
    InferenceOperatorWithModelPath.from_model_registry(registry)

    InferenceOperatorWithModelPath.__init__.assert_called_with(tmpdir)


@patch("requests.get")
def test_from_model_registry__forClassesNotAcceptingModelPaths_throws(mock_req, tmpdir):
    # the base InferenceOperator does not expect a model_or_path parameter, and should throw
    # a TypeError

    with pytest.raises(TypeError):
        InferenceOperator.from_model_registry(ModelRegistry())
