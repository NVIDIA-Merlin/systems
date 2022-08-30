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
from unittest.mock import MagicMock, patch

import pytest
import requests

# this needs to be before any modules that import protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from merlin.systems.dag.ops.operator import InferenceOperator  # noqa
from merlin.systems.model_registry import MLFlowModelRegistry, ModelRegistry  # noqa

ensemble = pytest.importorskip("merlin.systems.dag.ensemble")
workflow_op = pytest.importorskip("merlin.systems.dag.ops.workflow")


def test_from_model_registry_loads_model_from_path(tmpdir):
    class SimpleModelRegistry(ModelRegistry):
        def get_artifact_uri(self) -> str:
            return tmpdir

    registry = SimpleModelRegistry()

    # Make a new subclass of InferenceOperator so the mocks don't interfere with other tests.
    class InferenceOperatorWithModelPath(InferenceOperator):
        pass

    InferenceOperatorWithModelPath.from_path = MagicMock(return_value=None)

    # Now we can call from_model_registry and assert that from_path was called with the
    # proper model path.
    InferenceOperatorWithModelPath.from_model_registry(registry)
    InferenceOperatorWithModelPath.from_path.assert_called_with(tmpdir)


@patch("requests.get")
def test_mlflowregistry(mock_req, tmpdir):
    resp = requests.models.Response()
    resp.status_code = 200
    resp.json = lambda: {"artifact_uri": tmpdir}
    mock_req.return_value = resp

    registry = MLFlowModelRegistry("name", "version", "http://host:123")
    assert registry.get_artifact_uri() == tmpdir
