from unittest.mock import patch

import requests

from merlin.systems.model_registry import MLFlowModelRegistry


@patch("requests.get")
def test_mlflowregistry(mock_req, tmpdir):
    resp = requests.models.Response()
    resp.status_code = 200
    resp.json = lambda: {"artifact_uri": tmpdir}
    mock_req.return_value = resp

    registry = MLFlowModelRegistry("name", "version", "http://host:123")
    assert registry.artifact_uri() == tmpdir
