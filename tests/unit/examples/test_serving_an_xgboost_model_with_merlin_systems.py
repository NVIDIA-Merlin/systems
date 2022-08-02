from time import sleep

import pytest
import tritonclient.grpc as grpcclient
from testbook import testbook
from tritonclient.utils import InferenceServerException

from tests.conftest import REPO_ROOT

pytest.importorskip("tensorflow")
pytest.importorskip("merlin.models")
pytest.importorskip("xgboost")


@testbook(REPO_ROOT / "examples/Serving-An-XGboost-Model-With-Merlin-Systems.ipynb", execute=False)
def test_example_serving_xgboost(tb):
    NUM_OF_CELLS = len(tb.cells)
    tb.execute_cell(list(range(0, 22)))

    triton_client = grpcclient.InferenceServerClient(url="0.0.0.0:8001")
    while True:
        try:
            triton_client.is_server_live()
            break
        except InferenceServerException:
            sleep(1)

    tb.execute_cell(list(range(22, NUM_OF_CELLS)))
