import pytest
from testbook import testbook

from merlin.systems.triton.utils import run_triton_server
from tests.conftest import REPO_ROOT

pytest.importorskip("tensorflow")
pytest.importorskip("merlin.models")
pytest.importorskip("xgboost")


@testbook(REPO_ROOT / "examples/Serving-An-XGboost-Model-With-Merlin-Systems.ipynb", execute=False)
def test_example_serving_xgboost(tb):
    NUM_OF_CELLS = len(tb.cells)
    tb.execute_cell(list(range(0, 19)))

    with run_triton_server("ensemble"):
        tb.execute_cell(list(range(19, NUM_OF_CELLS)))
