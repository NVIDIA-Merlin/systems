import pytest
from testbook import testbook

from merlin.systems.triton.utils import run_triton_server
from tests.conftest import REPO_ROOT

pytest.importorskip("merlin.models")


@pytest.mark.notebook
@testbook(REPO_ROOT / "examples/Serving-An-Implicit-Model-With-Merlin-Systems.ipynb", execute=False)
def test_example_serving_implicit(tb):
    tb.inject(
        """
        from unittest.mock import patch
        from merlin.datasets.synthetic import generate_data
        mock_train, mock_valid = generate_data(
            input="movielens-100k",
            num_rows=1000,
            set_sizes=(0.8, 0.2)
        )
        p1 = patch(
            "merlin.datasets.entertainment.get_movielens",
            return_value=[mock_train, mock_valid]
        )
        p1.start()
        """
    )
    NUM_OF_CELLS = len(tb.cells)
    tb.execute_cell(list(range(0, 14)))
    # import pdb
    # pdb.set_trace()

    with run_triton_server("ensemble", grpc_port=8001):
        tb.execute_cell(list(range(14, NUM_OF_CELLS - 3)))
