import pytest
from testbook import testbook

from merlin.systems.triton.utils import run_triton_server
from tests.conftest import REPO_ROOT

pytest.importorskip("tensorflow")
pytest.importorskip("merlin.models")
pytest.importorskip("xgboost")


@testbook(REPO_ROOT / "examples/Serving-An-XGboost-Model-With-Merlin-Systems.ipynb", execute=False)
def test_example_serving_xgboost(tb):
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
    # TODO: the following line is a hack -- remove when merlin-models#624 gets fixed
    tb.cells[4].source = tb.cells[4].source.replace(
        "without(['rating_binary', 'title'])", "without(['rating_binary', 'title', 'userId_count'])"
    )
    tb.execute_cell(list(range(0, 14)))

    with run_triton_server("ensemble", grpc_port=8001):
        tb.execute_cell(list(range(14, NUM_OF_CELLS)))
