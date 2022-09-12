from distutils.spawn import find_executable  # pylint: disable=deprecated-module

import pytest
from testbook import testbook

from merlin.systems.triton.utils import run_triton_server
from tests.conftest import REPO_ROOT

pytest.importorskip("implicit")
pytest.importorskip("merlin.models")

try:
    # pylint: disable=unused-import
    import cudf  # noqa

    _TRAIN_ON_GPU = [True, False]
except ImportError:
    _TRAIN_ON_GPU = [False]

TRITON_SERVER_PATH = find_executable("tritonserver")


@pytest.mark.notebook
@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize("gpu", _TRAIN_ON_GPU)
@testbook(REPO_ROOT / "examples/Serving-An-Implicit-Model-With-Merlin-Systems.ipynb", execute=False)
def test_example_serving_implicit(tb, gpu):
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
        """,
        pop=True,
    )
    NUM_OF_CELLS = len(tb.cells)
    if not gpu:
        tb.cells[
            6
        ].source = (
            "model = BayesianPersonalizedRanking(use_gpu=False)\nmodel.fit(train_transformed)"
        )
    tb.execute_cell(list(range(0, 16)))

    with run_triton_server("ensemble", grpc_port=8001):
        tb.execute_cell(list(range(16, NUM_OF_CELLS - 2)))
