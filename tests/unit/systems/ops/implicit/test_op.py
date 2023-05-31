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

import shutil

import implicit
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from tritonclient import grpc as grpcclient

from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.ops.implicit import PredictImplicit
from merlin.systems.dag.runtimes.triton import TritonExecutorRuntime
from merlin.systems.triton.utils import run_triton_server

TRITON_SERVER_PATH = shutil.which("tritonserver")


triton = pytest.importorskip("merlin.systems.triton")


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize("runtime", [None, TritonExecutorRuntime()])
def test_als(tmpdir, runtime):
    run_ensemble_test(implicit.als.AlternatingLeastSquares, runtime, tmpdir)


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize("runtime", [None, TritonExecutorRuntime()])
def test_lmf(tmpdir, runtime):
    run_ensemble_test(implicit.lmf.LogisticMatrixFactorization, runtime, tmpdir)


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize("runtime", [None, TritonExecutorRuntime()])
def test_bpr(tmpdir, runtime):
    run_ensemble_test(implicit.bpr.BayesianPersonalizedRanking, runtime, tmpdir)


def run_ensemble_test(model_cls, runtime, model_repository):
    model = model_cls()
    n = 100
    user_items = csr_matrix(np.random.choice([0, 1], size=n * n, p=[0.9, 0.1]).reshape(n, n))
    model.fit(user_items)

    num_to_recommend = np.random.randint(1, n)

    user_items = None
    ids, scores = model.recommend(
        [0, 1], user_items, N=num_to_recommend, filter_already_liked_items=False
    )

    implicit_op = PredictImplicit(model, num_to_recommend=num_to_recommend)

    input_schema = Schema([ColumnSchema("user_id", dtype="int64", dims=(None, 1))])

    triton_chain = input_schema.column_names >> implicit_op

    triton_ens = Ensemble(triton_chain, input_schema)
    ensemble_config, _ = triton_ens.export(model_repository, runtime=runtime)

    input_user_id = np.array([[0], [1]], dtype=np.int64)
    inputs = [
        grpcclient.InferInput(
            "user_id", input_user_id.shape, triton.np_to_triton_dtype(input_user_id.dtype)
        ),
    ]
    inputs[0].set_data_from_numpy(input_user_id)
    outputs = [grpcclient.InferRequestedOutput("scores"), grpcclient.InferRequestedOutput("ids")]

    response = None

    with run_triton_server(model_repository) as client:
        response = client.infer(ensemble_config.name, inputs, outputs=outputs)

    response_ids = response.as_numpy("ids")
    response_scores = response.as_numpy("scores")

    np.testing.assert_array_equal(ids, response_ids)
    np.testing.assert_array_equal(scores, response_scores)
