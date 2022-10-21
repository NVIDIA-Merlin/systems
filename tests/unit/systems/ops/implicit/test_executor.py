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
from distutils.spawn import find_executable

import implicit
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from merlin.core.dispatch import make_df
from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.ops.implicit import PredictImplicit

TRITON_SERVER_PATH = find_executable("tritonserver")

tritonclient = pytest.importorskip("tritonclient")
grpcclient = pytest.importorskip("tritonclient.grpc")

from merlin.systems.triton.utils import run_ensemble_on_tritonserver  # noqa


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
def test_implicit_in_triton_executor_model(tmpdir):
    model = implicit.bpr.BayesianPersonalizedRanking()
    n = 100
    user_items = csr_matrix(np.random.choice([0, 1], size=n * n, p=[0.9, 0.1]).reshape(n, n))
    model.fit(user_items)

    request_schema = Schema([ColumnSchema("user_id", dtype="int64")])

    implicit_op = PredictImplicit(model, num_to_recommend=10)
    triton_chain = request_schema.column_names >> implicit_op

    ensemble = Ensemble(triton_chain, request_schema)
    ensemble.export(tmpdir, backend="executor")

    input_user_id = np.array([0, 1], dtype=np.int64)

    response = run_ensemble_on_tritonserver(
        tmpdir,
        request_schema,
        make_df({"user_id": input_user_id}),
        ensemble.output_schema.column_names,
        "executor_model",
    )
    assert response is not None
    assert len(response["ids"]) == len(input_user_id)
    assert len(response["scores"]) == len(input_user_id)
    assert len(response["ids"][0]) == 10
    assert len(response["scores"][0]) == 10
