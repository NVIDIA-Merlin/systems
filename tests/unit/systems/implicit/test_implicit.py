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
import json
from distutils.spawn import find_executable

import implicit
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.ops.implicit import PredictImplicit
from tests.unit.systems.utils.triton import _run_ensemble_on_tritonserver  # noqa

TRITON_SERVER_PATH = find_executable("tritonserver")


@pytest.mark.parametrize(
    "model_cls",
    [
        implicit.bpr.BayesianPersonalizedRanking,
        implicit.als.AlternatingLeastSquares,
        implicit.lmf.LogisticMatrixFactorization,
    ],
)
def test_predict_implcit(model_cls, tmpdir):
    model = model_cls()
    n = 10
    user_items = csr_matrix(np.random.choice([0, 1], size=n * n).reshape(n, n))
    model.fit(user_items)

    op = PredictImplicit(model)

    config = op.export(tmpdir, Schema(), Schema())

    node_config = json.loads(config.parameters[config.name].string_value)

    cls = PredictImplicit.from_config(
        node_config,
        model_repository=tmpdir,
        model_name=config.name,
        model_version=1,
    )
    reloaded_model = cls.model

    ids, scores = model.recommend(1, None, 10, filter_already_liked_items=False)

    reloaded_ids, reloaded_scores = reloaded_model.recommend(
        1, None, 10, filter_already_liked_items=False
    )

    np.testing.assert_array_equal(ids, reloaded_ids)
    np.testing.assert_array_equal(scores, reloaded_scores)


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize(
    "model_cls",
    [
        implicit.bpr.BayesianPersonalizedRanking,
        implicit.als.AlternatingLeastSquares,
        implicit.lmf.LogisticMatrixFactorization,
    ],
)
def test_ensemble(model_cls, tmpdir):
    model = model_cls()
    n = 10
    user_items = csr_matrix(np.random.choice([0, 1], size=n * n).reshape(n, n))
    model.fit(user_items)

    ids, scores = model.recommend(1, None, 10, filter_already_liked_items=False)

    implicit_op = PredictImplicit(model)

    input_schema = Schema([ColumnSchema("user_id", dtype="int64")])

    triton_chain = input_schema.column_names >> implicit_op

    triton_ens = Ensemble(triton_chain, input_schema)
    triton_ens.export(tmpdir)

    request_df = pd.DataFrame({"user_id": [1]})
    response = _run_ensemble_on_tritonserver(
        str(tmpdir), ["ids", "scores"], request_df, triton_ens.name
    )
    response_ids = response.as_numpy("ids")
    response_scores = response.as_numpy("scores")

    np.testing.assert_array_equal(ids, response_ids.ravel())
    np.testing.assert_array_equal(scores, response_scores.ravel())
