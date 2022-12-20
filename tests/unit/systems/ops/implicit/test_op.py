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

import implicit
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from merlin.schema import Schema
from merlin.systems.dag.ops.implicit import PredictImplicit


@pytest.mark.parametrize(
    "model_cls",
    [
        implicit.bpr.BayesianPersonalizedRanking,
        implicit.als.AlternatingLeastSquares,
        implicit.lmf.LogisticMatrixFactorization,
    ],
)
def test_reload_from_config(model_cls, tmpdir):
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

    num_to_recommend = np.random.randint(1, n)
    user_items = None
    ids, scores = model.recommend(
        1, user_items, N=num_to_recommend, filter_already_liked_items=False
    )

    reloaded_ids, reloaded_scores = reloaded_model.recommend(
        1, user_items, N=num_to_recommend, filter_already_liked_items=False
    )

    np.testing.assert_array_equal(ids, reloaded_ids)
    np.testing.assert_array_equal(scores, reloaded_scores)
