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
import cudf
import numpy as np
import nvtabular as nvt
import pytest
import tensorflow as tf
from merlin.core.dispatch import make_df
from nvtabular import ColumnSchema, Schema

from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.ops.session_filter import FilterCandidates
from merlin.systems.dag.ops.softmax_sampling import SoftmaxSampling
from merlin.systems.dag.ops.tensorflow import PredictTensorflow
from merlin.systems.dag.ops.unroll_features import UnrollFeatures
from tests.unit.systems.inference_utils import _run_ensemble_on_tritonserver

feast = pytest.importorskip("feast")
faiss = pytest.importorskip("faiss")

from merlin.systems.dag.ops.faiss import QueryFaiss, setup_faiss  # noqa
from merlin.systems.dag.ops.feast import QueryFeast  # noqa


def test_poc_ensemble(tmpdir):
    request_schema = Schema(
        [
            ColumnSchema("user_id", dtype=np.int64),
        ]
    )

    base_path = "/raid/workshared/systems/examples"
    faiss_index_path = tmpdir + "/index.faiss"
    feast_repo_path = base_path + "/feature_repo/"
    retrieval_model_path = base_path + "/query_tower/"
    ranking_model_path = base_path + "/dlrm/"

    item_embeddings = np.ascontiguousarray(
        cudf.read_parquet(base_path + "/item_embeddings").to_numpy()
    )

    feature_store = feast.FeatureStore(feast_repo_path)
    setup_faiss(item_embeddings, str(faiss_index_path))

    user_features = ["user_id"] >> QueryFeast.from_feature_view(
        store=feature_store,
        path=feast_repo_path,
        view="user_features",
        column="user_id",
        include_id=True
    )

    retrieval = (
        user_features
        >> PredictTensorflow(retrieval_model_path)
        >> QueryFaiss(faiss_index_path, topk=100)
    )

    item_features = retrieval["candidate_ids"] >> QueryFeast.from_feature_view(
        store=feature_store,
        path=feast_repo_path,
        view="item_features",
        column="candidate_ids",
        output_prefix="item",
        include_id=True,
    )

    user_features_to_unroll = [
        "user_id",
        "user_shops",
        "user_profile",
        "user_group",
        "user_gender",
        "user_age",
        "user_consumption_2",
        "user_is_occupied",
        "user_geography",
        "user_intentions",
        "user_brands",
        "user_categories",
    ]
    combined_features = item_features >> UnrollFeatures(
        "item_id", user_features[user_features_to_unroll]
    )

    ranking = combined_features >> PredictTensorflow(ranking_model_path)

    ordering = combined_features["item_id"] >> SoftmaxSampling(
        relevance_col=ranking["output_1"], topk=10, temperature=20.0
    )

    export_path = str("./test_poc")

    ensemble = Ensemble(ordering, request_schema)
    ens_config, node_configs = ensemble.export(export_path)

    request = make_df({"user_id": [1]})
    request["user_id"] = request["user_id"].astype(np.int64)

    response = _run_ensemble_on_tritonserver(
        export_path, ensemble.graph.output_schema.column_names, request, "ensemble_model"
    )

    assert response is not None
    assert len(response.as_numpy("ordered_ids")) == 10
