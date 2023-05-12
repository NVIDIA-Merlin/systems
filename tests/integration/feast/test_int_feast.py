#
# Copyright (c) 2023, NVIDIA CORPORATION.
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
import os
from datetime import datetime

import numpy as np
import pytest

from merlin.core.dispatch import make_df
from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag import Ensemble
from merlin.systems.dag.ops.feast import QueryFeast  # noqa
from merlin.table import TensorTable  # noqa

feast = pytest.importorskip("feast")  # noqa


def test_feast_integration(tmpdir):
    project_name = "test"
    os.system(f"cd {tmpdir} && feast init {project_name}")
    feast_repo = os.path.join(tmpdir, f"{project_name}")
    feature_repo_path = os.path.join(feast_repo, "feature_repo/")
    if os.path.exists(f"{feature_repo_path}/example_repo.py"):
        os.remove(f"{feature_repo_path}/example_repo.py")
    if os.path.exists(f"{feature_repo_path}/data/driver_stats.parquet"):
        os.remove(f"{feature_repo_path}/data/driver_stats.parquet")
    df_path = os.path.join(feature_repo_path, "data/", "item_features.parquet")
    feat_file_path = os.path.join(feature_repo_path, "item_features.py")

    item_features = make_df(
        {
            "item_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "item_id_raw": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "item_category": [
                [1, 11],
                [2, 12],
                [3, 13],
                [4, 14],
                [5, 15],
                [6, 16],
                [7, 17],
                [8, 18],
                [9, 19],
            ],
            "item_brand": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )
    item_features = TensorTable.from_df(item_features).to_df()
    item_features["datetime"] = datetime.now()
    item_features["datetime"] = item_features["datetime"].astype("datetime64[ns]")
    item_features["created"] = datetime.now()
    item_features["created"] = item_features["created"].astype("datetime64[ns]")

    item_features.to_parquet(df_path)

    with open(feat_file_path, "w", encoding="utf-8") as file:
        file.write(
            f"""
from datetime import timedelta
from feast import Entity, Field, FeatureView, ValueType
from feast.types import Int64, Array
from feast.infra.offline_stores.file_source import FileSource

item_features = FileSource(
    path="{df_path}",
    timestamp_field="datetime",
    created_timestamp_column="created",
)

item = Entity(name="item_id", value_type=ValueType.INT64, join_keys=["item_id"],)

item_features_view = FeatureView(
    name="item_features",
    entities=[item],
    ttl=timedelta(0),
    schema=[
        Field(name="item_category", dtype=Array(Int64)),
        Field(name="item_brand", dtype=Int64),
        Field(name="item_id_raw", dtype=Int64),
    ],
    online=True,
    source=item_features,
    tags=dict(),
)
"""
        )

    os.system(
        f"cd {feature_repo_path} && "
        "feast apply && "
        'CURRENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S") && '
        "feast materialize 1995-01-01T01:01:01 $CURRENT_TIME"
    )

    feature_store = feast.FeatureStore(feature_repo_path)

    # check the information is loaded and correctly querying
    feature_refs = [
        "item_features:item_id_raw",
        "item_features:item_category",
        "item_features:item_brand",
    ]
    feat_df = feature_store.get_historical_features(
        features=feature_refs,
        entity_df=make_df({"item_id": [1], "event_timestamp": [datetime.now()]}),
    ).to_df()
    assert all(feat_df["item_id_raw"] == 1)
    # feature_store.write_to_online_store("item_features", item_features)
    # create and run ensemble with feast operator
    request_schema = Schema([ColumnSchema("item_id", dtype=np.int64)])
    graph = ["item_id"] >> QueryFeast.from_feature_view(
        store=feature_store,
        view="item_features",
        column="item_id",
        output_prefix="item",
        include_id=True,
    )
    ensemble = Ensemble(graph, request_schema)
    result = ensemble.transform(TensorTable.from_df(make_df({"item_id": [1, 2]})))
    columns = ["item_id_raw", "item_brand", "item_category"]
    assert result.to_df()[columns].equals(item_features.iloc[0:2][columns])
