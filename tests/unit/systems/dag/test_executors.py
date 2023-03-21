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
import random
import shutil

import numpy as np
import pandas as pd
import pytest

from merlin.core.dispatch import HAS_GPU, make_df
from merlin.dag.executors import DaskExecutor, LocalExecutor
from merlin.io import Dataset
from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.ops.session_filter import FilterCandidates
from merlin.table import TensorTable

TRITON_SERVER_PATH = shutil.which("tritonserver")


def test_run_dag_on_tensor_table_with_local_executor():
    request_schema = Schema(
        [
            ColumnSchema("candidate_ids", dtype=np.int32),
            ColumnSchema("movie_ids", dtype=np.int32),
        ]
    )

    candidate_ids = np.array(random.sample(range(100000), 100), dtype=np.int32)
    movie_ids_1 = np.zeros(100, dtype=np.int32)
    movie_ids_1[:20] = np.unique(candidate_ids)[:20]

    combined_features = {
        "candidate_ids": candidate_ids,
        "movie_ids": movie_ids_1,
    }

    request_data = TensorTable(combined_features)

    filtering = ["candidate_ids"] >> FilterCandidates(filter_out=["movie_ids"])

    ensemble = Ensemble(filtering, request_schema)

    executor = LocalExecutor()
    response = executor.transform(request_data, [ensemble.graph.output_node])

    assert response is not None
    assert isinstance(response, TensorTable)
    assert len(response["filtered_ids"]) == 80


@pytest.mark.skipif(not HAS_GPU, reason="unable to find GPU")
def test_run_dag_on_dataframe_with_local_executor():
    import cudf

    request_schema = Schema(
        [
            ColumnSchema("candidate_ids", dtype=np.int32),
            ColumnSchema("movie_ids", dtype=np.int32),
        ]
    )

    candidate_ids = np.array(random.sample(range(100000), 100), dtype=np.int32)
    movie_ids_1 = np.zeros(100, dtype=np.int32)
    movie_ids_1[:20] = np.unique(candidate_ids)[:20]

    combined_features = {
        "candidate_ids": candidate_ids,
        "movie_ids": movie_ids_1,
    }

    request_data = make_df(combined_features)

    filtering = ["candidate_ids"] >> FilterCandidates(filter_out=["movie_ids"])
    ensemble = Ensemble(filtering, request_schema)

    executor = LocalExecutor()
    response = executor.transform(request_data, [ensemble.graph.output_node])

    assert response is not None
    assert isinstance(response, (cudf.DataFrame, pd.DataFrame))
    assert len(response["filtered_ids"]) == 80


@pytest.mark.skipif(not HAS_GPU, reason="unable to find GPU")
def test_run_dag_on_dataframe_with_dask_executor():
    import dask_cudf

    request_schema = Schema(
        [
            ColumnSchema("candidate_ids", dtype=np.int32),
            ColumnSchema("movie_ids", dtype=np.int32),
        ]
    )

    candidate_ids = np.array(random.sample(range(100000), 100), dtype=np.int32)
    movie_ids_1 = np.zeros(100, dtype=np.int32)
    movie_ids_1[:20] = np.unique(candidate_ids)[:20]

    combined_features = {
        "candidate_ids": candidate_ids,
        "movie_ids": movie_ids_1,
    }

    request_data = make_df(combined_features)
    request_dataset = Dataset(request_data)

    filtering = ["candidate_ids"] >> FilterCandidates(filter_out=["movie_ids"])
    ensemble = Ensemble(filtering, request_schema)

    executor = DaskExecutor()
    response = executor.transform(request_dataset.to_ddf(), [ensemble.graph.output_node])

    assert response is not None
    assert isinstance(response, dask_cudf.DataFrame)
    assert len(response["filtered_ids"]) == 80
