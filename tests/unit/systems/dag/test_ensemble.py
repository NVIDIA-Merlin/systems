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
import numpy as np

from merlin.dag.executors import LocalExecutor
from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag import DictArray
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.ops.session_filter import FilterCandidates


def test_ensemble_save_load(tmpdir):
    request_schema = Schema(
        [
            ColumnSchema("candidate_ids", dtype=np.int32),
            ColumnSchema("movie_ids", dtype=np.int32),
        ]
    )

    candidate_ids = np.random.randint(1, 100000, 100).astype(np.int32)
    movie_ids_1 = np.zeros(100, dtype=np.int32)
    movie_ids_1[:20] = np.unique(candidate_ids)[:20]

    combined_features = {
        "candidate_ids": candidate_ids,
        "movie_ids": movie_ids_1,
    }

    request_data = DictArray(combined_features)

    filtering = ["candidate_ids"] >> FilterCandidates(filter_out=["movie_ids"])

    ensemble = Ensemble(filtering, request_schema)
    ensemble.save(str(tmpdir))

    loaded_ensemble = Ensemble.load(str(tmpdir))

    executor = LocalExecutor()
    response = executor.transform(request_data, [ensemble.graph.output_node])

    loaded_response = executor.transform(request_data, [loaded_ensemble.graph.output_node])

    assert all(loaded_response.arrays["filtered_ids"] == response.arrays["filtered_ids"])
