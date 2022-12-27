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
import pytest

from merlin.schema import ColumnSchema, Schema  # noqa
from merlin.systems.dag import DictArray
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.ops.session_filter import FilterCandidates
from merlin.systems.dag.runtimes.triton import TritonExecutorRuntime
from merlin.systems.triton.utils import run_ensemble_on_tritonserver

triton = pytest.importorskip("merlin.systems.triton")
export = pytest.importorskip("merlin.systems.dag.ensemble")

TRITON_SERVER_PATH = shutil.which("tritonserver")


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize(
    ["runtime", "model_name", "expected_model_name"],
    [
        (TritonExecutorRuntime(), None, "executor_model"),
        (TritonExecutorRuntime(), "triton_model", "triton_model"),
    ],
)
def test_triton_runtime_export_and_run(runtime, model_name, expected_model_name, tmpdir):
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

    request_data = DictArray(combined_features)

    filtering = ["candidate_ids"] >> FilterCandidates(filter_out=["movie_ids"])

    ensemble = Ensemble(filtering, request_schema)
    ensemble_config, _ = ensemble.export(tmpdir, runtime=runtime, name=model_name)

    assert ensemble_config.name == expected_model_name
    response = run_ensemble_on_tritonserver(
        tmpdir,
        ensemble.input_schema,
        request_data.to_df(),
        ensemble.output_schema.column_names,
        ensemble_config.name,
    )
    assert response is not None
    # assert isinstance(response, DictArray)
    assert len(response["filtered_ids"]) == 80
