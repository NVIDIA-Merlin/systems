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

from merlin.core.compat import numpy as np
from merlin.dataloader.ops.padding import Padding
from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.triton.utils import run_ensemble_on_tritonserver
from merlin.table import TensorTable


def test_padding_op_no_triton(tmpdir):
    padding_size = 5
    padding_value = 0
    req_table = TensorTable(
        {"a": (np.array([1, 2, 3], dtype=np.int32), np.array([0, 1, 3], dtype=np.int32))}
    )
    schema = Schema(
        [ColumnSchema("a", dtype=np.int32, dims=req_table["a"].shape, is_list=True, is_ragged=True)]
    )

    graph = ["a"] >> Padding(padding_size, padding_value)
    triton_ens = Ensemble(graph, schema)
    result = triton_ens.transform(req_table)

    assert ["a"] == result.columns
    assert result["a"].values.shape == (2, padding_size)


def test_padding_op_triton(tmpdir):
    padding_size = 5
    padding_value = 0
    req_table = TensorTable(
        {"a": (np.array([1, 2, 3], dtype=np.int32), np.array([0, 1, 3], dtype=np.int32))}
    )
    schema = Schema(
        [ColumnSchema("a", dtype=np.int32, dims=req_table["a"].shape, is_list=True, is_ragged=True)]
    )

    graph = ["a"] >> Padding(padding_size, padding_value)

    triton_ens = Ensemble(graph, schema)
    ensemble_config, node_configs = triton_ens.export(str(tmpdir))
    response = run_ensemble_on_tritonserver(
        str(tmpdir), schema, req_table, ["a"], ensemble_config.name
    )

    assert "a" in response
    assert response["a"].shape == (2, padding_size)
