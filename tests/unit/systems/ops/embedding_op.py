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
from merlin.dataloader.ops.embeddings import NumpyEmbeddingOperator
from merlin.schema import ColumnSchema, Schema, Tags
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.triton.utils import run_ensemble_on_tritonserver
from merlin.table import TensorTable


def test_embedding_op_no_triton(tmpdir):
    embeddings = np.random.rand(100, 50)
    schema = Schema(
        [ColumnSchema("id", dtype=np.int32).with_tags([Tags.CATEGORICAL, Tags.EMBEDDING])]
    )

    graph = ["id"] >> NumpyEmbeddingOperator(embeddings)
    triton_ens = Ensemble(graph, schema)
    req_table = TensorTable({"id": np.array([1, 2, 3])})
    result = triton_ens.transform(req_table)
    assert ["id", "embeddings"] == result.columns
    assert result["embeddings"].shape.as_tuple == (3, 50)


def test_embedding_op_triton(tmpdir):
    embeddings = np.random.rand(100, 50)
    schema = Schema(
        [ColumnSchema("id", dtype=np.int32).with_tags([Tags.CATEGORICAL, Tags.EMBEDDING])]
    )

    graph = ["id"] >> NumpyEmbeddingOperator(embeddings)
    triton_ens = Ensemble(graph, schema)

    ensemble_config, node_configs = triton_ens.export(str(tmpdir))

    req_table = TensorTable({"id": np.array([1, 2, 3], dtype=np.int32)})
    response = run_ensemble_on_tritonserver(
        str(tmpdir), schema, req_table, ["id", "embeddings"], ensemble_config.name
    )

    assert "embeddings" in response
    assert response["embeddings"].shape == (3, 50)
