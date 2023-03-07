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
import shutil

import numpy as np
import pytest

from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.ops.faiss import QueryFaiss, setup_faiss

TRITON_SERVER_PATH = shutil.which("tritonserver")
pytest.importorskip("merlin.dataloader.tf_utils")
from merlin.dataloader.tf_utils import configure_tensorflow  # noqa

tritonclient = pytest.importorskip("tritonclient")
grpcclient = pytest.importorskip("tritonclient.grpc")

from merlin.systems.triton.utils import run_ensemble_on_tritonserver  # noqa

configure_tensorflow()

import tensorflow as tf  # noqa

from merlin.systems.dag.ops.tensorflow import PredictTensorflow  # noqa
from merlin.systems.dag.runtimes.triton import TritonExecutorRuntime  # noqa
from merlin.table import TensorTable  # noqa


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
def test_faiss_in_triton_executor_model(tmpdir):
    # Simulate a user vector with a TF model
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(name="user_id", dtype=tf.int32, shape=(1,)),
            tf.keras.layers.Dense(128, activation="relu", name="output"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
    )

    faiss_path = tmpdir / "faiss.index"
    item_ids = np.arange(0, 100).reshape(-1, 1)
    item_embeddings = np.ascontiguousarray(np.random.rand(100, 128))
    setup_faiss(np.concatenate((item_ids, item_embeddings), axis=1), faiss_path)

    request_schema = Schema(
        [
            ColumnSchema("user_id", dtype=np.int32, dims=(None, 1)),
        ]
    )

    request_data = TensorTable(
        {
            "user_id": np.array([[1]], dtype=np.int32),
        }
    )

    retrieval = ["user_id"] >> PredictTensorflow(model) >> QueryFaiss(faiss_path)

    ensemble = Ensemble(retrieval, request_schema)
    ensemble_config, node_configs = ensemble.export(tmpdir, runtime=TritonExecutorRuntime())

    response = run_ensemble_on_tritonserver(
        tmpdir,
        ensemble.input_schema,
        request_data,
        ensemble.output_schema.column_names,
        ensemble_config.name,
    )

    assert response is not None
    # assert isinstance(response, TensorTable)
    assert len(response["candidate_ids"]) == 10
