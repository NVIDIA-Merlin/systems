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
import pandas as pd
import pytest

from merlin.core.dispatch import HAS_GPU, make_df
from merlin.dag import ColumnSelector
from merlin.dag.dictarray import DictArray
from merlin.dag.executors import DaskExecutor, LocalExecutor
from merlin.io import Dataset
from merlin.schema import ColumnSchema, Schema, Tags

loader_tf_utils = pytest.importorskip("nvtabular.loader.tf_utils")

# everything tensorflow related must be imported after this.
loader_tf_utils.configure_tensorflow()
tf = pytest.importorskip("tensorflow")

from merlin.systems.dag.ensemble import Ensemble  # noqa
from merlin.systems.dag.ops.session_filter import FilterCandidates  # noqa
from merlin.systems.dag.ops.tensorflow import PredictTensorflow  # noqa
from merlin.systems.dag.ops.workflow import TransformWorkflow  # noqa
from merlin.systems.triton.utils import run_ensemble_on_tritonserver  # noqa
from nvtabular import Workflow  # noqa
from nvtabular import ops as wf_ops  # noqa


def test_run_dag_on_dictarray_with_local_executor():
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

    executor = LocalExecutor()
    response = executor.transform(request_data, [ensemble.graph.output_node])

    assert response is not None
    assert isinstance(response, DictArray)
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

    candidate_ids = np.random.randint(1, 100000, 100).astype(np.int32)
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

    candidate_ids = np.random.randint(1, 100000, 100).astype(np.int32)
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


def test_triton_executor_model(tmpdir):
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
    ensemble.export(tmpdir, backend="executor")

    response = run_ensemble_on_tritonserver(
        tmpdir,
        ensemble.input_schema,
        make_df(request_data.arrays),
        ensemble.output_schema.column_names,
        "executor_model",
    )
    assert response is not None
    # assert isinstance(response, DictArray)
    assert len(response["filtered_ids"]) == 80


@pytest.mark.parametrize("engine", ["parquet"])
def test_run_complex_dag_on_dataframe_with_dask_executor(tmpdir, dataset, engine):
    # Create a Workflow
    schema = dataset.schema
    for name in ["x", "y", "id"]:
        dataset.schema.column_schemas[name] = dataset.schema.column_schemas[name].with_tags(
            [Tags.USER]
        )
    selector = ColumnSelector(["x", "y", "id"])

    workflow_ops = selector >> wf_ops.Rename(postfix="_nvt")
    workflow = Workflow(workflow_ops["x_nvt"])
    workflow.fit(dataset)

    # Create Tensorflow Model
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(name="x_nvt", dtype=tf.float64, shape=(1,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, name="output"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
    )

    op_chain = selector >> TransformWorkflow(workflow, cats=["x_nvt"]) >> PredictTensorflow(model)
    ensemble = Ensemble(op_chain, schema)

    df = make_df({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "id": [7, 8, 9]})
    ds = Dataset(df)
    dask_exec = DaskExecutor(transform_method="transform_batch")
    response = dask_exec.transform(ds.to_ddf(), [ensemble.graph.output_node])
    assert len(response["output"]) == df.shape[0]
