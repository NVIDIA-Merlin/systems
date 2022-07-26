#
# Copyright (c) 2021, NVIDIA CORPORATION.
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

import numpy as np
import pytest

import nvtabular as nvt
from merlin.core.dispatch import make_df
from merlin.dag import ColumnSelector
from merlin.schema import ColumnSchema, Schema
from merlin.schema.tags import Tags
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.ops.hugectr import HugeCTR, PredictHugeCTR, _convert
from tests.unit.systems.utils.triton import _run_ensemble_on_tritonserver

try:
    import hugectr
    from hugectr.inference import CreateInferenceSession, InferenceParams
    from mpi4py import MPI  # noqa pylint: disable=unused-import
except ImportError:
    hugectr = None


triton = pytest.importorskip("merlin.systems.triton")
grpcclient = pytest.importorskip("tritonclient.grpc")
cudf = pytest.importorskip("cudf")


def _run_model(slot_sizes, source, dense_dim):
    solver = hugectr.CreateSolver(
        vvgpu=[[0]],
        batchsize=10,
        batchsize_eval=10,
        max_eval_batches=50,
        i64_input_key=True,
        use_mixed_precision=False,
        repeat_dataset=True,
    )
    # https://github.com/NVIDIA-Merlin/HugeCTR/blob/9e648f879166fc93931c676a5594718f70178a92/docs/source/api/python_interface.md#datareaderparams
    reader = hugectr.DataReaderParams(
        data_reader_type=hugectr.DataReaderType_t.Parquet,
        source=[os.path.join(source, "_file_list.txt")],
        eval_source=os.path.join(source, "_file_list.txt"),
        check_type=hugectr.Check_t.Non,
    )

    optimizer = hugectr.CreateOptimizer(optimizer_type=hugectr.Optimizer_t.Adam)
    model = hugectr.Model(solver, reader, optimizer)

    model.add(
        hugectr.Input(
            label_dim=1,
            label_name="label",
            dense_dim=dense_dim,
            dense_name="dense",
            data_reader_sparse_param_array=[
                hugectr.DataReaderSparseParam("data1", len(slot_sizes) + 1, True, len(slot_sizes))
            ],
        )
    )
    model.add(
        hugectr.SparseEmbedding(
            embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
            workspace_size_per_gpu_in_mb=107,
            embedding_vec_size=16,
            combiner="sum",
            sparse_embedding_name="sparse_embedding1",
            bottom_name="data1",
            slot_size_array=slot_sizes,
            optimizer=optimizer,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.InnerProduct,
            bottom_names=["dense"],
            top_names=["fc1"],
            num_output=512,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Reshape,
            bottom_names=["sparse_embedding1"],
            top_names=["reshape1"],
            leading_dim=48,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.InnerProduct,
            bottom_names=["reshape1", "fc1"],
            top_names=["fc2"],
            num_output=1,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
            bottom_names=["fc2", "label"],
            top_names=["loss"],
        )
    )
    model.compile()
    model.summary()
    model.fit(max_iter=20, display=100, eval_interval=200, snapshot=10)

    return model


@pytest.mark.skip(reason="More than one hugectr pytest results in segfault")
def test_hugectr_op(tmpdir):
    cat_dtypes = {"a": int, "b": int, "c": int}

    categorical_columns = list(cat_dtypes.keys())

    gdf = make_df(
        {
            "a": np.arange(64, dtype=np.int64),
            "b": np.arange(64, dtype=np.int64),
            "c": np.arange(64, dtype=np.int64),
            "d": np.random.rand(64).tolist(),
            "label": [0] * 64,
        },
    )
    gdf["label"] = gdf["label"].astype("float32")
    gdf["d"] = gdf["d"].astype("float32")
    train_dataset = nvt.Dataset(gdf)

    dense_columns = ["d"]

    dict_dtypes = {}
    for col in dense_columns:
        dict_dtypes[col] = np.float32

    for col in categorical_columns:
        dict_dtypes[col] = np.int64

    for col in ["label"]:
        dict_dtypes[col] = np.float32

    train_path = os.path.join(tmpdir, "train/")
    os.mkdir(train_path)

    train_dataset.to_parquet(
        output_path=train_path,
        shuffle=nvt.io.Shuffle.PER_PARTITION,
        cats=categorical_columns,
        conts=dense_columns,
        labels=["label"],
        dtypes=dict_dtypes,
    )

    embeddings = {"a": (64, 16), "b": (64, 16), "c": (64, 16)}

    total_cardinality = 0
    slot_sizes = []

    for column in cat_dtypes:
        slot_sizes.append(embeddings[column][0])
        total_cardinality += embeddings[column][0]

    # slot sizes = list of caridinalities per column, total is sum of individual
    model = _run_model(slot_sizes, train_path, len(dense_columns))

    model_op = HugeCTR(model, max_nnz=2, device_list=[0])

    model_repository_path = os.path.join(tmpdir, "model_repository")

    input_schema = Schema(
        [
            ColumnSchema("DES", dtype=np.float32),
            ColumnSchema("CATCOLUMN", dtype=np.int64),
            ColumnSchema("ROWINDEX", dtype=np.int32),
        ]
    )
    triton_chain = ColumnSelector(["DES", "CATCOLUMN", "ROWINDEX"]) >> model_op
    ens = Ensemble(triton_chain, input_schema)

    os.makedirs(model_repository_path)

    enc_config, node_configs = ens.export(model_repository_path)

    assert enc_config
    assert len(node_configs) == 1
    assert node_configs[0].name == "0_hugectr"

    df = train_dataset.to_ddf().compute()[:5]
    dense, cats, rowptr = _convert(df, slot_sizes, categorical_columns, labels=["label"])

    inputs = [
        grpcclient.InferInput("DES", dense.shape, triton.np_to_triton_dtype(dense.dtype)),
        grpcclient.InferInput("CATCOLUMN", cats.shape, triton.np_to_triton_dtype(cats.dtype)),
        grpcclient.InferInput("ROWINDEX", rowptr.shape, triton.np_to_triton_dtype(rowptr.dtype)),
    ]
    inputs[0].set_data_from_numpy(dense)
    inputs[1].set_data_from_numpy(cats)
    inputs[2].set_data_from_numpy(rowptr)

    response = _run_ensemble_on_tritonserver(
        model_repository_path,
        ["OUTPUT0"],
        inputs,
        "0_hugectr",
        backend_config=f"hugectr,ps={tmpdir}/model_repository/ps.json",
    )
    assert len(response.as_numpy("OUTPUT0")) == df.shape[0]

    model_config = node_configs[0].parameters["config"].string_value

    hugectr_name = node_configs[0].name
    dense_path = f"{tmpdir}/model_repository/{hugectr_name}/1/_dense_0.model"
    sparse_files = [f"{tmpdir}/model_repository/{hugectr_name}/1/0_sparse_0.model"]
    out_predict = _predict(
        dense, cats, rowptr, model_config, hugectr_name, dense_path, sparse_files
    )

    np.testing.assert_array_almost_equal(response.as_numpy("OUTPUT0"), np.array(out_predict))
    del model


def test_predict_hugectr(tmpdir):
    cat_dtypes = {"a": int, "b": int, "c": int}

    categorical_columns = ["a", "b", "c"]

    gdf = make_df(
        {
            "a": np.arange(64, dtype=np.int64),
            "b": np.arange(64, dtype=np.int64),
            "c": np.arange(64, dtype=np.int64),
            "d": np.random.rand(64).tolist(),
            "label": [0] * 64,
        },
    )
    gdf["label"] = gdf["label"].astype("float32")
    gdf["d"] = gdf["d"].astype("float32")
    train_dataset = nvt.Dataset(gdf)

    dense_columns = ["d"]

    dict_dtypes = {}
    col_schemas = train_dataset.schema.column_schemas
    for col in dense_columns:
        col_schemas[col] = col_schemas[col].with_tags(Tags.CONTINUOUS)
        dict_dtypes[col] = np.float32

    for col in categorical_columns:
        col_schemas[col] = col_schemas[col].with_tags(Tags.CATEGORICAL)
        dict_dtypes[col] = np.int64

    for col in ["label"]:
        col_schemas[col] = col_schemas[col].with_tags(Tags.TARGET)
        dict_dtypes[col] = np.float32

    train_path = os.path.join(tmpdir, "train/")
    os.mkdir(train_path)

    train_dataset.to_parquet(
        output_path=train_path,
        shuffle=nvt.io.Shuffle.PER_PARTITION,
        cats=categorical_columns,
        conts=dense_columns,
        labels=["label"],
        dtypes=dict_dtypes,
    )

    embeddings = {"a": (64, 16), "b": (64, 16), "c": (64, 16)}

    total_cardinality = 0
    slot_sizes = []

    for column in cat_dtypes:
        slot_sizes.append(embeddings[column][0])
        total_cardinality += embeddings[column][0]

    # slot sizes = list of caridinalities per column, total is sum of individual
    model = _run_model(slot_sizes, train_path, len(dense_columns))

    model_op = PredictHugeCTR(model, train_dataset.schema, max_nnz=2, device_list=[0])

    model_repository_path = os.path.join(tmpdir, "model_repository")

    input_schema = train_dataset.schema
    triton_chain = input_schema.column_names >> model_op
    ens = Ensemble(triton_chain, input_schema)

    os.makedirs(model_repository_path)

    enc_config, node_configs = ens.export(model_repository_path)

    assert enc_config
    assert len(node_configs) == 1
    assert node_configs[0].name == "0_predicthugectr"

    df = train_dataset.to_ddf().compute()[:5]
    dense, cats, rowptr = _convert(df, slot_sizes, categorical_columns, labels=["label"])

    response = _run_ensemble_on_tritonserver(
        model_repository_path,
        ["OUTPUT0"],
        df,
        "ensemble_model",
        backend_config=f"hugectr,ps={tmpdir}/model_repository/ps.json",
    )
    assert len(response.as_numpy("OUTPUT0")) == df.shape[0]

    model_config = f"{tmpdir}/model_repository/0_hugectr/1/0_hugectr.json"

    hugectr_name = "0_hugectr"
    dense_path = f"{tmpdir}/model_repository/{hugectr_name}/1/_dense_0.model"
    sparse_files = [f"{tmpdir}/model_repository/{hugectr_name}/1/0_sparse_0.model"]
    out_predict = _predict(
        dense, cats, rowptr, model_config, hugectr_name, dense_path, sparse_files
    )

    np.testing.assert_array_almost_equal(response.as_numpy("OUTPUT0"), np.array(out_predict))


def test_no_categoricals():
    with pytest.raises(ValueError) as exc_info:
        PredictHugeCTR(None, Schema())
    assert "HugeCTR require categorical columns." in str(exc_info.value)


def _predict(
    dense_features, embedding_columns, row_ptrs, config_file, model_name, dense_path, sparse_paths
):
    inference_params = InferenceParams(
        model_name=model_name,
        max_batchsize=64,
        hit_rate_threshold=0.5,
        dense_model_file=dense_path,
        sparse_model_files=sparse_paths,
        device_id=0,
        use_gpu_embedding_cache=True,
        cache_size_percentage=0.2,
        i64_input_key=True,
        use_mixed_precision=False,
    )
    inference_session = CreateInferenceSession(config_file, inference_params)
    output = inference_session.predict(
        dense_features[0].tolist(), embedding_columns[0].tolist(), row_ptrs[0].tolist()
    )
    return output
