import os

import numpy as np
import pytest
import tensorflow as tf
from testbook import testbook

import merlin.models.tf as mm
import nvtabular as nvt
from merlin.datasets.synthetic import generate_data
from merlin.io.dataset import Dataset
from merlin.schema import Schema, Tags
from merlin.schema.tags import Tags
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.ops.tensorflow import PredictTensorflow
from merlin.systems.dag.ops.workflow import TransformWorkflow
from merlin.systems.triton.utils import run_ensemble_on_tritonserver  # noqa
from nvtabular.workflow import Workflow
from tests.conftest import REPO_ROOT

pytest.importorskip("cudf")
pytest.importorskip("tensorflow")
pytest.importorskip("merlin.models")


def test_model_input_shapes():
    DATA_FOLDER = "/tmp/data/"
    NUM_ROWS = 1000000
    BATCH_SIZE = 512
    train, valid = generate_data("aliccp-raw", int(NUM_ROWS), set_sizes=(0.7, 0.3))
    train.to_ddf().to_parquet(os.path.join(DATA_FOLDER, "train"))
    valid.to_ddf().to_parquet(os.path.join(DATA_FOLDER, "valid"))
    train_path = os.path.join(DATA_FOLDER, "train", "*.parquet")
    valid_path = os.path.join(DATA_FOLDER, "valid", "*.parquet")
    output_path = os.path.join(DATA_FOLDER, "processed")
    user_id = ["user_id"] >> nvt.ops.Categorify() >> nvt.ops.TagAsUserID()
    item_id = ["item_id"] >> nvt.ops.Categorify() >> nvt.ops.TagAsItemID()
    targets = ["click"] >> nvt.ops.AddMetadata(tags=[Tags.BINARY_CLASSIFICATION, "target"])
    item_features = (
        ["item_category", "item_shop", "item_brand"]
        >> nvt.ops.Categorify()
        >> nvt.ops.TagAsItemFeatures()
    )
    user_features = (
        [
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
        >> nvt.ops.Categorify()
        >> nvt.ops.TagAsUserFeatures()
    )
    outputs = user_id + item_id + item_features + user_features + targets
    workflow = nvt.Workflow(outputs)
    train_dataset = nvt.Dataset(train_path)
    valid_dataset = nvt.Dataset(valid_path)
    workflow.fit(train_dataset)
    workflow.transform(train_dataset).to_parquet(output_path=output_path + "/train/")
    workflow.transform(valid_dataset).to_parquet(output_path=output_path + "/valid/")
    workflow.save("/tmp/data/workflow")
    train = Dataset(os.path.join(output_path, "train", "*.parquet"))
    valid = Dataset(os.path.join(output_path, "valid", "*.parquet"))
    schema = train.schema
    target_column = schema.select_by_tag(Tags.TARGET).column_names[0]
    model = mm.DLRMModel(
        schema,
        embedding_dim=64,
        bottom_block=mm.MLPBlock([128, 64]),
        top_block=mm.MLPBlock([128, 64, 32]),
        prediction_tasks=mm.BinaryClassificationTask(target_column),
    )
    model.compile("adam", run_eagerly=False, metrics=[tf.keras.metrics.AUC()])
    model.fit(train, validation_data=valid, batch_size=BATCH_SIZE)
    model.save("/tmp/data/dlrm")

    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    input_path = DATA_FOLDER

    workflow_stored_path = os.path.join(input_path, "workflow")

    workflow = Workflow.load(workflow_stored_path)

    label_columns = workflow.output_schema.select_by_tag(Tags.TARGET).column_names
    workflow.remove_inputs(label_columns)

    tf_model_path = os.path.join(input_path, "dlrm")

    model = tf.keras.models.load_model(tf_model_path)

    serving_operators = (
        workflow.input_schema.column_names
        >> TransformWorkflow(workflow)
        >> PredictTensorflow(model)
    )

    ensemble = Ensemble(serving_operators, workflow.input_schema)
    export_path = os.path.join(input_path, "ensemble")
    ens_conf, node_confs = ensemble.export(export_path)

    from merlin.core.dispatch import get_lib

    df_lib = get_lib()

    original_data_path = DATA_FOLDER

    # read in data for request
    batch = df_lib.read_parquet(
        os.path.join(original_data_path, "valid", "part.0.parquet"),
        num_rows=3,
        columns=workflow.input_schema.column_names,
    )

    response = run_ensemble_on_tritonserver(
        export_path,
        workflow.input_schema,
        batch,
        ensemble.graph.output_schema.column_names,
        ens_conf.name,
    )

    breakpoint()

    # for col in ensemble.graph.output_schema.column_names:
    #     print(col, response[col], response[col].shape)
