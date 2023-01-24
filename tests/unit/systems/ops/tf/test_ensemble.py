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
import os
from shutil import which

import numpy
import pytest

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from google.protobuf import text_format  # noqa

from merlin.core.dispatch import make_df  # noqa
from merlin.dag import ColumnSelector  # noqa
from merlin.schema import Schema, Tags  # noqa
from nvtabular import Workflow  # noqa
from nvtabular import ops as wf_ops  # noqa

loader_tf_utils = pytest.importorskip("nvtabular.loader.tf_utils")

# everything tensorflow related must be imported after this.
loader_tf_utils.configure_tensorflow()
tf = pytest.importorskip("tensorflow")

triton = pytest.importorskip("merlin.systems.triton")
export = pytest.importorskip("merlin.systems.dag.ensemble")

import tritonclient.grpc.model_config_pb2 as model_config  # noqa

from merlin.systems.dag.ensemble import Ensemble  # noqa
from merlin.systems.dag.ops.tensorflow import PredictTensorflow  # noqa
from merlin.systems.dag.ops.workflow import TransformWorkflow  # noqa
from merlin.systems.triton.utils import run_ensemble_on_tritonserver  # noqa
from tests.unit.systems.utils.tf import create_tf_model  # noqa

TRITON_SERVER_PATH = which("tritonserver")


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize("engine", ["parquet"])
def test_workflow_tf_e2e_config_verification(tmpdir, dataset, engine):
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

    # Creating Triton Ensemble
    triton_chain = (
        selector >> TransformWorkflow(workflow, cats=["x_nvt"]) >> PredictTensorflow(model)
    )
    triton_ens = Ensemble(triton_chain, schema)

    # Creating Triton Ensemble Config
    ensemble_config, node_configs = triton_ens.export(str(tmpdir))

    config_path = tmpdir / ensemble_config.name / "config.pbtxt"

    # Checking Triton Ensemble Config
    with open(config_path, "rb") as f:
        config = model_config.ModelConfig()
        raw_config = f.read()
        parsed = text_format.Parse(raw_config, config)

        # The config file contents are correct
        assert parsed.name == "executor_model"
        assert parsed.platform == "merlin_executor"
        assert hasattr(parsed, "ensemble_scheduling")

    df = make_df({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "id": [7, 8, 9]})

    request_schema = Schema([schema["x"], schema["y"], schema["id"]])

    output_columns = triton_ens.output_schema.column_names
    response = run_ensemble_on_tritonserver(
        str(tmpdir), request_schema, df, output_columns, ensemble_config.name
    )
    assert len(response["output"]) == df.shape[0]


def raise_(col):
    if (
        isinstance(col.dtype, (type(numpy.dtype("float64")), type(numpy.dtype("int64"))))
        and col.sum() != 0
    ):
        return col
    else:
        raise ValueError("Number Too High!!")


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize("engine", ["parquet"])
def test_workflow_tf_e2e_error_propagation(tmpdir, dataset, engine):
    # Create a Workflow
    schema = dataset.schema
    for name in ["x", "y", "id"]:
        dataset.schema.column_schemas[name] = dataset.schema.column_schemas[name].with_tags(
            [Tags.USER]
        )
    selector = ColumnSelector(["x", "y", "id"])

    workflow_ops = selector >> wf_ops.Rename(postfix="_nvt") >> wf_ops.LambdaOp(raise_)
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

    # Creating Triton Ensemble
    triton_chain = (
        selector >> TransformWorkflow(workflow, cats=["x_nvt"]) >> PredictTensorflow(model)
    )
    triton_ens = Ensemble(triton_chain, schema)

    # Creating Triton Ensemble Config
    ensemble_config, node_configs = triton_ens.export(str(tmpdir))

    config_path = tmpdir / ensemble_config.name / "config.pbtxt"

    # Checking Triton Ensemble Config
    with open(config_path, "rb") as f:
        config = model_config.ModelConfig()
        raw_config = f.read()
        parsed = text_format.Parse(raw_config, config)

        # The config file contents are correct
        assert parsed.name == "executor_model"
        assert parsed.platform == "merlin_executor"
        assert hasattr(parsed, "ensemble_scheduling")

    df = make_df({"x": [0.0, 0.0, 0.0], "y": [4.0, 5.0, 6.0], "id": [7, 8, 9]})

    request_schema = Schema([schema["x"], schema["y"], schema["id"]])

    output_columns = triton_ens.output_schema.column_names
    with pytest.raises(Exception) as exc:
        run_ensemble_on_tritonserver(
            str(tmpdir), request_schema, df, output_columns, ensemble_config.name
        )

    assert "Number Too High!!" in str(exc.value)


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize("engine", ["parquet"])
def test_workflow_tf_e2e_multi_op_run(tmpdir, dataset, engine):
    # Create a Workflow
    schema = dataset.schema
    for name in ["x", "y", "id"]:
        dataset.schema.column_schemas[name] = dataset.schema.column_schemas[name].with_tags(
            [Tags.USER]
        )

    workflow_ops = ["name-cat"] >> wf_ops.Categorify(cat_cache="host")
    workflow = Workflow(workflow_ops)
    workflow.fit(dataset)

    embedding_shapes_1 = wf_ops.get_embedding_sizes(workflow)

    cats = ["name-string"] >> wf_ops.Categorify(cat_cache="host")
    workflow_2 = Workflow(cats)
    workflow_2.fit(dataset)

    embedding_shapes = wf_ops.get_embedding_sizes(workflow_2)
    embedding_shapes_1.update(embedding_shapes)
    # Create Tensorflow Model
    model = create_tf_model(["name-cat", "name-string"], [], embedding_shapes_1)

    # Creating Triton Ensemble
    triton_chain_1 = ["name-cat"] >> TransformWorkflow(workflow)
    triton_chain_2 = ["name-string"] >> TransformWorkflow(workflow_2)
    triton_chain = (triton_chain_1 + triton_chain_2) >> PredictTensorflow(model)

    triton_ens = Ensemble(triton_chain, schema)

    # Creating Triton Ensemble Config
    ensemble_config, nodes_config = triton_ens.export(str(tmpdir))
    config_path = tmpdir / "executor_model" / "config.pbtxt"

    # Checking Triton Ensemble Config
    with open(config_path, "rb") as f:
        config = model_config.ModelConfig()
        raw_config = f.read()
        parsed = text_format.Parse(raw_config, config)

        # The config file contents are correct
        assert parsed.name == "executor_model"
        assert parsed.platform == "merlin_executor"
        assert hasattr(parsed, "ensemble_scheduling")

    df = dataset.to_ddf().compute()[["name-string", "name-cat"]].iloc[:3]
    request_schema = workflow.input_schema + workflow_2.input_schema

    response = run_ensemble_on_tritonserver(
        str(tmpdir), request_schema, df, ["output"], ensemble_config.name
    )
    assert len(response["output"]) == df.shape[0]


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("python", [False, True])
def test_workflow_tf_python_wrapper(tmpdir, dataset, engine, python):
    # Create a Workflow
    schema = dataset.schema
    for name in ["x", "y", "id"]:
        dataset.schema.column_schemas[name] = dataset.schema.column_schemas[name].with_tags(
            [Tags.USER]
        )

    workflow_ops = ["name-cat"] >> wf_ops.Categorify(cat_cache="host")
    workflow = Workflow(workflow_ops)
    workflow.fit(dataset)

    embedding_shapes_1 = wf_ops.get_embedding_sizes(workflow)

    cats = ["name-string"] >> wf_ops.Categorify(cat_cache="host")
    workflow_2 = Workflow(cats)
    workflow_2.fit(dataset)

    embedding_shapes = wf_ops.get_embedding_sizes(workflow_2)
    embedding_shapes_1.update(embedding_shapes)
    # Create Tensorflow Model
    model = create_tf_model(["name-cat", "name-string"], [], embedding_shapes_1)

    # Creating Triton Ensemble
    triton_chain_1 = ["name-cat"] >> TransformWorkflow(workflow)
    triton_chain_2 = ["name-string"] >> TransformWorkflow(workflow_2)
    triton_chain = (triton_chain_1 + triton_chain_2) >> PredictTensorflow(model)

    triton_ens = Ensemble(triton_chain, schema)

    # Creating Triton Ensemble Config
    ensemble_config, nodes_config = triton_ens.export(str(tmpdir))
    config_path = tmpdir / "executor_model" / "config.pbtxt"

    # Checking Triton Ensemble Config
    with open(config_path, "rb") as f:
        config = model_config.ModelConfig()
        raw_config = f.read()
        parsed = text_format.Parse(raw_config, config)

        # The config file contents are correct
        assert parsed.name == "executor_model"
        assert parsed.platform == "merlin_executor"
        assert hasattr(parsed, "ensemble_scheduling")

    df = dataset.to_ddf().compute()[["name-string", "name-cat"]].iloc[:3]
    request_schema = workflow.input_schema + workflow_2.input_schema

    response = run_ensemble_on_tritonserver(
        str(tmpdir), request_schema, df, ["output"], ensemble_config.name
    )
    assert len(response["output"]) == df.shape[0]
