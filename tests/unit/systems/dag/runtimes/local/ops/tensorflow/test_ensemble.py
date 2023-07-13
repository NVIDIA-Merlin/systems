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
import pytest

from merlin.dag import ColumnSelector
from merlin.dag.runtime import Runtime
from merlin.schema import Tags
from nvtabular import Workflow
from nvtabular import ops as wf_ops

loader_tf_utils = pytest.importorskip("nvtabular.loader.tf_utils")

# everything tensorflow related must be imported after this.
loader_tf_utils.configure_tensorflow()
tf = pytest.importorskip("tensorflow")

export = pytest.importorskip("merlin.systems.dag.ensemble")

from merlin.systems.dag.ensemble import Ensemble  # noqa
from merlin.systems.dag.ops.tensorflow import PredictTensorflow  # noqa
from merlin.systems.dag.ops.workflow import TransformWorkflow  # noqa
from merlin.table import TensorTable  # noqa
from tests.unit.systems.utils.tf import create_tf_model  # noqa


@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("runtime", [(Runtime())])
def test_workflow_tf_e2e_config_verification(tmpdir, dataset, engine, runtime):
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
    inputs_table = TensorTable(
        {"x": np.array([1.0, 2.0, 3.0]), "y": np.array([4.0, 5.0, 6.0]), "id": np.array([7, 8, 9])}
    )

    response = triton_ens.transform(inputs_table, runtime=runtime)

    assert response["output"].shape.dims[0] == inputs_table["x"].shape.dims[0]


@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("runtime", [(Runtime())])
def test_workflow_tf_e2e_multi_op_run(tmpdir, dataset, engine, runtime):
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

    df = dataset.to_ddf().compute()[["name-string", "name-cat"]].iloc[:3]

    response = triton_ens.transform(df, runtime=runtime)

    assert response["predictions"].shape[0] == df.shape[0]
    assert response["predictions"].shape[0] == df.shape[0]


@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("python", [False, True])
@pytest.mark.parametrize("runtime", [(Runtime())])
def test_workflow_tf_python_wrapper(tmpdir, dataset, engine, python, runtime):
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

    df = dataset.to_ddf().compute()[["name-string", "name-cat"]].iloc[:3]
    response = triton_ens.transform(df, runtime=runtime)

    assert len(response["predictions"]) == df.shape[0]
