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
from tritonclient import grpc as grpcclient

from merlin.core.dispatch import make_df  # noqa
from merlin.dag import ColumnSelector  # noqa
from merlin.io import Dataset
from merlin.schema import Schema, Tags  # noqa
from merlin.systems.dag.ops.workflow import TransformWorkflow  # noqa
from merlin.systems.triton.utils import (
    run_ensemble_on_tritonserver,
    run_triton_server,
    send_triton_request,
)
from merlin.table import TensorTable
from nvtabular import Workflow
from nvtabular import ops as wf_ops

TRITON_SERVER_PATH = shutil.which("tritonserver")

triton = pytest.importorskip("merlin.systems.triton")
ensemble = pytest.importorskip("merlin.systems.dag.ensemble")
workflow_op = pytest.importorskip("merlin.systems.dag.ops.workflow")


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize(
    ["model_name", "expected_model_name"],
    [
        (None, "executor_model"),
    ],
)
def test_workflow_op_serving_triton(tmpdir, dataset, engine, model_name, expected_model_name):
    input_columns = ["x", "y", "id"]

    # NVT
    workflow_ops = input_columns >> wf_ops.Rename(postfix="_nvt")
    workflow = Workflow(workflow_ops)
    workflow.fit(dataset)

    # Triton
    triton_op = "*" >> workflow_op.TransformWorkflow(
        workflow,
        conts=["x_nvt", "y_nvt"],
        cats=["id_nvt"],
    )

    wkflow_ensemble = ensemble.Ensemble(triton_op, workflow.input_schema)
    ens_config, node_configs = wkflow_ensemble.export(tmpdir, name=model_name)

    assert ens_config.name == expected_model_name

    input_data = {}
    inputs = []
    for col_name, col_schema in workflow.input_schema.column_schemas.items():
        col_dtype = col_schema.dtype.to_numpy
        input_data[col_name] = np.array([2, 3, 4]).astype(col_dtype)

        triton_input = grpcclient.InferInput(
            col_name, input_data[col_name].shape, triton.np_to_triton_dtype(col_dtype)
        )

        triton_input.set_data_from_numpy(input_data[col_name])

        inputs.append(triton_input)

    outputs = []
    for col_name in workflow.output_schema.column_names:
        outputs.append(grpcclient.InferRequestedOutput(col_name))

    response = None
    with run_triton_server(tmpdir) as client:
        response = client.infer(ens_config.name, inputs, outputs=outputs)

    for col_name in workflow.output_schema.column_names:
        assert response.as_numpy(col_name).shape[0] == input_data[col_name.split("_")[0]].shape[0]


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize("engine", ["parquet"])
def test_workflow_tf_e2e_error_propagation(tmpdir, dataset, engine):
    def raise_(col):
        if (
            isinstance(col.dtype, (type(np.dtype("float64")), type(np.dtype("int64"))))
            and col.sum() != 0
        ):
            return col
        else:
            raise ValueError("Number Too High!!")

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

    # Creating Triton Ensemble
    triton_chain = selector >> TransformWorkflow(workflow, cats=["x_nvt"])
    triton_ens = ensemble.Ensemble(triton_chain, schema)

    # Creating Triton Ensemble Config
    ensemble_config, node_configs = triton_ens.export(str(tmpdir))

    df = make_df({"x": [0.0, 0.0, 0.0], "y": [4.0, 5.0, 6.0], "id": [7, 8, 9]})

    request_schema = Schema([schema["x"], schema["y"], schema["id"]])

    output_columns = triton_ens.output_schema.column_names
    with pytest.raises(Exception) as exc:
        run_ensemble_on_tritonserver(
            str(tmpdir), request_schema, df, output_columns, ensemble_config.name
        )

    assert "Number Too High!!" in str(exc.value)


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
def test_workflow_with_ragged_output(tmpdir):
    df = make_df({"x": [100, 200, 300], "session_id": [1, 1, 2]})
    dataset = Dataset(df)
    workflow_ops = (
        ["x", "session_id"]
        >> wf_ops.Groupby(groupby_cols=["session_id"], aggs={"x": ["list"]}, name_sep="-")
        >> wf_ops.ValueCount()
    )
    workflow = Workflow(workflow_ops["x-list"])
    workflow.fit(dataset)

    workflow_node = workflow.input_schema.column_names >> workflow_op.TransformWorkflow(workflow)
    wkflow_ensemble = ensemble.Ensemble(workflow_node, workflow.input_schema)
    ensemble_config, node_configs = wkflow_ensemble.export(tmpdir)

    with run_triton_server(tmpdir) as client:
        for model_name in [ensemble_config.name, node_configs[0].name]:
            for request_dict, expected_response in [
                (
                    {"x": np.array([100], dtype="int64"), "session_id": np.array([1])},
                    {
                        "x-list__values": np.array([100], dtype="int64"),
                        "x-list__offsets": np.array([0, 1], dtype="int32"),
                    },
                ),
                (
                    {
                        "x": np.array([100, 200, 300], dtype="int64"),
                        "session_id": np.array([1, 1, 2]),
                    },
                    {
                        "x-list__values": np.array([100, 200, 300], dtype="int64"),
                        "x-list__offsets": np.array([0, 2, 3], dtype="int32"),
                    },
                ),
                (
                    {
                        "x": np.array([100, 200, 300, 400], dtype="int64"),
                        "session_id": np.array([1, 1, 2, 2]),
                    },
                    {
                        "x-list__values": np.array([100, 200, 300, 400], dtype="int64"),
                        "x-list__offsets": np.array([0, 2, 4], dtype="int32"),
                    },
                ),
            ]:
                schema = workflow.input_schema
                df = TensorTable(request_dict)
                output_names = ["x-list__values", "x-list__offsets"]
                response = send_triton_request(
                    schema, df, output_names, client=client, triton_model=model_name
                )
                for key, value in expected_response.items():
                    np.testing.assert_array_equal(response[key], value)


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
def test_workflow_with_padded_output(tmpdir):
    df = make_df({"x": [100, 200, 300], "session_id": [1, 1, 2]})
    dataset = Dataset(df)
    workflow_ops = ["x", "session_id"] >> wf_ops.Groupby(
        groupby_cols=["session_id"], aggs={"x": ["list"]}, name_sep="-"
    )
    workflow_ops = workflow_ops["x-list"] >> wf_ops.ListSlice(-3, pad=True)
    workflow = Workflow(workflow_ops)
    workflow.fit(dataset)

    workflow_node = workflow.input_schema.column_names >> workflow_op.TransformWorkflow(workflow)
    wkflow_ensemble = ensemble.Ensemble(workflow_node, workflow.input_schema)
    ensemble_config, node_configs = wkflow_ensemble.export(tmpdir)

    with run_triton_server(tmpdir) as client:
        for model_name in [ensemble_config.name, node_configs[0].name]:
            for request_dict, expected_response in [
                (
                    {"x": np.array([100], dtype="int64"), "session_id": np.array([1])},
                    {
                        "x-list": np.array([[100, 0, 0]], dtype="int64"),
                    },
                ),
                (
                    {
                        "x": np.array([100, 200, 300], dtype="int64"),
                        "session_id": np.array([1, 1, 2]),
                    },
                    {
                        "x-list": np.array([[100, 200, 0], [300, 0, 0]], dtype="int64"),
                    },
                ),
                (
                    {
                        "x": np.array([100, 200, 300, 400], dtype="int64"),
                        "session_id": np.array([1, 1, 2, 2]),
                    },
                    {
                        "x-list": np.array([[100, 200, 0], [300, 400, 0]], dtype="int64"),
                    },
                ),
            ]:
                schema = workflow.input_schema
                df = TensorTable(request_dict)
                output_names = ["x-list"]
                response = send_triton_request(
                    schema, df, output_names, client=client, triton_model=model_name
                )
                for key, value in expected_response.items():
                    np.testing.assert_array_equal(response[key], value)


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
def test_workflow_with_ragged_input_and_output(tmpdir):
    df = make_df({"x": [[100], [200], [300], [400]]})
    dataset = Dataset(df)
    workflow_ops = ["x"] >> wf_ops.Categorify()
    workflow = Workflow(workflow_ops)
    workflow.fit(dataset)

    workflow_node = workflow.input_schema.column_names >> workflow_op.TransformWorkflow(workflow)
    wkflow_ensemble = ensemble.Ensemble(workflow_node, workflow.input_schema)
    ensemble_config, node_configs = wkflow_ensemble.export(tmpdir)

    with run_triton_server(tmpdir) as client:
        for model_name in [ensemble_config.name, node_configs[0].name]:
            for request_dict, expected_response in [
                (
                    {
                        "x__values": np.array([100], dtype="int64"),
                        "x__offsets": np.array([0, 1], dtype="int32"),
                    },
                    {
                        "x__values": np.array([1], dtype="int64"),
                        "x__offsets": np.array([0, 1], dtype="int32"),
                    },
                ),
                (
                    {
                        "x__values": np.array([100, 200], dtype="int64"),
                        "x__offsets": np.array([0, 1, 2], dtype="int32"),
                    },
                    {
                        "x__values": np.array([1, 2], dtype="int64"),
                        "x__offsets": np.array([0, 1, 2], dtype="int32"),
                    },
                ),
                (
                    {
                        "x__values": np.array([100, 200, 300], dtype="int64"),
                        "x__offsets": np.array([0, 2, 3], dtype="int32"),
                    },
                    {
                        "x__values": np.array([1, 2, 3], dtype="int64"),
                        "x__offsets": np.array([0, 2, 3], dtype="int32"),
                    },
                ),
            ]:
                schema = workflow.input_schema
                input_table = TensorTable(request_dict)
                output_names = ["x__values", "x__offsets"]
                response = send_triton_request(
                    schema, input_table, output_names, client=client, triton_model=model_name
                )
                for key, value in expected_response.items():
                    np.testing.assert_array_equal(response[key], value)


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
def test_workflow_dtypes(tmpdir):
    """This test checks that the NVTabular Workflow Triton model outputs dtypes
    that match the workflow output schema even if the transformation results in outputs
    of a different dtype. We coerce the output to match the workflow output dtype.
    """
    df = make_df(
        {
            "a": list(np.array([[100], [200], [300], [400]], dtype="int64")),
            "b": np.array([2.0, 3.6, 4.3, 5.7], dtype="float64"),
        }
    )
    dataset = Dataset(df)
    workflow_ops = ["a", "b"] >> wf_ops.AddMetadata(tags=["my_tag"])
    workflow = Workflow(workflow_ops)
    workflow.fit(dataset)

    # change output workflow schema dtypes so that the output type is different from the schema
    # this is to check that the workflow runner coerces the output types to match the schema
    workflow.output_schema["a"] = workflow.output_schema["a"].with_dtype("int32")
    workflow.output_schema["b"] = workflow.output_schema["b"].with_dtype("float32")

    workflow_node = workflow.input_schema.column_names >> workflow_op.TransformWorkflow(workflow)
    wkflow_ensemble = ensemble.Ensemble(workflow_node, workflow.input_schema)
    ensemble_config, node_configs = wkflow_ensemble.export(tmpdir)

    with run_triton_server(tmpdir) as client:
        for model_name in [ensemble_config.name, node_configs[0].name]:
            for request_dict, expected_response in [
                (
                    {
                        "a__values": np.array([100], dtype="int64"),
                        "a__offsets": np.array([0, 1], dtype="int32"),
                        "b": np.array([0.2], dtype="float64"),
                    },
                    {
                        "a__values": np.array([100], dtype="int32"),
                        "a__offsets": np.array([0, 1], dtype="int32"),
                        "b": np.array([0.2], dtype="float32"),
                    },
                ),
                (
                    {
                        "a__values": np.array([100, 200], dtype="int64"),
                        "a__offsets": np.array([0, 1, 2], dtype="int32"),
                        "b": np.array([0.2, 0.5], dtype="float64"),
                    },
                    {
                        "a__values": np.array([100, 200], dtype="int32"),
                        "a__offsets": np.array([0, 1, 2], dtype="int32"),
                        "b": np.array([0.2, 0.5], dtype="float32"),
                    },
                ),
            ]:
                schema = workflow.input_schema
                input_table = TensorTable(request_dict)
                output_names = ["a__values", "a__offsets", "b"]
                response = send_triton_request(
                    schema, input_table, output_names, client=client, triton_model=model_name
                )
                for key, value in expected_response.items():
                    np.testing.assert_array_equal(response[key], value)
