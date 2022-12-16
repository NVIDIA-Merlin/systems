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
from distutils.spawn import find_executable  # pylint: disable=deprecated-module

import numpy as np
import pytest
from tritonclient import grpc as grpcclient

from merlin.systems.dag.runtimes.triton import TritonExecutorRuntime
from merlin.systems.triton.utils import run_triton_server
from nvtabular import Workflow
from nvtabular import ops as wf_ops

TRITON_SERVER_PATH = find_executable("tritonserver")

triton = pytest.importorskip("merlin.systems.triton")
ensemble = pytest.importorskip("merlin.systems.dag.ensemble")
workflow_op = pytest.importorskip("merlin.systems.dag.ops.workflow")


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize(
    ["runtime", "model_name", "expected_model_name"],
    [
        (TritonExecutorRuntime(), None, "executor_model"),
    ],
)
def test_workflow_op_serving_triton(
    tmpdir, dataset, engine, runtime, model_name, expected_model_name
):
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
    ens_config, node_configs = wkflow_ensemble.export(tmpdir, runtime=runtime, name=model_name)

    assert ens_config.name == expected_model_name

    input_data = {}
    inputs = []
    for col_name, col_schema in workflow.input_schema.column_schemas.items():
        col_dtype = col_schema.dtype
        input_data[col_name] = np.array([[2], [3], [4]]).astype(col_dtype)

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
