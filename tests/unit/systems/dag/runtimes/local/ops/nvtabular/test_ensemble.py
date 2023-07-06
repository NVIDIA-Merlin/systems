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

from merlin.dag.base_runtime import Runtime
from merlin.table import TensorTable
from nvtabular import Workflow
from nvtabular import ops as wf_ops

ensemble = pytest.importorskip("merlin.systems.dag.ensemble")
workflow_op = pytest.importorskip("merlin.systems.dag.ops.workflow")


@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("runtime", [Runtime()])
def test_workflow_op_serving_triton(tmpdir, dataset, engine, runtime):
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

    input_data = {}
    for col_name, col_schema in workflow.input_schema.column_schemas.items():
        col_dtype = col_schema.dtype
        input_data[col_name] = np.array([2, 3, 4]).astype(col_dtype.to_numpy)
    table = TensorTable(input_data)
    response = wkflow_ensemble.transform(table, runtime=runtime)

    for col_name in workflow.output_schema.column_names:
        assert response[col_name].shape.dims[0] == table[col_name.split("_")[0]].shape.dims[0]
