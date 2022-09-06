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
import pytest

from merlin.dag.node import postorder_iter_nodes
from merlin.dag.ops.concat_columns import ConcatColumns
from merlin.dag.ops.selection import SelectionOp
from merlin.schema import Schema
from nvtabular import Workflow
from nvtabular import ops as wf_ops

ensemble = pytest.importorskip("merlin.systems.dag.ensemble")
workflow_op = pytest.importorskip("merlin.systems.dag.ops.workflow")

from merlin.systems.dag.ops.workflow import TransformWorkflow  # noqa


def test_inference_schema_propagation():
    input_columns = ["a", "b", "c"]
    request_schema = Schema(input_columns)
    expected_schema = Schema(["a_nvt", "b_nvt", "c_nvt"])

    # NVT
    workflow_ops = input_columns >> wf_ops.Rename(postfix="_nvt")
    workflow = Workflow(workflow_ops)
    workflow.fit_schema(request_schema)

    assert workflow.input_schema == request_schema
    assert workflow.output_schema == expected_schema

    # Triton
    triton_ops = input_columns >> workflow_op.TransformWorkflow(workflow)
    ensemble_out = ensemble.Ensemble(triton_ops, request_schema)

    assert ensemble_out.input_schema == request_schema
    assert ensemble_out.output_schema == expected_schema


def test_graph_traverse_algo():
    chain_1 = ["name-cat"] >> TransformWorkflow(Workflow(["name-cat"] >> wf_ops.Categorify()))
    chain_2 = ["name-string"] >> TransformWorkflow(Workflow(["name-string"] >> wf_ops.Categorify()))

    triton_chain = chain_1 + chain_2

    ordered_list = list(postorder_iter_nodes(triton_chain))
    assert len(ordered_list) == 5
    assert isinstance(ordered_list[0].op, SelectionOp)
    assert isinstance(ordered_list[-1].op, ConcatColumns)
