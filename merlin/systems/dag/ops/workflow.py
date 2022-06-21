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
import pathlib
from typing import List

from merlin.dag import ColumnSelector
from merlin.schema import Schema
from merlin.systems.dag.ops.operator import InferenceOperator
from merlin.systems.triton.export import generate_nvtabular_model


class TransformWorkflow(InferenceOperator):
    """
    This operator takes a workflow and turns it into a ensemble operator so that we can
    execute feature engineering during ensemble on tritonserver.
    """

    def __init__(
        self,
        workflow,
        sparse_max: dict = None,
        max_batch_size: int = None,
        label_columns: List[str] = None,
        model_framework: str = None,
        cats: List[str] = None,
        conts: List[str] = None,
    ):
        """
        Creates a Transform Workflow operator for a target workflow.

        Parameters
        ----------
        workflow : Nvtabular.Workflow
            The workflow to transform data in ensemble.
        sparse_max : dict, optional
            Dictionary representing key(name)/val(max value) pairs of max sparsity, by default None
        max_batch_size : int, optional
            Maximum batch size, by default None
        label_columns : List[str], optional
            List of strings identifying the label columns, by default None
        model_framework : str, optional
            String representing the target framework
            (supported: hugectr, tensorflow, pytorch, python), by default None
        cats : List[str], optional
            List of strings identifying categorical columns, by default None
        conts : List[str], optional
            List of string identifying continuous columns, by default None
        """
        super().__init__()

        self.workflow = workflow
        self.sparse_max = sparse_max or {}
        self.max_batch_size = max_batch_size
        self.label_columns = label_columns or []
        self.model_framework = model_framework or ""
        self.cats = cats or []
        self.conts = conts or []
        super().__init__()

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        """Returns output schema of operator"""
        return self.workflow.output_schema

    def export(self, path, input_schema, output_schema, node_id=None, version=1):
        """Create a directory inside supplied path based on our export name"""
        modified_workflow = self.workflow.remove_inputs(self.label_columns)

        node_name = f"{node_id}_{self.export_name}" if node_id is not None else self.export_name

        node_export_path = pathlib.Path(path) / node_name
        node_export_path.mkdir(parents=True, exist_ok=True)

        return generate_nvtabular_model(
            modified_workflow,
            node_name,
            node_export_path,
            sparse_max=self.sparse_max,
            max_batch_size=self.max_batch_size,
            cats=self.cats,
            conts=self.conts,
        )
