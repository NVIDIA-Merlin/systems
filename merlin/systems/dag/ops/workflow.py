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
from typing import List

from merlin.core.protocols import Transformable
from merlin.dag import ColumnSelector
from merlin.schema import Schema
from merlin.systems.dag.ops.operator import InferenceOperator
from merlin.table import TensorTable


class TransformWorkflow(InferenceOperator):
    """
    This operator takes a workflow and turns it into a ensemble operator so that we can
    execute feature engineering during ensemble on tritonserver.
    """

    def __init__(
        self,
        workflow=None,
        sparse_max: dict = None,
        max_batch_size: int = None,
        label_columns: List[str] = None,
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
        cats : List[str], optional
            List of strings identifying categorical columns, by default None
        conts : List[str], optional
            List of string identifying continuous columns, by default None
        """
        super().__init__()

        self.workflow = workflow
        if label_columns:
            self.workflow = workflow.remove_inputs(label_columns)
        self._nvt_model_name = None
        self.sparse_max = sparse_max or {}
        self.max_batch_size = max_batch_size
        self.label_columns = label_columns or []
        self.cats = cats or []
        self.conts = conts or []

        if self.workflow is not None:
            self.input_schema = self.workflow.input_schema
            self.output_schema = self.workflow.output_schema

    @property
    def nvt_model_name(self):
        return self._nvt_model_name

    def set_nvt_model_name(self, nvt_model_name):
        self._nvt_model_name = nvt_model_name

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        """Returns output schema of operator"""
        return self.workflow.output_schema

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        """Run nvtabular workflow transformations.

        Parameters
        ----------
        col_selector : ColumnSelector
            Unused ColumunSelector input
        transformable : Transformable
            Input features to model

        Returns
        -------
        Transformable
            workflow transform
        """
        output_type = type(transformable)
        if isinstance(transformable, TensorTable):
            transformable = transformable.to_df()

        output = self.workflow._transform_df(transformable)

        if not isinstance(output, output_type):
            output = TensorTable.from_df(output)

        return output
