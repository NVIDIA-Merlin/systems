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
import json
import pathlib
from typing import List

from merlin.core.protocols import Transformable  # noqa
from merlin.dag import ColumnSelector
from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ops.compat import pb_utils  # noqa
from merlin.systems.dag.ops.operator import PipelineableInferenceOperator  # noqa
from merlin.systems.triton.export import generate_nvtabular_model


class TransformWorkflow(PipelineableInferenceOperator):
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
        model_framework: str = None,
        cats: List[str] = None,
        conts: List[str] = None,
        backend: str = "workflow",
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
        self._nvt_model_name = None
        self.sparse_max = sparse_max or {}
        self.max_batch_size = max_batch_size
        self.label_columns = label_columns or []
        self.model_framework = model_framework or ""
        self.cats = cats or []
        self.conts = conts or []
        self._python = backend == "python"

        if workflow is not None:
            self.input_schema = workflow.input_schema
            self.output_schema = workflow.output_schema

    def transform(self, col_selector: ColumnSelector, transformable: Transformable):
        # TODO: Validate that the inputs match the schema
        # TODO: Should we coerce the dtypes to match the schema here?
        input_tensors = []
        for col_name in self.input_schema.column_schemas.keys():
            input_tensors.append(pb_utils.Tensor(col_name, transformable[col_name]))

        inference_request = pb_utils.InferenceRequest(
            model_name=self.nvt_model_name,
            requested_output_names=self.output_schema.column_names,
            inputs=input_tensors,
        )
        inference_response = inference_request.exec()

        # TODO: Validate that the outputs match the schema
        outputs_dict = {}
        for out_col_name in self.output_schema.column_schemas.keys():
            output_val = pb_utils.get_output_tensor_by_name(
                inference_response, out_col_name
            ).as_numpy()
            outputs_dict[out_col_name] = output_val

        return type(transformable)(outputs_dict)

    @classmethod
    def from_config(cls, config: dict, **kwargs) -> "TransformWorkflow":
        """Instantiate the class from a dictionary representation.

        Expected structure:
        {
            "input_dict": str  # JSON dict with input names and schemas
            "params": str  # JSON dict with params saved at export
        }

        """
        # Input schema
        input_column_schemas = [
            ColumnSchema(name, **schema_properties)
            for name, schema_properties in json.loads(config["input_dict"]).items()
        ]
        input_schema = Schema(input_column_schemas)

        # Output schema
        output_column_schemas = [
            ColumnSchema(name, **schema_properties)
            for name, schema_properties in json.loads(config["output_dict"]).items()
        ]
        output_schema = Schema(output_column_schemas)

        # This method only runs when we're loading the operator to run in Python,
        # so it's safe to set a property on the object that tells us that
        cls_instance = cls(backend="python")
        cls_instance.input_schema = input_schema
        cls_instance.output_schema = output_schema

        params = json.loads(config["params"])
        cls_instance.set_nvt_model_name(params["nvt_model_name"])

        return cls_instance

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

    def export(
        self,
        path: str,
        input_schema: Schema,
        output_schema: Schema,
        params: dict = None,
        node_id: int = None,
        version: int = 1,
        backend: str = "ensemble",
    ):
        """Create a directory inside supplied path based on our export name"""
        modified_workflow = self.workflow.remove_inputs(self.label_columns)
        export_name = self.__class__.__name__.lower()
        node_name = f"{node_id}_{export_name}" if node_id is not None else export_name

        node_export_path = pathlib.Path(path) / node_name
        node_export_path.mkdir(parents=True, exist_ok=True)

        backend_model_config = generate_nvtabular_model(
            modified_workflow,
            node_name,
            node_export_path,
            sparse_max=self.sparse_max,
            max_batch_size=self.max_batch_size,
            cats=self.cats,
            conts=self.conts,
        )

        if self._python:
            self_params = {
                "nvt_model_name": f"{node_id}_{export_name}",
            }
            python_config = super().export(
                path, input_schema, output_schema, self_params, node_id, version
            )

            return python_config
        else:
            return backend_model_config

    @property
    def export_name(self):
        """
        Provides a clear common english identifier for this operator.

        Returns
        -------
        String
            Name of the current class as spelled in module.
        """
        cls_name = self.__class__.__name__.lower()
        if self._python:
            return f"py_{cls_name}"
        else:
            return cls_name
