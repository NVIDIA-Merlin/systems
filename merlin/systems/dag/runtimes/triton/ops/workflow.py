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
import importlib.resources
import json
import os
import pathlib
from shutil import copyfile

import tritonclient.grpc.model_config_pb2 as model_config
import tritonclient.utils
from google.protobuf import text_format

from merlin.core.protocols import Transformable
from merlin.dag import ColumnSelector
from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.runtimes.triton.ops.operator import TritonOperator
from merlin.systems.triton.conversions import (
    tensor_table_to_triton_request,
    triton_response_to_tensor_table,
)
from merlin.systems.triton.export import _add_model_param


class TransformWorkflowTriton(TritonOperator):
    """
    This operator takes a workflow and turns it into a ensemble operator so that we can
    execute feature engineering during ensemble on tritonserver.
    """

    def __init__(self, op):
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
        super().__init__(op)

        self._nvt_model_name = None

        if op.workflow is not None:
            self.input_schema = op.workflow.input_schema
            self.output_schema = op.workflow.output_schema

    def transform(self, col_selector: ColumnSelector, transformable: Transformable):
        """Transform the dataframe by applying this FIL operator to the set of input columns.

        Parameters
        -----------
        df: TensorTable
            A pandas or cudf dataframe that this operator will work on

        Returns
        -------
        TensorTable
            Returns a transformed dataframe for this operator"""

        output_names = []
        for col in self.output_schema:
            if col.is_ragged:
                output_names.append(f"{col.name}__values")
                output_names.append(f"{col.name}__offsets")
            else:
                output_names.append(col.name)

        inference_request = tensor_table_to_triton_request(
            self._nvt_model_name,
            transformable,
            self.input_schema.column_names,
            output_names,
        )

        inference_response = inference_request.exec()

        # check inference response for errors:
        if inference_response.has_error():
            # Cannot raise inference response error because it is not derived from BaseException
            raise tritonclient.utils.InferenceServerException(
                str(inference_response.error().message())
            )

        response_table = triton_response_to_tensor_table(
            inference_response, type(transformable), output_names
        )

        return response_table

    @classmethod
    def from_config(cls, config: dict, **kwargs) -> "TransformWorkflowTriton":
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
        cls_instance = cls(None)
        cls_instance.input_schema = input_schema
        cls_instance.output_schema = output_schema

        params = json.loads(config["params"])
        cls_instance.set_nvt_model_name(params["nvt_model_name"])

        return cls_instance

    @property
    def nvt_model_name(self):
        """The name of the model held by the operator"""
        return self._nvt_model_name

    def set_nvt_model_name(self, nvt_model_name):
        """Set the name of the model held by the operator"""
        self._nvt_model_name = nvt_model_name

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        """Returns output schema of operator"""
        return self.op.workflow.output_schema

    def export(
        self,
        path: str,
        input_schema: Schema,
        output_schema: Schema,
        params: dict = None,
        node_id: int = None,
        version: int = 1,
    ):
        """Create a directory inside supplied path based on our export name"""
        modified_workflow = self.op.workflow.remove_inputs(self.op.label_columns)
        export_name = self.__class__.__name__.lower()
        node_name = f"{node_id}_{export_name}" if node_id is not None else export_name
        self.set_nvt_model_name(node_name)
        node_export_path = pathlib.Path(path) / node_name
        node_export_path.mkdir(parents=True, exist_ok=True)

        backend_model_config = _generate_nvtabular_model(
            modified_workflow,
            node_name,
            node_export_path,
            sparse_max=self.op.sparse_max,
            max_batch_size=self.op.max_batch_size,
            cats=self.op.cats,
            conts=self.op.conts,
        )

        return backend_model_config


def _generate_nvtabular_model(
    workflow,
    name,
    output_path,
    version=1,
    max_batch_size=None,
    sparse_max=None,
    backend="python",
    cats=None,
    conts=None,
):
    """converts a workflow to a triton mode
    Parameters
    ----------
    sparse_max:
        Max length of the each row when the sparse data is converted to dense
    cats:
        Names of the categorical columns
    conts:
        Names of the continuous columns
    """
    workflow.save(os.path.join(output_path, str(version), "workflow"))
    config = _generate_nvtabular_config(
        workflow,
        name,
        output_path,
        max_batch_size,
        sparse_max=sparse_max,
        backend=backend,
        cats=cats,
        conts=conts,
    )

    # copy the model file over. note that this isn't necessary with the c++ backend, but
    # does provide us to use the python backend with just changing the 'backend' parameter
    with importlib.resources.path(
        "merlin.systems.triton.models", "workflow_model.py"
    ) as workflow_model:
        copyfile(
            workflow_model,
            os.path.join(output_path, str(version), "model.py"),
        )

    return config


def _generate_nvtabular_config(
    workflow,
    name,
    output_path,
    max_batch_size=None,
    sparse_max=None,
    backend="python",
    cats=None,
    conts=None,
):
    """given a workflow generates the trton modelconfig proto object describing the inputs
    and outputs to that workflow"""
    config = model_config.ModelConfig(
        name=name,
        backend=backend,
        max_batch_size=max_batch_size,
        instance_group=[
            model_config.ModelInstanceGroup(kind=model_config.ModelInstanceGroup.Kind.KIND_AUTO)
        ],
    )

    config.parameters["python_module"].string_value = "merlin.systems.triton.models.workflow_model"

    config.parameters["cats"].string_value = json.dumps(cats) if cats else ""
    config.parameters["conts"].string_value = json.dumps(conts) if conts else ""

    if sparse_max:
        # this assumes seq_length is same for each list column
        config.parameters["sparse_max"].string_value = json.dumps(sparse_max)

    for col_name, col_schema in workflow.input_schema.column_schemas.items():
        _add_model_param(col_schema, model_config.ModelInput, config.input)

    for col_name, col_schema in workflow.output_schema.column_schemas.items():
        if sparse_max and col_name in sparse_max.keys():
            # this assumes max_sequence_length is equal for all output columns
            dim = sparse_max[col_name]
            _add_model_param(col_schema, model_config.ModelOutput, config.output, [-1, dim])
        else:
            _add_model_param(col_schema, model_config.ModelOutput, config.output)

    with open(os.path.join(output_path, "config.pbtxt"), "w", encoding="utf-8") as o:
        text_format.PrintMessage(config, o)
    return config
