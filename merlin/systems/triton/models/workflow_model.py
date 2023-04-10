# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import logging
import pathlib

import nvtabular
from merlin.dag import ColumnSelector
from merlin.schema import Tags
from merlin.systems.dag.runtimes.nvtabular.runtime import NVTabularServingRuntime
from merlin.systems.triton.conversions import (
    tensor_table_to_triton_response,
    triton_request_to_tensor_table,
)
from merlin.systems.triton.utils import triton_error_handling, triton_multi_request
from merlin.table import TensorTable

LOG = logging.getLogger("merlin-systems")


class TritonPythonModel:
    """
    Triton model used for running NVT Workflows
    """

    def initialize(self, args):
        """
        Initialize a TritonPytonModel with an Nvtabular workflow.

        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # Arg parsing
        model_repo = args["model_repository"]
        repository_path = pathlib.Path(model_repo)

        # Handle bug in Tritonserver 22.06
        # model_repository argument became path to model.py
        if str(repository_path).endswith(".py"):
            repository_path = repository_path.parent.parent

        workflow_path = repository_path / str(args["model_version"]) / "workflow"

        model_device = args["model_instance_kind"]
        self.runtime = NVTabularServingRuntime(model_device)

        # Workflow instantiation
        workflow = nvtabular.Workflow.load(str(workflow_path))
        workflow.graph = self.runtime.convert(workflow.graph)
        self.workflow = workflow

        # Config loading and parsing
        model_config = json.loads(args["model_config"])

        mc_cats, mc_conts = _parse_mc_features(model_config)
        schema_cats, schema_conts = _parse_schema_features(self.workflow.output_schema)

        self.cats = mc_cats or schema_cats
        self.conts = mc_conts or schema_conts

        missing_cols = set(self.cats + self.conts) - set(self.workflow.output_schema.column_names)

        if missing_cols:
            raise ValueError(
                "The following requested columns were not found in the workflow's output: "
                f"{missing_cols}"
            )

    @triton_multi_request
    @triton_error_handling
    def execute(self, request):
        """Transforms the input batches by running through a NVTabular workflow.transform
        function.
        """

        try:
            input_columns = self.workflow.input_schema.column_names
            input_tensors = triton_request_to_tensor_table(request, input_columns)
            transformed = self.runtime.transform(self.workflow.graph, input_tensors)
            return tensor_table_to_triton_response(TensorTable(transformed))
        except BaseException as e:
            import traceback

            raise RuntimeError(
                f"Error: {type(e)} - {str(e)}, "
                f"Traceback: {traceback.format_tb(e.__traceback__)}"
            ) from e


def _parse_schema_features(schema):
    schema_cats = schema.apply(ColumnSelector(tags=[Tags.CATEGORICAL])).column_names
    schema_conts = schema.apply(ColumnSelector(tags=[Tags.CONTINUOUS])).column_names

    return schema_cats, schema_conts


def _parse_mc_features(model_config):
    mc_cats = json.loads(_get_param(model_config, "cats", "string_value", default="[]"))
    mc_conts = json.loads(_get_param(model_config, "conts", "string_value", default="[]"))

    return mc_cats, mc_conts


def _get_param(config, *args, default=None):
    config_element = config["parameters"]
    for key in args:
        config_element = config_element.get(key, {})
    return config_element or default
