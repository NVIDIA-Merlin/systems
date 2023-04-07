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
import pathlib

import nvtabular
from merlin.systems.triton.conversions import (
    tensor_table_to_triton_response,
    triton_request_to_tensor_table,
)
from merlin.systems.triton.utils import triton_error_handling, triton_multi_request
from merlin.systems.workflow.base import WorkflowRunner
from merlin.table import TensorTable


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

        # Workflow instantiation
        self.workflow = nvtabular.Workflow.load(str(workflow_path))

        # Config loading and parsing
        self.model_config = json.loads(args["model_config"])

        self.runner = WorkflowRunner(self.workflow, self.model_config, model_device)

    @triton_multi_request
    @triton_error_handling
    def execute(self, request):
        """Transforms the input batches by running through a NVTabular workflow.transform
        function.
        """

        try:
            input_columns = self.workflow.input_schema.column_names
            input_tensors = triton_request_to_tensor_table(request, input_columns)
            output_tensors = self.runner.run_workflow(input_tensors)
            return tensor_table_to_triton_response(TensorTable(output_tensors))
        except BaseException as e:
            import traceback

            raise RuntimeError(
                f"Error: {type(e)} - {str(e)}, "
                f"Traceback: {traceback.format_tb(e.__traceback__)}"
            ) from e
