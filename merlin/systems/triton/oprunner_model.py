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
import sys
import traceback
from typing import List

import triton_python_backend_utils as pb_utils
from triton_python_backend_utils import (
    InferenceRequest,
    InferenceResponse,
    Tensor,
    get_input_tensor_by_name,
)

from merlin.systems.dag.op_runner import OperatorRunner
from merlin.systems.dag.ops.operator import InferenceDataFrame


class TritonPythonModel:
    """Model for Triton Python Backend.

    Every Python model must have "TritonPythonModel" as the class name
    """

    def initialize(self, args):
        """Called only once when the model is being loaded. Allowing
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.model_config = json.loads(args["model_config"])
        self.runner = OperatorRunner(self.model_config)

    def execute(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Receives a list of pb_utils.InferenceRequest as the only argument. This
        function is called when an inference is requested for this model. Depending on the
        batching configuration (e.g. Dynamic Batching) used, `requests` may contain
        multiple requests. Every Python model, must create one pb_utils.InferenceResponse
        for every pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        params = self.model_config["parameters"]
        op_names = json.loads(params["operator_names"]["string_value"])
        first_operator_name = op_names[0]
        operator_params = json.loads(params[first_operator_name]["string_value"])
        input_column_names = list(json.loads(operator_params["input_dict"]).keys())

        responses = []

        for request in requests:
            try:
                # transform the triton tensors to a dict of name:numpy tensor
                input_tensors = {
                    name: get_input_tensor_by_name(request, name).as_numpy()
                    for name in input_column_names
                }

                inf_df = InferenceDataFrame(input_tensors)

                raw_tensor_tuples = self.runner.execute(inf_df)

                output_tensors = []
                for name, data in raw_tensor_tuples:
                    if isinstance(data, Tensor):
                        output_tensors.append(data)
                    data = data.get() if hasattr(data, "get") else data
                    tensor = Tensor(name, data)
                    output_tensors.append(tensor)

                responses.append(InferenceResponse(output_tensors))

            except Exception:  # pylint: disable=broad-except
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_string = repr(traceback.extract_tb(exc_traceback))
                responses.append(
                    pb_utils.InferenceResponse(
                        tensors=[], error=f"{exc_type}, {exc_value}, {tb_string}"
                    )
                )

        return responses
