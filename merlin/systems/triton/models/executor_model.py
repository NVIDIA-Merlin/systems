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
import pathlib
import sys
import traceback
from pathlib import Path

import triton_python_backend_utils as pb_utils

from merlin.dag import postorder_iter_nodes
from merlin.systems.dag import Ensemble
from merlin.systems.dag.runtimes.triton import TritonExecutorRuntime
from merlin.systems.triton.conversions import (
    dict_array_to_triton_response,
    triton_request_to_dict_array,
)


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
        # Arg parsing
        model_repo = args["model_repository"]
        repository_path = _parse_model_repository(model_repo)

        ensemble_path = (
            Path(repository_path) / args["model_name"] / str(args["model_version"]) / "ensemble"
        )

        self.ensemble = Ensemble.load(str(ensemble_path))

        for node in list(postorder_iter_nodes(self.ensemble.graph.output_node)):
            if hasattr(node.op, "load_artifacts"):
                node.op.load_artifacts(str(ensemble_path))

    def execute(self, requests):
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
        responses = []

        for request in requests:
            try:
                inputs = triton_request_to_dict_array(
                    request, self.ensemble.input_schema.column_names
                )
                outputs = self.ensemble.transform(inputs, runtime=TritonExecutorRuntime())
                output_tensors = dict_array_to_triton_response(outputs)
                responses.append(output_tensors)

            except Exception:  # pylint: disable=broad-except
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_string = repr(traceback.extract_tb(exc_traceback))
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(f"{exc_type}, {exc_value}, {tb_string}"),
                    )
                )

        return responses


def _parse_model_repository(model_repository: str) -> str:
    """
    Extract the model repository path from the model_repository value
    passed to the TritonPythonModel initialize method.
    """
    # Handle bug in Tritonserver 22.06
    # model_repository argument became path to model.py
    # instead of path to model directory within the model repository
    if model_repository.endswith(".py"):
        return str(pathlib.Path(model_repository).parent.parent.parent)
    else:
        return str(pathlib.Path(model_repository).parent)
