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
import itertools
import json

import numpy as np

from merlin.core.compat import cupy
from merlin.systems.workflow.base import WorkflowRunner


class TensorflowWorkflowRunner(WorkflowRunner):
    def __init__(self, workflow, output_dtypes, model_config, model_device):
        super().__init__(workflow, output_dtypes, model_config, model_device)

        self.offsets = None

    def _transform_outputs(self, tensors):
        # Load extra info needed for the Transformer4Rec (if exists)
        sparse_feat = None
        params = self.model_config["parameters"]
        if "sparse_max" in params.keys():
            sparse_feat = json.loads(self.model_config["parameters"]["sparse_max"]["string_value"])

        # transforms outputs for both pytorch and tensorflow
        output_tensors = []

        for name in self.cats + self.conts:
            value = tensors[name]
            if sparse_feat and name in sparse_feat.keys():
                # convert sparse tensors to dense representations
                d = value[0].astype(self.output_dtypes[name])
                col_dim = sparse_feat[name]
                row_dim = d.shape[0] // col_dim
                d = d.reshape(row_dim, col_dim)
                output_tensors.append((name, d))
            elif isinstance(value, tuple):
                values, offsets = value
                if self.workflow.output_schema[name].is_ragged:
                    values = values.astype(self.output_dtypes[name + "__values"])
                    offsets = offsets.astype(self.output_dtypes[name + "__offsets"])
                    output_tensors.append((name + "__values", values))
                    output_tensors.append((name + "__offsets", offsets))
                else:
                    raise ValueError(
                        f"ColumnSchema for list column '{name}' describes a fixed size list. "
                        "Found a ragged list output. If this workflow outputs a ragged list, "
                        "Please check the output schema has correctly set the column shape. "
                    )
            else:
                if self.workflow.output_schema[name].is_ragged:
                    values, offsets = _to_ragged(value)
                    values = values.astype(self.output_dtypes[name + "__values"])
                    offsets = offsets.astype(self.output_dtypes[name + "__offsets"])
                    output_tensors.append((name + "__values", values))
                    output_tensors.append((name + "__offsets", offsets))
                else:
                    value = value.astype(self.output_dtypes[name])
                    output_tensors.append((name, value))

        return output_tensors


def _to_ragged(array):
    """Convert Array to Ragged representation

    Parameters
    ----------
    array : numpy.ndarray or cupy.ndarray
        Array to convert

    Returns
    -------
    values, offsets
        Tuple of values and offsets
    """
    num_rows = array.shape[0]
    row_lengths = [array.shape[1]] * num_rows
    offsets = [0] + list(itertools.accumulate(row_lengths))
    array_lib = cupy if cupy and isinstance(array, cupy.ndarray) else np
    offsets = array_lib.array(offsets, dtype="int32")
    values = array.reshape(-1, *array.shape[2:])
    return values, offsets
