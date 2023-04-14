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

import functools
import itertools
import json
import logging

import numpy as np

from merlin.core.compat import cupy
from merlin.dag import ColumnSelector, Supports
from merlin.schema import Tags
from merlin.systems.dag.runtimes.nvtabular.runtime import NVTabularServingRuntime
from merlin.systems.triton.conversions import convert_format
from merlin.table import TensorColumn, TensorTable

LOG = logging.getLogger("merlin-systems")


class WorkflowRunner:
    def __init__(self, workflow, output_dtypes, model_config, model_device):

        self.runtime = NVTabularServingRuntime(model_device)

        self.workflow = workflow
        self.output_dtypes = output_dtypes
        self.model_config = model_config
        self.device = model_device

        output_schema = self.workflow.output_schema

        schema_cats = output_schema.apply(ColumnSelector(tags=[Tags.CATEGORICAL])).column_names
        schema_conts = output_schema.apply(ColumnSelector(tags=[Tags.CONTINUOUS])).column_names

        mc_cats = json.loads(self._get_param(model_config, "cats", "string_value", default="[]"))
        mc_conts = json.loads(self._get_param(model_config, "conts", "string_value", default="[]"))

        self.cats = mc_cats or schema_cats
        self.conts = mc_conts or schema_conts

        workflow_outputs = set(workflow.output_schema.column_names)
        requested_cols = set(self.cats + self.conts)
        missing_cols = requested_cols - workflow_outputs

        if missing_cols:
            raise ValueError(
                f"The following columns were not found in the workflow's output: {missing_cols}"
            )

        # recurse over all column groups, initializing operators for inference pipeline
        self._initialize_ops(self.workflow.output_node)

    def _initialize_ops(self, workflow_node, visited=None):
        if visited is None:
            visited = set()

        if workflow_node.op and hasattr(workflow_node.op, "inference_initialize"):
            inference_op = workflow_node.op.inference_initialize(
                workflow_node.selector, self.model_config
            )
            if inference_op:
                workflow_node.op = inference_op

            supported = workflow_node.op.supports

            # if we're running on the CPU only, mask off support for GPU data formats
            if self.device == "CPU":
                supported = functools.reduce(
                    lambda a, b: a | b,
                    (v for v in list(Supports) if v & supported and "CPU" in str(v)),
                )
            # the 'supports' property is readonly, and we can't always attach a new property
            # to some of the operators (C++ categorify etc). set on the workflow_node instead
            workflow_node.inference_supports = supported

        for parent in workflow_node.parents_with_dependencies:
            if parent not in visited:
                visited.add(parent)
                self._initialize_ops(parent, visited)

    def run_workflow(self, input_tensors):
        # use our NVTabular workflow to transform the dataset
        transformed, kind = self.runtime.transform(self.workflow.graph, input_tensors)

        # if we don't have tensors in numpy format, convert back so that the we can return
        # to triton
        if kind != Supports.CPU_DICT_ARRAY:
            transformed, kind = convert_format(transformed, kind, Supports.CPU_DICT_ARRAY)

        output_table = TensorTable(transformed)

        for col in self.workflow.output_schema:
            if col.is_ragged and output_table[col.name].offsets is None:
                values, offsets = _to_ragged(output_table[col.name].values)
                output_table[col.name] = TensorColumn(values, offsets=offsets)

        output_dict = output_table.to_dict()

        for key, value in output_dict.items():
            output_dict[key] = value.astype(self.output_dtypes[key])

        return output_dict

    def _get_param(self, config, *args, default=None):
        config_element = config["parameters"]
        for key in args:
            config_element = config_element.get(key, {})
        return config_element or default


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
