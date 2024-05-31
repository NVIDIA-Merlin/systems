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
import json
import logging
import os

from merlin.dag import ColumnSelector, DataFormats, Supports
from merlin.dag.executors import LocalExecutor, _convert_format, _data_format
from merlin.schema import Tags
from merlin.systems.triton.conversions import match_representations
from merlin.table import TensorTable

LOG = logging.getLogger("merlin-systems")


class WorkflowRunner:
    def __init__(self, workflow, output_dtypes, model_config, model_device):
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
        self.offsets = None

        workflow_outputs = set(workflow.output_schema.column_names)
        requested_cols = set(self.cats + self.conts)
        missing_cols = requested_cols - workflow_outputs

        if missing_cols:
            raise ValueError(
                f"The following columns were not found in the workflow's output: {missing_cols}"
            )

        # recurse over all column groups, initializing operators for inference pipeline.
        # (disabled everything other than operators that are specifically listed
        # by the `NVT_CPP_OPS` environment variable while we sort out whether
        # and how we want to use C++ implementations of NVTabular operators for
        # performance optimization)
        _nvt_cpp_ops = os.environ.get("NVT_CPP_OPS", "").split(",")
        self._initialize_ops(self.workflow.output_node, restrict=_nvt_cpp_ops)

    def _initialize_ops(self, workflow_node, visited=None, restrict=None):
        restrict = restrict or []

        if visited is None:
            visited = set()

        if (
            workflow_node.op
            and hasattr(workflow_node.op, "inference_initialize")
            and (not restrict or workflow_node.op.label in restrict)
        ):
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
                self._initialize_ops(parent, visited=visited, restrict=restrict)

    def run_workflow(self, input_tensors):
        transformable = TensorTable(input_tensors).to_df()
        transformed = LocalExecutor().transform(transformable, self.workflow.graph)

        if _data_format(transformed) != DataFormats.NUMPY_DICT_ARRAY:
            transformed = _convert_format(transformed, DataFormats.NUMPY_DICT_ARRAY)

        return match_representations(self.workflow.output_schema, transformed)

    def _get_param(self, config, *args, default=None):
        config_element = config["parameters"]
        for key in args:
            config_element = config_element.get(key, {})
        return config_element or default
