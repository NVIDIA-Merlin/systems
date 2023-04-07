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

from merlin.dag import ColumnSelector
from merlin.schema import Tags
from merlin.systems.dag.runtimes.nvtabular.runtime import NVTabularServingRuntime
from merlin.table import TensorTable

LOG = logging.getLogger("merlin-systems")


class WorkflowRunner:
    def __init__(self, workflow, model_config, model_device):
        self.runtime = NVTabularServingRuntime(model_device)
        workflow.graph = self.runtime.convert(workflow.graph)

        self.workflow = workflow

        schema_cats, schema_conts = _parse_schema_features(self.workflow.output_schema)
        mc_cats, mc_conts = _parse_mc_features(model_config)

        self.cats = mc_cats or schema_cats
        self.conts = mc_conts or schema_conts
        self.offsets = None

        missing_cols = set(self.cats + self.conts) - set(workflow.output_schema.column_names)

        if missing_cols:
            raise ValueError(
                "The following requested columns were not found in the workflow's output: "
                f"{missing_cols}"
            )

    def run_workflow(self, input_tensors):
        transformed = self.runtime.transform(self.workflow.graph, input_tensors)
        return TensorTable(transformed).to_dict()


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
