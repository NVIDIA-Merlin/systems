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
from typing import List

from merlin.systems.dag.ops.operator import InferenceOperator, PipelineableInferenceOperator


class TritonOperator(PipelineableInferenceOperator):
    """Base class for Triton operators."""

    def __init__(self, base_op: InferenceOperator):
        """Construct TritonOperator from a base operator.

        Parameters
        ----------
        base_op : merlin.systems.dag.ops.operator.InfereneOperator
            Base operator used to construct this Triton op.
        """
        self.op = base_op

    @property
    def export_name(self):
        """
        Provides a clear common english identifier for this operator.

        Returns
        -------
        String
            Name of the current class as spelled in module.
        """
        return self.__class__.__name__.lower()

    @property
    def exportable_backends(self) -> List[str]:
        """Returns list of supported backends.

        Returns
        -------
        List[str]
            List of supported backends
        """
        return ["ensemble", "executor"]
