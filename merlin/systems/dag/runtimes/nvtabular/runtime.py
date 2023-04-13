#
# Copyright (c) 2023, NVIDIA CORPORATION.
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
import logging

import nvtabular_cpp

from merlin.systems.dag.runtimes import Runtime
from merlin.systems.dag.runtimes.nvtabular.executor import NVTabularServingExecutor
from merlin.systems.dag.runtimes.op_table import OpTable
from nvtabular.ops import Categorify, FillMissing

LOG = logging.getLogger("merlin-systems")


NVTABULAR_OP_TABLE = OpTable()
NVTABULAR_OP_TABLE.register(
    Categorify, nvtabular_cpp.inference.CategorifyTransform, lambda op: op.encode_type != "combo"
)
NVTABULAR_OP_TABLE.register(
    FillMissing, nvtabular_cpp.inference.FillTransform, lambda op: not op.add_binary_cols
)


class NVTabularServingRuntime(Runtime):
    def __init__(self, device: str):
        super().__init__(executor=NVTabularServingExecutor(device))
        self.op_table = NVTABULAR_OP_TABLE
