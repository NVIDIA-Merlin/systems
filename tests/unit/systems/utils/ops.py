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

import pytest

from merlin.core.protocols import Transformable
from merlin.dag import ColumnSelector
from merlin.systems.dag import DictArray

inf_op = pytest.importorskip("merlin.systems.dag.ops.operator")


class PlusTwoOp(inf_op.PipelineableInferenceOperator):
    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        result = DictArray()

        for name, column in transformable.items():
            result[f"{name}_plus_2"] = column.values + 2

        return result

    def column_mapping(self, col_selector):
        column_mapping = {}
        for col_name in col_selector.names:
            column_mapping[f"{col_name}_plus_2"] = [col_name]
        return column_mapping

    @classmethod
    def from_config(cls, config, **kwargs):
        return PlusTwoOp()
