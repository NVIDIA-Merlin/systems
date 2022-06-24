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

from merlin.features.df import VirtualDataFrame

inf_op = pytest.importorskip("merlin.systems.dag.ops.operator")


class PlusTwoOp(inf_op.PipelineableInferenceOperator):
    def transform(self, df: VirtualDataFrame) -> VirtualDataFrame:
        focus_df = df
        new_df = VirtualDataFrame()

        for col_name in focus_df:
            new_df[f"{col_name}_plus_2"] = focus_df[col_name] + 2

        return new_df

    def column_mapping(self, col_selector):
        column_mapping = {}
        for col_name in col_selector.names:
            column_mapping[f"{col_name}_plus_2"] = [col_name]
        return column_mapping

    @classmethod
    def from_config(cls, config):
        return PlusTwoOp()
