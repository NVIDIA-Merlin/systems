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
import json

import numpy as np

from merlin.core.protocols import Transformable
from merlin.dag import Node
from merlin.dag.selector import ColumnSelector
from merlin.schema import Schema
from merlin.systems.dag.ops.operator import PipelineableInferenceOperator


class UnrollFeatures(PipelineableInferenceOperator):
    """
    This operator takes a target column and joins the "unroll" columns to the target. This helps
    when broadcasting a series of user features to a set of items.
    """

    def __init__(self, item_id_col, unroll_cols, unrolled_prefix=""):
        self.item_id_col = item_id_col
        self.unroll_cols = Node.construct_from(unroll_cols)
        self.unrolled_prefix = unrolled_prefix
        super().__init__()

    @classmethod
    def from_config(cls, config, **kwargs) -> "UnrollFeatures":
        """Load operator and properties from Triton config"""
        parameters = json.loads(config.get("params", ""))
        candidate_col = parameters["item_id_col"]
        unroll_cols = parameters["unroll_cols"]
        unrolled_prefix = parameters["unrolled_prefix"]
        return UnrollFeatures(candidate_col, unroll_cols, unrolled_prefix)

    def export(
        self,
        path: str,
        input_schema: Schema,
        output_schema: Schema,
        params: dict = None,
        node_id: int = None,
        version: int = 1,
        backend: str = "ensemble",
    ):
        """Write out a Triton model config directory"""
        params = params or {}
        self_params = {
            "item_id_col": self.item_id_col,
            "unroll_cols": self._unroll_col_names,
            "unrolled_prefix": self.unrolled_prefix,
        }
        self_params.update(params)
        return super().export(path, input_schema, output_schema, self_params, node_id, version)

    @property
    def dependencies(self):
        return self.unroll_cols

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        schema = super().compute_output_schema(input_schema, col_selector, prev_output_schema)

        for col_name, col_schema in self.unroll_cols.output_schema.column_schemas.items():
            schema.column_schemas.pop(col_name, None)
            col_name = f"{self.unrolled_prefix}_{col_name}" if self.unrolled_prefix else col_name
            schema[col_name] = col_schema.with_name(col_name)

        return schema

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        num_items = transformable[self.item_id_col].shape[0]
        outputs = {}
        for col_name, col_value in transformable.items():
            outputs[col_name] = col_value

        for col in self._unroll_col_names:
            target = outputs.pop(col)
            col_name = f"{self.unrolled_prefix}_{col}" if self.unrolled_prefix else col
            outputs[col_name] = np.repeat(target, num_items, axis=0)

        return type(transformable)(outputs)

    @property
    def _unroll_col_names(self):
        if self.unroll_cols.selector:
            return self.unroll_cols.selector.names
        else:
            return self.unroll_cols.output_columns.names
