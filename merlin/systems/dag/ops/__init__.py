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


def compute_dims(col_schema, scalar_shape=None):
    """
    Compute Triton dimensions for a column from its schema

    Parameters
    ----------
    col_schema : ColumnSchema
        Schema of the column to compute dimensions for
    scalar_shape : List[int], optional
        The shape of a single scalar element, by default None

    Returns
    -------
    List[int]
        Triton dimensions for the column
    """
    batch_dim = [-1]

    default_scalar_shape = col_schema.properties.get("triton_scalar_shape", [1])
    column_dims = scalar_shape if scalar_shape is not None else default_scalar_shape
    assert isinstance(column_dims, list)

    if col_schema.is_list:
        value_count = col_schema.properties.get("value_count", None)
        if value_count and value_count["max"] > 0 and value_count["min"] == value_count["max"]:
            column_dims = [value_count["max"]]
        else:
            column_dims = [-1]

    return batch_dim + column_dims
