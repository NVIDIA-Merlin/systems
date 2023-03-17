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


def compute_dims(col_schema):
    """
    Compute Triton dimensions for a column from its schema

    Parameters
    ----------
    col_schema : ColumnSchema
        Schema of the column to compute dimensions for

    Returns
    -------
    List[int]
        Triton dimensions for the column
    """
    dims = [-1]

    if col_schema.shape is not None and col_schema.shape.dims is not None:
        for dim in col_schema.shape.as_tuple[1:]:
            dim = dim if isinstance(dim, int) else -1
            dims.append(dim)

    return dims
