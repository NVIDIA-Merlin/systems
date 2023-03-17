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
import os

import pandas as pd

# this needs to be before any modules that import protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tritonclient.grpc as grpcclient  # noqa
from tritonclient.utils import np_to_triton_dtype  # noqa

import merlin.dtypes as md  # noqa
from merlin.core.dispatch import is_string_dtype, make_df  # noqa
from merlin.systems.dag.ops import compute_dims  # noqa
from merlin.table import NumpyColumn, TensorTable  # noqa


def convert_table_to_triton_input(schema, batch, input_class=grpcclient.InferInput, dtype="int32"):
    """
    Convert a dataframe to a set of Triton inputs

    Parameters
    ----------
    schema : Schema
        Schema of the input data
    batch : TensorTable
        The input data itself
    input_class : Triton input class, optional
        The Triton input class to use, by default grpcclient.InferInput
    dtype : str, optional
        The dtype for lengths/offsets values, by default "int32"

    Returns
    -------
    List[input_class]
        A list of Triton inputs of the requested input class
    """
    cpu_table = batch.cpu()

    numpy_cols = {}
    for col_name, col_values in cpu_table.items():
        np_col = NumpyColumn(col_values) if not isinstance(col_values, NumpyColumn) else col_values
        numpy_cols[col_name] = np_col
    numpy_table = TensorTable(numpy_cols)

    inputs = []
    for col_name, col_values in numpy_table.to_dict().items():
        triton_input = _convert_array_to_triton_input(col_name, col_values, input_class)
        inputs.append(triton_input)

    return inputs


def convert_df_to_triton_input(schema, batch, input_class=grpcclient.InferInput, dtype="int32"):
    """
    Convert a dataframe to a set of Triton inputs

    Parameters
    ----------
    schema : Schema
        Schema of the input data
    batch : DataFrame
        The input data itself
    input_class : Triton input class, optional
        The Triton input class to use, by default grpcclient.InferInput
    dtype : str, optional
        The dtype for lengths/offsets values, by default "int32"

    Returns
    -------
    List[input_class]
        A list of Triton inputs of the requested input class
    """
    df_dict = _convert_df_to_dict(schema, batch, dtype)
    inputs = [
        _convert_array_to_triton_input(col_name, col_values, input_class)
        for col_name, col_values in df_dict.items()
    ]
    return inputs


def _convert_array_to_triton_input(col_name, col_values, input_class=grpcclient.InferInput):
    # Triton's mapping of numpy types to Triton types doesn't know how to handle string types,
    # so we need to map them to object ourselves before we call np_to_triton_dtype
    col_dtype = md.dtype(col_values.dtype).to_numpy
    dtype = np_to_triton_dtype(col_dtype)
    input_tensor = input_class(col_name, col_values.shape, dtype)

    # set_data_from_numpy checks the type against what was supplied when we created the tensor
    # using np_to_triton_dtype, so the workaround above isn't enough to make them match here.
    # Do one last `astype` cast to make absolutely sure the dtypes match.
    col_values = col_values.astype(col_dtype)
    input_tensor.set_data_from_numpy(col_values)

    return input_tensor


def convert_triton_output_to_df(columns, response):
    """
    Convert a Triton response to a dataframe

    Parameters
    ----------
    columns : List[str]
        Names of the response columns to include in the dataframe
    response : Triton output class
        The Triton response itself

    Returns
    -------
    DataFrame
        A dataframe with the requested columns
    """
    return make_df({col: response.as_numpy(col) for col in columns})


def get_column_types(path):
    """
    Load column types from a JSON file

    Parameters
    ----------
    path : str
        Path of the directory containing `column_types.json` file

    Returns
    -------
    dict
        JSON loaded from the file
    """
    path = os.path.join(path, "column_types.json")
    return json.load(open(path, encoding="utf-8"))


def _convert_tensor(t):
    out = t.as_numpy()
    if len(out.shape) == 2:
        out = out[:, 0]
    # cudf doesn't seem to handle dtypes like |S15 or object that well
    if is_string_dtype(out.dtype):
        out = out.astype("str")
    return out


def _convert_df_to_dict(schema, batch, dtype="int32"):
    df_dict = {}
    for col_name, col_schema in schema.column_schemas.items():
        col = batch[col_name]
        shape = compute_dims(col_schema)
        shape[0] = len(col)

        if col_schema.is_list:
            if isinstance(col, pd.Series):
                raise ValueError("this function doesn't support CPU list values yet")

            if col_schema.is_ragged:
                df_dict[col_name + "__values"] = col.list.leaves.values_host.astype(
                    col_schema.dtype.to_numpy
                )
                offsets = col._column.offsets.values_host.astype(dtype)
                df_dict[col_name + "__offsets"] = offsets
            else:
                values = col.list.leaves.values_host
                values = values.reshape(*shape).astype(col_schema.dtype.to_numpy)
                df_dict[col_name] = values

        else:
            values = col.values if isinstance(col, pd.Series) else col.values_host
            values = values.reshape(*shape).astype(col_schema.dtype.to_numpy)
            df_dict[col_name] = values
    return df_dict
