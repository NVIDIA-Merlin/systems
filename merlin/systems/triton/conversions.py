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

import itertools
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import merlin.dtypes as md
from merlin.core.compat import cudf
from merlin.core.compat import cupy as cp
from merlin.core.dispatch import build_cudf_list_column, is_list_dtype
from merlin.dag import Supports
from merlin.schema import Schema
from merlin.systems.dag.ops.compat import pb_utils
from merlin.table import TensorTable


def tensor_names(schema: Schema) -> List[str]:
    """
    Compute the expected tensor names from a Merlin schema

    This takes the columns from a schema, checks whether the columns are ragged or not,
    and translates ragged columns to two separate tensor names for the values/offsets
    representation.

    Parameters
    ----------
    schema : Schema
        Schema to compute tensor names for

    Returns
    -------
    List[str]
        A list of the tensors implied by the schema
    """
    tensor_names = []
    for col_name, col_schema in schema.column_schemas.items():
        if col_schema.is_ragged:
            tensor_names.append(f"{col_name}__values")
            tensor_names.append(f"{col_name}__offsets")
        else:
            tensor_names.append(col_name)
    return tensor_names


def match_representations(schema: Schema, dict_array: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert values-only tensors to values/offsets when indicated by the schema

    Parameters
    ----------
    schema : Schema
        Downstream input schema to match
    dict_array : Dict[str, Any]
        A dictionary of NumPy or CuPy ndarrays

    Returns
    -------
    Dict[str, Any]
        A dictionary of NumPy or CuPy ndarrays with representations adjusted
    """
    aligned = {}
    for col_name, col_schema in schema.column_schemas.items():
        dtype = col_schema.dtype

        vals_name = f"{col_name}__values"
        offs_name = f"{col_name}__offsets"

        if col_schema.is_ragged:
            try:
                # Look for values and offsets that already exist
                aligned[vals_name] = dict_array[vals_name]
                aligned[offs_name] = dict_array[offs_name]
            except KeyError:
                # If you don't find them, create the offsets
                values, offsets = _to_values_offsets(dict_array[col_name])
                aligned[vals_name] = values
                aligned[offs_name] = offsets

            if dtype != md.unknown:
                aligned[vals_name] = aligned[vals_name].astype(dtype.to_numpy)
        else:
            try:
                # Look for values and offsets that already exist,
                # then reshape accordingly
                aligned[col_name] = dict_array[vals_name]

                new_shape = [-1]
                new_shape.extend(col_schema.shape.as_tuple[1:])

                aligned[col_name] = aligned[col_name].reshape(new_shape)
            except KeyError:
                # If you don't find them, just use the values
                aligned[col_name] = dict_array[col_name]

            if dtype != md.unknown:
                aligned[col_name] = aligned[col_name].astype(dtype.to_numpy)

    return aligned


def _to_values_offsets(array):
    """Convert array to values/offsets representation

    Parameters
    ----------
    array : numpy.ndarray or cupy.ndarray
        Array to convert

    Returns
    -------
    values, offsets
        Tuple of values and offsets
    """
    num_rows = array.shape[0]
    row_lengths = [array.shape[1]] * num_rows
    offsets = [0] + list(itertools.accumulate(row_lengths))
    array_lib = cp if cp and isinstance(array, cp.ndarray) else np
    offsets = array_lib.array(offsets, dtype="int32")
    values = array.reshape(-1, *array.shape[2:])
    return values, offsets


def triton_request_to_tensor_table(request, schema):
    """
    Turns a Triton request into a TensorTable by extracting individual tensors
    from the request using pb_utils.

    Parameters
    ----------
    request : TritonInferenceRequest
        The incoming request for predictions
    column_names : List[str]
        List of the input columns to extract from the request

    Returns
    -------
    TensorTable
        Dictionary-like representation of the input columns
    """
    return TensorTable(
        {name: _array_from_triton_tensor(request, name) for name in tensor_names(schema)}
    )


def tensor_table_to_triton_response(tensor_table, schema):
    """
    Turns a TensorTable into a Triton response that can be returned
    to resolve an incoming request.

    Parameters
    ----------
    tensor_table : TensorTable
        Dictionary-like representation of the output columns

    Returns
    -------
    response : TritonInferenceResponse
        The output response for predictions
    """
    aligned = match_representations(schema, tensor_table.to_dict())
    return pb_utils.InferenceResponse(
        [_triton_tensor_from_array(name, array) for name, array in aligned.items()]
    )


def tensor_table_to_triton_request(model_name, tensor_table, input_schema, output_schema):
    """
    Turns a TensorTable into a Triton request that can, for example, be used to make a
    Business Logic Scripting call to a Triton model on the same Triton instance.

    Parameters
    ----------
    model_name : String
        Name of model registered in triton
    tensor_table : TensorTable
        Dictionary-like representation of the output columns
    input_col_names : List[str]
        List of the input columns to create triton request
    output_col_names : List[str]
        List of the output columns to extract from the response

    Returns
    -------
    TritonInferenceRequest
        The TensorTable reformatted as a Triton request
    """
    aligned = match_representations(input_schema, tensor_table.to_dict())
    input_tensors = [_triton_tensor_from_array(name, tensor) for name, tensor in aligned.items()]

    return pb_utils.InferenceRequest(
        model_name=model_name,
        requested_output_names=tensor_names(output_schema),
        inputs=input_tensors,
    )


def triton_response_to_tensor_table(response, transformable_type, schema):
    """
    Turns a Triton response into a TensorTable by extracting individual tensors
    from the request using pb_utils.

    Parameters
    ----------
    response : pb_utils.InferenceResponse
        Response received from triton containing prediction
    transformable_type : Union[pd.DataFrame, cudf.DataFrame, TensorTable]
        The specific type of object matching the Transformable protocol to create
    output_col_names : List[str]
        List of the output columns to extract from the response

    Returns
    -------
    Transformable
        A TensorTable or DataFrame representing the response columns from a Triton request
    """
    return transformable_type(
        {name: _array_from_triton_tensor(response, name) for name in tensor_names(schema)}
    )


def _triton_tensor_from_array(name, array):
    # The .get() here handles variations across Numpy versions, some of which
    # require .get() to be used here and some of which don't.
    array = array.get() if hasattr(array, "get") else array
    if not isinstance(array, np.ndarray):
        # TODO: Find a way to keep GPU arrays on the GPU instead of forcing a move to CPU here
        # This move is a workaround for CuPy and Triton dlpack implementations not working together
        tensor = pb_utils.Tensor(name, cp.asnumpy(array))
    else:
        tensor = pb_utils.Tensor(name, array)
    return tensor


def _array_from_triton_tensor(triton_obj, name):
    if isinstance(triton_obj, pb_utils.InferenceRequest):
        tensor = pb_utils.get_input_tensor_by_name(triton_obj, name)
    elif isinstance(triton_obj, pb_utils.InferenceResponse):
        tensor = pb_utils.get_output_tensor_by_name(triton_obj, name)
    else:
        raise TypeError(
            "Can only convert Triton tensors from InferenceRequest and "
            f"InferenceResponse to Numpy/CuPy arrays, but found type {type(triton_obj)}"
        )

    if tensor is None:
        raise ValueError(f"Column {name} not found in {type(triton_obj)}")

    return _to_array_lib(tensor)


def _to_array_lib(triton_tensor):
    if triton_tensor.is_cpu():
        return triton_tensor.as_numpy()
    elif cp:
        return cp.fromDlpack(triton_tensor.to_dlpack())
    else:
        raise TypeError(
            "Can't convert Triton GPU tensors to CuPy tensors without CuPy available. "
            "Is it installed?"
        )


def convert_format(tensors, kind, target_kind):
    """Converts data from format 'kind' to one of the formats specified in 'target_kind'
    This allows us to convert data to/from dataframe representations for operators that
    only support certain reprentations
    """

    # this is all much more difficult because of multihot columns, which don't have
    # great representations in dicts of cpu/gpu arrays. we're representing multihots
    # as tuples of (values, offsets) tensors in this case - but have to do work at
    # each step in terms of converting.
    if kind & target_kind:
        return tensors, kind

    elif target_kind & Supports.GPU_DICT_ARRAY:
        if kind == Supports.CPU_DICT_ARRAY:
            return _convert_array(tensors, cp.array), Supports.GPU_DICT_ARRAY
        elif kind == Supports.CPU_DATAFRAME:
            return _pandas_to_array(tensors, False), Supports.GPU_DICT_ARRAY
        elif kind == Supports.GPU_DATAFRAME:
            return _cudf_to_array(tensors, False), Supports.GPU_DICT_ARRAY

    elif target_kind & Supports.CPU_DICT_ARRAY:
        if kind == Supports.GPU_DICT_ARRAY:
            return _convert_array(tensors, cp.asnumpy), Supports.CPU_DICT_ARRAY
        elif kind == Supports.CPU_DATAFRAME:
            return _pandas_to_array(tensors, True), Supports.CPU_DICT_ARRAY
        elif kind == Supports.GPU_DATAFRAME:
            return _cudf_to_array(tensors, True), Supports.CPU_DICT_ARRAY

    elif cudf and target_kind & Supports.GPU_DATAFRAME:
        if kind == Supports.CPU_DATAFRAME:
            return cudf.DataFrame(tensors), Supports.GPU_DATAFRAME
        return _array_to_cudf(tensors), Supports.GPU_DATAFRAME

    elif target_kind & Supports.CPU_DATAFRAME:
        if kind == Supports.GPU_DATAFRAME:
            return tensors.to_pandas(), Supports.CPU_DATAFRAME
        elif kind == Supports.CPU_DICT_ARRAY:
            return _array_to_pandas(tensors), Supports.CPU_DATAFRAME
        elif kind == Supports.GPU_DICT_ARRAY:
            return (
                _array_to_pandas(_convert_array(tensors, cp.asnumpy)),
                Supports.CPU_DATAFRAME,
            )

    raise ValueError("unsupported target for converting tensors", target_kind)


def _convert_array(tensors, converter):
    output = {}
    for name, tensor in tensors.items():
        if isinstance(tensor, tuple):
            output[name] = tuple(converter(t) for t in tensor)
        else:
            output[name] = converter(tensor)
    return output


def _array_to_pandas(tensors):
    output = pd.DataFrame()
    for name, tensor in tensors.items():
        if isinstance(tensor, tuple):
            values, offsets = tensor
            output[name] = [values[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]
        else:
            output[name] = tensor
    return output


def _array_to_cudf(tensors):
    output = cudf.DataFrame()
    for name, tensor in tensors.items():
        if isinstance(tensor, tuple):
            output[name] = build_cudf_list_column(tensor[0], tensor[1].astype("int32"))
        else:
            output[name] = tensor
    return output


def _pandas_to_array(df, cpu=True):
    array_type = np.array if cpu else cp.array

    output = {}
    for name in df.columns:
        col = df[name]
        if pd.api.types.is_list_like(col.values[0]):
            values = array_type(list(itertools.chain(*col)))
            row_lengths = col.map(len)
            if all(row_lengths == row_lengths[0]):
                output[name] = values.reshape((-1, row_lengths[0]))
            else:
                offsets = pd.Series([0]).append(row_lengths.cumsum()).values
                if not cpu:
                    offsets = cp.array(offsets)
                output[name] = (values, offsets)
        else:
            values = col.values
            if not cpu:
                values = cp.array(values)
            output[name] = values

    return output


def _cudf_to_array(df, cpu=True):
    output = {}
    for name in df.columns:
        col = df[name]
        if is_list_dtype(col.dtype):
            values = col.list.leaves.values_host if cpu else col.list.leaves.values
            offsets = col._column.offsets.values_host if cpu else col._column.offsets.values

            row_lengths = offsets[1:] - offsets[:-1]
            if all(row_lengths == row_lengths[0]):
                output[name] = values.reshape((-1, row_lengths[0]))
            else:
                output[name] = (values, offsets)
        else:
            output[name] = col.values_host if cpu else col.values

    return output
