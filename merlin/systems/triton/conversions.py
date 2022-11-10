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

try:
    import cudf
    import cupy as cp
except ImportError:
    cudf = cp = None

import numpy as np
import pandas as pd

from merlin.core.dispatch import build_cudf_list_column, is_list_dtype
from merlin.dag import Supports
from merlin.systems.dag import DictArray
from merlin.systems.dag.ops.compat import pb_utils


def triton_request_to_dict_array(request, column_names):
    """
    Turns a Triton request into a DictArray by extracting individual tensors
    from the request using pb_utils.

    Parameters
    ----------
    request : TritonInferenceRequest
        The incoming request for predictions
    column_names : List[str]
        List of the input columns to extract from the request

    Returns
    -------
    DictArray
        Dictionary-like representation of the input columns
    """
    dict_inputs = {}
    for name in column_names:
        try:
            values = _get_triton_tensor(request, f"{name}__values")
            lengths = _get_triton_tensor(request, f"{name}__lengths")
            dict_inputs[name] = (values, lengths)
        except AttributeError:
            dict_inputs[name] = _get_triton_tensor(request, name)

    return DictArray(dict_inputs)


def _get_triton_tensor(request, name):
    return pb_utils.get_input_tensor_by_name(request, name).as_numpy()


def dict_array_to_triton_response(dictarray):
    """
    Turns a DictArray into a Triton response that can be returned
    to resolve an incoming request.

    Parameters
    ----------
    dictarray : DictArray
        Dictionary-like representation of the output columns

    Returns
    -------
    response : TritonInferenceResponse
        The output response for predictions
    """
    output_tensors = []
    for name, column in dictarray.items():
        if column.is_list:
            values = _make_triton_tensor(f"{name}__values", column.values)
            lengths = _make_triton_tensor(f"{name}__lengths", column.row_lengths)
            output_tensors.extend([values, lengths])
        else:
            col_tensor = _make_triton_tensor(name, column.values)
            output_tensors.append(col_tensor)

    return pb_utils.InferenceResponse(output_tensors)


def _make_triton_tensor(name, array):
    # The .get() here handles variations across Numpy versions, some of which
    # require .get() to be used here and some of which don't.
    array = array.get() if hasattr(array, "get") else array
    if not isinstance(array, np.ndarray):
        tensor = pb_utils.Tensor(name, cp.asnumpy(array))
    else:
        tensor = pb_utils.Tensor(name, array)
    return tensor


def dict_array_to_triton_request(model_name, dictarray, input_col_names, output_col_names):
    """
    Turns a DictArray into a Triton request that can, for example, be used to make a
    Business Logic Scripting call to a Triton model on the same Triton instance.

    Parameters
    ----------
    model_name : String
        Name of model registered in triton
    dictarray : DictArray
        Dictionary-like representation of the output columns
    input_col_names : List[str]
        List of the input columns to create triton request
    output_col_names : List[str]
        List of the output columns to extract from the response

    Returns
    -------
    TritonInferenceRequest
        The DictArray reformatted as a Triton request
    """
    input_tensors = []
    for col_name in input_col_names:
        input_tensors.append(pb_utils.Tensor(col_name, dictarray[col_name].values))

    return pb_utils.InferenceRequest(
        model_name=model_name,
        requested_output_names=output_col_names,
        inputs=input_tensors,
    )


def triton_response_to_dict_array(inference_response, transformable_type, output_column_names):
    """
    Turns a Triton response into a DictArray by extracting individual tensors
    from the request using pb_utils.

    Parameters
    ----------
    inference_response : pb_utils.InferenceResponse
        Response received from triton containing prediction
    transformable_type : Union[pd.DataFrame, cudf.DataFrame, DictArray]
        The specific type of object matching the Transformable protocol to create
    output_col_names : List[str]
        List of the output columns to extract from the response

    Returns
    -------
    Transformable
        A DictArray or DataFrame representing the response columns from a Triton request
    """
    outputs_dict = {}
    for out_col_name in output_column_names:
        output_val = pb_utils.get_output_tensor_by_name(inference_response, out_col_name)

        if output_val.is_cpu():
            output_val = output_val.as_numpy()
        elif cp:
            output_val = cp.fromDlpack(output_val.to_dlpack())

        outputs_dict[out_col_name] = output_val

    return transformable_type(outputs_dict)


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

    elif target_kind & Supports.GPU_DATAFRAME:
        if kind == Supports.CPU_DATAFRAME:
            return cudf.DataFrame(tensors), Supports.GPU_DATAFRAME
        return _array_to_cudf(tensors), Supports.GPU_DATAFRAME

    elif target_kind & Supports.CPU_DATAFRAME:
        if kind == Supports.GPU_DATAFRAME:
            return tensors.to_pandas(), Supports.CPU_DATAFRAME
        elif kind == Supports.CPU_DICT_ARRAY:
            return _array_to_pandas(tensors), Supports.CPU_DATAFRAME
        elif kind == Supports.GPU_DICT_ARRAY:
            return _array_to_pandas(_convert_array(tensors, cp.asnumpy)), Supports.CPU_DATAFRAME

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
            offsets = pd.Series([0]).append(col.map(len).cumsum()).values
            if not cpu:
                offsets = cp.array(offsets)
            values = array_type(list(itertools.chain(*col)))
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
            offsets = col._column.offsets.values_host if cpu else col._column.offsets.values
            values = col.list.leaves.values_host if cpu else col.list.leaves.values
            output[name] = (values, offsets)
        else:
            output[name] = col.values_host if cpu else col.values

    return output
