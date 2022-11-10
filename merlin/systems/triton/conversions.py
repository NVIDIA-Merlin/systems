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
    input_tensors = {
        name: pb_utils.get_input_tensor_by_name(request, name).as_numpy() for name in column_names
    }
    return DictArray(input_tensors)


def dict_array_to_triton_response(dictarray):
    output_tensors = []
    for name, data in dictarray.items():
        if isinstance(data, pb_utils.Tensor):
            output_tensors.append(data)
            continue
        # The .get() here handles variations across Numpy versions, some of which
        # require .get() to be used here and some of which don't.
        data = data.get() if hasattr(data, "get") else data
        if not isinstance(data.values, np.ndarray):
            tensor = pb_utils.Tensor(name, cp.asnumpy(data.values))
        else:
            tensor = pb_utils.Tensor(name, data.values)
        output_tensors.append(tensor)

    return pb_utils.InferenceResponse(output_tensors)


def dict_array_to_triton_request(model_name, dictarray, input_col_names, output_col_names):
    input_tensors = []
    for col_name in input_col_names:
        input_tensors.append(pb_utils.Tensor(col_name, dictarray[col_name].values))

    return pb_utils.InferenceRequest(
        model_name=model_name,
        requested_output_names=output_col_names,
        inputs=input_tensors,
    )


def triton_response_to_dict_array(inference_response, transformable_type, output_column_names):
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
