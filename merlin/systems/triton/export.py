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
import os

# this needs to be before any modules that import protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tritonclient.grpc.model_config_pb2 as model_config  # noqa

import merlin.dtypes as md  # noqa
from merlin.core.dispatch import is_string_dtype  # noqa
from merlin.systems.dag.ops import compute_dims  # noqa


def _add_model_param(col_schema, paramclass, params, dims=None):
    dims = dims if dims is not None else compute_dims(col_schema)
    if col_schema.is_list and col_schema.is_ragged:
        params.append(
            paramclass(
                name=col_schema.name + "__values",
                data_type=_convert_dtype(col_schema.dtype),
                dims=[-1],
            )
        )
        params.append(
            paramclass(
                name=col_schema.name + "__offsets", data_type=model_config.TYPE_INT32, dims=[-1]
            )
        )
    else:
        params.append(
            paramclass(name=col_schema.name, data_type=_convert_dtype(col_schema.dtype), dims=dims)
        )


def _convert_dtype(dtype):
    """converts a dtype to the appropriate triton proto type"""
    dtype = md.dtype(dtype)
    try:
        return dtype.to("triton")
    except ValueError:
        dtype = dtype.to_numpy

    if is_string_dtype(dtype):
        return model_config.TYPE_STRING
    else:
        raise ValueError(f"Can't convert {dtype} to a Triton dtype")


def _convert_string2pytorch_dtype(dtype):
    """converts a dtype to the appropriate torch type"""

    import torch

    if not isinstance(dtype, str):
        dtype_name = dtype.name
    else:
        dtype_name = dtype

    dtypes = {
        "TYPE_FP64": torch.float64,
        "TYPE_FP32": torch.float32,
        "TYPE_FP16": torch.float16,
        "TYPE_INT64": torch.int64,
        "TYPE_INT32": torch.int32,
        "TYPE_INT16": torch.int16,
        "TYPE_INT8": torch.int8,
        "TYPE_UINT8": torch.uint8,
        "TYPE_BOOL": torch.bool,
    }

    if is_string_dtype(dtype):
        return model_config.TYPE_STRING
    elif dtype_name in dtypes:
        return dtypes[dtype_name]
    else:
        raise ValueError(f"Can't convert dtype {dtype})")
