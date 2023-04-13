#
# Copyright (c) 2023, NVIDIA CORPORATION.
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
import functools
import itertools
import logging

from merlin.core.compat import cudf
from merlin.core.compat import cupy as cp
from merlin.core.compat import numpy as np
from merlin.core.compat import pandas
from merlin.core.dispatch import build_cudf_list_column, concat_columns, is_list_dtype
from merlin.dag import Graph, Node, Supports
from merlin.dag.executors import LocalExecutor
from merlin.table import CupyColumn, NumpyColumn, TensorTable

LOG = logging.getLogger("merlin-systems")


class NVTabularServingExecutor(LocalExecutor):
    """
    An executor for running Merlin operator DAGs locally
    """

    def __init__(self, device: str):
        self.device = device

    def transform(
        self,
        transformable,
        graph,
        output_dtypes=None,
        additional_columns=None,
        capture_dtypes=False,
        strict=False,
        output_format=Supports.CPU_DICT_ARRAY,
    ):
        """
        Transforms a single dataframe (possibly a partition of a Dask Dataframe)
        by applying the operators from a collection of Nodes
        """
        nodes = []
        if isinstance(graph, Graph):
            nodes.append(graph.output_node)
        elif isinstance(graph, Node):
            nodes.append(graph)
        elif isinstance(graph, list):
            nodes = graph
        else:
            raise TypeError(
                f"LocalExecutor detected unsupported type of input for graph: {type(graph)}."
                " `graph` argument must be either a `Graph` object (preferred)"
                " or a list of `Node` objects (deprecated, but supported for backward "
                " compatibility.)"
            )

        output_data = None

        for node in nodes:
            transformed_data = self._execute_node(node, transformable)
            output_data = self._combine_node_outputs(node, transformed_data, output_data)

        if additional_columns:
            output_data = concat_columns(
                [output_data, transformable[_get_unique(additional_columns)]]
            )

        format_ = _data_format(output_data)
        if format_ != output_format:
            output_data = _convert_format(output_data, output_format)

        return output_data

    def _execute_node(self, workflow_node, input_tensors, capture_dtypes=False, strict=False):
        upstream_outputs = self._run_upstream_transforms(workflow_node, input_tensors)
        upstream_outputs = self._merge_addl_root_columns(
            workflow_node, input_tensors, upstream_outputs
        )
        tensors = self._standardize_formats(workflow_node, upstream_outputs)

        transform_input = _concat_tensors(tensors)
        # TODO: In order to replace the line above with the line below, we first have to replace
        #       dictionaries with TensorTables
        # transform_input = self._merge_upstream_columns(tensors, merge_fn=_concat_tensors)
        transform_output = self._run_node_transform(workflow_node, transform_input)

        return transform_output

    def _merge_addl_root_columns(self, workflow_node, input_tensors, upstream_outputs):
        if workflow_node.selector:
            selector_columns = workflow_node.selector.names
            to_remove = []
            for upstream_tensors in upstream_outputs or []:
                for col in selector_columns:
                    if col in upstream_tensors:
                        to_remove.append(col)
            for col in set(to_remove):
                selector_columns.remove(col)

            if selector_columns:
                selected_tensors = {c: input_tensors[c] for c in selector_columns}
                upstream_outputs.append(selected_tensors)

        return upstream_outputs

    def _standardize_formats(self, workflow_node, node_input_data):
        # Get the supported formats
        op = workflow_node.op
        if op and hasattr(op, "inference_initialize"):
            supported_formats = _maybe_mask_cpu_only(op.supports, self.device)
        else:
            supported_formats = Supports.CPU_DICT_ARRAY

        # Convert the first thing into a supported format
        tensors = _convert_format(node_input_data[0], supported_formats)
        target_format = _data_format(tensors)

        # Convert the whole list into the same format
        formatted_tensors = []
        for upstream_tensors in node_input_data:
            upstream_tensors = _convert_format(upstream_tensors, target_format)
            formatted_tensors.append(upstream_tensors)

        return formatted_tensors


def _concat_tensors(tensors):
    format_ = _data_format(tensors[0])

    if format_ & (Supports.GPU_DATAFRAME | Supports.CPU_DATAFRAME):
        return concat_columns(tensors)
    else:
        output = tensors[0]
        for tensor in tensors[1:]:
            output.update(tensor)
        return output


def _maybe_mask_cpu_only(supported, device):
    # if we're running on the CPU only, mask off support for GPU data formats
    if device == "CPU":
        supported = functools.reduce(
            lambda a, b: a | b,
            (v for v in list(Supports) if v & supported and "CPU" in str(v)),
        )

    return supported


def _get_unique(cols):
    # Need to preserve order in unique-column list
    return list({x: x for x in cols}.keys())


def _data_format(transformable):
    data = TensorTable(transformable) if isinstance(transformable, dict) else transformable

    if cudf and isinstance(data, cudf.DataFrame):
        return Supports.GPU_DATAFRAME
    elif pandas and isinstance(data, pandas.DataFrame):
        return Supports.CPU_DATAFRAME
    elif data.column_type is CupyColumn:
        return Supports.GPU_DICT_ARRAY
    elif data.column_type is NumpyColumn:
        return Supports.CPU_DICT_ARRAY
    else:
        if isinstance(data, TensorTable):
            raise TypeError(f"Unknown type: {data.column_type}")
        else:
            raise TypeError(f"Unknown type: {type(data)}")


def _convert_format(tensors, target_format):
    """
    Converts data to one of the formats specified in 'target_format'

    This allows us to convert data to/from dataframe representations for operators that
    only support certain reprentations
    """
    format_ = _data_format(tensors)

    # this is all much more difficult because of multihot columns, which don't have
    # great representations in dicts of cpu/gpu arrays. we're representing multihots
    # as tuples of (values, offsets) tensors in this case - but have to do work at
    # each step in terms of converting.
    if format_ & target_format:
        return tensors

    elif target_format & Supports.GPU_DICT_ARRAY:
        if format_ == Supports.CPU_DICT_ARRAY:
            return _convert_array(tensors, cp.array)
        elif format_ == Supports.CPU_DATAFRAME:
            return _pandas_to_array(tensors, False)
        elif format_ == Supports.GPU_DATAFRAME:
            return _cudf_to_array(tensors, False)

    elif target_format & Supports.CPU_DICT_ARRAY:
        if format_ == Supports.GPU_DICT_ARRAY:
            return _convert_array(tensors, cp.asnumpy)
        elif format_ == Supports.CPU_DATAFRAME:
            return _pandas_to_array(tensors, True)
        elif format_ == Supports.GPU_DATAFRAME:
            return _cudf_to_array(tensors, True)

    elif cudf and target_format & Supports.GPU_DATAFRAME:
        if format_ == Supports.CPU_DATAFRAME:
            return cudf.DataFrame(tensors)
        return _array_to_cudf(tensors)

    elif target_format & Supports.CPU_DATAFRAME:
        if format_ == Supports.GPU_DATAFRAME:
            return tensors.to_pandas()
        elif format_ == Supports.CPU_DICT_ARRAY:
            return _array_to_pandas(tensors)
        elif format_ == Supports.GPU_DICT_ARRAY:
            return _array_to_pandas(_convert_array(tensors, cp.asnumpy))

    raise ValueError("unsupported target for converting tensors", target_format)


def _convert_array(tensors, converter):
    output = {}
    for name, tensor in tensors.items():
        if isinstance(tensor, tuple):
            output[name] = tuple(converter(t) for t in tensor)
        else:
            output[name] = converter(tensor)
    return output


def _array_to_pandas(tensors):
    output = pandas.DataFrame()
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
        if pandas.api.types.is_list_like(col.values[0]):
            values = array_type(list(itertools.chain(*col)))
            row_lengths = col.map(len)
            if all(row_lengths == row_lengths[0]):
                output[name] = values.reshape((-1, row_lengths[0]))
            else:
                offsets = pandas.Series([0]).append(row_lengths.cumsum()).values
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
