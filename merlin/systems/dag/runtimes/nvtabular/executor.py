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
import logging
from enum import Enum

from merlin.core.compat import HAS_GPU
from merlin.core.dispatch import concat_columns
from merlin.dag import DataFormats, Graph, Node
from merlin.dag.executors import LocalExecutor, _convert_format, _data_format

LOG = logging.getLogger("merlin-systems")


class Device(Enum):
    CPU = 0
    GPU = 1


class NVTabularServingExecutor(LocalExecutor):
    """
    An executor for running Merlin operator DAGs locally
    """

    def __init__(self, device=Device.GPU):
        self.device = device if HAS_GPU else Device.CPU

    def transform(
        self,
        transformable,
        graph,
        output_dtypes=None,
        additional_columns=None,
        capture_dtypes=False,
        strict=False,
        output_format=DataFormats.NUMPY_DICT_ARRAY,
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
        upstream_outputs = self._append_addl_root_columns(
            workflow_node, input_tensors, upstream_outputs
        )
        tensors = self._standardize_formats(workflow_node, upstream_outputs)
        transform_input = self._merge_upstream_columns(tensors, merge_fn=_concat_tensors)
        transform_output = self._run_node_transform(workflow_node, transform_input)

        return transform_output


def _concat_tensors(tensors):
    format_ = _data_format(tensors[0])

    if format_ & (DataFormats.CUDF_DATAFRAME | DataFormats.PANDAS_DATAFRAME):
        return concat_columns(tensors)
    else:
        output = tensors[0]
        for tensor in tensors[1:]:
            output.update(tensor)
        return output


def _get_unique(cols):
    # Need to preserve order in unique-column list
    return list({x: x for x in cols}.keys())
