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
import logging

import nvtabular_cpp

from merlin.core.dispatch import concat_columns
from merlin.dag import Graph, Node, Supports
from merlin.dag.executors import LocalExecutor
from merlin.systems.dag.runtimes import Runtime
from merlin.systems.triton.conversions import convert_format
from nvtabular.ops import Categorify, FillMissing

NVTABULAR_OP_TABLE = {}

# TODO: Figure out how evaluate a conditional and do the swap or not

# # we don't currently support 'combo'
# if self.encode_type == "combo":
#     warnings.warn("Falling back to unoptimized inference path for encode_type 'combo' ")
#     return None
NVTABULAR_OP_TABLE[Categorify] = nvtabular_cpp.inference.CategorifyTransform

# if self.add_binary_cols:
#     return None
NVTABULAR_OP_TABLE[FillMissing] = nvtabular_cpp.inference.FillTransform

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
            transformed_data, kind = self._transform_tensors(transformable, node)
            output_data = self._combine_node_outputs(node, transformed_data, output_data)

        # if we don't have tensors in numpy format, convert back so that the we can return
        # to triton
        if kind != Supports.CPU_DICT_ARRAY:
            output_data, kind = convert_format(output_data, kind, Supports.CPU_DICT_ARRAY)

        return output_data

    def _transform_tensors(self, input_tensors, workflow_node):
        upstream_inputs = []

        # Gather inputs from the parents and dependency nodes
        if workflow_node.parents_with_dependencies:
            for parent in workflow_node.parents_with_dependencies:
                upstream_tensors, upstream_kind = self._transform_tensors(input_tensors, parent)
                if upstream_tensors is not None and upstream_kind:
                    upstream_inputs.append((upstream_tensors, upstream_kind))

        # Gather additional input columns from the original input tensors
        if workflow_node.selector:
            selector_columns = workflow_node.selector.names
            to_remove = []
            for upstream_tensors, upstream_kind in upstream_inputs:
                for col in selector_columns:
                    if col in upstream_tensors:
                        to_remove.append(col)
            for col in set(to_remove):
                selector_columns.remove(col)

            if selector_columns:
                selected_tensors = {c: input_tensors[c] for c in selector_columns}
                selected_kinds = Supports.CPU_DICT_ARRAY
                upstream_inputs.append((selected_tensors, selected_kinds))

        # Standardize the formats
        tensors, kind = None, None
        for upstream_tensors, upstream_kind in upstream_inputs:
            if tensors is None:
                tensors, kind = upstream_tensors, upstream_kind
            else:
                if kind != upstream_kind:
                    # we have multiple different kinds of data here (dataframe/array on cpu/gpu)
                    # we need to convert to a common format here first before concatenating.
                    op = workflow_node.op
                    if op and hasattr(op, "inference_initialize"):
                        target_kind = self._maybe_mask_cpu_only(op.supports)
                    else:
                        target_kind = Supports.CPU_DICT_ARRAY
                    # note : the 2nd convert_format call needs to be stricter in what the kind is
                    # (exact match rather than a bitmask of values)
                    tensors, kind = convert_format(tensors, kind, target_kind)
                    upstream_tensors, _ = convert_format(upstream_tensors, upstream_kind, kind)

                tensors = self.concat_tensors([tensors, upstream_tensors], kind)

        # Run the transform
        if tensors is not None and kind and workflow_node.op:
            try:
                inference_supports = self._maybe_mask_cpu_only(workflow_node.op.supports)

                # if the op doesn't support the current kind - we need to convert
                if (
                    hasattr(workflow_node.op, "inference_initialize")
                    and not inference_supports & kind
                ):
                    tensors, kind = convert_format(tensors, kind, inference_supports)

                tensors = workflow_node.op.transform(
                    workflow_node.input_columns,
                    tensors,
                )

            except Exception:
                LOG.exception("Failed to transform operator %s", workflow_node.op)
                raise

        return tensors, kind

    def concat_tensors(self, tensors, kind):
        if kind & (Supports.GPU_DATAFRAME | Supports.CPU_DATAFRAME):
            return concat_columns(tensors)
        else:
            output = tensors[0]
            for tensor in tensors[1:]:
                output.update(tensor)
            return output

    def _maybe_mask_cpu_only(self, supported):
        # if we're running on the CPU only, mask off support for GPU data formats
        if self.device == "CPU":
            supported = functools.reduce(
                lambda a, b: a | b,
                (v for v in list(Supports) if v & supported and "CPU" in str(v)),
            )

        return supported


class NVTabularServingRuntime(Runtime):
    def __init__(self, device: str):
        super().__init__(executor=NVTabularServingExecutor(device))
        self.op_table = NVTABULAR_OP_TABLE
