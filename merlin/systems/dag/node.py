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
import os
from typing import Union

from merlin.dag import Node
from merlin.schema import Schema


class InferenceNode(Node):
    """Specialized node class used in Triton Ensemble DAGs"""

    def exportable(self, backend: str = None):
        """
        Determine whether the current node's operator is exportable for a given back-end

        Parameters
        ----------
        backend : str, optional
            The Merlin Systems (not Triton) back-end to use,
            either "ensemble" or "executor", by default None

        Returns
        -------
        bool
            True if the node's operator is exportable for the supplied back-end
        """
        backends = getattr(self.op, "exportable_backends", [])

        return hasattr(self.op, "export") and backend in backends

    def export(
        self,
        output_path: Union[str, os.PathLike],
        node_id: int = None,
        version: int = 1,
        backend="ensemble",
    ):
        """
        Export a Triton config directory for this node.

        Parameters
        ----------
        output_path : Union[str, os.PathLike]
            The base path to write this node's config directory.
        node_id : int, optional
            The id of this node in a larger graph (for disambiguation), by default None.
        version : int, optional
            The Triton model version to use for this config export, by default 1.

        Returns
        -------
        ModelConfig
            Triton model config corresponding to this node.
        """
        return self.op.export(
            output_path,
            self.input_schema,
            self.output_schema,
            node_id=node_id,
            version=version,
            backend=backend,
        )

    @property
    def export_name(self):
        """
        Name for the exported Triton config directory.

        Returns
        -------
        str
            Name supplied by this node's operator.
        """
        return self.op.export_name

    def validate_schemas(self, root_schema, strict_dtypes=False):
        """
        Checks that the output schema is valid given the previous
        nodes in the graph and following nodes in the graph, as
        well as any additional root inputs.

        Parameters
        ----------
        root_schema : Schema
            Schema of selection from the original data supplied
        strict_dtypes : bool, optional
            If True, raises an error when the dtypes in the input data
            do not match the dtypes in the schema, by default False

        Raises
        ------
        ValueError
            If an output column is produced but not used by child nodes
        """
        super().validate_schemas(root_schema, strict_dtypes)

        if self.children:
            childrens_schema = Schema()
            for elem in self.children:
                childrens_schema += elem.input_schema

            for col_name, col_schema in self.output_schema.column_schemas.items():
                sink_col_schema = childrens_schema.get(col_name)

                if not sink_col_schema:
                    raise ValueError(
                        f"Output column '{col_name}' not detected in any "
                        f"child inputs for '{self.op.__class__.__name__}'."
                    )
