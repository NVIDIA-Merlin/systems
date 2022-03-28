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

    def export(self, output_path: Union[str, os.PathLike], node_id: int = None, version: int = 1):
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
            output_path, self.input_schema, self.output_schema, node_id=node_id, version=version
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
