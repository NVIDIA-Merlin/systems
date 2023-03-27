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
from abc import ABCMeta, abstractmethod

import tritonclient.grpc.model_config_pb2 as model_config

from merlin.core.protocols import Transformable
from merlin.dag.selector import ColumnSelector
from merlin.schema import Schema
from merlin.systems.dag.ops.operator import InferenceOperator
from merlin.systems.triton.export import _convert_dtype


class TritonOperator(InferenceOperator, metaclass=ABCMeta):
    """Base class for Triton operators."""

    def __init__(self, base_op: InferenceOperator):
        """Construct TritonOperator from a base operator.

        Parameters
        ----------
        base_op : merlin.systems.dag.ops.operator.InfereneOperator
            Base operator used to construct this Triton op.
        """
        super().__init__()
        self.op = base_op

    @property
    def export_name(self):
        """
        Provides a clear common english identifier for this operator.

        Returns
        -------
        String
            Name of the current class as spelled in module.
        """
        return self.__class__.__name__.lower()

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        """Transform the dataframe by applying this operator to the set of input columns

        Parameters
        -----------
        df: Dataframe
            A pandas or cudf dataframe that this operator will work on

        Returns
        -------
        DataFrame
            Returns a transformed dataframe for this operator
        """
        return transformable

    @abstractmethod
    def export(
        self,
        path: str,
        input_schema: Schema,
        output_schema: Schema,
        params: dict = None,
        node_id: int = None,
        version: int = 1,
    ):
        """
        Export the Operator to as a Triton Model at the path corresponding to the model repository.

        Parameters
        ----------
        path : str
            Artifact export path
        input_schema : Schema
            A schema with information about the inputs to this operator.
        output_schema : Schema
            A schema with information about the outputs of this operator.
        params : dict, optional
            Parameters dictionary of key, value pairs stored in exported config, by default None.
        node_id : int, optional
            The placement of the node in the graph (starts at 1), by default None.
        version : int, optional
            The version of the model, by default 1.

        Returns
        -------
        model_config: ModelConfig
            The config for the operator (model) if defined.
        """


def add_model_param(params, paramclass, col_schema, dims=None):
    if col_schema.is_list and col_schema.is_ragged:
        params.append(
            paramclass(
                name=col_schema.name + "__values",
                data_type=_convert_dtype(col_schema.dtype),
                dims=dims[1:],
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
