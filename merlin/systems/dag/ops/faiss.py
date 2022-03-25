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

import json
import os
from shutil import copy2
from typing import List

import faiss
import numpy as np
from merlin.dag import ColumnSelector
from merlin.schema import ColumnSchema, Schema

from merlin.systems.dag.ops.operator import InferenceDataFrame, PipelineableInferenceOperator


class QueryFaiss(PipelineableInferenceOperator):
    """
    This operator creates an interface between a FAISS[1] Approximate Nearest Neighbors (ANN)
    Index and Triton infrence server. It allows users to perform different supported types[2]
    of Nearest Neighbor search to your ensemble. For input query vector, we do an ANN search
    query to find the ids of top-k nearby nodes in the index.

    References
    ----------
    [1] https://github.com/facebookresearch/faiss)
    [2] https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
    """

    def __init__(self, index_path, topk=10):
        self.index_path = str(index_path)
        self.topk = topk
        self._index = None
        super().__init__()

    @classmethod
    def from_config(cls, config: dict) -> "QueryFaiss":
        """
        Instantiate a class object given a config.

        Parameters
        ----------
        config : dict


        Returns
        -------
        QueryFaiss
            class object instantiated with config values
        """
        parameters = json.loads(config.get("params", ""))
        index_path = parameters["index_path"]
        topk = parameters["topk"]

        operator = QueryFaiss(index_path, topk=topk)
        operator._index = faiss.read_index(str(index_path))

        return operator

    def export(
        self,
        path: str,
        input_schema: Schema,
        output_schema: Schema,
        params: dict = None,
        node_id: int = None,
        version: int = 1,
    ) -> List[dict, list]:
        """
        Export the class object as a config and all related files to the user defined path.

        Parameters
        ----------
        path : str
            Artifact export path
        input_schema : Schema
            A schema with information about the inputs to this operator
        output_schema : Schema
            A schema with information about the outputs of this operator
        params : dict, optional
            Parameters dictionary of key, value pairs stored in exported config, by default None
        node_id : int, optional
            The placement of the node in the graph (starts at 1), by default None
        version : int, optional
            The version of the model, by default 1

        Returns
        -------
        Ensemble_config: dict
        Node_configs: list
        """
        params = params or {}

        # TODO: Copy the index into the export directory

        self_params = {
            # TODO: Write the (relative) path from inside the export directory
            "index_path": self.index_path,
            "topk": self.topk,
        }
        self_params.update(params)
        index_filename = os.path.basename(os.path.realpath(self.index_path))

        # set index path to new path after export
        new_index_path = os.path.join(
            path, f"{node_id}_{QueryFaiss.__name__.lower()}", str(version), index_filename
        )
        copy2(self.index_path, new_index_path)
        self.index_path = new_index_path
        return super().export(path, input_schema, output_schema, self_params, node_id, version)

    def transform(self, df: InferenceDataFrame) -> InferenceDataFrame:
        """
        Transform input dataframe to output dataframe using function logic.

        Parameters
        ----------
        df : InferenceDataFrame
            Input tensor dictionary, data that will be manipulated

        Returns
        -------
        InferenceDataFrame
            Transformed tensor dictionary
        """
        user_vector = list(df.tensors.values())[0]

        _, indices = self._index.search(user_vector, self.topk)
        # distances, indices = self.index.search(user_vector, self.topk)

        candidate_ids = np.array(indices).T.astype(np.int32)

        return InferenceDataFrame({"candidate_ids": candidate_ids})

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:
        """
        Compute the input schema of this node given the root, parents and dependencies schemas of
        all ancestor nodes.

        Parameters
        ----------
        root_schema : Schema
            The schema representing the input columns to the graph
        parents_schema : Schema
            A schema representing all the output columns of the ancestors of this node.
        deps_schema : Schema
            A schema representing the dependencies of this node.
        selector : ColumnSelector
            A column selector representing a target subset of columns necessary for this node's
            operator

        Returns
        -------
        Schema
            A schema that has the correct representation of all the incoming columns necessary for
            this node's operator to complete its transform.

        Raises
        ------
        ValueError
            Cannot receive more than one input for this node
        """
        input_schema = super().compute_input_schema(
            root_schema, parents_schema, deps_schema, selector
        )
        if len(input_schema.column_schemas) > 1:
            raise ValueError(
                "More than one input has been detected for this node,"
                / f"inputs received: {input_schema.column_names}"
            )
        return input_schema

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        """
        Compute the input schema of this node given the root, parents and dependencies schemas of
        all ancestor nodes.

        Parameters
        ----------
        input_schema : Schema
            The schema representing the input columns to the graph
        col_selector : ColumnSelector
            A column selector representing a target subset of columns necessary for this node's
            operator
        prev_output_schema : Schema
            A schema representing the output of the previous node.

        Returns
        -------
        Schema
            A schema object representing all outputs of this node.
        """
        return Schema(
            [
                ColumnSchema("candidate_ids", dtype=np.int32),
            ]
        )


def setup_faiss(item_vector, output_path: str):
    """_summary_

    Parameters
    ----------
    item_vector : Numpy.ndarray
        This is a matrix representing all the nodes embeddings, represented as a numpy ndarray.
    output_path : string
        target output path
    """
    index = faiss.IndexFlatL2(item_vector[0].shape[0])
    index.add(item_vector)
    faiss.write_index(index, str(output_path))
