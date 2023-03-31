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
from pathlib import Path
from shutil import copy2

import faiss
import numpy as np

from merlin.core.dispatch import HAS_GPU
from merlin.core.protocols import Transformable
from merlin.dag import ColumnSelector
from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ops.operator import InferenceOperator


class QueryFaiss(InferenceOperator):
    """
    This operator creates an interface between a FAISS[1] Approximate Nearest Neighbors (ANN)
    Index and Triton Infrence Server. The operator allows users to perform different supported
    types[2] of Nearest Neighbor search to your ensemble. For input query vector, we do an ANN
    search query to find the ids of top-k nearby nodes in the index.

    References
    ----------
    [1] https://github.com/facebookresearch/faiss)
    [2] https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
    """

    def __init__(self, index_path, topk=10):
        """
        Creates a QueryFaiss Pipelineable Inference Operator.

        Parameters
        ----------
        index_path : str
            A path to an already setup index
        topk : int, optional
            The number of results we should receive from query to Faiss as output, by default 10
        """
        super().__init__()

        self.index_path = str(index_path)
        self.topk = topk
        self._index = None

    def load_artifacts(self, artifact_path: str) -> None:
        filename = Path(self.index_path).name
        path_artifact = Path(artifact_path)
        if path_artifact.is_file():
            path_artifact = path_artifact.parent
        full_index_path = str(path_artifact / filename)
        index = faiss.read_index(full_index_path)

        if HAS_GPU:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        self._index = index

    def save_artifacts(self, artifact_path: str) -> None:
        index_filename = os.path.basename(os.path.realpath(self.index_path))
        new_index_path = Path(artifact_path) / index_filename
        copy2(self.index_path, new_index_path)

    def __getstate__(self) -> dict:
        """Return state of instance when pickled.

        Returns
        -------
        dict
            Returns object state excluding index attribute.
        """
        return {k: v for k, v in self.__dict__.items() if k != "_index"}

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        """
        Transform input dataframe to output dataframe using function logic.

        Parameters
        ----------
        df : TensorTable
            Input tensor dictionary, data that will be manipulated

        Returns
        -------
        TensorTable
            Transformed tensor dictionary
        """
        user_vector = list(transformable.values())[0]

        _, indices = self._index.search(user_vector.values, self.topk)

        candidate_ids = np.array(indices).astype(np.int32).flatten()

        return type(transformable)({"candidate_ids": candidate_ids})

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
                ColumnSchema("candidate_ids", dtype=np.int32, dims=(None, self.topk)),
            ]
        )

    def validate_schemas(
        self, parents_schema, deps_schema, input_schema, output_schema, strict_dtypes=False
    ):
        if len(input_schema.column_schemas) > 1:
            raise ValueError(
                "More than one input has been detected for this node,"
                / f"inputs received: {input_schema.column_names}"
            )


def setup_faiss(item_vector, output_path: str, metric=faiss.METRIC_INNER_PRODUCT):
    """
    Utiltiy function that will create a Faiss index from a set of embedding vectors

    Parameters
    ----------
    item_vector : Numpy.ndarray
        This is a matrix representing all the nodes embeddings, represented as a numpy ndarray.
    output_path : string
        target output path
    """
    ids = item_vector[:, 0].astype(np.int64)
    item_vectors = np.ascontiguousarray(item_vector[:, 1:].astype(np.float32))

    index = faiss.index_factory(item_vectors.shape[1], "IVF32,Flat", metric)
    index.nprobe = 8

    index.train(item_vectors)
    index.add_with_ids(item_vectors, ids)
    faiss.write_index(index, str(output_path))
