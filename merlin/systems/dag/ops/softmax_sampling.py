import json
from typing import List, Optional

import numpy as np

from merlin.dag.node import Node
from merlin.dag.selector import ColumnSelector
from merlin.schema import Schema
from merlin.systems.dag.ops.operator import InferenceDataFrame, PipelineableInferenceOperator


class SoftmaxSampling(PipelineableInferenceOperator):
    """
    Given inputs of ID and prediction, this operator will sort all
    inputs in descending order.
    """

    def __init__(
        self, relevance_col, temperature=20.0, topk=10, _input_cols: Optional[List[str]] = None
    ):
        """
        Create a SoftmaxSampling Pipelineable Inference Operator.

        Parameters
        ----------
        relevance_col : string
            The column to judge sorting order with.
        temperature : float, optional
            Value which will be used to effect the weights used in sorting, by default 20.0
        topk : int, optional
            The max number of results you wish to receive as output, by default 10
        _input_cols : List[str], optional
            The column(s) whose values will be sorted, by default None.
        """
        self.relevance_col = Node.construct_from(relevance_col)
        self.temperature = temperature
        self.topk = topk
        self._input_col_names: List[str] = _input_cols or []
        self._relevance_col_name = relevance_col
        super().__init__()

    @classmethod
    def from_config(cls, config, **kwargs) -> "SoftmaxSampling":
        """Load operator and properties from Triton config"""
        parameters = json.loads(config.get("params", ""))
        relevance_col = parameters["relevance_col"]
        input_cols = parameters["input_col"]
        if isinstance(input_cols, str):
            input_cols = [input_cols]
        temperature = parameters["temperature"]
        topk = parameters["topk"]

        return SoftmaxSampling(
            relevance_col, temperature=temperature, topk=topk, _input_cols=input_cols
        )

    @property
    def dependencies(self):
        return self.relevance_col

    def export(self, path, input_schema, output_schema, params=None, node_id=None, version=1):
        """Write out a Triton model config directory"""
        params = params or {}
        self_params = {
            "input_cols": self._input_col_names,
            "relevance_col": self._relevance_col_name,
            "temperature": self.temperature,
            "topk": self.topk,
        }
        self_params.update(params)
        return super().export(path, input_schema, output_schema, self_params, node_id, version)

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:
        input_schema = super().compute_input_schema(
            root_schema, parents_schema, deps_schema, selector
        )

        self._input_col_names = parents_schema.column_names
        self._relevance_col_name = deps_schema.column_names[0]
        return input_schema

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        """
        Describes the operator's outputs, which are computed from the input schema.

        The output schema should be only the selected column(s) from the input.

        Parameters
        ----------
        input_schema : Schema
            The input schema
        col_selector : ColumnSelector
            ColumnSelector indicating which column(s) you would like to return, in sorted order.
        prev_output_schema : Schema, optional
            Not used.
        """
        col_schemas = sorted(
            [
                schema
                for name, schema in input_schema.column_schemas.items()
                if name in col_selector.names
            ],
            key=lambda cs: cs.name,
        )
        return Schema(col_schemas)

    def transform(self, df: InferenceDataFrame) -> InferenceDataFrame:
        """Transform the dataframe by applying this operator to the set of input columns"""
        # Extract parameters from the request

        candidate_ids = df[self._input_col_names]

        predicted_scores = df[self._relevance_col_name].reshape(-1)

        # Exponential sort trick for sampling from a distribution without replacement from:

        # Pavlos S. Efraimidis, Paul G. Spirakis, Weighted random sampling with a reservoir,
        # Information Processing Letters, Volume 97, Issue 5, 2006, Pages 181-185, ISSN 0020-0190,
        # https://doi.org/10.1016/j.ipl.2005.11.003.

        # As implemented by Tim Vieira in "Algorithms for sampling without replacement"
        # https://timvieira.github.io/blog/post/2019/09/16/algorithms-for-sampling-without-replacement/

        # The weights for the sampling distribution are the softmax of the scores
        weights = np.exp(self.temperature * predicted_scores) / np.sum(predicted_scores)

        # This is the core of the exponential sampling trick, which creates a
        # set of values that depend on both the predicted scores and random
        # variables, resulting in a set of values that will sort into an order
        # that reflects sampling without replacement according to the weight
        # distribution

        # TODO we take max but these should all be the same size.
        num_items = max(len(v) for _, v in candidate_ids)
        exponentials = -np.log(np.random.uniform(0, 1, size=(num_items,)))

        exponentials /= weights

        # This is just bookkeeping to produce the final ordered list of recs
        sorted_indices = np.argsort(exponentials)
        topk_item_ids = InferenceDataFrame(
            {
                col: candidate_ids[col][sorted_indices][: self.topk]
                for col in candidate_ids.tensors.keys()
            }
        )

        return topk_item_ids
