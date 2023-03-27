import numpy as np

from merlin.core.protocols import Transformable
from merlin.dag.node import Node
from merlin.dag.selector import ColumnSelector
from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ops.operator import InferenceOperator


class SoftmaxSampling(InferenceOperator):
    """
    Given inputs of ID and prediction, this operator will sort all
    inputs in descending order.
    """

    def __init__(self, relevance_col, temperature=20.0, topk=10, _input_col=None):
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
        _input_col : _type_, optional
            The column whose values will be sorted, by default None.
        """
        self.relevance_col = Node.construct_from(relevance_col)
        self.temperature = temperature
        self.topk = topk
        self._input_col_name = _input_col
        self._relevance_col_name = relevance_col
        super().__init__()

    @property
    def dependencies(self):
        return self.relevance_col

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
        if len(parents_schema.column_schemas) > 1:
            raise ValueError(
                "More than one input has been detected for this node,"
                f" inputs received: {input_schema.column_names}"
            )

        self._input_col_name = parents_schema.column_names[0]
        self._relevance_col_name = deps_schema.column_names[0]
        return input_schema

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        """Describe the operator's outputs"""
        return Schema(
            [
                ColumnSchema(
                    "ordered_ids", dtype=input_schema[self._input_col_name].dtype, dims=(None, 1)
                ),
                ColumnSchema(
                    "ordered_scores",
                    dtype=input_schema[self._relevance_col_name].dtype,
                    dims=(None, 1),
                ),
            ]
        )

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        """Transform the dataframe by applying this operator to the set of input columns"""
        # Extract parameters from the request
        candidate_ids = transformable[self._input_col_name].values.reshape(-1)
        predicted_scores = transformable[self._relevance_col_name].values.reshape(-1)

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
        num_items = candidate_ids.shape[0]
        exponentials = -np.log(np.random.uniform(0, 1, size=(num_items,)))
        exponentials /= weights

        # This is just bookkeeping to produce the final ordered list of recs
        sorted_indices = np.argsort(exponentials)
        topk_item_ids = candidate_ids[sorted_indices][: self.topk]
        topk_item_scores = predicted_scores[sorted_indices][: self.topk]
        ordered_item_ids = topk_item_ids.reshape(1, -1)
        ordered_item_scores = topk_item_scores.reshape(1, -1)

        return type(transformable)(
            {"ordered_ids": ordered_item_ids, "ordered_scores": ordered_item_scores}
        )
