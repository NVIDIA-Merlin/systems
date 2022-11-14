import json
from typing import List

import numpy as np
from feast import FeatureStore, ValueType

from merlin.core.protocols import Transformable
from merlin.dag import ColumnSelector
from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ops.operator import PipelineableInferenceOperator

# Feast_key: (numpy dtype, is_list, is_ragged)
feast_2_numpy = {
    ValueType.INT64: (np.int64, False, False),
    ValueType.INT32: (np.int32, False, False),
    ValueType.FLOAT: (np.float32, False, False),
    ValueType.INT64_LIST: (np.int64, True, True),
    ValueType.INT32_LIST: (np.int32, True, True),
    ValueType.FLOAT_LIST: (np.float32, True, True),
}


class QueryFeast(PipelineableInferenceOperator):
    """
    The QueryFeast operator is responsible for ensuring that your feast feature store [1]
    can communicate correctly with tritonserver for the ensemble feast feature look ups.

    References
    ----------
    [1] https://docs.feast.dev/
    """

    @classmethod
    def from_feature_view(
        cls,
        store: FeatureStore,
        view: str,
        column: str,
        output_prefix: str = None,
        include_id: bool = False,
    ):
        """
        Allows for the creation of a QueryFeast operator from already created Feast artifacts.

        Parameters
        ----------
        store : FeatureStore
            Previously loaded feast feature store.
        path : str
            Path to the feast feature repo.
        view : str
            The feature view you want to pull feature from.
        column : str
            The column that input data will match against.
        output_prefix : str, optional
            A column prefix that can be added to each output column, by default None
        include_id : bool, optional
            A boolean to decide to include the input column in output, by default False

        Returns
        -------
        QueryFeast
            Class object
        """
        feature_view = store.get_feature_view(view)
        entity_id = feature_view.entities[0]

        entity_dtype = np.int64
        ent_is_list = False
        ent_is_ragged = False
        for idx, entity in enumerate(store.list_entities()):
            if entity.name == entity_id:
                entity_dtype, ent_is_list, ent_is_ragged = feast_2_numpy[
                    store.list_entities()[idx].value_type
                ]

        features = []
        mh_features = []

        input_schema = Schema(
            [ColumnSchema(column, dtype=entity_dtype, is_list=ent_is_list, is_ragged=ent_is_ragged)]
        )

        output_schema = Schema([])
        for feature in feature_view.features:
            feature_dtype, is_list, is_ragged = feast_2_numpy[feature.dtype]

            if is_list:
                mh_features.append(feature.name)
            else:
                features.append(feature.name)

            name = cls._prefixed_name(output_prefix, feature.name)
            output_schema[name] = ColumnSchema(
                name, dtype=feature_dtype, is_list=is_list, is_ragged=is_ragged
            )

        if include_id:
            output_schema[entity_id] = ColumnSchema(
                entity_id, dtype=entity_dtype, is_list=ent_is_list, is_ragged=ent_is_ragged
            )
        return QueryFeast(
            str(store.repo_path),
            entity_id,
            view,
            column,
            features,
            mh_features,
            input_schema,
            output_schema,
            include_id=include_id,
            output_prefix=output_prefix or "",
        )

    def __init__(
        self,
        repo_path: str,
        entity_id: str,
        entity_view: str,
        entity_column: str,
        features: List[str],
        mh_features: List[str],
        input_schema: Schema,
        output_schema: Schema,
        include_id: bool = False,
        output_prefix: str = "",
    ):
        """
        Create a new QueryFeast operator to handle link between tritonserver ensemble
        and a Feast feature store. This operator will create your feature store as well
        as the QueryFeast operator.

        Parameters
        ----------
        repo_path : str
            _description_
        entity_id : str
            _description_
        entity_view : str
            _description_
        entity_column : str
            _description_
        features : List[str]
            _description_
        mh_features : List[str]
            _description_
        input_schema : Schema
            _description_
        output_schema : Schema
            _description_
        include_id : bool, optional
            _description_, by default False
        output_prefix : str, optional
            _description_, by default ""
        """
        self.repo_path = repo_path
        self.entity_id = entity_id
        self.entity_view = entity_view
        self.entity_column = entity_column

        self.features = features
        self.mh_features = mh_features
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.include_id = include_id
        self.output_prefix = output_prefix

        self.store = FeatureStore(repo_path=repo_path)
        super().__init__()

    def __getstate__(self):
        # feature store objects aren't picklable - exclude from saved representation
        return {k: v for k, v in self.__dict__.items() if k != "store"}

    def load_artifacts(self, artifact_path):
        self.store = FeatureStore(self.repo_path)

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        """Compute the output schema for the operator."""
        return self.output_schema

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:
        """Compute the input schema for the operator."""
        return self.input_schema

    @classmethod
    def from_config(cls, config, **kwargs) -> "QueryFeast":
        """Create the operator from a config."""
        parameters = json.loads(config.get("params", ""))
        entity_id = parameters["entity_id"]
        entity_view = parameters["entity_view"]
        entity_column = parameters["entity_column"]
        repo_path = parameters["feast_repo_path"]
        features = parameters["features"]
        mh_features = parameters["mh_features"]
        in_dict = json.loads(config.get("input_dict", "{}"))
        out_dict = json.loads(config.get("output_dict", "{}"))
        include_id = parameters["include_id"]
        output_prefix = parameters["output_prefix"]

        in_schema = Schema([])
        for col_name, col_rep in in_dict.items():
            in_schema[col_name] = ColumnSchema(
                col_name,
                dtype=col_rep["dtype"],
                is_list=col_rep["is_list"],
                is_ragged=col_rep["is_ragged"],
            )
        out_schema = Schema([])
        for col_name, col_rep in out_dict.items():
            out_schema[col_name] = ColumnSchema(
                col_name,
                dtype=col_rep["dtype"],
                is_list=col_rep["is_list"],
                is_ragged=col_rep["is_ragged"],
            )

        return QueryFeast(
            repo_path,
            entity_id,
            entity_view,
            entity_column,
            features,
            mh_features,
            in_schema,
            out_schema,
            include_id,
            output_prefix,
        )

    def export(
        self,
        path: str,
        input_schema: Schema,
        output_schema: Schema,
        params: dict = None,
        node_id: int = None,
        version: int = 1,
        backend: str = "ensemble",
    ):
        params = params or {}
        self_params = {
            "entity_id": self.entity_id,
            "entity_view": self.entity_view,
            "entity_column": self.entity_column,
            "features": self.features,
            "mh_features": self.mh_features,
            "feast_repo_path": self.repo_path,
            "include_id": self.include_id,
            "output_prefix": self.output_prefix,
        }
        self_params.update(params)
        return super().export(path, input_schema, output_schema, self_params, node_id, version)

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        """
        Transform input dataframe to output dataframe using function logic.

        Parameters
        ----------
        df : DictArray
            Input tensor dictionary, data that will be manipulated

        Returns
        -------
        DictArray
            Transformed tensor dictionary
        """
        entity_ids = transformable[self.entity_column]
        array_lib = entity_ids._array_lib

        if len(entity_ids) < 1:
            raise ValueError(
                "No entity ids provided when querying Feast. Must provide "
                "at least one id in order to fetch features."
            )
        entity_rows = [{self.entity_id: int(entity_id)} for entity_id in entity_ids]

        feature_names = self.features + self.mh_features
        feature_refs = [
            ":".join([self.entity_view, feature_name]) for feature_name in feature_names
        ]

        feast_response = self.store.get_online_features(
            features=feature_refs,
            entity_rows=entity_rows,
        ).to_dict()

        output_tensors = {}
        if self.include_id:
            output_tensors[self.entity_id] = entity_ids

        # Numerical and single-hot categorical
        for feature_name in self.features:
            prefixed_name = self.__class__._prefixed_name(self.output_prefix, feature_name)

            feature_value = feast_response[feature_name]
            feature_array = array_lib.array([feature_value]).T.astype(
                self.output_schema[prefixed_name].dtype
            )
            output_tensors[prefixed_name] = feature_array

        # Multi-hot categorical
        for feature_name in self.mh_features:
            feature_value = feast_response[feature_name]
            prefixed_name = self.__class__._prefixed_name(self.output_prefix, feature_name)

            row_lengths = None
            if isinstance(feature_value[0], list) and self.output_schema[prefixed_name].is_ragged:
                # concatenate lists we got back from Feast
                flattened_value = []
                for val in feature_value:
                    flattened_value.extend(val)

                # get the lengths of the lists
                row_lengths = [len(vals) for vals in feature_value]

                # wrap the flattened values with a list to get the shape right
                feature_value = [flattened_value]

            # create a numpy array
            feature_array = array_lib.array(feature_value).T.astype(
                self.output_schema[prefixed_name].dtype
            )

            # if we're a list but not ragged, construct row lengths
            if not row_lengths:
                row_lengths = [len(feature_array)]

            feature_row_lengths = array_lib.array(
                [row_lengths], dtype=self.output_schema[prefixed_name].dtype
            ).T

            output_tensors[prefixed_name] = (feature_array, feature_row_lengths)

        return type(transformable)(output_tensors)

    @classmethod
    def _prefixed_name(cls, output_prefix, col_name):
        if output_prefix and col_name and not col_name.startswith(output_prefix):
            return f"{output_prefix}_{col_name}"
        else:
            return col_name
