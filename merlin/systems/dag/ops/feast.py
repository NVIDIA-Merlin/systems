import json
from dataclasses import dataclass
from typing import List

import numpy as np
from feast import FeatureStore, ValueType

from merlin.dag import ColumnSelector
from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ops.operator import InferenceDataFrame, PipelineableInferenceOperator

# Feast_key: (numpy dtype, is_list, is_ragged)


@dataclass
class MerlinDtype:
    dtype: np.dtype
    is_list: bool = False
    is_ragged: bool = False


feast_2_merlin = {
    ValueType.INT64: MerlinDtype(np.int64),
    ValueType.INT32: MerlinDtype(np.int32),
    ValueType.FLOAT: MerlinDtype(np.float),
    ValueType.INT64_LIST: MerlinDtype(np.int64, True, True),
    ValueType.INT32_LIST: MerlinDtype(np.int32, True, True),
    ValueType.FLOAT_LIST: MerlinDtype(np.float, True, True),
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

        entity_type = MerlinDtype(np.int64)
        for idx, entity in enumerate(store.list_entities()):
            if entity.name == entity_id:
                entity_type = feast_2_merlin[store.list_entities()[idx].value_type]

        features = []
        mh_features = []

        input_schema = Schema([_col_schema(column, entity_type)])

        output_schema = Schema([])
        for feature in feature_view.features:
            feature_type = feast_2_merlin[feature.dtype]

            if feature_type.is_list:
                mh_features.append(feature.name)

                values_name = cls._prefixed_name(output_prefix, f"{feature.name}_1")
                nnzs_name = cls._prefixed_name(output_prefix, f"{feature.name}_2")
                output_schema[values_name] = _col_schema(values_name, feature_type)

                nnzs_type = MerlinDtype(dtype=np.int32, is_list=True, is_ragged=False)
                output_schema[nnzs_name] = _col_schema(nnzs_name, nnzs_type)
            else:
                features.append(feature.name)
                name = cls._prefixed_name(output_prefix, feature.name)
                output_schema[name] = _col_schema(name, feature_type)

        if include_id:
            output_schema[entity_id] = _col_schema(entity_id, entity_type)

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
            suffix_int=1,
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
        suffix_int: int = 1,
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
        suffix_int : int, optional
            _description_, by default 1
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
        self.suffix_int = suffix_int

        self.store = FeatureStore(repo_path=repo_path)
        super().__init__()

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
        suffix_int = parameters["suffix_int"]

        in_schema = Schema([])
        for col_name, col_rep in in_dict.items():
            dtype = MerlinDtype(
                dtype=col_rep["dtype"],
                is_list=col_rep["is_list"],
                is_ragged=col_rep["is_ragged"],
            )
            in_schema[col_name] = _col_schema(col_name, dtype)
        out_schema = Schema([])
        for col_name, col_rep in out_dict.items():
            dtype = MerlinDtype(
                dtype=col_rep["dtype"],
                is_list=col_rep["is_list"],
                is_ragged=col_rep["is_ragged"],
            )
            out_schema[col_name] = _col_schema(col_name, dtype)

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
            suffix_int,
        )

    def export(self, path, input_schema, output_schema, params=None, node_id=None, version=1):
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
            "suffix_int": self.suffix_int,
        }
        self_params.update(params)
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
        entity_ids = df[self.entity_column]

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
            feature_array = np.array([feature_value]).T.astype(
                self.output_schema[prefixed_name].dtype
            )
            output_tensors[prefixed_name] = feature_array

        # Multi-hot categorical
        for feature_name in self.mh_features:
            feature_value = feast_response[feature_name]

            prefixed_name = self.__class__._prefixed_name(self.output_prefix, feature_name)
            feature_out_name = f"{prefixed_name}_{self.suffix_int}"

            nnzs = None
            if (
                isinstance(feature_value[0], list)
                and self.output_schema[feature_out_name].is_ragged
            ):
                flattened_value = []
                for val in feature_value:
                    flattened_value.extend(val)

                nnzs = [len(vals) for vals in feature_value]
                feature_value = [flattened_value]

            feature_array = np.array(feature_value).T.astype(
                self.output_schema[feature_out_name].dtype
            )
            if not nnzs:
                nnzs = [len(feature_array)]
            feature_out_nnz = f"{prefixed_name}_{self.suffix_int+1}"
            feature_nnzs = np.array([nnzs], dtype=self.output_schema[feature_out_nnz].dtype).T

            output_tensors[feature_out_name] = feature_array
            output_tensors[feature_out_nnz] = feature_nnzs

        return InferenceDataFrame(output_tensors)

    @classmethod
    def _prefixed_name(cls, output_prefix, col_name):
        if output_prefix and col_name and not col_name.startswith(output_prefix):
            return f"{output_prefix}_{col_name}"
        else:
            return col_name


def _col_schema(name: str, dtype: MerlinDtype):
    return ColumnSchema(
        name,
        dtype=dtype.dtype,
        is_list=dtype.is_list,
        is_ragged=dtype.is_ragged,
    )
