import json
from datetime import timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from merlin.dag import ColumnSelector
from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag import DictArray

feast = pytest.importorskip("feast")  # noqa

from feast.online_response import OnlineResponse  # noqa
from feast.protos.feast.serving import ServingService_pb2  # noqa
from feast.protos.feast.types import Value_pb2  # noqa

from merlin.systems.dag.ops.feast import QueryFeast  # noqa


def test_feast_config_round_trip(tmpdir):
    """
    This builds a QueryFeast op via the constructor, exports the config, and then builds another
    QueryFeast op via from_config to ensure that the constructor arguments match the original.
    """
    input_schema = Schema()
    output_schema = Schema()

    # We need to mock the FeatureStore constructor so it doesn't look for a real store, and the
    # QueryFeast constructor so that we can ensure that it gets called with the correct args.
    # These must be done in a context manager so we don't modify the classes globally and
    # affect other tests.
    with patch("feast.FeatureStore.__init__", MagicMock(return_value=None)), patch(
        "merlin.systems.dag.ops.feast.QueryFeast", MagicMock(side_effect=QueryFeast)
    ) as qf_init:

        # Define the args & kwargs. We want to ensure the round-tripped version uses these same
        # arguments.
        args = [
            "repo_path",
            "entity_id",
            "entity_view",
            "entity_column",
            ["features"],
            ["mh_features"],
            input_schema,
            output_schema,
            True,  # include_id
            "prefix",  # output_prefix
        ]
        feast_op = QueryFeast(*args)

        created_config = feast_op.export(tmpdir + "/export_path/", input_schema, output_schema)
        created_config_dict = json.loads(created_config.parameters["queryfeast"].string_value)

        # now mock the QueryFeast constructor so we can inspect its arguments.
        QueryFeast.from_config(created_config_dict)
        qf_init.assert_called_with(*args)


def test_feast_from_feature_view(tmpdir):

    with patch("feast.FeatureStore.__init__", MagicMock(return_value=None)), patch(
        "feast.feature_store.Registry.__init__", MagicMock(return_value=None)
    ), patch(
        "merlin.systems.dag.ops.feast.QueryFeast",
        MagicMock(side_effect=QueryFeast),
    ) as qf_init:
        input_source = feast.FileSource(
            path=tmpdir,
            event_timestamp_column="datetime",
            created_timestamp_column="created",
        )
        feature_view = feast.FeatureView(
            name="item_features",
            entities=["item_id"],
            ttl=timedelta(seconds=100),
            features=[
                feast.Feature(name="int_feature", dtype=feast.ValueType.INT32),
                feast.Feature(name="float_feature", dtype=feast.ValueType.FLOAT),
                feast.Feature(name="int_list_feature", dtype=feast.ValueType.INT32_LIST),
                feast.Feature(name="float_list_feature", dtype=feast.ValueType.FLOAT_LIST),
            ],
            online=True,
            input=input_source,
            tags={},
        )
        fs = feast.FeatureStore("repo_path")
        fs.repo_path = "repo_path"
        fs._registry = feast.feature_store.Registry(None, None)
        fs.list_entities = MagicMock(
            return_value=[feast.Entity(name="item_id", value_type=feast.ValueType.INT32)]
        )
        fs.get_feature_view = MagicMock(return_value=feature_view)
        fs._registry.get_feature_view = MagicMock(return_value=feature_view)

        expected_input_schema = Schema(
            column_schemas=[ColumnSchema(name="item_id", dtype=np.int32)]
        )

        # The expected output is all of the feature columns plus the item_id column
        expected_output_schema = Schema(
            column_schemas=[
                ColumnSchema(name="prefix_int_feature", dtype=np.int32),
                ColumnSchema(name="prefix_float_feature", dtype=np.float32),
                ColumnSchema(
                    name="prefix_int_list_feature",
                    dtype=np.int32,
                    is_list=True,
                    is_ragged=True,
                ),
                ColumnSchema(
                    name="prefix_float_list_feature",
                    dtype=np.float32,
                    is_list=True,
                    is_ragged=True,
                ),
                ColumnSchema(name="item_id", dtype=np.int32),
            ]
        )

        feast_op: QueryFeast = QueryFeast.from_feature_view(
            fs,
            "item_features",
            "item_id",
            output_prefix="prefix",
            include_id=True,
        )

        assert feast_op.input_schema == expected_input_schema
        assert feast_op.output_schema == expected_output_schema

        args = [
            "repo_path",
            "item_id",
            "item_features",
            "item_id",
            ["int_feature", "float_feature"],
            ["int_list_feature", "float_list_feature"],
            expected_input_schema,
            expected_output_schema,
        ]
        kwargs = {"include_id": True, "output_prefix": "prefix"}
        qf_init.assert_called_with(*args, **kwargs)


@pytest.mark.parametrize("is_ragged", [True, False])
@pytest.mark.parametrize("prefix", ["prefix", ""])
def test_feast_transform(prefix, is_ragged):
    mocked_resp = OnlineResponse(
        online_response_proto=ServingService_pb2.GetOnlineFeaturesResponse(
            metadata=ServingService_pb2.GetOnlineFeaturesResponseMetadata(
                feature_names=ServingService_pb2.FeatureList(
                    val=["entity_id", "feature", "mh_feature"]
                )
            ),
            results=[
                ServingService_pb2.GetOnlineFeaturesResponse.FeatureVector(
                    values=[
                        Value_pb2.Value(int32_val=1),
                        Value_pb2.Value(float_val=1.0),
                        Value_pb2.Value(float_list_val=Value_pb2.FloatList(val=[1.0, 2.0, 3.0])),
                    ]
                )
            ],
        )
    )

    with patch("feast.FeatureStore.__init__", MagicMock(return_value=None)), patch(
        "feast.FeatureStore.get_online_features", MagicMock(return_value=mocked_resp)
    ):

        # names of the features with prefix
        feature_name = f"{prefix}_feature" if prefix else "feature"
        feature_mh = f"{prefix}_mh_feature" if prefix else "mh_feature"

        input_schema = Schema(
            [ColumnSchema("feature"), ColumnSchema("mh_feature", is_list=True, is_ragged=True)]
        )
        output_schema = Schema(
            [
                ColumnSchema(feature_name),
                ColumnSchema(feature_mh, is_list=True, is_ragged=is_ragged),
            ]
        )

        feast_op = QueryFeast(
            "repo_path",
            "entity_id",
            "entity_view",
            "entity_id",
            ["feature"],
            ["mh_feature"],
            input_schema,
            output_schema,
            include_id=True,
            output_prefix=prefix,
        )

        df = DictArray({"entity_id": [1]})
        resp = feast_op.transform(ColumnSelector("*"), df)

        array_lib = resp["entity_id"]._array_lib
        assert resp["entity_id"].values == array_lib.array([1])  # pylint: disable=W0143
        assert resp[feature_name].values == array_lib.array([[1.0]])
        assert np.all(resp[feature_mh].values == array_lib.array([[1.0], [2.0], [3.0]]))
        assert resp[feature_mh].row_lengths == array_lib.array([[3.0]])
