import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from feast.online_response import OnlineResponse
from feast.protos.feast.serving import ServingService_pb2
from feast.protos.feast.types import Value_pb2

from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ops.feast import QueryFeast
from merlin.systems.dag.ops.operator import InferenceDataFrame


def test_feast_config_round_trip():
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
        "merlin.systems.dag.ops.feast.__init__", MagicMock(return_value=None)
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
        ]
        kwargs = {"include_id": True, "output_prefix": "prefix"}
        feast_op = QueryFeast(*args, **kwargs)

        created_config = feast_op.export("export_path", input_schema, output_schema)
        created_config_dict = json.loads(created_config.parameters["queryfeast"].string_value)

        # now mock the QueryFeast constructor so we can inspect its arguments.
        QueryFeast.from_config(created_config_dict)
        assert qf_init.called_with(*args, **kwargs)


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

        # names of the features with prefix/suffix
        feature_name = f"{prefix}_feature" if prefix else "feature"
        feature_mh_1 = f"{prefix}_mh_feature_1" if prefix else "mh_feature_1"
        feature_mh_2 = f"{prefix}_mh_feature_2" if prefix else "mh_feature_2"

        input_schema = Schema([ColumnSchema("feature"), ColumnSchema("mh_feature")])
        output_schema = Schema(
            [
                ColumnSchema(feature_name),
                ColumnSchema(feature_mh_1, is_list=True, is_ragged=is_ragged),
                ColumnSchema(feature_mh_2, is_list=True, is_ragged=is_ragged),
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

        df = InferenceDataFrame({"entity_id": [1]})
        resp = feast_op.transform(df)
        assert resp["entity_id"] == [1]
        assert resp[feature_name] == np.array([[1.0]])
        assert np.all(resp[feature_mh_1] == np.array([[1.0], [2.0], [3.0]]))
        assert resp[feature_mh_2] == np.array([[3.0]])
