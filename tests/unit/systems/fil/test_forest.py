import json

import numpy as np
import pytest
import sklearn.datasets
import xgboost
from google.protobuf import text_format

from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ops.fil import Forest

tritonclient = pytest.importorskip("tritonclient")
import tritonclient.grpc.model_config_pb2 as model_config  # noqa


def test_load_from_config(tmpdir):
    rows = 200
    num_features = 16
    X, y = sklearn.datasets.make_regression(
        n_samples=rows,
        n_features=num_features,
        n_informative=num_features // 3,
        random_state=0,
    )
    model = xgboost.XGBRegressor()
    model.fit(X, y)
    feature_names = [str(i) for i in range(num_features)]
    input_schema = Schema([ColumnSchema(col, dtype=np.float32) for col in feature_names])
    output_schema = Schema([ColumnSchema("output__0", dtype=np.float32)])
    config = Forest(model, input_schema).export(tmpdir, input_schema, output_schema, node_id=2)
    node_config = json.loads(config.parameters[config.name].string_value)

    assert json.loads(node_config["output_dict"]) == {
        "output__0": {"dtype": "float32", "is_list": False, "is_ragged": False}
    }

    cls = Forest.from_config(node_config)
    assert cls.fil_model_name == "2_fil"


def read_config(config_path):
    with open(config_path, "rb") as f:
        config = model_config.ModelConfig()
        raw_config = f.read()
        return text_format.Parse(raw_config, config)


def test_export(tmpdir):
    rows = 200
    num_features = 16
    X, y = sklearn.datasets.make_regression(
        n_samples=rows,
        n_features=num_features,
        n_informative=num_features // 3,
        random_state=0,
    )
    model = xgboost.XGBRegressor()
    model.fit(X, y)
    feature_names = [str(i) for i in range(num_features)]
    input_schema = Schema([ColumnSchema(col, dtype=np.float32) for col in feature_names])
    output_schema = Schema([ColumnSchema("output__0", dtype=np.float32)])
    _ = Forest(model, input_schema).export(tmpdir, input_schema, output_schema, node_id=2)

    config_path = tmpdir / "2_forest" / "config.pbtxt"
    parsed_config = read_config(config_path)
    assert parsed_config.name == "2_forest"
    assert parsed_config.backend == "python"

    config_path = tmpdir / "2_fil" / "config.pbtxt"
    parsed_config = read_config(config_path)
    assert parsed_config.name == "2_fil"
    assert parsed_config.backend == "fil"