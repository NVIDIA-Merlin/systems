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
import pathlib
from copy import deepcopy

import lightgbm
import pytest
import sklearn.datasets
import sklearn.ensemble
import xgboost

import merlin.systems.dag.ops.fil as fil_op
from merlin.dag import ColumnSelector, Graph
from merlin.schema import Schema

# this needs to be before any modules that import protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from google.protobuf import text_format  # noqa
from tritonclient.grpc import model_config_pb2 as model_config  # noqa


def export_op(export_dir, triton_op) -> model_config.ModelConfig:
    triton_op.export(export_dir, None, None)

    # Export creates directory
    export_path = pathlib.Path(export_dir) / triton_op.export_name
    assert export_path.exists()

    # Export creates the config file
    config_path = export_path / "config.pbtxt"
    assert config_path.exists()

    # Read the config file back in from proto
    with open(config_path, "rb") as f:
        config = model_config.ModelConfig()
        raw_config = f.read()
        parsed = text_format.Parse(raw_config, config)
        return parsed


def get_classification_data(classes=2, rows=200, cols=16):
    return sklearn.datasets.make_classification(
        n_samples=rows,
        n_features=cols,
        n_informative=cols // 3,
        n_classes=classes,
        random_state=0,
    )


def get_regression_data(rows=200, cols=16, **kwargs):
    return sklearn.datasets.make_regression(
        n_samples=rows, n_features=cols, n_informative=cols // 3, random_state=0, **kwargs
    )


def xgboost_train(X, y, **params):
    data = xgboost.DMatrix(X, label=y)
    model = xgboost.train(params, data)
    return model


def xgboost_regressor(X, y, **params):
    model = xgboost.XGBRegressor(**params)
    model.fit(X, y)
    return model


def xgboost_classifier(X, y, **params):
    model = xgboost.XGBClassifier(**params)
    model.fit(X, y)
    return model


def lightgbm_train(X, y, **params):
    data = lightgbm.Dataset(X, label=y.ravel())
    model = lightgbm.train({**params, "verbose": -1}, data, 100)
    return model


def lightgbm_classifier(X, y, **params):
    model = lightgbm.LGBMClassifier(**params)
    model.fit(X, y)
    return model


def lightgbm_regressor(X, y, **params):
    model = lightgbm.LGBMRegressor(**params)
    model.fit(X, y.ravel())
    return model


def sklearn_forest_classifier(X, y, **params):
    params = {
        "max_depth": 25,
        "n_estimators": 100,
        "random_state": 0,
        **params,
    }
    model = sklearn.ensemble.RandomForestClassifier(**params)
    model.fit(X, y)
    return model


def sklearn_forest_regressor(X, y, **params):
    params = {
        "max_depth": 25,
        "n_estimators": 100,
        "random_state": 0,
        **params,
    }
    model = sklearn.ensemble.RandomForestRegressor(**params)
    model.fit(X, y.ravel())
    return model


@pytest.mark.parametrize(
    ["get_model_fn", "get_model_params"],
    [
        (xgboost_train, {"objective": "binary:logistic"}),
        (xgboost_classifier, {}),
        (lightgbm_train, {"objective": "binary"}),
        (lightgbm_classifier, {}),
        (sklearn_forest_classifier, {}),
    ],
)
def test_binary_classifier_default(get_model_fn, get_model_params, tmpdir):
    X, y = get_classification_data(classes=2)
    model = get_model_fn(X, y, **get_model_params)
    triton_op = fil_op.FIL(model)
    config = export_op(tmpdir, triton_op)
    assert config.parameters["output_class"].string_value == "false"
    assert config.parameters["predict_proba"].string_value == "false"
    assert config.output[0].dims == [1]


@pytest.mark.parametrize(
    ["get_model_fn", "get_model_params"],
    [
        (xgboost_train, {"objective": "binary:logistic"}),
        (xgboost_classifier, {}),
        (lightgbm_train, {"objective": "binary"}),
        (lightgbm_classifier, {}),
        (sklearn_forest_classifier, {}),
    ],
)
def test_binary_classifier_with_proba(get_model_fn, get_model_params, tmpdir):
    X, y = get_classification_data(classes=2)
    model = get_model_fn(X, y, **get_model_params)
    triton_op = fil_op.FIL(model, predict_proba=True, output_class=True)
    config = export_op(tmpdir, triton_op)
    assert config.parameters["output_class"].string_value == "true"
    assert config.parameters["predict_proba"].string_value == "true"
    assert config.output[0].dims == [2]


@pytest.mark.parametrize(
    ["get_model_fn", "get_model_params"],
    [
        (xgboost_train, {"objective": "multi:softmax", "num_class": 8}),
        (xgboost_classifier, {}),
        (lightgbm_train, {"objective": "multiclass", "num_class": 8}),
        (lightgbm_classifier, {}),
        (sklearn_forest_classifier, {}),
    ],
)
def test_multi_classifier(get_model_fn, get_model_params, tmpdir):
    X, y = get_classification_data(classes=8)
    model = get_model_fn(X, y, **get_model_params)
    triton_op = fil_op.FIL(model, predict_proba=True, output_class=True)
    config = export_op(tmpdir, triton_op)
    assert config.parameters["output_class"].string_value == "true"
    assert config.parameters["predict_proba"].string_value == "true"
    assert config.output[0].dims == [8]


@pytest.mark.parametrize(
    ["get_model_fn", "get_model_params"],
    [
        (xgboost_train, {"objective": "reg:squarederror"}),
        (xgboost_regressor, {}),
        (lightgbm_train, {"objective": "regression"}),
        (lightgbm_regressor, {}),
        (sklearn_forest_regressor, {}),
    ],
)
def test_regressor(get_model_fn, get_model_params, tmpdir):
    X, y = get_regression_data()
    model = get_model_fn(X, y, **get_model_params)
    triton_op = fil_op.FIL(model)
    config = export_op(tmpdir, triton_op)
    assert config.parameters["output_class"].string_value == "false"
    assert config.parameters["predict_proba"].string_value == "false"
    assert config.output[0].dims == [1]


@pytest.mark.parametrize(
    ["get_model_fn", "expected_model_filename"],
    [
        (xgboost_regressor, "xgboost.json"),
        (lightgbm_regressor, "model.txt"),
        (sklearn_forest_regressor, "checkpoint.tl"),
    ],
)
def test_model_file(get_model_fn, expected_model_filename, tmpdir):
    X, y = get_regression_data()
    model = get_model_fn(X, y)
    triton_op = fil_op.FIL(model)
    _ = export_op(tmpdir, triton_op)
    model_path = pathlib.Path(tmpdir) / "fil" / "1" / expected_model_filename
    assert model_path.is_file()


def test_fil_op_exports_own_config(tmpdir):
    X, y = get_regression_data()
    model = xgboost_train(X, y, objective="reg:squarederror")

    # Triton
    triton_op = fil_op.FIL(model)
    config = export_op(tmpdir, triton_op)

    assert config.name == triton_op.export_name
    assert config.backend == "fil"
    assert config.max_batch_size == 8192
    assert config.input[0].name == "input__0"
    assert config.output[0].name == "output__0"
    assert config.output[0].dims == [1]


def test_fil_op_compute_schema():
    X, y = get_regression_data()
    model = xgboost_train(X, y, objective="reg:squarederror")

    # Triton
    triton_op = fil_op.FIL(model)

    out_schema = triton_op.compute_output_schema(
        Schema(["input__0"]), ColumnSelector(["input__0"]), None
    )
    assert out_schema.column_names == ["output__0"]


def test_fil_schema_validation():
    X, y = get_regression_data()
    model = xgboost_train(X, y, objective="reg:squarederror")

    # Triton
    fil_node = [] >> fil_op.FIL(model)
    fil_graph = Graph(fil_node)

    with pytest.raises(ValueError) as exception_info:
        deepcopy(fil_graph).construct_schema(Schema([]))
    assert "Missing column 'input__0'" in str(exception_info.value)

    with pytest.raises(ValueError) as exception_info:
        deepcopy(fil_graph).construct_schema(Schema(["not_input"]))
    assert "Missing column 'input__0'" in str(exception_info.value)

    with pytest.raises(ValueError) as exception_info:
        deepcopy(fil_graph).construct_schema(Schema(["input__0", "not_input"]))
    assert "Mismatched dtypes for column 'input__0'" in str(exception_info.value)
