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

import numpy as np
import pandas as pd
import pytest
import sklearn.datasets
import xgboost
from google.protobuf import text_format
from tritonclient.grpc import model_config_pb2 as model_config

from merlin.core.utils import Distributed
from merlin.dag import ColumnSelector
from merlin.io import Dataset
from merlin.schema import ColumnSchema, Schema, Tags
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.ops.fil import PredictForest
from merlin.systems.dag.ops.workflow import TransformWorkflow
from merlin.systems.dag.runtimes.triton.ops.fil import PredictForestTriton
from nvtabular import Workflow
from nvtabular import ops as wf_ops


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
    config = PredictForestTriton(PredictForest(model, input_schema)).export(
        tmpdir, input_schema, output_schema, node_id=2
    )
    node_config = json.loads(config.parameters[config.name].string_value)

    assert json.loads(node_config["output_dict"]) == {
        "output__0": {"dtype": "float32", "is_list": False, "is_ragged": False}
    }

    cls = PredictForestTriton.from_config(node_config)
    assert "2_fil" in cls.fil_model_name


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
    _ = PredictForestTriton(PredictForest(model, input_schema)).export(
        tmpdir, input_schema, output_schema, node_id=2
    )

    config_path = tmpdir / "2_predictforesttriton" / "config.pbtxt"
    parsed_config = read_config(config_path)
    assert "2_predictforest" in parsed_config.name
    assert parsed_config.backend == "python"

    config_path = tmpdir / "2_filtriton" / "config.pbtxt"
    parsed_config = read_config(config_path)
    assert "2_fil" in parsed_config.name
    assert parsed_config.backend == "fil"


def test_export_merlin_models(tmpdir):
    merlin_xgb = pytest.importorskip("merlin.models.xgb")

    # create a merlin.io.Dataset
    rows = 200
    num_features = 16
    X, y = sklearn.datasets.make_regression(
        n_samples=rows,
        n_features=num_features,
        n_informative=num_features // 3,
        random_state=0,
    )
    df = pd.DataFrame({f"col_{i}": col for i, col in enumerate(X.T)})
    df["target"] = y
    ds = Dataset(df)
    ds.schema["target"] = ds.schema["target"].with_tags([Tags.TARGET, Tags.REGRESSION])

    # train a XGB model using merlin-models
    model = merlin_xgb.XGBoost(ds.schema)
    with Distributed(cluster_type="cpu"):
        model.fit(ds)

    # make sure we can export the merlin-model xgb wrapper using merlin-systems
    feature_names = [str(i) for i in range(num_features)]
    input_schema = Schema([ColumnSchema(col, dtype=np.float32) for col in feature_names])
    output_schema = Schema([ColumnSchema("output__0", dtype=np.float32)])
    _ = PredictForestTriton(PredictForest(model, input_schema)).export(
        tmpdir, input_schema, output_schema, node_id=2
    )

    config_path = tmpdir / "2_predictforesttriton" / "config.pbtxt"
    parsed_config = read_config(config_path)
    assert "2_predictforest" in parsed_config.name
    assert parsed_config.backend == "python"

    config_path = tmpdir / "2_filtriton" / "config.pbtxt"
    parsed_config = read_config(config_path)
    assert "2_fil" in parsed_config.name
    assert parsed_config.backend == "fil"


def test_ensemble(tmpdir):
    rows = 200
    num_features = 16
    X, y = sklearn.datasets.make_regression(
        n_samples=rows,
        n_features=num_features,
        n_informative=num_features // 3,
        random_state=0,
    )
    feature_names = [str(i) for i in range(num_features)]
    df = pd.DataFrame(X, columns=feature_names)
    dataset = Dataset(df)

    # Fit GBDT Model
    model = xgboost.XGBRegressor()
    model.fit(X, y)

    input_schema = Schema([ColumnSchema(col, dtype=np.float32) for col in feature_names])
    selector = ColumnSelector(feature_names)

    workflow_ops = ["0", "1", "2"] >> wf_ops.LogOp()
    workflow = Workflow(workflow_ops)
    workflow.fit(dataset)

    triton_chain = selector >> TransformWorkflow(workflow) >> PredictForest(model, input_schema)

    triton_ens = Ensemble(triton_chain, input_schema)

    triton_ens.export(tmpdir)

    config_path = tmpdir / "1_predictforesttriton" / "config.pbtxt"
    parsed_config = read_config(config_path)
    assert "1_predictforest" in parsed_config.name
    assert parsed_config.backend == "python"

    config_path = tmpdir / "1_filtriton" / "config.pbtxt"
    parsed_config = read_config(config_path)
    assert "1_fil" in parsed_config.name
    assert parsed_config.backend == "fil"

    config_path = tmpdir / "0_transformworkflowtriton" / "config.pbtxt"
    parsed_config = read_config(config_path)
    assert "0_transformworkflow" in parsed_config.name
    assert parsed_config.backend == "python"

    config_path = tmpdir / "ensemble_model" / "config.pbtxt"
    parsed_config = read_config(config_path)
    assert parsed_config.name == "ensemble_model"
    assert parsed_config.platform == "ensemble"
