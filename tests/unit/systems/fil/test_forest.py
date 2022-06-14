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

from merlin.dag import ColumnSelector
from merlin.io import Dataset
from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.ops.fil import Forest
from merlin.systems.dag.ops.workflow import TransformWorkflow
from nvtabular import Workflow
from nvtabular import ops as wf_ops

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

    triton_chain = selector >> TransformWorkflow(workflow) >> Forest(model, input_schema)

    triton_ens = Ensemble(triton_chain, input_schema)

    triton_ens.export(tmpdir)

    config_path = tmpdir / "1_forest" / "config.pbtxt"
    parsed_config = read_config(config_path)
    assert parsed_config.name == "1_forest"
    assert parsed_config.backend == "python"

    config_path = tmpdir / "1_fil" / "config.pbtxt"
    parsed_config = read_config(config_path)
    assert parsed_config.name == "1_fil"
    assert parsed_config.backend == "fil"

    config_path = tmpdir / "0_transformworkflow" / "config.pbtxt"
    parsed_config = read_config(config_path)
    assert parsed_config.name == "0_transformworkflow"
    assert parsed_config.backend == "nvtabular"

    config_path = tmpdir / "ensemble_model" / "config.pbtxt"
    parsed_config = read_config(config_path)
    assert parsed_config.name == "ensemble_model"
    assert parsed_config.platform == "ensemble"
