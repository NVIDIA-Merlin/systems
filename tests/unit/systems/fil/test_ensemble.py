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
from distutils.spawn import find_executable

import numpy as np
import pandas as pd
import pytest
import sklearn.datasets
import xgboost

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from merlin.dag import ColumnSelector  # noqa
from merlin.io import Dataset  # noqa
from merlin.schema import ColumnSchema, Schema  # noqa
from merlin.systems.dag.ops.fil import PredictForest  # noqa
from nvtabular import Workflow  # noqa
from nvtabular import ops as wf_ops  # noqa

triton = pytest.importorskip("merlin.systems.triton")
export = pytest.importorskip("merlin.systems.dag.ensemble")

from merlin.systems.dag.ensemble import Ensemble  # noqa
from merlin.systems.dag.ops.workflow import TransformWorkflow  # noqa
from tests.unit.systems.utils.triton import _run_ensemble_on_tritonserver  # noqa

tritonclient = pytest.importorskip("tritonclient")
grpcclient = pytest.importorskip("tritonclient.grpc")

TRITON_SERVER_PATH = find_executable("tritonserver")


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
def test_workflow_with_forest_inference(tmpdir):
    rows = 200
    num_features = 16
    X, y = sklearn.datasets.make_regression(
        n_samples=rows,
        n_features=num_features,
        n_informative=num_features // 3,
        random_state=0,
    )
    feature_names = [str(i) for i in range(num_features)]
    df = pd.DataFrame(X, columns=feature_names, dtype=np.float32)
    dataset = Dataset(df)

    # Fit GBDT Model
    model = xgboost.XGBRegressor()
    model.fit(X, y)

    input_column_schemas = [ColumnSchema(col, dtype=np.float32) for col in feature_names]
    input_schema = Schema(input_column_schemas)
    selector = ColumnSelector(feature_names)

    workflow_ops = feature_names >> wf_ops.LogOp()
    workflow = Workflow(workflow_ops)
    workflow.fit(dataset)

    triton_chain = selector >> TransformWorkflow(workflow) >> PredictForest(model, input_schema)

    triton_ens = Ensemble(triton_chain, input_schema)

    request_df = df[:5]
    triton_ens.export(tmpdir)

    response = _run_ensemble_on_tritonserver(
        str(tmpdir), ["output__0"], request_df, triton_ens.name
    )
    assert response.as_numpy("output__0").shape == (5,)
