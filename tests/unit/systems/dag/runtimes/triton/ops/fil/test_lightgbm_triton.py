from distutils.spawn import find_executable  # pylint: disable=W0402

import numpy as np
import pandas as pd
import pytest

from merlin.dag import ColumnSelector
from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.ops.fil import PredictForest
from merlin.systems.dag.runtimes.triton import TritonExecutorRuntime
from merlin.systems.triton.utils import run_ensemble_on_tritonserver

sklearn_datasets = pytest.importorskip("sklearn.datasets")
lightgbm = pytest.importorskip("lightgbm")
triton = pytest.importorskip("merlin.systems.triton")
export = pytest.importorskip("merlin.systems.dag.ensemble")

TRITON_SERVER_PATH = find_executable("tritonserver")


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize(
    ["runtime", "model_name", "expected_model_name"],
    [
        (TritonExecutorRuntime(), None, "executor_model"),
    ],
)
def test_lightgbm_regressor_forest_inference(runtime, model_name, expected_model_name, tmpdir):
    rows = 200
    num_features = 16
    X, y = sklearn_datasets.make_regression(
        n_samples=rows,
        n_features=num_features,
        n_informative=num_features // 3,
        random_state=0,
    )
    feature_names = [str(i) for i in range(num_features)]
    df = pd.DataFrame(X, columns=feature_names, dtype=np.float32)
    for column in df.columns:
        df[column] = np.log(df[column] + 1).fillna(0.5)

    # Fit GBDT Model
    model = lightgbm.LGBMRegressor()
    model.fit(X, y)

    input_column_schemas = [ColumnSchema(col, dtype=np.float32) for col in feature_names]
    input_schema = Schema(input_column_schemas)
    selector = ColumnSelector(feature_names)

    triton_chain = selector >> PredictForest(model, input_schema)

    ensemble = Ensemble(triton_chain, input_schema)

    request_df = df[:5]
    ensemble_config, _ = ensemble.export(tmpdir, runtime=runtime, name=model_name)

    assert ensemble_config.name == expected_model_name

    response = run_ensemble_on_tritonserver(
        str(tmpdir), input_schema, request_df, ["output__0"], ensemble_config.name
    )
    assert response["output__0"].shape == (5,)


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize(
    ["runtime", "model_name", "expected_model_name"],
    [
        (TritonExecutorRuntime(), None, "executor_model"),
    ],
)
def test_lightgbm_classify_forest_inference(runtime, model_name, expected_model_name, tmpdir):
    rows = 200
    num_features = 16
    X, y = sklearn_datasets.make_classification(
        n_samples=rows,
        n_features=num_features,
        n_informative=num_features // 3,
        random_state=0,
    )
    feature_names = [str(i) for i in range(num_features)]
    df = pd.DataFrame(X, columns=feature_names, dtype=np.float32)
    for column in df.columns:
        df[column] = np.log(df[column] + 1).fillna(0.5)

    # Fit GBDT Model
    model = lightgbm.LGBMClassifier()
    model.fit(X, y)

    input_column_schemas = [ColumnSchema(col, dtype=np.float32) for col in feature_names]
    input_schema = Schema(input_column_schemas)
    selector = ColumnSelector(feature_names)

    triton_chain = selector >> PredictForest(model, input_schema)

    ensemble = Ensemble(triton_chain, input_schema)

    request_df = df[:5]
    ensemble_config, _ = ensemble.export(tmpdir, runtime=runtime, name=model_name)

    assert ensemble_config.name == expected_model_name

    response = run_ensemble_on_tritonserver(
        str(tmpdir), input_schema, request_df, ["output__0"], ensemble_config.name
    )
    assert response["output__0"].shape == (5,)
