import numpy as np
import pandas as pd
import pytest

from merlin.core.compat import HAS_GPU
from merlin.dag import ColumnSelector
from merlin.dag.base_runtime import Runtime
from merlin.dtypes.shape import Shape
from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.ops.fil import PredictForest
from merlin.table import TensorTable

sklearn_datasets = pytest.importorskip("sklearn.datasets")
xgboost = pytest.importorskip("xgboost")
export = pytest.importorskip("merlin.systems.dag.ensemble")


@pytest.mark.parametrize("runtime", [Runtime()])
@pytest.mark.skipif(not HAS_GPU, reason="no gpu detected")
def test_xgboost_regressor_forest_inference(runtime, tmpdir):
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
    model = xgboost.XGBRegressor()
    model.fit(X, y)

    input_column_schemas = [ColumnSchema(col, dtype=np.float32) for col in feature_names]
    input_schema = Schema(input_column_schemas)
    selector = ColumnSelector(feature_names)

    triton_chain = selector >> PredictForest(model, input_schema)

    ensemble = Ensemble(triton_chain, input_schema)

    request_df = df[:5]

    tensor_table = TensorTable.from_df(request_df)
    response = ensemble.transform(tensor_table, runtime=runtime)

    assert response["output__0"].shape == Shape((5,))


@pytest.mark.skipif(not HAS_GPU, reason="no gpu detected")
@pytest.mark.parametrize("runtime", [Runtime()])
def test_xgboost_classify_forest_inference(runtime, tmpdir):
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
    model = xgboost.XGBClassifier()
    model.fit(X, y)

    input_column_schemas = [ColumnSchema(col, dtype=np.float32) for col in feature_names]
    input_schema = Schema(input_column_schemas)
    selector = ColumnSelector(feature_names)

    triton_chain = selector >> PredictForest(model, input_schema)

    ensemble = Ensemble(triton_chain, input_schema)

    request_df = df[:5]

    tensor_table = TensorTable.from_df(request_df)
    response = ensemble.transform(tensor_table, runtime=runtime)

    assert response["output__0"].shape == Shape((5,))
