from distutils.spawn import find_executable  # pylint: disable=W0402

import numpy as np
import pandas as pd
import pytest
import sklearn.datasets
import xgboost

from merlin.dag import ColumnSelector  # noqa
from merlin.io import Dataset  # noqa
from merlin.schema import ColumnSchema, Schema  # noqa
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.ops.fil import PredictForest  # noqa
from merlin.systems.dag.ops.workflow import TransformWorkflow  # noqa
from merlin.systems.dag.runtimes.triton import TritonEnsembleRuntime
from merlin.systems.triton.utils import run_ensemble_on_tritonserver
from nvtabular import Workflow  # noqa
from nvtabular import ops as wf_ops  # noqa

triton = pytest.importorskip("merlin.systems.triton")
export = pytest.importorskip("merlin.systems.dag.ensemble")

TRITON_SERVER_PATH = find_executable("tritonserver")


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize(
    ["runtime", "model_name", "expected_model_name"],
    [
        (None, None, "ensemble_model"),
        (TritonEnsembleRuntime(), None, "ensemble_model"),
        (TritonEnsembleRuntime(), "triton_model", "triton_model"),
        # (TritonExecutorRuntime(), None, "executor_model"),
        # (TritonExecutorRuntime(), "triton_model", "triton_model"),
    ],
)
def test_workflow_with_forest_inference(runtime, model_name, expected_model_name, tmpdir):
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

    ensemble = Ensemble(triton_chain, input_schema)

    request_df = df[:5]
    ensemble_config, _ = ensemble.export(tmpdir, runtime=runtime, name=model_name)

    assert ensemble_config.name == expected_model_name

    response = run_ensemble_on_tritonserver(
        str(tmpdir), input_schema, request_df, ["output__0"], ensemble_config.name
    )
    assert response["output__0"].shape == (5,)
