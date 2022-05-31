import pytest
import xgboost as xgb


@pytest.fixture
def xgboost_binary_classifier():
    params = {"objective": "binary:logistic"}
    X = [[1, 2], [2, 4], [3, 5]]
    y = [0, 1, 0]
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data)
    return model


@pytest.fixture
def xgboost_mutli_classifier():
    params = {"objective": "multi:softmax"}
    X = [[1, 2], [2, 4], [3, 5]]
    y = [0, 1, 2]
    model = xgb.XGBClassifier(**params)
    model.fit(X, y)
    return model
