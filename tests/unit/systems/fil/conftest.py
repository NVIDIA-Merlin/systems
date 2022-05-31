import lightgbm
import numpy as np
import pytest
import sklearn.ensemble
import xgboost


@pytest.fixture
def xgboost_binary_classifier():
    params = {"objective": "binary:logistic"}
    X = [[1, 2], [2, 4], [3, 5]]
    y = [0, 1, 0]
    data = xgboost.DMatrix(X, label=y)
    model = xgboost.train(params, data)
    return model


@pytest.fixture
def xgboost_mutli_classifier():
    params = {"objective": "multi:softmax"}
    X = [[1, 2], [2, 4], [3, 5]]
    y = [0, 1, 2]
    model = xgboost.XGBClassifier(**params)
    model.fit(X, y)
    return model


@pytest.fixture
def lightgbm_binary_classifier():
    params = {"objective": "binary", "verbose": -1}
    X = np.array([[1, 2], [2, 4], [3, 5]])
    y = [0, 1, 0]
    data = lightgbm.Dataset(X, label=y)
    model = lightgbm.train(params, data, 100)
    return model


@pytest.fixture
def sklearn_binary_classifier():
    X = [[1, 2], [2, 4], [3, 5]]
    y = [0, 1, 0]
    model = sklearn.ensemble.RandomForestClassifier(max_depth=25, n_estimators=100, random_state=0)
    model.fit(X, y)
    return model
