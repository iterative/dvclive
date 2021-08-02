import os

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from funcy import first
from sklearn import datasets

import dvclive
from dvclive.xgb import DvcLiveCallback
from tests.test_main import read_logs

# pylint: disable=redefined-outer-name, unused-argument


@pytest.fixture
def train_params():
    return {"objective": "multi:softmax", "num_class": 3, "seed": 0}


@pytest.fixture
def iris_data():
    iris = datasets.load_iris()
    x = pd.DataFrame(iris["data"], columns=iris["feature_names"])
    y = iris["target"]
    return xgb.DMatrix(x, y)


def test_xgb_integration(tmp_dir, train_params, iris_data):
    dvclive.init("logs")
    xgb.train(
        train_params,
        iris_data,
        callbacks=[DvcLiveCallback("eval_data")],
        num_boost_round=5,
        evals=[(iris_data, "eval_data")],
    )

    assert os.path.exists("logs")

    logs, _ = read_logs("logs")
    assert len(logs) == 1
    assert len(first(logs.values())) == 5


def test_xgb_model_file(tmp_dir, train_params, iris_data):
    dvclive.init("logs")
    model = xgb.train(
        train_params,
        iris_data,
        callbacks=[DvcLiveCallback("eval_data", model_file="model_xgb.json")],
        num_boost_round=5,
        evals=[(iris_data, "eval_data")],
    )

    preds = model.predict(iris_data)
    model2 = xgb.Booster(model_file="model_xgb.json")
    preds2 = model2.predict(iris_data)
    assert np.sum(np.abs(preds2 - preds)) == 0
