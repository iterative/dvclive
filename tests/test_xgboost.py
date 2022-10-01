import os

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn import datasets

from dvclive import Live
from dvclive.utils import parse_scalars
from dvclive.xgb import DvcLiveCallback

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
    callback = DvcLiveCallback("eval_data")
    xgb.train(
        train_params,
        iris_data,
        callbacks=[callback],
        num_boost_round=5,
        evals=[(iris_data, "eval_data")],
    )

    assert os.path.exists("dvclive")

    logs, _ = parse_scalars(callback.dvclive)
    assert len(logs) == 1
    assert len(list(logs.values())[0]) == 5


def test_xgb_model_file(tmp_dir, train_params, iris_data):
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


def test_xgb_pass_logger():
    logger = Live("train_logs")

    assert DvcLiveCallback("eval_data").dvclive is not logger
    assert DvcLiveCallback("eval_data", dvclive=logger).dvclive is logger
