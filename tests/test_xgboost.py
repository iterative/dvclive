import os

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn import datasets

from dvclive import Live
from dvclive.utils import parse_metrics
from dvclive.xgb import DVCLiveCallback

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
    callback = DVCLiveCallback("eval_data")
    xgb.train(
        train_params,
        iris_data,
        callbacks=[callback],
        num_boost_round=5,
        evals=[(iris_data, "eval_data")],
    )

    assert os.path.exists("dvclive")

    logs, _ = parse_metrics(callback.live)
    assert len(logs) == 1
    assert len(list(logs.values())[0]) == 5


def test_xgb_model_file(tmp_dir, train_params, iris_data):
    model = xgb.train(
        train_params,
        iris_data,
        callbacks=[DVCLiveCallback("eval_data", model_file="model_xgb.json")],
        num_boost_round=5,
        evals=[(iris_data, "eval_data")],
    )

    preds = model.predict(iris_data)
    model2 = xgb.Booster(model_file="model_xgb.json")
    preds2 = model2.predict(iris_data)
    assert np.sum(np.abs(preds2 - preds)) == 0


def test_xgb_pass_logger():
    logger = Live("train_logs")

    assert DVCLiveCallback("eval_data").live is not logger
    assert DVCLiveCallback("eval_data", live=logger).live is logger
