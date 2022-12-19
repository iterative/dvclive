import os
from sys import platform

import lightgbm as lgbm
import numpy as np
import pandas as pd
import pytest
from sklearn import datasets
from sklearn.model_selection import train_test_split

from dvclive import Live
from dvclive.lgbm import DVCLiveCallback
from dvclive.utils import parse_metrics

# pylint: disable=redefined-outer-name, unused-argument


@pytest.fixture
def model_params():
    return {"objective": "multiclass", "n_estimators": 5, "seed": 0}


@pytest.fixture
def iris_data():
    iris = datasets.load_iris()
    x = pd.DataFrame(iris["data"], columns=iris["feature_names"])
    y = iris["target"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, random_state=42
    )
    return (x_train, y_train), (x_test, y_test)


@pytest.mark.skipif(
    platform == "darwin", reason="LIBOMP Segmentation fault on MacOS"
)
def test_lgbm_integration(tmp_dir, model_params, iris_data):
    model = lgbm.LGBMClassifier()
    model.set_params(**model_params)

    callback = DVCLiveCallback()
    model.fit(
        iris_data[0][0],
        iris_data[0][1],
        eval_set=(iris_data[1][0], iris_data[1][1]),
        eval_metric=["multi_logloss"],
        callbacks=[callback],
    )

    assert os.path.exists("dvclive")

    logs, _ = parse_metrics(callback.live)
    assert len(logs) == 1
    assert len(list(logs.values())[0]) == 5


@pytest.mark.skipif(
    platform == "darwin", reason="LIBOMP Segmentation fault on MacOS"
)
def test_lgbm_model_file(tmp_dir, model_params, iris_data):
    model = lgbm.LGBMClassifier()
    model.set_params(**model_params)

    model.fit(
        iris_data[0][0],
        iris_data[0][1],
        eval_set=(iris_data[1][0], iris_data[1][1]),
        eval_metric=["multi_logloss"],
        callbacks=[DVCLiveCallback("lgbm_model")],
    )

    preds = model.predict(iris_data[1][0])
    model2 = lgbm.Booster(model_file="lgbm_model")
    preds2 = model2.predict(iris_data[1][0])
    preds2 = np.argmax(preds2, axis=1)
    assert np.sum(np.abs(preds2 - preds)) == 0


def test_lgbm_pass_logger():
    logger = Live("train_logs")

    assert DVCLiveCallback().live is not logger
    assert DVCLiveCallback(live=logger).live is logger
