import os

import lightgbm as lgbm
import numpy as np
import pandas as pd
import pytest
from funcy import first
from sklearn import datasets
from sklearn.model_selection import train_test_split

from dvclive.data.scalar import Scalar
from dvclive.lgbm import DvcLiveCallback
from tests.test_main import read_logs

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


def test_lgbm_integration(tmp_dir, model_params, iris_data):
    model = lgbm.LGBMClassifier()
    model.set_params(**model_params)

    model.fit(
        iris_data[0][0],
        iris_data[0][1],
        eval_set=(iris_data[1][0], iris_data[1][1]),
        eval_metric=["multi_logloss"],
        callbacks=[DvcLiveCallback()],
    )

    assert os.path.exists("dvclive")

    logs, _ = read_logs(tmp_dir / "dvclive" / Scalar.subfolder)
    assert len(logs) == 1
    assert len(first(logs.values())) == 5


def test_lgbm_model_file(tmp_dir, model_params, iris_data):
    model = lgbm.LGBMClassifier()
    model.set_params(**model_params)

    model.fit(
        iris_data[0][0],
        iris_data[0][1],
        eval_set=(iris_data[1][0], iris_data[1][1]),
        eval_metric=["multi_logloss"],
        callbacks=[DvcLiveCallback("lgbm_model")],
    )

    preds = model.predict(iris_data[1][0])
    model2 = lgbm.Booster(model_file="lgbm_model")
    preds2 = model2.predict(iris_data[1][0])
    preds2 = np.argmax(preds2, axis=1)
    assert np.sum(np.abs(preds2 - preds)) == 0
