import os
from sys import platform

import pytest

from dvclive import Live
from dvclive.utils import parse_metrics

try:
    import lightgbm as lgbm
    import pandas as pd
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    from dvclive.lgbm import DVCLiveCallback
except ImportError:
    pytest.skip("skipping lightgbm tests", allow_module_level=True)


@pytest.fixture()
def model_params():
    return {"objective": "multiclass", "n_estimators": 5, "seed": 0}


@pytest.fixture()
def iris_data():
    iris = datasets.load_iris()
    x = pd.DataFrame(iris["data"], columns=iris["feature_names"])
    y = iris["target"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, random_state=42
    )
    return (x_train, y_train), (x_test, y_test)


@pytest.mark.skipif(platform == "darwin", reason="LIBOMP Segmentation fault on MacOS")
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
    assert "dvclive/plots/metrics/multi_logloss.tsv" in logs
    assert len(logs) == 1
    assert len(next(iter(logs.values()))) == 5


@pytest.mark.skipif(platform == "darwin", reason="LIBOMP Segmentation fault on MacOS")
def test_lgbm_integration_multi_eval(tmp_dir, model_params, iris_data):
    model = lgbm.LGBMClassifier()
    model.set_params(**model_params)

    callback = DVCLiveCallback()
    model.fit(
        iris_data[0][0],
        iris_data[0][1],
        eval_set=[
            (iris_data[0][0], iris_data[0][1]),
            (iris_data[1][0], iris_data[1][1]),
        ],
        eval_metric=["multi_logloss"],
        callbacks=[callback],
    )

    assert os.path.exists("dvclive")

    logs, _ = parse_metrics(callback.live)
    assert "dvclive/plots/metrics/training/multi_logloss.tsv" in logs
    assert "dvclive/plots/metrics/valid_1/multi_logloss.tsv" in logs
    assert len(logs) == 2
    assert len(next(iter(logs.values()))) == 5


def test_lgbm_pass_logger():
    logger = Live("train_logs")

    assert DVCLiveCallback().live is not logger
    assert DVCLiveCallback(live=logger).live is logger
