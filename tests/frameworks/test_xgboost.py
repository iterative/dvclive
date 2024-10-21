import os
from contextlib import nullcontext

import pytest

from dvclive import Live
from dvclive.plots.metric import Metric
from dvclive.utils import parse_metrics

try:
    import pandas as pd
    import xgboost as xgb
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    from dvclive.xgb import DVCLiveCallback
except ImportError:
    pytest.skip("skipping xgboost tests", allow_module_level=True)


@pytest.fixture
def train_params():
    return {"objective": "multi:softmax", "num_class": 3, "seed": 0}


@pytest.fixture
def iris_data():
    iris = datasets.load_iris()
    x = pd.DataFrame(iris["data"], columns=iris["feature_names"])
    y = iris["target"]
    return xgb.DMatrix(x, y)


@pytest.fixture
def iris_train_eval_data():
    iris = datasets.load_iris()
    x_train, x_eval, y_train, y_eval = train_test_split(
        iris.data, iris.target, random_state=0
    )
    return (xgb.DMatrix(x_train, y_train), xgb.DMatrix(x_eval, y_eval))


@pytest.mark.parametrize(
    ("metric_data", "subdirs", "context"),
    [
        (
            "eval",
            ("",),
            pytest.warns(DeprecationWarning, match="`metric_data`.+deprecated"),
        ),
        (None, ("train", "eval"), nullcontext()),
    ],
)
def test_xgb_integration(
    tmp_dir, train_params, iris_train_eval_data, metric_data, subdirs, context, mocker
):
    with context:
        callback = DVCLiveCallback(metric_data)
    live = callback.live
    spy = mocker.spy(live, "end")
    data_train, data_eval = iris_train_eval_data
    xgb.train(
        train_params,
        data_train,
        callbacks=[callback],
        num_boost_round=5,
        evals=[(data_train, "train"), (data_eval, "eval")],
    )
    spy.assert_called_once()

    assert os.path.exists("dvclive")

    logs, _ = parse_metrics(callback.live)
    assert len(logs) == len(subdirs)
    assert list(map(len, logs.values())) == [5] * len(logs)
    scalars = os.path.join(callback.live.plots_dir, Metric.subfolder)
    assert all(
        os.path.join(scalars, subdir, "mlogloss.tsv") in logs for subdir in subdirs
    )


def test_xgb_pass_logger():
    logger = Live("train_logs")

    assert DVCLiveCallback("eval_data").live is not logger
    assert DVCLiveCallback("eval_data", live=logger).live is logger
