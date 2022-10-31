import os

import pytest
from fastai.tabular.all import (
    Categorify,
    Normalize,
    ProgressCallback,
    TabularDataLoaders,
    accuracy,
    tabular_learner,
)

from dvclive import Live
from dvclive.fastai import DVCLiveCallback
from dvclive.plots.metric import Metric

# pylint: disable=redefined-outer-name, unused-argument


@pytest.fixture
def data_loader():
    from pandas import DataFrame

    d = {
        "x1": [1, 1, 0, 0, 1, 1, 0, 0],
        "x2": [1, 0, 1, 0, 1, 0, 1, 0],
        "y": [1, 0, 0, 1, 1, 0, 0, 1],
    }
    df = DataFrame(d)
    xor_loader = TabularDataLoaders.from_df(
        df,
        valid_idx=[4, 5, 6, 7],
        batch_size=2,
        cont_names=["x1", "x2"],
        procs=[Categorify, Normalize],
        y_names="y",
    )
    return xor_loader


def test_fastai_callback(tmp_dir, data_loader):
    learn = tabular_learner(data_loader, metrics=accuracy)
    learn.remove_cb(ProgressCallback)
    learn.model_dir = os.path.abspath("./")
    callback = DVCLiveCallback("model")
    live = callback.live
    learn.fit_one_cycle(2, cbs=[callback])

    assert os.path.exists(live.dir)

    metrics_path = tmp_dir / live.plots_dir / Metric.subfolder
    train_path = metrics_path / "train"
    valid_path = metrics_path / "eval"

    assert train_path.is_dir()
    assert valid_path.is_dir()
    assert (metrics_path / "accuracy.tsv").exists()


def test_fastai_model_file(tmp_dir, data_loader):
    learn = tabular_learner(data_loader, metrics=accuracy)
    learn.remove_cb(ProgressCallback)
    learn.model_dir = os.path.abspath("./")
    learn.fit_one_cycle(2, cbs=[DVCLiveCallback("model")])
    assert (tmp_dir / "model.pth").is_file()


def test_fastai_pass_logger():
    logger = Live("train_logs")

    assert DVCLiveCallback().live is not logger
    assert DVCLiveCallback(live=logger).live is logger
