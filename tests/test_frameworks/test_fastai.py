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


def test_fastai_callback(tmp_dir, data_loader, mocker):
    learn = tabular_learner(data_loader, metrics=accuracy)
    learn.remove_cb(ProgressCallback)
    callback = DVCLiveCallback()
    live = callback.live

    spy = mocker.spy(live, "end")
    learn.fit_one_cycle(2, cbs=[callback])
    spy.assert_called_once()

    assert (tmp_dir / live.dir).exists()
    assert (tmp_dir / live.params_file).exists()
    assert (tmp_dir / live.params_file).read_text() == (
        "model: TabularModel\nbatch_size: 2\nbatch_per_epoch: 2\nfrozen: false"
        "\nfrozen_idx: 0\ntransforms: None\n"
    )

    metrics_path = tmp_dir / live.plots_dir / Metric.subfolder
    train_path = metrics_path / "train"
    valid_path = metrics_path / "eval"

    assert train_path.is_dir()
    assert valid_path.is_dir()
    assert (metrics_path / "accuracy.tsv").exists()
    assert not (metrics_path / "epoch.tsv").exists()


def test_fastai_model_file(tmp_dir, data_loader, mocker):
    learn = tabular_learner(data_loader, metrics=accuracy)
    learn.remove_cb(ProgressCallback)
    learn.model_dir = os.path.abspath("./")
    save = mocker.spy(learn, "save")
    learn.fit_one_cycle(2, cbs=[DVCLiveCallback("model", with_opt=True)])
    assert (tmp_dir / "model.pth").is_file()
    save.assert_called_with("model", with_opt=True)


def test_fastai_pass_logger():
    logger = Live("train_logs")

    assert DVCLiveCallback().live is not logger
    assert DVCLiveCallback(live=logger).live is logger


def test_fast_ai_resume(tmp_dir, data_loader, mocker):
    learn = tabular_learner(data_loader, metrics=accuracy)
    learn.remove_cb(ProgressCallback)
    callback = DVCLiveCallback()
    live = callback.live

    spy = mocker.spy(live, "next_step")
    end = mocker.spy(live, "end")
    learn.fit_one_cycle(2, cbs=[callback])
    assert spy.call_count == 2
    assert end.call_count == 1

    callback = DVCLiveCallback(resume=True)
    live = callback.live
    spy = mocker.spy(live, "next_step")
    learn.fit_one_cycle(3, cbs=[callback], start_epoch=live.step)
    assert spy.call_count == 1


def test_fast_ai_avoid_unnecessary_end_calls(tmp_dir, data_loader, mocker):
    """
    `after_fit` might be called from different points and not all mean that the
    training has ended.
    """
    learn = tabular_learner(data_loader, metrics=accuracy)
    learn.remove_cb(ProgressCallback)
    callback = DVCLiveCallback()
    live = callback.live

    end = mocker.spy(live, "end")
    after_fit = mocker.spy(callback, "after_fit")
    learn.fine_tune(2, cbs=[callback])
    assert end.call_count == 1
    assert after_fit.call_count == 2
