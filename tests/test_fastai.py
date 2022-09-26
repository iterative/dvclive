import os

import pytest
from fastai.tabular.all import (
    Categorify,
    Normalize,
    TabularDataLoaders,
    accuracy,
    tabular_learner,
)

from dvclive.data.scalar import Scalar
from dvclive.fastai import DvcLiveCallback

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
    learn.model_dir = os.path.abspath("./")
    learn.fit_one_cycle(2, cbs=[DvcLiveCallback("model")])

    assert os.path.exists("dvclive")

    train_path = tmp_dir / "dvclive" / Scalar.subfolder / "train"
    valid_path = tmp_dir / "dvclive" / Scalar.subfolder / "eval"

    assert train_path.is_dir()
    assert valid_path.is_dir()
    assert (tmp_dir / "dvclive" / Scalar.subfolder / "accuracy.tsv").exists()


def test_fastai_model_file(tmp_dir, data_loader):
    learn = tabular_learner(data_loader, metrics=accuracy)
    learn.model_dir = os.path.abspath("./")
    learn.fit_one_cycle(2, cbs=[DvcLiveCallback("model")])
    assert (tmp_dir / "model.pth").is_file()
