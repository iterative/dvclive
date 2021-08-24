import os

import pytest
from catalyst import dl
from catalyst.contrib.datasets import MNIST
from catalyst.data import ToTensor
from catalyst.utils.torch import get_available_engine
from torch import nn, optim
from torch.utils.data import DataLoader

import dvclive
from dvclive.catalyst import DvcLiveCallback

# pylint: disable=redefined-outer-name, unused-argument


@pytest.fixture
def loaders():
    train_data = MNIST(
        os.getcwd(), train=True, download=True, transform=ToTensor()
    )
    valid_data = MNIST(
        os.getcwd(), train=False, download=True, transform=ToTensor()
    )
    return {
        "train": DataLoader(train_data, batch_size=32),
        "valid": DataLoader(valid_data, batch_size=32),
    }


@pytest.fixture
def runner():
    return dl.SupervisedRunner(
        engine=get_available_engine(),
        input_key="features",
        output_key="logits",
        target_key="targets",
        loss_key="loss",
    )


def test_catalyst_callback(tmp_dir, runner, loaders):
    dvclive.init("dvc_logs")

    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02)

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=2,
        callbacks=[
            dl.AccuracyCallback(input_key="logits", target_key="targets"),
            DvcLiveCallback(),
        ],
        logdir="./logs",
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
        load_best_on_end=True,
    )

    assert os.path.exists("dvc_logs")

    train_path = tmp_dir / "dvc_logs/train"
    valid_path = tmp_dir / "dvc_logs/valid"

    assert train_path.is_dir()
    assert valid_path.is_dir()
    assert (train_path / "accuracy.tsv").exists()


def test_catalyst_model_file(tmp_dir, runner, loaders):
    dvclive.init("dvc_logs")

    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02)

    runner.train(
        model=model,
        engine=runner.engine,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=2,
        callbacks=[
            dl.AccuracyCallback(input_key="logits", target_key="targets"),
            DvcLiveCallback("model.pth"),
        ],
        logdir="./logs",
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
        load_best_on_end=True,
    )
    assert (tmp_dir / "model.pth").is_file()
