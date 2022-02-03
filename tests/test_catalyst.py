import os

import pytest
from catalyst import dl
from catalyst.contrib.datasets import MNIST
from catalyst.utils.torch import get_available_engine
from torch import nn, optim
from torch.utils.data import DataLoader

from dvclive.catalyst import DvcLiveCallback
from dvclive.data import Scalar

# pylint: disable=redefined-outer-name, unused-argument


@pytest.fixture(scope="session")
def loaders(tmp_path_factory):
    path = tmp_path_factory.mktemp("catalyst_mnist")
    train_data = MNIST(path, train=True, download=True)
    valid_data = MNIST(path, train=False, download=True)
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

    assert os.path.exists("dvclive")

    train_path = tmp_dir / "dvclive" / Scalar.subfolder / "train"
    valid_path = tmp_dir / "dvclive" / Scalar.subfolder / "valid"

    assert train_path.is_dir()
    assert valid_path.is_dir()
    assert any("accuracy" in x.name for x in train_path.iterdir())
    assert any("accuracy" in x.name for x in valid_path.iterdir())


def test_catalyst_model_file(tmp_dir, runner, loaders):
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
