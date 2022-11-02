import os

import catalyst
import pytest
import torch
from catalyst import dl

from dvclive import Live
from dvclive.catalyst import DVCLiveCallback
from dvclive.plots import Metric

# pylint: disable=redefined-outer-name, unused-argument


@pytest.fixture
def runner():
    return dl.SupervisedRunner(
        engine=catalyst.utils.torch.get_available_engine(cpu=True),
        input_key="features",
        output_key="logits",
        target_key="targets",
        loss_key="loss",
    )


# see:
# https://github.com/catalyst-team/catalyst/blob/e99f9/tests/catalyst/callbacks/test_batch_overfit.py
@pytest.fixture
def runner_params():
    from torch.utils.data import DataLoader, TensorDataset

    catalyst.utils.set_global_seed(42)
    num_samples, num_features = int(32e1), int(1e1)
    X, y = torch.rand(num_samples, num_features), torch.rand(num_samples)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=0)
    loaders = {"train": loader, "valid": loader}

    model = torch.nn.Linear(num_features, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6])
    return {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loaders": loaders,
    }


def test_catalyst_callback(tmp_dir, runner, runner_params):
    runner.train(
        **runner_params,
        num_epochs=2,
        callbacks=[
            dl.AccuracyCallback(input_key="logits", target_key="targets"),
            DVCLiveCallback(),
        ],
        logdir="./logs",
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
        load_best_on_end=True,
    )

    assert os.path.exists("dvclive")

    train_path = tmp_dir / "dvclive" / "plots" / Metric.subfolder / "train"
    valid_path = tmp_dir / "dvclive" / "plots" / Metric.subfolder / "valid"

    assert train_path.is_dir()
    assert valid_path.is_dir()
    assert any("accuracy" in x.name for x in train_path.iterdir())
    assert any("accuracy" in x.name for x in valid_path.iterdir())


def test_catalyst_model_file(tmp_dir, runner, runner_params):
    runner.train(
        **runner_params,
        num_epochs=2,
        callbacks=[
            dl.AccuracyCallback(input_key="logits", target_key="targets"),
            DVCLiveCallback("model.pth"),
        ],
        logdir="./logs",
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
        load_best_on_end=True,
    )
    assert (tmp_dir / "model.pth").is_file()


def test_catalyst_pass_logger():
    logger = Live("train_logs")

    assert DVCLiveCallback().live is not logger
    assert DVCLiveCallback(live=logger).live is logger
