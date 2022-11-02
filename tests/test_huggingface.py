import os

import numpy as np
import pytest
import torch
from torch import nn
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

from dvclive import Live
from dvclive.huggingface import DVCLiveCallback
from dvclive.plots.metric import Metric
from dvclive.utils import parse_metrics

# pylint: disable=redefined-outer-name, unused-argument, no-value-for-parameter


def compute_metrics(eval_preds):
    """https://github.com/iterative/dvclive/pull/321#issuecomment-1266916039"""
    import time

    time.sleep(time.get_clock_info("time").resolution)
    return {"foo": 1}


# From transformers/tests/trainer


class RegressionDataset:
    def __init__(self, a=2, b=3, length=64, seed=42, label_names=None):
        np.random.seed(seed)
        self.label_names = ["labels"] if label_names is None else label_names
        self.length = length
        self.x = np.random.normal(size=(length,)).astype(np.float32)
        self.ys = [
            a * self.x + b + np.random.normal(scale=0.1, size=(length,))
            for _ in self.label_names
        ]
        self.ys = [y.astype(np.float32) for y in self.ys]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        result = {name: y[i] for name, y in zip(self.label_names, self.ys)}
        result["input_x"] = self.x[i]
        return result


class RegressionModelConfig(PretrainedConfig):
    def __init__(
        self, a=0, b=0, double_output=False, random_torch=True, **kwargs
    ):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.double_output = double_output
        self.random_torch = random_torch
        self.hidden_size = 1


class RegressionPreTrainedModel(PreTrainedModel):
    config_class = RegressionModelConfig
    base_model_prefix = "regression"

    def __init__(self, config):
        super().__init__(config)
        self.a = nn.Parameter(torch.tensor(config.a).float())
        self.b = nn.Parameter(torch.tensor(config.b).float())
        self.double_output = config.double_output

    def forward(self, input_x, labels=None, **kwargs):
        y = input_x * self.a + self.b
        if labels is None:
            return (y, y) if self.double_output else (y,)
        loss = nn.functional.mse_loss(y, labels)
        return (loss, y, y) if self.double_output else (loss, y)


@pytest.fixture
def data():
    return RegressionDataset(), RegressionDataset()


@pytest.fixture
def model():
    config = RegressionModelConfig()
    return RegressionPreTrainedModel(config)


@pytest.fixture
def args():
    return TrainingArguments(
        "foo",
        evaluation_strategy="epoch",
        num_train_epochs=2,
    )


def test_huggingface_integration(tmp_dir, model, args, data):
    trainer = Trainer(
        model,
        args,
        train_dataset=data[0],
        eval_dataset=data[1],
        compute_metrics=compute_metrics,
    )
    callback = DVCLiveCallback()
    trainer.add_callback(callback)
    trainer.train()

    live = callback.live
    assert os.path.exists(live.dir)

    logs, _ = parse_metrics(live)

    assert len(logs) == 10

    scalars = os.path.join(live.plots_dir, Metric.subfolder)
    assert os.path.join(scalars, "eval", "foo.tsv") in logs
    assert os.path.join(scalars, "eval", "loss.tsv") in logs
    assert os.path.join(scalars, "train", "loss.tsv") in logs
    assert len(logs[os.path.join(scalars, "epoch.tsv")]) == 3
    assert len(logs[os.path.join(scalars, "eval", "loss.tsv")]) == 2


def test_huggingface_model_file(tmp_dir, model, args, data, mocker):
    model_path = tmp_dir / "model_hf"
    model_save = mocker.spy(model, "save_pretrained")
    trainer = Trainer(
        model,
        args,
        train_dataset=data[0],
        eval_dataset=data[1],
        compute_metrics=compute_metrics,
    )
    trainer.add_callback(DVCLiveCallback(model_file=model_path))
    trainer.train()

    assert model_path.is_dir()

    assert (model_path / "pytorch_model.bin").exists()
    assert (model_path / "config.json").exists()
    assert model_save.call_count == 2


def test_huggingface_pass_logger():
    logger = Live("train_logs")

    assert DVCLiveCallback().live is not logger
    assert DVCLiveCallback(live=logger).live is logger
