import os

import pytest

from dvclive import Live
from dvclive.plots.metric import Metric
from dvclive.serialize import load_yaml
from dvclive.utils import parse_metrics

try:
    import numpy as np
    import torch
    from torch import nn
    from transformers import (
        PretrainedConfig,
        PreTrainedModel,
        Trainer,
        TrainingArguments,
    )
    from transformers.integrations import DVCLiveCallback as ExternalCallback

    from dvclive.huggingface import DVCLiveCallback as InternalCallback
except ImportError:
    pytest.skip("skipping huggingface tests", allow_module_level=True)


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
    def __init__(self, a=0, b=0, double_output=False, random_torch=True, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.double_output = double_output
        self.random_torch = random_torch
        self.hidden_size = 1


class RegressionPreTrainedModel(PreTrainedModel):
    config_class = RegressionModelConfig  # type: ignore[assignment]
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
        eval_strategy="epoch",
        num_train_epochs=2,
        save_strategy="epoch",
        report_to="none",  # Disable auto-reporting to avoid duplication
        use_cpu=True,
    )


@pytest.mark.parametrize("callback", [ExternalCallback, InternalCallback])
def test_huggingface_integration(tmp_dir, model, args, data, mocker, callback):
    trainer = Trainer(
        model,
        args,
        train_dataset=data[0],
        eval_dataset=data[1],
        compute_metrics=compute_metrics,
    )
    callback = callback()
    spy = mocker.spy(Live, "end")
    trainer.add_callback(callback)
    trainer.train()
    spy.assert_called_once()

    live = callback.live
    assert os.path.exists(live.dir)

    logs, _ = parse_metrics(live)

    scalars = os.path.join(live.plots_dir, Metric.subfolder)
    assert os.path.join(scalars, "eval", "foo.tsv") in logs
    assert os.path.join(scalars, "eval", "loss.tsv") in logs
    assert os.path.join(scalars, "train", "loss.tsv") in logs
    assert len(logs[os.path.join(scalars, "epoch.tsv")]) == 3
    assert len(logs[os.path.join(scalars, "eval", "loss.tsv")]) == 2

    params = load_yaml(live.params_file)
    assert params["num_train_epochs"] == 2


@pytest.mark.parametrize("log_model", ["all", True, False, None])
@pytest.mark.parametrize("best", [True, False])
@pytest.mark.parametrize("callback", [ExternalCallback, InternalCallback])
def test_huggingface_log_model(
    tmp_dir,
    mocked_dvc_repo,
    model,
    data,
    args,
    monkeypatch,
    mocker,
    log_model,
    best,
    callback,
):
    live = Live()
    log_artifact = mocker.patch.object(live, "log_artifact")
    if callback == ExternalCallback:
        monkeypatch.setenv("HF_DVCLIVE_LOG_MODEL", str(log_model))
        live_callback = callback(live=live)
    else:
        live_callback = callback(live=live, log_model=log_model)

    args.load_best_model_at_end = best
    args.metric_for_best_model = "loss"

    trainer = Trainer(
        model,
        args,
        train_dataset=data[0],
        eval_dataset=data[1],
        compute_metrics=compute_metrics,
    )
    trainer.add_callback(live_callback)
    trainer.train()

    expected_call_count = {
        "all": 2,
        True: 1,
        False: 0,
        None: 0,
    }
    assert log_artifact.call_count == expected_call_count[log_model]

    if log_model is True:
        name = "best" if best else "last"
        log_artifact.assert_called_with(
            os.path.join(args.output_dir, name),
            name=name,
            type="model",
            copy=True,
        )


@pytest.mark.parametrize("callback", [ExternalCallback, InternalCallback])
def test_huggingface_pass_logger(callback):
    logger = Live("train_logs")

    assert callback().live is not logger
    assert callback(live=logger).live is logger


@pytest.mark.parametrize("report_to", ["all", "dvclive", "none"])
def test_huggingface_report_to(model, report_to):
    args = TrainingArguments("foo", report_to=report_to)
    trainer = Trainer(
        model,
        args,
    )
    live_cbs = [
        cb
        for cb in trainer.callback_handler.callbacks
        if isinstance(cb, ExternalCallback)
    ]
    if report_to == "none":
        assert not any(live_cbs)
    else:
        assert any(live_cbs)
