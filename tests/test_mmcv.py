import logging

import torch
from mmcv.runner import build_runner

import dvclive
from dvclive.mmcv import DVCLiveLoggerHook
from tests.test_main import read_logs

# pylint: disable=unused-argument


def test_mmcv_hook(tmp_dir, mocker):
    work_dir = tmp_dir / "work_dir"
    runner = _build_demo_runner(str(work_dir))

    hook = DVCLiveLoggerHook()
    runner.register_hook(hook, priority="VERY_LOW")

    next_step = mocker.spy(dvclive.metrics.MetricLogger, "next_step")
    log = mocker.spy(dvclive.metrics.MetricLogger, "log")
    loader = torch.utils.data.DataLoader(torch.ones((5, 2)))

    dvclive.init("logs")

    runner.run([loader, loader], [("train", 1), ("val", 1)])

    assert next_step.call_count == 5
    assert log.call_count == 12

    logs, _ = read_logs("logs")
    assert "learning_rate" in logs
    assert "momentum" in logs


def test_mmcv_model_file(tmp_dir, mocker):
    work_dir = tmp_dir / "work_dir"
    runner = _build_demo_runner(str(work_dir))

    hook = DVCLiveLoggerHook(model_file=tmp_dir / "model.pth")
    runner.register_hook(hook, priority="VERY_LOW")

    save_checkpoint = mocker.spy(runner, "save_checkpoint")
    loader = torch.utils.data.DataLoader(torch.ones((5, 2)))

    dvclive.init("logs")

    runner.run([loader, loader], [("train", 1), ("val", 1)])

    assert save_checkpoint.call_count == 1
    assert (tmp_dir / "model.pth").is_file()


def _build_demo_runner(
    workdir, runner_type="EpochBasedRunner", max_epochs=1, max_iters=None
):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)
            self.conv = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x):
            return self.linear(x)

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

        def val_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

    model = Model()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.95)

    runner = build_runner(
        dict(type=runner_type),
        default_args=dict(
            model=model,
            work_dir=workdir,
            optimizer=optimizer,
            logger=logging.getLogger(),
            max_epochs=max_epochs,
            max_iters=max_iters,
        ),
    )
    log_config = dict(interval=1, hooks=[dict(type="TextLoggerHook")])
    runner.register_logger_hooks(log_config)

    return runner
