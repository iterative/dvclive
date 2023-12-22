import os

import pytest

from dvclive import Live
from dvclive.utils import read_history, read_latest


@pytest.mark.parametrize("metric", ["m1", os.path.join("train", "m1")])
def test_allow_step_override(tmp_dir, metric):
    dvclive = Live("logs")

    dvclive.log_metric(metric, 1.0)
    dvclive.log_metric(metric, 2.0)


def test_custom_steps(tmp_dir):
    dvclive = Live("logs")

    steps = [0, 62, 1000]
    metrics = [0.9, 0.8, 0.7]

    for step, metric in zip(steps, metrics):
        dvclive.step = step
        dvclive.log_metric("m", metric)
        dvclive.make_summary()

    assert read_history(dvclive, "m") == (steps, metrics)
    assert read_latest(dvclive, "m") == (steps[-1], metrics[-1])


def test_log_reset_with_set_step(tmp_dir):
    dvclive = Live()

    for i in range(3):
        dvclive.step = i
        dvclive.log_metric("train_m", 1)
        dvclive.make_summary()

    for i in range(3):
        dvclive.step = i
        dvclive.log_metric("val_m", 1)
        dvclive.make_summary()

    assert read_history(dvclive, "train_m") == ([0, 1, 2], [1, 1, 1])
    assert read_history(dvclive, "val_m") == ([0, 1, 2], [1, 1, 1])
    assert read_latest(dvclive, "train_m") == (2, 1)
    assert read_latest(dvclive, "val_m") == (2, 1)


def test_get_step_resume(tmp_dir):
    dvclive = Live()

    for metric in [0.9, 0.8]:
        dvclive.log_metric("metric", metric)
        dvclive.next_step()

    assert dvclive.step == 2

    dvclive = Live(resume=True)
    assert dvclive.step == 2

    dvclive = Live(resume=False)
    assert dvclive.step == 0


def test_get_step_custom_steps(tmp_dir):
    dvclive = Live()

    steps = [0, 62, 1000]
    metrics = [0.9, 0.8, 0.7]

    for step, metric in zip(steps, metrics):
        dvclive.step = step
        dvclive.log_metric("x", metric)
        assert dvclive.step == step


def test_get_step_control_flow(tmp_dir):
    dvclive = Live()

    while dvclive.step < 10:
        dvclive.log_metric("i", dvclive.step)
        dvclive.next_step()

    steps, values = read_history(dvclive, "i")
    assert steps == list(range(10))
    assert values == [float(x) for x in range(10)]


def test_set_step_only(tmp_dir):
    dvclive = Live()
    dvclive.step = 1
    dvclive.end()

    assert dvclive.read_latest() == {"step": 1}
    assert not os.path.exists(os.path.join(tmp_dir, "dvclive", "plots"))


def test_step_on_end(tmp_dir):
    dvclive = Live()
    for metric in range(3):
        dvclive.log_metric("m", metric)
        dvclive.next_step()
    dvclive.end()
    assert dvclive.step == metric

    assert dvclive.read_latest() == {"step": metric, "m": metric}
