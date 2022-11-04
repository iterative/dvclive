# pylint: disable=protected-access
# pylint: disable=unused-argument
import json
import os

import pytest
from funcy import last

from dvclive import Live, env
from dvclive.error import (
    DataAlreadyLoggedError,
    InvalidDataTypeError,
    InvalidParameterTypeError,
)
from dvclive.plots import Metric
from dvclive.serialize import load_yaml
from dvclive.utils import parse_metrics


def read_history(live, metric):
    history, _ = parse_metrics(live)
    steps = []
    values = []
    name = os.path.join(live.plots_dir, Metric.subfolder, f"{metric}.tsv")
    for e in history[name]:
        steps.append(int(e["step"]))
        values.append(float(e[metric]))
    return steps, values


def read_latest(live, metric_name):
    _, latest = parse_metrics(live)
    return latest["step"], latest[metric_name]


def test_logging_no_step(tmp_dir):
    dvclive = Live("logs")

    dvclive.log_metric("m1", 1)
    dvclive.make_summary()

    assert not (tmp_dir / "logs" / "m1.tsv").is_file()
    assert (tmp_dir / dvclive.metrics_file).is_file()

    s = load_yaml(dvclive.metrics_file)
    assert s["m1"] == 1
    assert "step" not in s


@pytest.mark.parametrize(
    "param_name,param_value",
    [
        ("param_string", "value"),
        ("param_int", 42),
        ("param_float", 42.0),
        ("param_bool_true", True),
        ("param_bool_false", False),
        ("param_list", [1, 2, 3]),
        (
            "param_dict_simple",
            {"str": "value", "int": 42, "bool": True, "list": [1, 2, 3]},
        ),
        (
            "param_dict_nested",
            {
                "str": "value",
                "int": 42,
                "bool": True,
                "list": [1, 2, 3],
                "dict": {"nested-str": "value", "nested-int": 42},
            },
        ),
    ],
)
def test_log_param(tmp_dir, param_name, param_value):
    dvclive = Live()

    dvclive.log_param(param_name, param_value)

    s = load_yaml(dvclive.params_file)
    assert s[param_name] == param_value


def test_log_params(tmp_dir):
    dvclive = Live()
    params = {
        "param_string": "value",
        "param_int": 42,
        "param_float": 42.0,
        "param_bool_true": True,
        "param_bool_false": False,
    }

    dvclive.log_params(params)

    s = load_yaml(dvclive.params_file)
    assert s == params


@pytest.mark.parametrize("resume", (False, True))
def test_log_params_resume(tmp_dir, resume):
    dvclive = Live(resume=resume)
    dvclive.log_param("param", 42)

    dvclive = Live(resume=resume)
    assert ("param" in dvclive._params) == resume


def test_log_param_custom_obj(tmp_dir):
    dvclive = Live("logs")

    class Dummy:
        val = 42

    param_value = Dummy()

    with pytest.raises(InvalidParameterTypeError):
        dvclive.log_param("param_complex", param_value)


@pytest.mark.parametrize("path", ["logs", os.path.join("subdir", "logs")])
def test_logging_step(tmp_dir, path):
    dvclive = Live(path)
    dvclive.log_metric("m1", 1)
    dvclive.next_step()
    assert (tmp_dir / dvclive.dir).is_dir()
    assert (
        tmp_dir / dvclive.plots_dir / Metric.subfolder / "m1.tsv"
    ).is_file()
    assert (tmp_dir / dvclive.metrics_file).is_file()

    s = load_yaml(dvclive.metrics_file)
    assert s["m1"] == 1
    assert s["step"] == 0


def test_nested_logging(tmp_dir):
    dvclive = Live("logs")

    out = tmp_dir / dvclive.plots_dir / Metric.subfolder

    dvclive.log_metric("train/m1", 1)
    dvclive.log_metric("val/val_1/m1", 1)
    dvclive.log_metric("val/val_1/m2", 1)

    dvclive.next_step()

    assert (out / "val" / "val_1").is_dir()
    assert (out / "train" / "m1.tsv").is_file()
    assert (out / "val" / "val_1" / "m1.tsv").is_file()
    assert (out / "val" / "val_1" / "m2.tsv").is_file()

    summary = load_yaml(dvclive.metrics_file)

    assert summary["train"]["m1"] == 1
    assert summary["val"]["val_1"]["m1"] == 1
    assert summary["val"]["val_1"]["m2"] == 1


@pytest.mark.parametrize(
    "html",
    [True, False],
)
def test_cleanup(tmp_dir, html):
    dvclive = Live("logs", report="html" if html else None)
    dvclive.log_metric("m1", 1)
    dvclive.next_step()

    html_path = tmp_dir / dvclive.report_file
    if html:
        html_path.touch()

    (tmp_dir / "logs" / "some_user_file.txt").touch()

    assert (
        tmp_dir / dvclive.plots_dir / Metric.subfolder / "m1.tsv"
    ).is_file()
    assert (tmp_dir / dvclive.metrics_file).is_file()
    assert html_path.is_file() == html

    dvclive = Live("logs")

    assert (tmp_dir / "logs" / "some_user_file.txt").is_file()
    assert not (tmp_dir / dvclive.plots_dir / Metric.subfolder).exists()
    assert not (tmp_dir / dvclive.metrics_file).is_file()
    assert not (html_path).is_file()


def test_cleanup_params(tmp_dir):
    dvclive = Live("logs")
    dvclive.log_param("param", 42)

    assert os.path.isfile(dvclive.params_file)

    dvclive = Live("logs")
    assert not os.path.exists(dvclive.params_file)


@pytest.mark.parametrize(
    "resume, steps, metrics",
    [(True, [0, 1, 2, 3], [0.9, 0.8, 0.7, 0.6]), (False, [0, 1], [0.7, 0.6])],
)
def test_continue(tmp_dir, resume, steps, metrics):
    dvclive = Live("logs")

    for metric in [0.9, 0.8]:
        dvclive.log_metric("metric", metric)
        dvclive.next_step()

    assert read_history(dvclive, "metric") == ([0, 1], [0.9, 0.8])
    assert read_latest(dvclive, "metric") == (1, 0.8)

    dvclive = Live("logs", resume=resume)

    for new_metric in [0.7, 0.6]:
        dvclive.log_metric("metric", new_metric)
        dvclive.next_step()

    assert read_history(dvclive, "metric") == (steps, metrics)
    assert read_latest(dvclive, "metric") == (last(steps), last(metrics))


def test_resume_on_first_init(tmp_dir):
    dvclive = Live(resume=True)

    assert dvclive._step == 0


def test_resume_env_var(tmp_dir, monkeypatch):
    assert not Live()._resume

    monkeypatch.setenv(env.DVCLIVE_RESUME, True)
    assert Live()._resume


@pytest.mark.parametrize("metric", ["m1", os.path.join("train", "m1")])
def test_require_step_update(tmp_dir, metric):
    dvclive = Live("logs")

    dvclive.log_metric(metric, 1.0)

    with pytest.raises(
        DataAlreadyLoggedError,
        match="has already been logged with step '0'",
    ):
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
    assert read_latest(dvclive, "m") == (last(steps), last(metrics))


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


@pytest.mark.parametrize("invalid_type", [{0: 1}, [0, 1], "foo", (0, 1)])
def test_invalid_metric_type(tmp_dir, invalid_type):
    dvclive = Live()

    with pytest.raises(
        InvalidDataTypeError,
        match=f"Data 'm' has not supported type {type(invalid_type)}",
    ):
        dvclive.log_metric("m", invalid_type)


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


def test_logger(tmp_dir, mocker, monkeypatch):
    logger = mocker.patch("dvclive.live.logger")
    monkeypatch.setenv(env.DVCLIVE_LOGLEVEL, "DEBUG")

    live = Live()
    msg = "Report file (if generated)"
    assert msg in logger.info.call_args[0][0]
    live.log_metric("foo", 0)
    logger.debug.assert_called_with("Logged foo: 0")
    live.next_step()
    logger.debug.assert_called_with("Step: 1")

    live = Live(resume=True)
    logger.info.assert_called_with("Resumed from step 0")


def test_make_summary_without_calling_log(tmp_dir):
    dvclive = Live()

    dvclive.summary["foo"] = 1.0
    dvclive.make_summary()

    assert json.loads((tmp_dir / dvclive.metrics_file).read_text()) == {
        # no `step`
        "foo": 1.0
    }
    log_file = tmp_dir / dvclive.plots_dir / Metric.subfolder / "foo.tsv"
    assert not log_file.exists()


@pytest.mark.parametrize("timestamp", (True, False))
def test_log_metric_timestamp(timestamp):
    live = Live()
    live.log_metric("foo", 1.0, timestamp=timestamp)
    live.next_step()

    history, _ = parse_metrics(live)
    logged = next(iter(history.values()))
    assert ("timestamp" in logged[0]) == timestamp


def test_make_summary_is_called_on_end(tmp_dir):
    live = Live()

    live.summary["foo"] = 1.0
    live.end()

    assert json.loads((tmp_dir / live.metrics_file).read_text()) == {
        # no `step`
        "foo": 1.0
    }
    log_file = tmp_dir / live.plots_dir / Metric.subfolder / "foo.tsv"
    assert not log_file.exists()


def test_context_manager(tmp_dir):
    with Live() as live:
        live.summary["foo"] = 1.0

    assert json.loads((tmp_dir / live.metrics_file).read_text()) == {
        # no `step`
        "foo": 1.0
    }
    log_file = tmp_dir / live.plots_dir / Metric.subfolder / "foo.tsv"
    assert not log_file.exists()
    report_file = tmp_dir / live.report_file
    assert report_file.exists()
