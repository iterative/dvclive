# pylint: disable=protected-access
import json
import os
import re
from pathlib import Path

import pytest
from funcy import last

from dvclive import Live, env
from dvclive.data import Scalar

# pylint: disable=unused-argument
from dvclive.error import (
    ConfigMismatchError,
    DataAlreadyLoggedError,
    InvalidDataTypeError,
    InvalidParameterTypeError,
    ParameterAlreadyLoggedError,
)
from dvclive.serialize import load_yaml
from dvclive.utils import parse_tsv


def read_logs(path_: str):
    path = Path(path_)
    assert path.is_dir()
    history = {}
    for metric_file in path.rglob("*.tsv"):
        metric_name = str(metric_file).replace(str(path) + os.path.sep, "")
        metric_name = metric_name.replace(".tsv", "")
        history[metric_name] = parse_tsv(metric_file)
    latest = _parse_json(str(path.parent) + ".json")
    return history, latest


def read_history(path, metric):
    history, _ = read_logs(path)
    steps = []
    values = []
    for e in history[metric]:
        steps.append(int(e["step"]))
        values.append(float(e[metric]))
    return steps, values


def read_latest(path, metric_name):
    _, latest = read_logs(path)
    return latest["step"], latest[metric_name]


def _parse_json(path):
    with open(path, "r", encoding="utf-8") as fd:
        return json.load(fd)


def test_logging_no_step(tmp_dir):
    dvclive = Live("logs")

    dvclive.log("m1", 1)

    assert not (tmp_dir / "logs" / "m1.tsv").is_file()
    assert (tmp_dir / dvclive.summary_path).is_file()

    s = _parse_json(dvclive.summary_path)
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

    s = load_yaml(dvclive.params_path)
    assert s[param_name] == param_value


def test_log_param_already_logged(tmp_dir):
    dvclive = Live()

    dvclive.log_param("param", 42)
    with pytest.raises(
        ParameterAlreadyLoggedError,
        match="Parameter 'param=42' has already been logged.",
    ):
        dvclive.log_param("param", 42)

    with pytest.raises(
        ParameterAlreadyLoggedError,
        match=re.escape(
            "Parameter 'param=1' has already been logged (previous value=42)."
        ),
    ):
        dvclive.log_param("param", 1)


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

    s = load_yaml(dvclive.params_path)
    assert s == params


@pytest.mark.parametrize("resume", (False, True))
def test_log_params_resume(tmp_dir, resume):
    dvclive = Live(resume=resume)
    dvclive.log_param("param", 42)

    dvclive = Live(resume=resume)
    assert ("param" in dvclive._params) == resume

    if resume:
        with pytest.raises(ParameterAlreadyLoggedError):
            dvclive.log_param("param", 42)


@pytest.mark.parametrize("resume", (False, True))
def test_log_params_resume_log(tmp_dir, resume, caplog, mocker):
    dvclive = Live(resume=resume)
    dvclive.log_param("param", 42)

    dvclive = Live(resume=resume)
    assert ("param" in dvclive._params) == resume
    dvclive.next_step()

    if resume:
        spy = mocker.spy(dvclive, "_dump_params")
        dvclive.log_param("param", 42)
        assert (
            caplog.messages[-1]
            == "Resuming previous dvclive session, not logging params."
        )
        assert not spy.called


def test_log_params_already_logged(tmp_dir):
    dvclive = Live()
    params = {
        "param_string": "string_value",
        "param_int": 42,
        "param_float": 42.0,
        "param_bool_true": True,
        "param_bool_false": False,
    }

    dvclive.log_params(params)
    assert os.path.isfile(dvclive.params_path)

    with pytest.raises(
        ParameterAlreadyLoggedError,
        match="Parameter 'param_string=string_value' has already been logged.",
    ):

        dvclive.log_param("param_string", "string_value")

    with pytest.raises(
        ParameterAlreadyLoggedError,
        match=re.escape(
            "Parameter 'param_string=other_value' has already been logged (previous value=string_value)."  # noqa
        ),
    ):

        dvclive.log_param("param_string", "other_value")


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
    dvclive.log("m1", 1)
    dvclive.next_step()
    assert (tmp_dir / dvclive.dir).is_dir()
    assert (tmp_dir / dvclive.dir / Scalar.subfolder / "m1.tsv").is_file()
    assert (tmp_dir / dvclive.summary_path).is_file()

    s = load_yaml(dvclive.summary_path)
    assert s["m1"] == 1
    assert s["step"] == 0


def test_nested_logging(tmp_dir):
    dvclive = Live("logs")

    out = tmp_dir / dvclive.dir / Scalar.subfolder

    dvclive.log("train/m1", 1)
    dvclive.log("val/val_1/m1", 1)
    dvclive.log("val/val_1/m2", 1)

    dvclive.next_step()

    assert (out / "val" / "val_1").is_dir()
    assert (out / "train" / "m1.tsv").is_file()
    assert (out / "val" / "val_1" / "m1.tsv").is_file()
    assert (out / "val" / "val_1" / "m2.tsv").is_file()

    summary = load_yaml(dvclive.summary_path)

    assert summary["train"]["m1"] == 1
    assert summary["val"]["val_1"]["m1"] == 1
    assert summary["val"]["val_1"]["m2"] == 1


@pytest.mark.parametrize(
    "html",
    [True, False],
)
def test_cleanup(tmp_dir, html):
    dvclive = Live("logs", report="html" if html else None)
    dvclive.log("m1", 1)
    dvclive.next_step()

    html_path = tmp_dir / dvclive.report_path
    if html:
        html_path.touch()

    (tmp_dir / "logs" / "some_user_file.txt").touch()

    assert (tmp_dir / dvclive.dir / Scalar.subfolder / "m1.tsv").is_file()
    assert (tmp_dir / dvclive.summary_path).is_file()
    assert html_path.is_file() == html

    dvclive = Live("logs")

    assert (tmp_dir / "logs" / "some_user_file.txt").is_file()
    assert not (tmp_dir / dvclive.dir / Scalar.subfolder).exists()
    assert not (tmp_dir / dvclive.summary_path).is_file()
    assert not (html_path).is_file()


def test_cleanup_params(tmp_dir):
    dvclive = Live("logs")
    dvclive.log_param("param", 42)

    assert os.path.isfile(dvclive.params_path)

    dvclive = Live("logs")
    assert not os.path.exists(dvclive.params_path)


@pytest.mark.parametrize(
    "resume, steps, metrics",
    [(True, [0, 1, 2, 3], [0.9, 0.8, 0.7, 0.6]), (False, [0, 1], [0.7, 0.6])],
)
def test_continue(tmp_dir, resume, steps, metrics):
    dvclive = Live("logs")

    out = tmp_dir / dvclive.dir / Scalar.subfolder

    for metric in [0.9, 0.8]:
        dvclive.log("metric", metric)
        dvclive.next_step()

    assert read_history(out, "metric") == ([0, 1], [0.9, 0.8])
    assert read_latest(out, "metric") == (1, 0.8)

    dvclive = Live("logs", resume=resume)

    for new_metric in [0.7, 0.6]:
        dvclive.log("metric", new_metric)
        dvclive.next_step()

    assert read_history(out, "metric") == (steps, metrics)
    assert read_latest(out, "metric") == (last(steps), last(metrics))


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

    dvclive.log(metric, 1.0)

    with pytest.raises(
        DataAlreadyLoggedError,
        match="has already been logged with step 'None'",
    ):
        dvclive.log(metric, 2.0)


def test_custom_steps(tmp_dir):
    dvclive = Live("logs")

    out = tmp_dir / dvclive.dir / Scalar.subfolder

    steps = [0, 62, 1000]
    metrics = [0.9, 0.8, 0.7]

    for step, metric in zip(steps, metrics):
        dvclive.set_step(step)
        dvclive.log("m", metric)

    assert read_history(out, "m") == (steps, metrics)
    assert read_latest(out, "m") == (last(steps), last(metrics))


def test_log_reset_with_set_step(tmp_dir):
    dvclive = Live()
    out = tmp_dir / dvclive.dir / Scalar.subfolder

    for i in range(3):
        dvclive.set_step(i)
        dvclive.log("train_m", 1)

    for i in range(3):
        dvclive.set_step(i)
        dvclive.log("val_m", 1)

    assert read_history(out, "train_m") == ([0, 1, 2], [1, 1, 1])
    assert read_history(out, "val_m") == ([0, 1, 2], [1, 1, 1])
    assert read_latest(out, "train_m") == (2, 1)
    assert read_latest(out, "val_m") == (2, 1)


@pytest.mark.parametrize("html", [True, False])
def test_init_from_env(tmp_dir, html, monkeypatch):
    monkeypatch.setenv(env.DVCLIVE_PATH, "logs")
    monkeypatch.setenv(env.DVCLIVE_HTML, str(int(html)))

    dvclive = Live()
    assert dvclive._path == "logs"
    if html:
        html_path = str(dvclive.dir) + "_dvc_plots/index.html"
        assert dvclive._report == "html"
        assert dvclive.report_path == html_path
    else:
        assert dvclive._report is None
        assert dvclive.report_path == ""


def test_fail_on_conflict(tmp_dir, monkeypatch):
    monkeypatch.setenv(env.DVCLIVE_PATH, "logs")

    with pytest.raises(ConfigMismatchError):
        Live("dvclive")


@pytest.mark.parametrize("invalid_type", [{0: 1}, [0, 1], "foo", (0, 1)])
def test_invalid_metric_type(tmp_dir, invalid_type):
    dvclive = Live()

    with pytest.raises(
        InvalidDataTypeError,
        match=f"Data 'm' has not supported type {type(invalid_type)}",
    ):
        dvclive.log("m", invalid_type)


def test_get_step_resume(tmp_dir):
    dvclive = Live()

    for metric in [0.9, 0.8]:
        dvclive.log("metric", metric)
        dvclive.next_step()

    assert dvclive.get_step() == 2

    dvclive = Live(resume=True)
    assert dvclive.get_step() == 2

    dvclive = Live(resume=False)
    assert dvclive.get_step() == 0


def test_get_step_custom_steps(tmp_dir):
    dvclive = Live()

    steps = [0, 62, 1000]
    metrics = [0.9, 0.8, 0.7]

    for step, metric in zip(steps, metrics):
        dvclive.set_step(step)
        dvclive.log("x", metric)
        assert dvclive.get_step() == step


def test_get_step_control_flow(tmp_dir):
    dvclive = Live()

    out = tmp_dir / dvclive.dir / Scalar.subfolder

    while dvclive.get_step() < 10:
        dvclive.log("i", dvclive.get_step())
        dvclive.next_step()

    steps, values = read_history(out, "i")
    assert steps == list(range(10))
    assert values == [float(x) for x in range(10)]


def test_make_checkpoint(tmp_dir, mocker, monkeypatch):
    make_checkpoint = mocker.patch("dvclive.live.make_checkpoint")

    dvclive = Live()
    dvclive.log("foo", 1)
    dvclive.next_step()
    assert not make_checkpoint.called

    monkeypatch.setenv(env.DVC_CHECKPOINT, True)
    dvclive = Live()
    dvclive.log("foo", 1)
    dvclive.next_step()
    assert make_checkpoint.called


def test_logger(tmp_dir, mocker, monkeypatch):
    logger = mocker.patch("dvclive.live.logger")
    monkeypatch.setenv(env.DVCLIVE_LOGLEVEL, "DEBUG")

    live = Live()
    msg = "Report path (if generated)"
    assert msg in logger.info.call_args[0][0]
    live.log("foo", 0)
    logger.debug.assert_called_with("Logged foo: 0")
    live.next_step()
    logger.debug.assert_called_with("Step: 1")

    live = Live(resume=True)
    logger.info.assert_called_with("Resumed from step 0")
