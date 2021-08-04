import csv
import json
import os
from pathlib import Path

import pytest
from funcy import last

import dvclive
from dvclive import env

# pylint: disable=unused-argument
from dvclive.dvc import SIGNAL_FILE
from dvclive.error import (
    ConfigMismatchError,
    DvcLiveError,
    InitializationError,
)


def read_logs(path: str):
    assert os.path.isdir(path)
    history = {}
    for metric_file in Path(path).rglob("*.tsv"):
        metric_name = str(metric_file).replace(path + os.path.sep, "")
        metric_name = metric_name.replace(".tsv", "")
        history[metric_name] = _parse_tsv(metric_file)
    latest = _parse_json(path + ".json")
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


def _parse_tsv(path):
    with open(path, "r") as fd:
        reader = csv.DictReader(fd, delimiter="\t")
        return list(reader)


def _parse_json(path):
    with open(path, "r") as fd:
        return json.load(fd)


@pytest.mark.parametrize("path", ["logs", os.path.join("subdir", "logs")])
def test_create_logs_dir(tmp_dir, path):
    dvclive.init(path)

    assert (tmp_dir / path).is_dir()


@pytest.mark.parametrize("summary", [True, False])
def test_logging(tmp_dir, summary):
    dvclive.init("logs", summary=summary)

    dvclive.log("m1", 1)

    assert (tmp_dir / "logs").is_dir()
    assert (tmp_dir / "logs" / "m1.tsv").is_file()
    assert not (tmp_dir / "logs.json").is_file()

    dvclive.next_step()

    assert (tmp_dir / "logs.json").is_file() == summary


def test_nested_logging(tmp_dir):
    dvclive.init("logs", summary=True)

    dvclive.log("train/m1", 1)
    dvclive.log("val/val_1/m1", 1)

    assert (tmp_dir / "logs").is_dir()
    assert (tmp_dir / "logs" / "train").is_dir()
    assert (tmp_dir / "logs" / "val" / "val_1").is_dir()
    assert (tmp_dir / "logs" / "train" / "m1.tsv").is_file()
    assert (tmp_dir / "logs" / "val" / "val_1" / "m1.tsv").is_file()

    dvclive.next_step()

    _, summary = read_logs("logs")

    assert summary["train"]["m1"] == 1
    assert summary["val"]["val_1"]["m1"] == 1


@pytest.mark.parametrize(
    "dvc_repo,html,signal_exists",
    [
        (True, False, False),
        (True, True, True),
        (False, True, False),
        (False, False, False),
    ],
)
def test_html(tmp_dir, dvc_repo, html, signal_exists, monkeypatch):
    if dvc_repo:
        from dvc.repo import Repo

        Repo.init(no_scm=True)
    monkeypatch.setenv(env.DVCLIVE_PATH, "logs")
    monkeypatch.setenv(env.DVCLIVE_HTML, str(int(html)))

    dvclive.log("m1", 1)
    dvclive.next_step()

    assert (tmp_dir / ".dvc" / "tmp" / SIGNAL_FILE).is_file() == signal_exists


@pytest.mark.parametrize(
    "summary,html",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_cleanup(tmp_dir, summary, html):
    dvclive.init("logs", summary=summary)
    dvclive.log("m1", 1)
    dvclive.next_step()
    if html:
        (tmp_dir / "logs.html").touch()

    (tmp_dir / "logs" / "some_user_file.txt").touch()

    assert (tmp_dir / "logs" / "m1.tsv").is_file()
    assert (tmp_dir / "logs.json").is_file() == summary
    assert (tmp_dir / "logs.html").is_file() == html

    dvclive.init("logs")

    assert (tmp_dir / "logs" / "some_user_file.txt").is_file()
    assert not (tmp_dir / "logs" / "m1.tsv").is_file()
    assert not (tmp_dir / "logs.json").is_file()
    assert not (tmp_dir / "logs.html").is_file()


@pytest.mark.parametrize(
    "resume, steps, metrics",
    [(True, [0, 1, 2, 3], [0.9, 0.8, 0.7, 0.6]), (False, [0, 1], [0.7, 0.6])],
)
def test_continue(tmp_dir, resume, steps, metrics):
    dvclive.init("logs")

    for metric in [0.9, 0.8]:
        dvclive.log("metric", metric)
        dvclive.next_step()

    assert read_history("logs", "metric") == ([0, 1], [0.9, 0.8])
    assert read_latest("logs", "metric") == (1, 0.8)

    dvclive.init("logs", resume=resume)

    for new_metric in [0.7, 0.6]:
        dvclive.log("metric", new_metric)
        dvclive.next_step()

    assert read_history("logs", "metric") == (steps, metrics)
    assert read_latest("logs", "metric") == (last(steps), last(metrics))


def test_infer_next_step(tmp_dir, mocker):
    dvclive.init("logs")

    m = mocker.spy(dvclive.metrics.MetricLogger, "next_step")
    dvclive.log("m1", 1.0)
    dvclive.log("m1", 2.0)
    dvclive.log("m1", 3.0)

    assert read_history("logs", "m1") == ([0, 1, 2], [1.0, 2.0, 3.0])
    assert m.call_count == 2


def test_custom_steps(tmp_dir):
    dvclive.init("logs")

    steps = [0, 62, 1000]
    metrics = [0.9, 0.8, 0.7]

    for step, metric in zip(steps, metrics):
        dvclive.log("m", metric, step=step)

    assert read_history("logs", "m") == (steps, metrics)


def test_log_reset_with_step_0(tmp_dir):
    for i in range(3):
        dvclive.log("train_m", 1, step=i)

    for i in range(3):
        dvclive.log("val_m", 1, step=i)

    assert read_history("dvclive", "train_m") == ([0, 1, 2], [1, 1, 1])
    assert read_history("dvclive", "val_m") == ([0, 1, 2], [1, 1, 1])


@pytest.mark.parametrize("html", [True, False])
@pytest.mark.parametrize("summary", [True, False])
def test_init_from_env(tmp_dir, summary, html, monkeypatch):
    monkeypatch.setenv(env.DVCLIVE_PATH, "logs")
    monkeypatch.setenv(env.DVCLIVE_SUMMARY, str(int(summary)))
    monkeypatch.setenv(env.DVCLIVE_HTML, str(int(html)))

    dvclive.log("m", 0.1)

    assert dvclive._metric_logger._path == "logs"
    assert dvclive._metric_logger._summary == summary
    assert dvclive._metric_logger._html == html


@pytest.mark.parametrize("summary", [True, False])
def test_init_overrides_env(tmp_dir, summary, monkeypatch):
    monkeypatch.setenv(env.DVCLIVE_PATH, "FOO")
    monkeypatch.setenv(env.DVCLIVE_SUMMARY, str(int(not summary)))

    dvclive.init("logs", summary=summary)

    assert dvclive._metric_logger._path == "logs"
    assert dvclive._metric_logger._summary == summary


def test_no_init(tmp_dir):
    dvclive.log("m", 0.1)

    assert os.path.isdir("dvclive")


def test_fail_on_conflict(tmp_dir, monkeypatch):
    dvclive.init("some_dir")
    monkeypatch.setenv(env.DVCLIVE_PATH, "logs")

    with pytest.raises(ConfigMismatchError):
        dvclive.log("m", 0.1)


@pytest.mark.parametrize("invalid_type", [{0: 1}, [0, 1], "foo", (0, 1)])
def test_invalid_metric_type(tmp_dir, invalid_type):

    with pytest.raises(DvcLiveError, match="has not supported type"):
        dvclive.log("m", invalid_type)


def test_initialization_error(tmp_dir):
    with pytest.raises(InitializationError):
        dvclive.next_step()
