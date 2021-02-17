import csv
import json
import os

import pytest
from funcy import last

import dvclive
from dvclive import env

# pylint: disable=unused-argument
from dvclive.dvc import SIGNAL_FILE


def read_logs(path):
    assert os.path.isdir(path)
    history = {}
    for p in os.listdir(path):
        metric_name = os.path.splitext(p)[0]
        history[metric_name] = _parse_tsv(os.path.join(path, p))
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


@pytest.mark.parametrize(
    "dvc_repo,html,signal_exists",
    [
        (True, False, False),
        (True, True, True),
        (False, True, False),
        (False, False, False),
    ],
)
def test_html(tmp_dir, dvc_repo, html, signal_exists):
    if dvc_repo:
        from dvc.repo import Repo

        Repo.init(no_scm=True)

    dvclive.init("logs", html=html)

    dvclive.log("m1", 1)
    dvclive.next_step()

    assert (tmp_dir / ".dvc" / "tmp" / SIGNAL_FILE).is_file() == signal_exists


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


@pytest.mark.parametrize("html", [True, False])
@pytest.mark.parametrize("summary", [True, False])
def test_init_from_env(tmp_dir, summary, html):
    os.environ[env.DVCLIVE_PATH] = "logs"
    os.environ[env.DVCLIVE_SUMMARY] = str(int(summary))
    os.environ[env.DVCLIVE_HTML] = str(int(html))

    dvclive.log("m", 0.1)

    assert dvclive._metric_logger._path == "logs"
    assert dvclive._metric_logger._summary == summary
    assert dvclive._metric_logger._html == html
