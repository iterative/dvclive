import csv
import json
import os

import pytest
from funcy import last

from dvclive import DVCLIVE_PATH, DVCLIVE_SUMMARY, DvcLive, dvclive, init


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
    init(path)

    assert (tmp_dir / path).is_dir()


@pytest.mark.parametrize("dump_latest", [True, False])
def test_logging(tmp_dir, dump_latest):
    init("logs", dump_latest=dump_latest)

    dvclive.log("m1", 1)

    assert (tmp_dir / "logs").is_dir()
    assert (tmp_dir / "logs" / "m1.tsv").is_file()
    assert not (tmp_dir / "logs.json").is_file()

    dvclive.next_step()

    assert (tmp_dir / "logs.json").is_file() == dump_latest


@pytest.mark.parametrize("report", [True, False])
def test_dvc_summary(tmp_dir, report):
    init("logs", report=report)

    dvclive.log("m1", 1)
    dvclive.next_step()

    assert (tmp_dir / "logs.html").is_file() == report


@pytest.mark.parametrize(
    "is_continue, steps, metrics",
    [(True, [0, 1, 2, 3], [0.9, 0.8, 0.7, 0.6]), (False, [0, 1], [0.7, 0.6])],
)
def test_continue(
    tmp_dir, is_continue, steps, metrics
):  # pylint: disable=unused-argument
    init("logs")

    for metric in [0.9, 0.8]:
        dvclive.log("metric", metric)
        dvclive.next_step()

    assert read_history("logs", "metric") == ([0, 1], [0.9, 0.8])
    assert read_latest("logs", "metric") == (1, 0.8)

    init("logs", is_continue=is_continue)

    for new_metric in [0.7, 0.6]:
        dvclive.log("metric", new_metric)
        dvclive.next_step()

    assert read_history("logs", "metric") == (steps, metrics)
    assert read_latest("logs", "metric") == (last(steps), last(metrics))


def test_infer_next_step(tmp_dir, mocker):  # pylint: disable=unused-argument
    init("logs")

    m = mocker.spy(dvclive, "next_step")
    dvclive.log("m1", 1.0)
    dvclive.log("m1", 2.0)
    dvclive.log("m1", 3.0)

    assert read_history("logs", "m1") == ([0, 1, 2], [1.0, 2.0, 3.0])
    assert m.call_count == 2


def test_custom_steps(tmp_dir):  # pylint: disable=unused-argument
    init("logs")

    steps = [0, 62, 1000]
    metrics = [0.9, 0.8, 0.7]

    for step, metric in zip(steps, metrics):
        dvclive.log("m", metric, step=step)

    assert read_history("logs", "m") == (steps, metrics)


@pytest.mark.parametrize("summary", [True, False])
def test_init_from_env(tmp_dir, summary):  # pylint: disable=unused-argument
    logger = DvcLive()
    os.environ[DVCLIVE_PATH] = "logs"
    os.environ[DVCLIVE_SUMMARY] = str(summary)

    logger.log("m", 0.1)

    assert logger._dir == "logs"  # pylint: disable=protected-access
    assert logger._dump_latest == summary  # pylint: disable=protected-access
