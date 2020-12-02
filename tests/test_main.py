import os

import pytest
from funcy import last

from dvclive import dvclive, init
from dvclive.serialize import read_logs


def read_history(logs_dir, metric_name):
    history, _ = read_logs(logs_dir)
    steps = []
    values = []
    for e in history[metric_name + ".tsv"]:
        steps.append(int(e["step"]))
        values.append(float(e[metric_name]))
    return steps, values


def read_latest(logs_dir, metric_name):
    _, latest = read_logs(logs_dir)
    return latest["step"], latest[metric_name]


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
