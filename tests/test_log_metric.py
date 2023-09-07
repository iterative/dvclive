import math
import os

import numpy as np
import pytest

from dvclive import Live
from dvclive.error import InvalidDataTypeError
from dvclive.plots import Metric
from dvclive.serialize import load_yaml
from dvclive.utils import parse_metrics, parse_tsv


def test_logging_no_step(tmp_dir):
    dvclive = Live("logs")

    dvclive.log_metric("m1", 1, plot=False)
    dvclive.make_summary()

    assert not (tmp_dir / "logs" / "plots" / "metrics" / "m1.tsv").is_file()
    assert (tmp_dir / dvclive.metrics_file).is_file()

    s = load_yaml(dvclive.metrics_file)
    assert s["m1"] == 1
    assert "step" not in s


@pytest.mark.parametrize("path", ["logs", os.path.join("subdir", "logs")])
def test_logging_step(tmp_dir, path):
    dvclive = Live(path)
    dvclive.log_metric("m1", 1)
    dvclive.next_step()
    assert (tmp_dir / dvclive.dir).is_dir()
    assert (tmp_dir / dvclive.plots_dir / Metric.subfolder / "m1.tsv").is_file()
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

    assert "m1" in parse_tsv(out / "train" / "m1.tsv")[0]
    assert "m1" in parse_tsv(out / "val" / "val_1" / "m1.tsv")[0]
    assert "m2" in parse_tsv(out / "val" / "val_1" / "m2.tsv")[0]

    summary = load_yaml(dvclive.metrics_file)

    assert summary["train"]["m1"] == 1
    assert summary["val"]["val_1"]["m1"] == 1
    assert summary["val"]["val_1"]["m2"] == 1


@pytest.mark.parametrize("timestamp", [True, False])
def test_log_metric_timestamp(tmp_dir, timestamp):
    live = Live()
    live.log_metric("foo", 1.0, timestamp=timestamp)
    live.next_step()

    history, _ = parse_metrics(live)
    logged = next(iter(history.values()))
    assert ("timestamp" in logged[0]) == timestamp


@pytest.mark.parametrize("invalid_type", [{0: 1}, [0, 1], (0, 1)])
def test_invalid_metric_type(tmp_dir, invalid_type):
    dvclive = Live()

    with pytest.raises(
        InvalidDataTypeError,
        match=f"Data 'm' has not supported type {type(invalid_type)}",
    ):
        dvclive.log_metric("m", invalid_type)


@pytest.mark.parametrize(
    ("val"),
    [math.inf, math.nan, np.nan, np.inf],
)
def test_log_metric_inf_nan(tmp_dir, val):
    with Live() as live:
        live.log_metric("metric", val)
    assert live.summary["metric"] == str(val)


def test_log_metic_str(tmp_dir):
    with Live() as live:
        live.log_metric("metric", "foo")
    assert live.summary["metric"] == "foo"
