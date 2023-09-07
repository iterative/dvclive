import json

from dvclive import Live
from dvclive.plots import Metric


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


def test_make_summary_on_end_dont_increment_step(tmp_dir):
    with Live() as live:
        for i in range(2):
            live.log_metric("foo", i)
            live.next_step()

    assert json.loads((tmp_dir / live.metrics_file).read_text()) == {
        "foo": 1.0,
        "step": 1,
    }
