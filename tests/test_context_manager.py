import json

from dvclive import Live
from dvclive.plots import Metric


def test_context_manager(tmp_dir):
    with Live(report="html") as live:
        live.summary["foo"] = 1.0

    assert json.loads((tmp_dir / live.metrics_file).read_text()) == {
        # no `step`
        "foo": 1.0
    }
    log_file = tmp_dir / live.plots_dir / Metric.subfolder / "foo.tsv"
    assert not log_file.exists()
    report_file = tmp_dir / live.report_file
    assert report_file.exists()


def test_context_manager_skips_end_calls(tmp_dir):
    with Live() as live:
        live.summary["foo"] = 1.0
        live.end()
        assert not (tmp_dir / live.metrics_file).exists()
    assert (tmp_dir / live.metrics_file).exists()
