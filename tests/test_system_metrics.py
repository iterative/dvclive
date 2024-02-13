import time

import pytest

from dvclive import Live
from dvclive.system_metrics import _get_cpus_metrics
from dvclive.utils import parse_metrics


def mock_psutil(mocker):
    mocker.patch(
        "dvclive.system_metrics.psutil.cpu_percent",
        return_value=[10, 20, 30, 40, 50, 60],
    )
    mocker.patch("dvclive.system_metrics.psutil.cpu_count", return_value=6)

    mocked_virtual_memory = mocker.MagicMock()
    mocked_virtual_memory.percent = 20
    mocked_virtual_memory.total = 4 * 1024**3

    mocked_disk_io_counters = mocker.MagicMock()
    mocked_disk_io_counters.read_bytes = 2 * 1024**2
    mocked_disk_io_counters.read_time = 1
    mocked_disk_io_counters.write_bytes = 2 * 1024**2
    mocked_disk_io_counters.write_time = 1

    mocking_dict = {
        "virtual_memory": mocked_virtual_memory,
        "disk_io_counters": mocked_disk_io_counters,
    }
    for function_name, return_value in mocking_dict.items():
        mocker.patch(
            f"dvclive.system_metrics.psutil.{function_name}",
            return_value=return_value,
        )


@pytest.mark.parametrize(
    ("metric_name"),
    [
        "system/cpu/usage (%)",
        "system/cpu/count",
        "system/cpu/parallelization (%)",
        "system/ram/usage (%)",
        "system/ram/usage (GB)",
        "system/ram/total (GB)",
        "system/io/read speed (MB)",
        "system/io/write speed (MB)",
    ],
)
def test_get_cpus_metrics(mocker, metric_name):
    mock_psutil(mocker)
    metrics = _get_cpus_metrics()
    assert metric_name in metrics


def test_monitor_system(tmp_dir):
    with Live(
        tmp_dir,
        save_dvc_exp=False,
        monitor_system=True,
    ) as live:
        time.sleep(5 + 1)  # allow the thread to finish
        live.next_step()
        time.sleep(5 + 1)  # allow the thread to finish
        timeseries, latest = parse_metrics(live)

    assert "system" in latest
    assert "cpu" in latest["system"]
    assert "usage (%)" in latest["system"]["cpu"]
    assert "count" in latest["system"]["cpu"]
    assert "parallelization (%)" in latest["system"]["cpu"]
    assert "ram" in latest["system"]
    assert "usage (%)" in latest["system"]["ram"]
    assert "usage (GB)" in latest["system"]["ram"]
    assert "total (GB)" in latest["system"]["ram"]
    assert "io" in latest["system"]
    assert "read speed (MB)" in latest["system"]["io"]
    assert "write speed (MB)" in latest["system"]["io"]

    assert any("usage (%).tsv" in key for key in timeseries)
    assert any("parallelization (%).tsv" in key for key in timeseries)
    assert any("usage (GB).tsv" in key for key in timeseries)
    assert any("read speed (MB).tsv" in key for key in timeseries)
    assert any("write speed (MB).tsv" in key for key in timeseries)
    assert all(len(timeseries[key]) == 2 for key in timeseries if "system" in key)

    # not plot for constant values
    assert all("count.tsv" not in key for key in timeseries)
    assert all("total (GB).tsv" not in key for key in timeseries)
