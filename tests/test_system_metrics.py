import time

import pytest

from dvclive import Live
from dvclive.system_metrics import CPUMetricsCallback, get_cpus_metrics
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
        "system/cpu/usage_avg_percent",
        "system/cpu/usage_max_percent",
        "system/cpu/count",
        "system/cpu/parallelism_percent",
        "system/cpu/ram_usage_percent",
        "system/cpu/ram_total_GB",
        "system/io/read_speed_MB",
        "system/io/write_speed_MB",
    ],
)
def test_get_cpus_metrics(mocker, metric_name):
    mock_psutil(mocker)
    metrics = get_cpus_metrics()
    assert metric_name in metrics


@pytest.mark.parametrize(
    ("duration", "interval", "plot"),
    [
        (1, 0.5, True),
        (2.0, 1, True),
    ],
)
def test_cpumetricscallback_with_plot(duration, interval, plot, tmpdir):
    with Live(
        tmpdir,
        save_dvc_exp=False,
        callbacks=[CPUMetricsCallback(duration, interval, plot)],
    ) as live:
        time.sleep(duration * 2)
        live.next_step()
        time.sleep(duration * 2 + 0.1)  # allow the thread to finish
        history, latest = parse_metrics(live)

    assert "system" in latest
    assert "cpu" in latest["system"]
    assert "usage_avg_percent" in latest["system"]["cpu"]
    assert "usage_max_percent" in latest["system"]["cpu"]
    assert "count" in latest["system"]["cpu"]
    assert "parallelism_percent" in latest["system"]["cpu"]
    assert "ram_usage_percent" in latest["system"]["cpu"]
    assert "ram_total_GB" in latest["system"]["cpu"]
    assert "io" in latest["system"]
    assert "read_speed_MB" in latest["system"]["io"]
    assert "write_speed_MB" in latest["system"]["io"]

    prefix = f"{tmpdir}/plots/metrics/system"
    assert f"{prefix}/cpu/usage_avg_percent.tsv" in history
    assert f"{prefix}/cpu/usage_max_percent.tsv" in history
    assert f"{prefix}/cpu/parallelism_percent.tsv" in history
    assert f"{prefix}/cpu/ram_usage_percent.tsv" in history
    assert f"{prefix}/io/write_speed_MB.tsv" in history
    assert f"{prefix}/io/read_speed_MB.tsv" in history
    assert len(history[f"{prefix}/cpu/ram_usage_percent.tsv"]) == 4

    assert f"{prefix}/cpu/count.tsv" not in history  # no plot for count
    assert f"{prefix}/cpu/ram_total_GB.tsv" not in history


@pytest.mark.parametrize(
    ("duration", "interval", "plot"),
    [
        (1, 0.5, False),
        (2.0, 1, False),
    ],
)
def test_cpumetricscallback_without_plot(duration, interval, plot, tmpdir):
    with Live(
        tmpdir,
        save_dvc_exp=False,
        callbacks=[CPUMetricsCallback(duration, interval, plot)],
    ) as live:
        time.sleep(duration * 2)
        live.next_step()
        time.sleep(duration * 2 + 0.1)  # allow the thread to finish
        history, latest = parse_metrics(live)

    assert "system" in latest
    assert "cpu" in latest["system"]
    assert "usage_avg_percent" in latest["system"]["cpu"]
    assert "usage_max_percent" in latest["system"]["cpu"]
    assert "count" in latest["system"]["cpu"]
    assert "parallelism_percent" in latest["system"]["cpu"]
    assert "ram_usage_percent" in latest["system"]["cpu"]
    assert "ram_total_GB" in latest["system"]["cpu"]
    assert "io" in latest["system"]
    assert "read_speed_MB" in latest["system"]["io"]
    assert "write_speed_MB" in latest["system"]["io"]

    prefix = f"{tmpdir}/plots/metrics/system"
    assert f"{prefix}/cpu/usage_avg_percent.tsv" not in history
    assert f"{prefix}/cpu/usage_max_percent.tsv" not in history
    assert f"{prefix}/cpu/count.tsv" not in history
    assert f"{prefix}/cpu/parallelism_percent.tsv" not in history
    assert f"{prefix}/cpu/ram_usage_percent.tsv" not in history
    assert f"{prefix}/cpu/ram_total_GB.tsv" not in history
    assert f"{prefix}/cpu/write_speed_MB.tsv" not in history
    assert f"{prefix}/cpu/read_speed_MB.tsv" not in history
