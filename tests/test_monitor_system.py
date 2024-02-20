import time
from pathlib import Path
import pytest

import dpath
from pytest_voluptuous import S

from dvclive import Live
from dvclive.monitor_system import (
    CPUMonitor,
    GPUMonitor,
    METRIC_CPU_COUNT,
    METRIC_CPU_USAGE_PERCENT,
    METRIC_CPU_PARALLELIZATION_PERCENT,
    METRIC_RAM_USAGE_PERCENT,
    METRIC_RAM_USAGE_GB,
    METRIC_RAM_TOTAL_GB,
    METRIC_DISK_USAGE_PERCENT,
    METRIC_DISK_USAGE_GB,
    METRIC_DISK_TOTAL_GB,
    GIGABYTES_DIVIDER,
)
from dvclive.utils import parse_metrics


def mock_psutil_cpu(mocker):
    mocker.patch(
        "dvclive.monitor_system.psutil.cpu_percent",
        return_value=[10, 10, 10, 40, 50, 60],
    )
    mocker.patch("dvclive.monitor_system.psutil.cpu_count", return_value=6)


def mock_psutil_ram(mocker):
    mocked_ram = mocker.MagicMock()
    mocked_ram.percent = 50
    mocked_ram.used = 2 * GIGABYTES_DIVIDER
    mocked_ram.total = 4 * GIGABYTES_DIVIDER
    mocker.patch(
        "dvclive.monitor_system.psutil.virtual_memory", return_value=mocked_ram
    )


def mock_psutil_disk(mocker):
    mocked_disk = mocker.MagicMock()
    mocked_disk.percent = 50
    mocked_disk.used = 16 * GIGABYTES_DIVIDER
    mocked_disk.total = 32 * GIGABYTES_DIVIDER
    mocker.patch("dvclive.monitor_system.psutil.disk_usage", return_value=mocked_disk)


def mock_psutil_disk_with_oserror(mocker):
    mocked_disk = mocker.MagicMock()
    mocked_disk.percent = 50
    mocked_disk.used = 16 * GIGABYTES_DIVIDER
    mocked_disk.total = 32 * GIGABYTES_DIVIDER
    mocker.patch(
        "dvclive.monitor_system.psutil.disk_usage",
        side_effect=[
            mocked_disk,
            OSError,
            mocked_disk,
            OSError,
        ],
    )


def test_monitor_system_is_false(tmp_dir, mocker):
    mock_psutil_cpu(mocker)
    mock_psutil_ram(mocker)
    mock_psutil_disk(mocker)
    cpu_monitor_mock = mocker.patch("dvclive.live.CPUMonitor")
    with Live(
        tmp_dir,
        save_dvc_exp=False,
        monitor_system=False,
    ) as live:
        assert live.cpu_monitor is None

    cpu_monitor_mock.assert_not_called()


def test_monitor_system_is_true(tmp_dir, mocker):
    mock_psutil_cpu(mocker)
    mock_psutil_ram(mocker)
    mock_psutil_disk(mocker)
    cpu_monitor_mock = mocker.patch("dvclive.live.CPUMonitor", spec=CPUMonitor)

    with Live(
        tmp_dir,
        save_dvc_exp=False,
        monitor_system=True,
    ) as live:
        cpu_monitor = live.cpu_monitor

        assert isinstance(cpu_monitor, CPUMonitor)
        cpu_monitor_mock.assert_called_once()

        end_spy = mocker.spy(cpu_monitor, "end")
        end_spy.assert_not_called()

    # check the monitoring thread is stopped
    end_spy.assert_called_once()


def test_ignore_non_existent_directories(tmp_dir, mocker):
    mock_psutil_cpu(mocker)
    mock_psutil_ram(mocker)
    mock_psutil_disk_with_oserror(mocker)
    with Live(
        tmp_dir,
        save_dvc_exp=False,
        monitor_system=False,
    ) as live:
        non_existent_disk = "/non-existent"
        monitor = CPUMonitor(
            directories_to_monitor={"main": "/", "non-existent": non_existent_disk}
        )
        monitor(live)
        metrics = monitor._get_metrics()
        monitor.end()

    assert not Path(non_existent_disk).exists()

    assert f"{METRIC_DISK_USAGE_PERCENT}/non-existent" not in metrics
    assert f"{METRIC_DISK_USAGE_GB}/non-existent" not in metrics
    assert f"{METRIC_DISK_TOTAL_GB}/non-existent" not in metrics


@pytest.mark.timeout(2)
def test_monitor_system_metrics(tmp_dir, mocker):
    mock_psutil_cpu(mocker)
    mock_psutil_ram(mocker)
    mock_psutil_disk(mocker)
    with Live(
        tmp_dir,
        save_dvc_exp=False,
        monitor_system=False,
    ) as live:
        live.cpu_monitor = CPUMonitor(interval=0.05, num_samples=4)
        # wait for the metrics to be logged.
        # METRIC_DISK_TOTAL_GB is the last metric to be logged.
        while len(dpath.search(live.summary, METRIC_DISK_TOTAL_GB)) == 0:
            time.sleep(0.001)
        live.next_step()

        _, latest = parse_metrics(live)

    schema = {}
    for name, value in {
        "step": 0,
        METRIC_CPU_COUNT: 6,
        METRIC_CPU_USAGE_PERCENT: 30.0,
        METRIC_CPU_PARALLELIZATION_PERCENT: 50.0,
        METRIC_RAM_USAGE_PERCENT: 50.0,
        METRIC_RAM_USAGE_GB: 2.0,
        METRIC_RAM_TOTAL_GB: 4.0,
        f"{METRIC_DISK_USAGE_PERCENT}/main": 50.0,
        f"{METRIC_DISK_USAGE_GB}/main": 16.0,
        f"{METRIC_DISK_TOTAL_GB}/main": 32.0,
    }.items():
        dpath.new(schema, name, value)

    assert latest == S(schema)


@pytest.mark.timeout(2)
def test_monitor_system_timeseries(tmp_dir, mocker):
    mock_psutil_cpu(mocker)
    mock_psutil_ram(mocker)
    mock_psutil_disk(mocker)
    with Live(
        tmp_dir,
        save_dvc_exp=False,
        monitor_system=False,
    ) as live:
        live.cpu_monitor = CPUMonitor(interval=0.05, num_samples=4)

        # wait for the metrics to be logged.
        # METRIC_DISK_TOTAL_GB is the last metric to be logged.
        while len(dpath.search(live.summary, METRIC_DISK_TOTAL_GB)) == 0:
            time.sleep(0.001)

        live.next_step()

        timeseries, _ = parse_metrics(live)

    def timeserie_schema(name):
        return [{name: str, "timestamp": str, "step": str(0)}]

    # timeseries contains all the system metrics
    prefix = Path(tmp_dir) / "plots/metrics"
    assert timeseries == S(
        {
            str(prefix / f"{METRIC_CPU_USAGE_PERCENT}.tsv"): timeserie_schema(
                METRIC_CPU_USAGE_PERCENT.split("/")[-1]
            ),
            str(prefix / f"{METRIC_CPU_PARALLELIZATION_PERCENT}.tsv"): timeserie_schema(
                METRIC_CPU_PARALLELIZATION_PERCENT.split("/")[-1]
            ),
            str(prefix / f"{METRIC_RAM_USAGE_PERCENT}.tsv"): timeserie_schema(
                METRIC_RAM_USAGE_PERCENT.split("/")[-1]
            ),
            str(prefix / f"{METRIC_RAM_USAGE_GB}.tsv"): timeserie_schema(
                METRIC_RAM_USAGE_GB.split("/")[-1]
            ),
            str(prefix / f"{METRIC_DISK_USAGE_PERCENT}/main.tsv"): timeserie_schema(
                "main"
            ),
            str(prefix / f"{METRIC_DISK_USAGE_GB}/main.tsv"): timeserie_schema("main"),
        }
    )


def mock_pynvml(mocker, num_gpus=2, crash_index=None):
    mocker.patch("dvclive.monitor_system.GPU_AVAILABLE", return_value=num_gpus)
    mocker.patch("dvclive.monitor_system.nvmlDeviceGetCount", return_value=num_gpus)
    mocker.patch("dvclive.monitor_system.nvmlInit", return_value=None)
    mocker.patch("dvclive.monitor_system.nvmlShutdown", return_value=None)

    mocked_memory_info = mocker.MagicMock()
    mocked_memory_info.used = 3 * 1024**3
    mocked_memory_info.total = 5 * 1024**3

    mocked_utilization_rate = mocker.MagicMock()
    mocked_utilization_rate.memory = 5
    mocked_utilization_rate.gpu = 10

    mocking_dict = {
        "nvmlDeviceGetHandleByIndex": None,
        "nvmlDeviceGetMemoryInfo": mocked_memory_info,
        "nvmlDeviceGetUtilizationRates": mocked_utilization_rate,
    }

    for function_name, return_value in mocking_dict.items():
        mocker.patch(
            f"dvclive.monitor_system.{function_name}",
            return_value=return_value,
        )


def test_get_gpus_metrics_mocker(mocker, tmp_dir):
    mock_pynvml(mocker, num_gpus=2)
    with Live(
        tmp_dir,
        save_dvc_exp=False,
        monitor_system=False,
    ) as live:
        monitor = GPUMonitor()
        monitor(live)
        metrics = monitor._get_metrics()
        monitor.end()
    assert "system/gpu/usage (%)/0" in metrics
    assert "system/gpu/usage (%)/1" in metrics
    assert "system/vram/usage (%)/0" in metrics
    assert "system/vram/usage (%)/1" in metrics
    assert "system/vram/usage (GB)/0" in metrics
    assert "system/vram/usage (GB)/1" in metrics
    assert "system/vram/total (GB)/0" in metrics
    assert "system/vram/total (GB)/1" in metrics


def test_monitor_gpu_system(tmp_dir, mocker):
    mock_psutil_cpu(mocker)
    mock_psutil_ram(mocker)
    mock_psutil_disk(mocker)
    mock_pynvml(mocker, num_gpus=1)
    with Live(
        tmp_dir,
        save_dvc_exp=False,
        monitor_system=True,
    ) as live:
        time.sleep(5 + 1)  # allow the thread to finish
        live.next_step()
        time.sleep(5 + 1)  # allow the thread to finish
        timeseries, latest = parse_metrics(live)

    # metrics.json records CPU and RAM metrics if GPU detected
    assert "system" in latest
    assert "cpu" in latest["system"]
    assert "ram" in latest["system"]
    assert "disk" in latest["system"]

    # metrics.json file contains all the system metrics
    assert "gpu" in latest["system"]
    assert "count" in latest["system"]["gpu"]
    assert "usage (%)" in latest["system"]["gpu"]
    assert "0" in latest["system"]["gpu"]["usage (%)"]
    assert "vram" in latest["system"]
    assert "usage (%)" in latest["system"]["vram"]
    assert "0" in latest["system"]["vram"]["usage (%)"]
    assert "usage (GB)" in latest["system"]["vram"]
    assert "0" in latest["system"]["vram"]["usage (GB)"]
    assert "total (GB)" in latest["system"]["vram"]
    assert "0" in latest["system"]["vram"]["total (GB)"]

    # timeseries contains all the system metrics
    assert any(str(Path("system/gpu/usage (%)/0.tsv")) in key for key in timeseries)
    assert any(str(Path("system/vram/usage (%)/0.tsv")) in key for key in timeseries)
    assert any(str(Path("system/vram/usage (GB)/0.tsv")) in key for key in timeseries)
    assert all(len(timeseries[key]) == 2 for key in timeseries if "system" in key)

    # blacklisted timeseries
    assert all(str(Path("system/gpu/count.tsv")) not in key for key in timeseries)
    assert all(str(Path("system/vram/total (GB).tsv")) not in key for key in timeseries)
