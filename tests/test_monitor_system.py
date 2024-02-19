import time
from pathlib import Path
import pytest

import dpath
from voluptuous import Union
from pytest_voluptuous import S

from dvclive import Live
from dvclive.monitor_system import (
    CPUMonitor,
    METRIC_CPU_COUNT,
    METRIC_CPU_USAGE_PERCENT,
    METRIC_CPU_PARALLELIZATION_PERCENT,
    METRIC_RAM_USAGE_PERCENT,
    METRIC_RAM_USAGE_GB,
    METRIC_RAM_TOTAL_GB,
    METRIC_DISK_USAGE_PERCENT,
    METRIC_DISK_USAGE_GB,
    METRIC_DISK_TOTAL_GB,
)
from dvclive.utils import parse_metrics


def mock_psutil(mocker):
    mocker.patch(
        "dvclive.monitor_system.psutil.cpu_percent",
        return_value=[10, 20, 30, 40, 50, 60],
    )
    mocker.patch("dvclive.monitor_system.psutil.cpu_count", return_value=6)

    mocked_virtual_memory = mocker.MagicMock()
    mocked_virtual_memory.percent = 20
    mocked_virtual_memory.total = 4 * 1024**3

    mocked_disk_usage = mocker.MagicMock()
    mocked_disk_usage.used = 16 * 1024**3
    mocked_disk_usage.percent = 50
    mocked_disk_usage.total = 32 * 1024**3

    mocking_dict = {
        "virtual_memory": mocked_virtual_memory,
        "disk_usage": mocked_disk_usage,
    }
    for function_name, return_value in mocking_dict.items():
        mocker.patch(
            f"dvclive.monitor_system.psutil.{function_name}",
            return_value=return_value,
        )


def mock_psutil_with_oserror(mocker):
    mocker.patch(
        "dvclive.monitor_system.psutil.cpu_percent",
        return_value=[10, 20, 30, 40, 50, 60],
    )
    mocker.patch("dvclive.monitor_system.psutil.cpu_count", return_value=6)

    mocked_virtual_memory = mocker.MagicMock()
    mocked_virtual_memory.percent = 20
    mocked_virtual_memory.total = 4 * 1024**3

    mocked_disk_usage = mocker.MagicMock()
    mocked_disk_usage.used = 16 * 1024**3
    mocked_disk_usage.percent = 50
    mocked_disk_usage.total = 32 * 1024**3

    mocker.patch(
        "dvclive.monitor_system.psutil.virtual_memory",
        return_value=mocked_virtual_memory,
    )
    mocker.patch(
        "dvclive.monitor_system.psutil.disk_usage",
        side_effect=[
            mocked_disk_usage,
            OSError,
            mocked_disk_usage,
            mocked_disk_usage,
        ],
    )


def test_monitor_system_is_false(tmp_dir, mocker):
    mock_psutil(mocker)

    cpu_monitor_mock = mocker.patch("dvclive.live.CPUMonitor")
    with Live(
        tmp_dir,
        save_dvc_exp=False,
        monitor_system=False,
    ) as live:
        assert live.cpu_monitor is None

    cpu_monitor_mock.assert_not_called()


def test_monitor_system_is_true(tmp_dir, mocker):
    mock_psutil(mocker)
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
    mock_psutil_with_oserror(mocker)

    with Live(
        tmp_dir,
        save_dvc_exp=False,
        monitor_system=False,
    ) as live:
        non_existent_disk = "/non-existent"
        monitor = CPUMonitor(
            folders_to_monitor={"main": "/", "non-existent": non_existent_disk}
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
    mock_psutil(mocker)
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
    for name in [
        "step",
        METRIC_CPU_COUNT,
        METRIC_CPU_USAGE_PERCENT,
        METRIC_CPU_PARALLELIZATION_PERCENT,
        METRIC_RAM_USAGE_PERCENT,
        METRIC_RAM_USAGE_GB,
        METRIC_RAM_TOTAL_GB,
        f"{METRIC_DISK_USAGE_PERCENT}/main",
        f"{METRIC_DISK_USAGE_GB}/main",
        f"{METRIC_DISK_TOTAL_GB}/main",
    ]:
        dpath.new(schema, name, Union(int, float))

    assert latest == S(schema)


@pytest.mark.timeout(2)
def test_monitor_system_timeseries(tmp_dir, mocker):
    mock_psutil(mocker)
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
