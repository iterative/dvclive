import time
from pathlib import Path

from dvclive import Live
from dvclive.monitor_system import CPUMonitor
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


def test_get_cpus_metrics_mocker(tmp_dir, mocker):
    mock_psutil(mocker)
    with Live(
        tmp_dir,
        save_dvc_exp=False,
        monitor_system=False,
    ) as live:
        monitor = CPUMonitor(disks_to_monitor=["/", "/home"])
        monitor(live)
        metrics = monitor._get_metrics()
        monitor.end()

    assert "system/cpu/usage (%)" in metrics
    assert "system/cpu/count" in metrics
    assert "system/cpu/parallelization (%)" in metrics
    assert "system/ram/usage (%)" in metrics
    assert "system/ram/usage (GB)" in metrics
    assert "system/ram/total (GB)" in metrics
    assert "system/disk/usage (%)" in metrics
    assert "system/disk/usage (%)/home" in metrics
    assert "system/disk/usage (GB)" in metrics
    assert "system/disk/usage (GB)/home" in metrics
    assert "system/disk/total (GB)" in metrics
    assert "system/disk/total (GB)/home" in metrics


def test_ignore_non_existent_directories(tmp_dir, mocker):
    mock_psutil(mocker)
    with Live(
        tmp_dir,
        save_dvc_exp=False,
        monitor_system=False,
    ) as live:
        non_existent_directories = "/non-existent"
        monitor = CPUMonitor(disks_to_monitor=["/", non_existent_directories])
        monitor(live)
        metrics = monitor._get_metrics()
        monitor.end()

    assert not Path(non_existent_directories).exists()

    assert "system/disk/usage (%)/non-existent" not in metrics
    assert "system/disk/usage (GB)/non-existent" not in metrics
    assert "system/disk/total (GB)/non-existent" not in metrics


def test_monitor_system(tmp_dir, mocker):
    mock_psutil(mocker)
    with Live(
        tmp_dir,
        save_dvc_exp=False,
        monitor_system=True,
    ) as live:
        time.sleep(5 + 1)  # allow the thread to finish
        live.next_step()
        time.sleep(5 + 1)  # allow the thread to finish
        timeseries, latest = parse_metrics(live)

    # metrics.json file contains all the system metrics
    assert "system" in latest
    assert "cpu" in latest["system"]
    assert "usage (%)" in latest["system"]["cpu"]
    assert "count" in latest["system"]["cpu"]
    assert "parallelization (%)" in latest["system"]["cpu"]
    assert "ram" in latest["system"]
    assert "usage (%)" in latest["system"]["ram"]
    assert "usage (GB)" in latest["system"]["ram"]
    assert "total (GB)" in latest["system"]["ram"]
    assert "disk" in latest["system"]
    assert "usage (%)" in latest["system"]["disk"]
    assert "usage (GB)" in latest["system"]["disk"]
    assert "total (GB)" in latest["system"]["disk"]

    # timeseries contains all the system metrics
    assert any(str(Path("system/cpu/usage (%).tsv")) in key for key in timeseries)
    assert any(
        str(Path("system/cpu/parallelization (%).tsv")) in key for key in timeseries
    )
    assert any(str(Path("system/ram/usage (%).tsv")) in key for key in timeseries)
    assert any(str(Path("system/ram/usage (GB).tsv")) in key for key in timeseries)
    assert any(str(Path("system/disk/usage (%).tsv")) in key for key in timeseries)
    assert any(str(Path("system/disk/usage (GB).tsv")) in key for key in timeseries)
    assert all(len(timeseries[key]) == 2 for key in timeseries if "system" in key)

    # blacklisted timeseries
    assert all(str(Path("system/cpu/count.tsv")) not in key for key in timeseries)
    assert all(str(Path("system/ram/total (GB).tsv")) not in key for key in timeseries)
    assert all(str(Path("system/disk/total (GB).tsv")) not in key for key in timeseries)
