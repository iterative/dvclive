import time
from pathlib import Path

from dvclive import Live
from dvclive.monitor_system import MonitorCPU, MonitorGPU
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
        monitor = MonitorCPU(directories_to_monitor=["/", "/"])
        monitor(live)
        metrics = monitor._get_metrics()
        monitor.end()

    assert "system/cpu/usage (%)" in metrics
    assert "system/cpu/count" in metrics
    assert "system/cpu/parallelization (%)" in metrics
    assert "system/ram/usage (%)" in metrics
    assert "system/ram/usage (GB)" in metrics
    assert "system/ram/total (GB)" in metrics
    assert "system/disk/usage (%)/0" in metrics
    assert "system/disk/usage (%)/1" in metrics
    assert "system/disk/usage (GB)/0" in metrics
    assert "system/disk/usage (GB)/1" in metrics
    assert "system/disk/total (GB)/0" in metrics
    assert "system/disk/total (GB)/1" in metrics


def test_ignore_missing_directories(tmp_dir, mocker):
    mock_psutil(mocker)
    with Live(
        tmp_dir,
        save_dvc_exp=False,
        monitor_system=False,
    ) as live:
        missing_directories = "______"
        monitor = MonitorCPU(directories_to_monitor=["/", missing_directories])
        monitor(live)
        metrics = monitor._get_metrics()
        monitor.end()

    assert not Path(missing_directories).exists()

    assert "system/disk/usage (%)/1" not in metrics
    assert "system/disk/usage (GB)/1" not in metrics
    assert "system/disk/total (GB)/1" not in metrics


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
    assert "0" in latest["system"]["disk"]["usage (%)"]
    assert "usage (GB)" in latest["system"]["disk"]
    assert "total (GB)" in latest["system"]["disk"]

    # timeseries contains all the system metrics
    assert any(str(Path("system/cpu/usage (%).tsv")) in key for key in timeseries)
    assert any(
        str(Path("system/cpu/parallelization (%).tsv")) in key for key in timeseries
    )
    assert any(str(Path("system/ram/usage (%).tsv")) in key for key in timeseries)
    assert any(str(Path("system/ram/usage (GB).tsv")) in key for key in timeseries)
    assert any(str(Path("system/disk/usage (%)/0.tsv")) in key for key in timeseries)
    assert any(str(Path("system/disk/usage (GB)/0.tsv")) in key for key in timeseries)
    assert all(len(timeseries[key]) == 2 for key in timeseries if "system" in key)

    # blacklisted timeseries
    assert all(str(Path("system/cpu/count.tsv")) not in key for key in timeseries)
    assert all(str(Path("system/ram/total (GB).tsv")) not in key for key in timeseries)
    assert all(
        str(Path("system/disk/total (GB)/0.tsv")) not in key for key in timeseries
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
        monitor = MonitorGPU()
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
    mock_psutil(mocker)
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
