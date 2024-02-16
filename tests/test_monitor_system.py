import time
from pathlib import Path

from dvclive import Live
from dvclive.monitor_system import CPUMonitor
from dvclive.utils import parse_metrics

from voluptuous import Union
from pytest_voluptuous import S


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
        monitor = CPUMonitor(disks_to_monitor={"main": "/", "home": "/"})
        monitor(live)
        metrics = monitor._get_metrics()
        monitor.end()

    assert metrics == S(
        {
            "system/cpu/usage (%)": Union(int, float),
            "system/cpu/count": int,
            "system/cpu/parallelization (%)": Union(int, float),
            "system/ram/usage (%)": Union(int, float),
            "system/ram/usage (GB)": Union(int, float),
            "system/ram/total (GB)": Union(int, float),
            "system/disk/usage (%)/main": Union(int, float),
            "system/disk/usage (%)/home": Union(int, float),
            "system/disk/usage (GB)/main": Union(int, float),
            "system/disk/usage (GB)/home": Union(int, float),
            "system/disk/total (GB)/main": Union(int, float),
            "system/disk/total (GB)/home": Union(int, float),
        }
    )


def test_monitor_system_is_false(tmp_dir, mocker):
    mock_psutil(mocker)
    with Live(
        tmp_dir,
        save_dvc_exp=False,
        monitor_system=False,
    ) as live:
        assert live._cpu_monitor is None


def test_monitor_system_is_true(tmp_dir, mocker):
    mock_psutil(mocker)
    with Live(
        tmp_dir,
        save_dvc_exp=False,
        monitor_system=True,
    ) as live:
        cpu_monitor = live._cpu_monitor
        assert isinstance(live._cpu_monitor, CPUMonitor)
        # start is called but end not called
        assert cpu_monitor._shutdown_event._flag is False

    # end was called
    assert cpu_monitor._shutdown_event._flag is True


def test_ignore_non_existent_directories(tmp_dir, mocker):
    mock_psutil(mocker)
    with Live(
        tmp_dir,
        save_dvc_exp=False,
        monitor_system=False,
    ) as live:
        non_existent_disk = "/non-existent"
        monitor = CPUMonitor(
            disks_to_monitor={"main": "/", "non-existent": non_existent_disk}
        )
        monitor(live)
        metrics = monitor._get_metrics()
        monitor.end()

    assert not Path(non_existent_disk).exists()

    assert "system/disk/usage (%)/non-existent" not in metrics
    assert "system/disk/usage (GB)/non-existent" not in metrics
    assert "system/disk/total (GB)/non-existent" not in metrics


def test_monitor_system_metrics(tmp_dir, mocker):
    mock_psutil(mocker)
    interval = 0.05
    num_samples = 4
    with Live(
        tmp_dir,
        save_dvc_exp=False,
        monitor_system=False,
    ) as live:
        live.cpu_monitor = CPUMonitor(interval=0.05, num_samples=4)
        time.sleep(interval * num_samples + interval)  # log metrics once
        live.next_step()
        time.sleep(interval * num_samples + interval)  # log metrics twice
        live.make_summary()

        _, latest = parse_metrics(live)

    # metrics.json file contains all the system metrics
    assert latest == S(
        {
            "step": Union(int, float),
            "system": {
                "cpu": {
                    "usage (%)": Union(int, float),
                    "count": Union(int, float),
                    "parallelization (%)": Union(int, float),
                },
                "ram": {
                    "usage (%)": Union(int, float),
                    "usage (GB)": Union(int, float),
                    "total (GB)": Union(int, float),
                },
                "disk": {
                    "usage (%)": {"main": Union(int, float)},
                    "usage (GB)": {"main": Union(int, float)},
                    "total (GB)": {"main": Union(int, float)},
                },
            },
        }
    )


def test_monitor_system_timeseries(tmp_dir, mocker):
    mock_psutil(mocker)
    interval = 0.05
    num_samples = 4
    with Live(
        tmp_dir,
        save_dvc_exp=False,
        monitor_system=False,
    ) as live:
        live.cpu_monitor = CPUMonitor(interval=0.05, num_samples=4)
        time.sleep(interval * num_samples + interval)  # log metrics once
        live.next_step()
        time.sleep(interval * num_samples + interval)  # log metrics twice
        live.make_summary()

        timeseries, _ = parse_metrics(live)

    def timeserie_metric_schema(name):
        return [{name: str, "timestamp": str, "step": str} for _ in range(2)]

    # timeseries contains all the system metrics
    assert timeseries == S(
        {
            str(
                Path(tmp_dir) / "plots/metrics" / "system/cpu/usage (%).tsv"
            ): timeserie_metric_schema("usage (%)"),
            str(
                Path(tmp_dir) / "plots/metrics" / "system/cpu/parallelization (%).tsv"
            ): timeserie_metric_schema("parallelization (%)"),
            str(
                Path(tmp_dir) / "plots/metrics" / "system/ram/usage (%).tsv"
            ): timeserie_metric_schema("usage (%)"),
            str(
                Path(tmp_dir) / "plots/metrics" / "system/ram/usage (GB).tsv"
            ): timeserie_metric_schema("usage (GB)"),
            str(
                Path(tmp_dir) / "plots/metrics" / "system/disk/usage (%)/main.tsv"
            ): timeserie_metric_schema("main"),
            str(
                Path(tmp_dir) / "plots/metrics" / "system/disk/usage (GB)/main.tsv"
            ): timeserie_metric_schema("main"),
        }
    )
