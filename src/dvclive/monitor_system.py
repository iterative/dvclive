import abc
import logging
import os
from typing import Dict, Union, Optional, Tuple
from pathlib import Path

import psutil
from statistics import mean
from threading import Event, Thread
from funcy import merge_with


logger = logging.getLogger("dvclive")
MEGABYTES_DIVIDER = 1024.0**2
GIGABYTES_DIVIDER = 1024.0**3

MINIMUM_CPU_USAGE_TO_BE_ACTIVE = 20

METRIC_CPU_COUNT = "system/cpu/count"
METRIC_CPU_USAGE_PERCENT = "system/cpu/usage (%)"
METRIC_CPU_PARALLELIZATION_PERCENT = "system/cpu/parallelization (%)"

METRIC_RAM_USAGE_PERCENT = "system/ram/usage (%)"
METRIC_RAM_USAGE_GB = "system/ram/usage (GB)"
METRIC_RAM_TOTAL_GB = "system/ram/total (GB)"

METRIC_DISK_USAGE_PERCENT = "system/disk/usage (%)"
METRIC_DISK_USAGE_GB = "system/disk/usage (GB)"
METRIC_DISK_TOTAL_GB = "system/disk/total (GB)"


class _SystemMonitor(abc.ABC):
    """
    Monitor system resources and log them to DVC Live.
    Use a separate thread to call a `_get_metrics` function at fix interval and
    aggregate the results of this sampling using the average.
    """

    _plot_blacklist_prefix: Tuple = ()

    def __init__(
        self,
        interval: float = 0.5,
        num_samples: int = 10,
        plot: bool = True,
    ):
        if not isinstance(interval, (int, float)):
            raise TypeError(  # noqa: TRY003
                "System monitoring `interval` should be an `int` or a `float`, but got "
                f"'{type(interval)}'"
            )
        if not isinstance(num_samples, int):
            raise TypeError(  # noqa: TRY003
                "System monitoring `num_samples` should be an `int`, but got "
                f"'{type(num_samples)}'"
            )
        if not isinstance(plot, bool):
            raise TypeError(  # noqa: TRY003
                f"System monitoring `plot` should be a `bool`, but got '{type(plot)}'"
            )

        self._interval = interval  # seconds
        self._nb_samples = num_samples
        self._plot = plot
        self._warn_user = True

    def __call__(self, live):
        self._live = live
        self._shutdown_event = Event()
        Thread(
            target=self._monitoring_loop,
        ).start()

    def _monitoring_loop(self):
        while not self._shutdown_event.is_set():
            self._metrics = {}
            for _ in range(self._nb_samples):
                last_metrics = {}
                try:
                    last_metrics = self._get_metrics()
                except psutil.Error:
                    if self._warn_user:
                        logger.exception("Failed to monitor CPU metrics")
                        self._warn_user = False

                self._metrics = merge_with(sum, self._metrics, last_metrics)
                self._shutdown_event.wait(self._interval)
                if self._shutdown_event.is_set():
                    break
            for name, values in self._metrics.items():
                blacklisted = any(
                    name.startswith(prefix) for prefix in self._plot_blacklist_prefix
                )
                self._live.log_metric(
                    name,
                    values / self._nb_samples,
                    timestamp=True,
                    plot=None if blacklisted else self._plot,
                )

    @abc.abstractmethod
    def _get_metrics(self) -> Dict[str, Union[float, int]]:
        pass

    def end(self):
        self._shutdown_event.set()


class CPUMonitor(_SystemMonitor):
    _plot_blacklist_prefix: Tuple = (
        METRIC_CPU_COUNT,
        METRIC_RAM_TOTAL_GB,
        "system/disk/total (GB)",
    )

    def __init__(
        self,
        interval: float = 0.5,
        num_samples: int = 10,
        disks_to_monitor: Optional[Dict[str, str]] = None,
        plot: bool = True,
    ):
        """Monitor CPU resources and log them to DVC Live.

        Args:
            interval (float): interval in seconds between two measurements.
                Defaults to 0.5.
            num_samples (int): number of samples to average. Defaults to 10.
            disks_to_monitor (Optional[Dict[str, str]]): paths to the disks or
                partitions to monitor disk usage statistics. The key is the name that
                will be displayed in the metric name. The value is the path to the disk
                or partition. Defaults to {"main": "/"}.
            plot (bool): should the system metrics be saved as plots. Defaults to True.

        Raises:
            TypeError: if the arguments passed to the function don't have a
                supported type.
        """
        super().__init__(interval=interval, num_samples=num_samples, plot=plot)
        disks_to_monitor = (
            {"main": "/"} if disks_to_monitor is None else disks_to_monitor
        )
        for disk_name, disk_path in disks_to_monitor.items():
            if not isinstance(disk_name, str):
                raise TypeError(  # noqa: TRY003
                    "Keys for `partitions_to_monitor` should be a `str`"
                    f", but got '{type(disk_name)}'"
                )
            if not isinstance(disk_path, str):
                raise TypeError(  # noqa: TRY003
                    "Value for `partitions_to_monitor` should be a `str`"
                    f", but got '{type(disk_path)}'"
                )
            if disk_name != os.path.normpath(disk_name):
                raise ValueError(  # noqa: TRY003
                    "Keys for `partitions_to_monitor` should be a valid name"
                    f", but got '{disk_name}'"
                )
        self._disks_to_monitor = disks_to_monitor

    def _get_metrics(self) -> Dict[str, Union[float, int]]:
        ram_info = psutil.virtual_memory()
        nb_cpus = psutil.cpu_count()
        cpus_percent = psutil.cpu_percent(percpu=True)
        result = {
            METRIC_CPU_COUNT: nb_cpus,
            METRIC_CPU_USAGE_PERCENT: mean(cpus_percent),
            METRIC_CPU_PARALLELIZATION_PERCENT: len(
                [
                    percent
                    for percent in cpus_percent
                    if percent >= MINIMUM_CPU_USAGE_TO_BE_ACTIVE
                ]
            )
            * 100
            / nb_cpus,
            METRIC_RAM_USAGE_PERCENT: ram_info.percent,
            METRIC_RAM_USAGE_GB: (ram_info.percent / 100)
            * (ram_info.total / GIGABYTES_DIVIDER),
            METRIC_RAM_TOTAL_GB: ram_info.total / GIGABYTES_DIVIDER,
        }
        for disk_name, disk_path in self._disks_to_monitor.items():
            if not Path(disk_path).exists():
                continue
            disk_info = psutil.disk_usage(disk_path)
            disk_metrics = {
                f"{METRIC_DISK_USAGE_PERCENT}/{disk_name}": disk_info.percent,
                f"{METRIC_DISK_USAGE_GB}/{disk_name}": disk_info.used
                / GIGABYTES_DIVIDER,
                f"{METRIC_DISK_TOTAL_GB}/{disk_name}": disk_info.total
                / GIGABYTES_DIVIDER,
            }
            disk_metrics = {k.rstrip("/"): v for k, v in disk_metrics.items()}
            result.update(disk_metrics)
        return result
