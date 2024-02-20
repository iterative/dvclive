import abc
import logging
import os
from typing import Dict, Union, Optional, Tuple

import psutil
from statistics import mean
from threading import Event, Thread
from funcy import merge_with


try:
    from pynvml import (
        nvmlInit,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetUtilizationRates,
        nvmlShutdown,
    )

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger("dvclive")
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
        interval: float,
        num_samples: int,
        plot: bool = True,
    ):
        max_interval = 0.1
        if interval > max_interval:
            interval = max_interval
            logger.warning(
                f"System monitoring `interval` should be less than {max_interval} "
                f"seconds. Setting `interval` to {interval} seconds."
            )

        min_num_samples = 1
        max_num_samples = 30
        if min_num_samples < num_samples < max_num_samples:
            num_samples = max(min(num_samples, max_num_samples), min_num_samples)
            logger.warning(
                f"System monitoring `num_samples` should be between {min_num_samples} "
                f"and {max_num_samples}. Setting `num_samples` to {num_samples}."
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
        interval: float = 0.1,
        num_samples: int = 20,
        directories_to_monitor: Optional[Dict[str, str]] = None,
        plot: bool = True,
    ):
        """Monitor CPU resources and log them to DVC Live.

        Args:
            interval (float): interval in seconds between two measurements.
                Defaults to 0.5.
            num_samples (int): number of samples to average. Defaults to 10.
            directories_to_monitor (Optional[Dict[str, str]]): monitor disk usage
                statistics about the partition which contains the given paths. The
                statistics include total and used space in gygabytes and percent.
                This argument expect a dict where the key is the name that will be used
                in the metric's name and the value is the path to the directory to
                monitor. Defaults to {"main": "/"}.
            plot (bool): should the system metrics be saved as plots. Defaults to True.

        Raises:
            ValueError: if the arguments passed to the function don't have a
                supported type.
        """
        super().__init__(interval=interval, num_samples=num_samples, plot=plot)
        directories_to_monitor = (
            {"main": "/"} if directories_to_monitor is None else directories_to_monitor
        )
        self._disks_to_monitor = {}
        for disk_name, disk_path in directories_to_monitor.items():
            if disk_name != os.path.normpath(disk_name):
                raise ValueError(  # noqa: TRY003
                    "Keys for `directories_to_monitor` should be a valid name"
                    f", but got '{disk_name}'."
                )
            self._disks_to_monitor[disk_name] = disk_path

        self._warn_disk_doesnt_exist: Dict[str, bool] = {}

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
            METRIC_RAM_USAGE_GB: ram_info.used / GIGABYTES_DIVIDER,
            METRIC_RAM_TOTAL_GB: ram_info.total / GIGABYTES_DIVIDER,
        }
        for disk_name, disk_path in self._disks_to_monitor.items():
            try:
                disk_info = psutil.disk_usage(disk_path)
            except OSError:
                if self._warn_disk_doesnt_exist.get(disk_name, True):
                    logger.warning(
                        f"Couldn't find directory '{disk_path}', ignoring it."
                    )
                    self._warn_disk_doesnt_exist[disk_name] = False
                continue
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


class GPUMonitor(_SystemMonitor):
    _plot_blacklist_prefix = ("system/gpu/count", "system/vram/total (GB)")

    def __init__(
        self,
        interval: float = 0.5,
        num_samples: int = 10,
        plot: bool = True,
    ):
        """Monitor GPU resources and log them to DVC Live.

        Args:
            interval (float): interval in seconds between two measurements.
                Defaults to 0.5.
            num_samples (int): number of samples to average. Defaults to 10.
            plot (bool): should the system metrics be saved as plots. Defaults to True.

        Raises:
            InvalidDataTypeError: if the arguments passed to the function don't have a
                supported type.
        """
        super().__init__(interval=interval, num_samples=num_samples, plot=plot)

    def _get_metrics(self) -> Dict[str, Union[float, int]]:
        nvmlInit()
        num_gpus = nvmlDeviceGetCount()
        gpu_metrics = {
            "system/gpu/count": num_gpus,
        }

        for gpu_idx in range(num_gpus):
            gpu_handle = nvmlDeviceGetHandleByIndex(gpu_idx)
            memory_info = nvmlDeviceGetMemoryInfo(gpu_handle)
            usage_info = nvmlDeviceGetUtilizationRates(gpu_handle)

            gpu_metrics[f"system/gpu/usage (%)/{gpu_idx}"] = (
                100 * usage_info.memory / usage_info.gpu if usage_info.gpu else 0
            )
            gpu_metrics[f"system/vram/usage (%)/{gpu_idx}"] = (
                100 * memory_info.used / memory_info.total
            )
            gpu_metrics[f"system/vram/usage (GB)/{gpu_idx}"] = (
                memory_info.used / GIGABYTES_DIVIDER
            )
            gpu_metrics[f"system/vram/total (GB)/{gpu_idx}"] = (
                memory_info.total / GIGABYTES_DIVIDER
            )
        nvmlShutdown()
        return gpu_metrics
