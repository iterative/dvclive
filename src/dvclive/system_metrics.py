import logging
from typing import Dict, Union, Optional, List
from pathlib import Path

import psutil
from statistics import mean
from threading import Event, Thread
from funcy import merge_with

logger = logging.getLogger("dvclive")
MEGABYTES_DIVIDER = 1024.0**2
GIGABYTES_DIVIDER = 1024.0**3

MINIMUM_CPU_USAGE_TO_BE_ACTIVE = 20


class _SystemMetrics:
    _plot_blacklist_prefix = ()

    def __init__(
        self,
        interval: float = 0.5,
        nb_samples: int = 10,
        plot: bool = True,
    ):
        self._interval = interval  # seconds
        self._nb_samples = nb_samples
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

    def _get_metrics() -> Dict[str, Union[float, int]]:
        pass

    def end(self):
        self._shutdown_event.set()


class CPUMetrics(_SystemMetrics):
    _plot_blacklist_prefix = (
        "system/cpu/count",
        "system/ram/total (GB)",
        "system/disk/total (GB)",
    )

    def __init__(
        self,
        interval: float = 0.5,
        nb_samples: int = 10,
        directories_to_monitor: Optional[List[str]] = None,
        plot: bool = True,
    ):
        super().__init__(interval=interval, nb_samples=nb_samples, plot=plot)
        self._directories_to_monitor = (
            ["."] if directories_to_monitor is None else directories_to_monitor
        )

    def _get_metrics(self) -> Dict[str, Union[float, int]]:
        ram_info = psutil.virtual_memory()
        nb_cpus = psutil.cpu_count()
        cpus_percent = psutil.cpu_percent(percpu=True)
        result = {
            "system/cpu/usage (%)": mean(cpus_percent),
            "system/cpu/count": nb_cpus,
            "system/cpu/parallelization (%)": len(
                [
                    percent
                    for percent in cpus_percent
                    if percent >= MINIMUM_CPU_USAGE_TO_BE_ACTIVE
                ]
            )
            * 100
            / nb_cpus,
            "system/ram/usage (%)": ram_info.percent,
            "system/ram/usage (GB)": (ram_info.percent / 100)
            * (ram_info.total / GIGABYTES_DIVIDER),
            "system/ram/total (GB)": ram_info.total / GIGABYTES_DIVIDER,
        }
        for idx, directory in enumerate(self._directories_to_monitor):
            if not Path(directory).exists():
                continue
            disk_info = psutil.disk_usage(directory)
            result[f"system/disk/usage (%)/{idx}"] = disk_info.percent
            result[f"system/disk/usage (GB)/{idx}"] = disk_info.used / GIGABYTES_DIVIDER
            result[f"system/disk/total (GB)/{idx}"] = (
                disk_info.total / GIGABYTES_DIVIDER
            )
        return result
