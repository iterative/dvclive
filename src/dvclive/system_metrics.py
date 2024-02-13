import logging
from typing import Dict, Union

import psutil
from statistics import mean
from threading import Event, Thread
from funcy import merge_with

logger = logging.getLogger("dvclive")
MEGABYTES_DIVIDER = 1024.0**2
GIGABYTES_DIVIDER = 1024.0**3

MINIMUM_CPU_USAGE_TO_BE_ACTIVE = 20


class CPUMetrics:
    def __init__(
        self,
        interval: float = 0.5,
        nb_samples: int = 10,
        plot: bool = True,
    ):
        self._interval = interval  # seconds
        self._nb_samples = nb_samples
        self._plot = plot
        self._no_plot_metrics = ["system/cpu/count", "system/ram/total (GB)"]
        self._warn_user = True

    def __call__(self, live):
        self._live = live
        self._shutdown_event = Event()
        Thread(
            target=self._monitoring_loop,
        ).start()

    def _monitoring_loop(self):
        while not self._shutdown_event.is_set():
            self._cpu_metrics = {}
            for _ in range(self._nb_samples):
                last_cpu_metrics = {}
                try:
                    last_cpu_metrics = _get_cpus_metrics()
                except psutil.Error:
                    if self._warn_user:
                        logger.exception("Failed to monitor CPU metrics")
                        self._warn_user = False

                self._cpu_metrics = merge_with(sum, self._cpu_metrics, last_cpu_metrics)
                self._shutdown_event.wait(self._interval)
                if self._shutdown_event.is_set():
                    break
            for metric_name, metric_values in self._cpu_metrics.items():
                self._live.log_metric(
                    metric_name,
                    metric_values / self._nb_samples,
                    timestamp=True,
                    plot=self._plot
                    if metric_name not in self._no_plot_metrics
                    else False,
                )

    def end(self):
        self._shutdown_event.set()


def _get_cpus_metrics() -> Dict[str, Union[float, int]]:
    ram_info = psutil.virtual_memory()
    io_info = psutil.disk_io_counters()
    nb_cpus = psutil.cpu_count()
    cpus_percent = psutil.cpu_percent(percpu=True)
    return {
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
        "system/io/read speed (MB)": io_info.read_bytes
        / (io_info.read_time * MEGABYTES_DIVIDER),
        "system/io/write speed (MB)": io_info.write_bytes
        / (io_info.write_time * MEGABYTES_DIVIDER),
    }
