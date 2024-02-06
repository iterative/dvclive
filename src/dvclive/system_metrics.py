import logging
from typing import Dict, Union

import psutil
from statistics import mean
from threading import Event, Thread

from .utils import append_dict

logger = logging.getLogger("dvclive")
MEGABYTES_DIVIDER = 1024.0**2

MINIMUM_CPU_USAGE_TO_BE_ACTIVE = 30


class CPUMetricsCallback:
    def __init__(
        self,
        duration: Union[int, float] = 30,
        interval: Union[int, float] = 2.0,
        plot: bool = True,
    ):
        self.duration = duration
        self.interval = interval
        self.plot = plot
        self._no_plot_metrics = ["system/cpu/count"]

    def __call__(self, live):
        self._live = live
        self._shutdown_event = Event()
        Thread(
            target=self.monitoring_loop,
        ).start()

    def monitoring_loop(self):
        while not self._shutdown_event.is_set():
            self._cpu_metrics = {}
            for _ in range(int(self.duration // self.interval)):
                last_cpu_metrics = {}
                try:
                    last_cpu_metrics = get_cpus_metrics()
                except psutil.Error:
                    logger.exception("Failed to monitor CPU metrics:")
                self._cpu_metrics = append_dict(self._cpu_metrics, last_cpu_metrics)
                self._shutdown_event.wait(self.interval)
                if self._shutdown_event.is_set():
                    break
            for metric_name, metric_values in self._cpu_metrics.items():
                self._live.log_metric(
                    metric_name,
                    mean(metric_values),
                    timestamp=True,
                    plot=self.plot
                    if metric_name not in self._no_plot_metrics
                    else False,
                )

    def end(self):
        self._shutdown_event.set()


def get_cpus_metrics() -> Dict[str, Union[float, int]]:
    ram_info = psutil.virtual_memory()
    io_info = psutil.disk_io_counters()
    nb_cpus = psutil.cpu_count()
    cpus_percent = psutil.cpu_percent(percpu=True)
    return {
        "system/cpu/usage_avg_percent": mean(cpus_percent),
        "system/cpu/usage_max_percent": max(cpus_percent),
        "system/cpu/count": nb_cpus,
        "system/cpu/parallelism_percent": len(
            [
                percent
                for percent in cpus_percent
                if percent >= MINIMUM_CPU_USAGE_TO_BE_ACTIVE
            ]
        )
        / nb_cpus,
        "system/cpu/ram_usage_percent": ram_info.percent,
        "system/io/read_speed_MB": io_info.read_bytes
        / (io_info.read_time * MEGABYTES_DIVIDER),
        "system/io/write_speed_MB": io_info.write_bytes
        / (io_info.write_time * MEGABYTES_DIVIDER),
    }
