import json
import logging
import os
import shutil
import time
from collections import OrderedDict
from typing import Dict

from dvc import env

from .error import DvcLiveError
from .serialize import update_tsv, write_json

logger = logging.getLogger(__name__)


class MetricLogger:
    DEFAULT_DIR = "dvclive"

    def __init__(
        self,
        path: str,
        resume: bool = False,
        step: int = 0,
        summary=True,
        html=True,
    ):
        self._path = path
        self._step = step
        self._html = html
        self._summary = summary
        self._metrics: Dict[str, float] = OrderedDict()

        if resume and self.exists:
            if step == 0:
                self._step = self.read_step() + 1
            else:
                self._step = step
        else:
            shutil.rmtree(self.dir, ignore_errors=True)
            try:
                os.makedirs(self.dir, exist_ok=True)
            except Exception as ex:
                raise DvcLiveError(
                    "dvc-live cannot create log dir - '{}'".format(ex),
                )

    @staticmethod
    def from_env():
        if env.DVCLIVE_PATH in os.environ:
            directory = os.environ[env.DVCLIVE_PATH]
            dump_latest = bool(int(os.environ.get(env.DVCLIVE_SUMMARY, "0")))
            report = bool(int(os.environ.get(env.DVCLIVE_REPORT, "0")))
            return MetricLogger(directory, summary=dump_latest, html=report)
        return None

    @property
    def dir(self):
        return self._path

    @property
    def exists(self):
        return os.path.isdir(self.dir)

    @property
    def history_path(self):
        if not self.exists:
            os.mkdir(self.dir)
        return self.dir

    @property
    def summary_path(self):
        return self.dir + ".json"

    def next_step(self):
        if self._summary:
            metrics = OrderedDict({"step": self._step})
            metrics.update(self._metrics)
            write_json(metrics, self.summary_path)

        if self._html:
            from dvc.api.live import summary

            summary(self.dir)

        self._metrics.clear()

        self._step += 1

    def log(self, name: str, val: float, step: int = None):
        if name in self._metrics.keys():
            logger.info(
                f"Found {name} in metrics dir, assuming new epoch started"
            )
            self.next_step()

        if not isinstance(val, (int, float)):
            raise DvcLiveError(
                "Metrics '{}' has not supported type {}".format(
                    name, type(val)
                )
            )

        if step:
            self._step = step

        metric_history_path = os.path.join(self.history_path, name + ".tsv")
        self._metrics[name] = val

        ts = int(time.time() * 1000)
        d = OrderedDict([("timestamp", ts), ("step", self._step), (name, val)])
        update_tsv(d, metric_history_path)

    def read_step(self):
        if self.exists:
            latest = self.read_latest()
            return int(latest["step"])
        return 0

    def read_latest(self):
        with open(self.summary_path, "r") as fobj:
            return json.load(fobj)
