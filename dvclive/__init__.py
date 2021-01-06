import json
import logging
import os
import shutil
import time
from collections import OrderedDict
from typing import Dict

from dvclive import env
from dvclive.error import DvcLiveError, InitializationError
from dvclive.serialize import update_tsv, write_json

logger = logging.getLogger(__name__)

__version__ = "0.0.1"
_metric_logger = None


class DvcLive:
    DEFAULT_DIR = "dvclive"

    def __init__(
        self,
        directory: str = None,
        is_continue: bool = False,
        step: int = 0,
        report=True,
        dump_latest=True,
    ):
        self._dir = directory
        self._step = step
        self._report = report
        self._dump_latest = dump_latest
        self._metrics: Dict[str, float] = {}

        if is_continue and self.exists:
            if step == 0:
                self._step = self.read_step() + 1
            else:
                self._step = step
        elif self.dir:
            shutil.rmtree(self.dir, ignore_errors=True)
            try:
                os.makedirs(self.dir, exist_ok=True)
            except Exception as ex:
                raise DvcLiveError(
                    "dvc-live cannot create log dir - {}".format(ex)
                )

    @staticmethod
    def from_env():
        if env.DVCLIVE_PATH in os.environ:
            directory = os.environ[env.DVCLIVE_PATH]
            dump_latest = bool(int(os.environ.get(env.DVCLIVE_SUMMARY, "0")))
            report = bool(int(os.environ.get(env.DVCLIVE_REPORT, "0")))
            return DvcLive(directory, dump_latest=dump_latest, report=report)
        return None

    @property
    def dir(self):
        return self._dir

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
        if self._dump_latest:
            write_json(
                {"step": self._step, **self._metrics}, self.summary_path
            )

        if self._report:
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
        with open(self.summary_path, "r") as fd:
            return json.load(fd)


def init(
    directory: str = None,
    is_continue: bool = False,
    step: int = 0,
    report=True,
    dump_latest=True,
) -> DvcLive:
    global _metric_logger  # pylint: disable=global-statement
    _metric_logger = DvcLive(
        directory=directory or DvcLive.DEFAULT_DIR,
        is_continue=is_continue,
        step=step,
        report=report,
        dump_latest=dump_latest,
    )
    return _metric_logger


def log(name: str, val: float, step: int = None):
    global _metric_logger  # pylint: disable=global-statement
    if not _metric_logger:
        _metric_logger = DvcLive.from_env()
    if not _metric_logger:
        raise InitializationError()

    _metric_logger.log(name=name, val=val, step=step)


def next_step():
    global _metric_logger  # pylint: disable=global-statement
    if not _metric_logger:
        raise InitializationError()
    _metric_logger.next_step()
