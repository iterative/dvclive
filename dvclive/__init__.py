import json
import logging
import os
import shutil
import time
from collections import OrderedDict

from dvclive.error import DvcLiveError, InitializationError
from dvclive.serialize import update_tsv, write_json

logger = logging.getLogger(__name__)

__version__ = "0.0.1"
DEFAULT_DIR = "dvclive"
DVCLIVE_PATH = "DVCLIVE_PATH"
DVCLIVE_SUMMARY = "DVCLIVE_SUMMARY"


class DvcLive:
    def __init__(self):
        self._dir = None
        self._step = None
        self._metrics = {}
        self._report = True
        self._dump_latest = True

    def init(
        self,
        directory: str = DEFAULT_DIR,
        is_continue: bool = False,
        step: int = 0,
        report=True,
        dump_latest=True,
    ):
        self._dir = directory
        self._step = step
        self._report = report
        self._dump_latest = dump_latest

        if is_continue and self.exists:
            if step == 0:
                self._step = self.read_step() + 1
            else:
                self._step = step
        else:
            shutil.rmtree(directory, ignore_errors=True)
            try:
                os.makedirs(self._dir, exist_ok=True)
            except Exception as ex:
                raise DvcLiveError(
                    "dvc-live cannot create log dir - {}".format(ex)
                )

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
            from dvc.api.dvclive import summary

            summary(self.dir)

        self._metrics.clear()

        self._step += 1

    def _from_env(self):
        directory = os.environ[DVCLIVE_PATH]
        dump_latest = os.environ.get(DVCLIVE_SUMMARY, "true").lower() == "true"
        self.init(directory, dump_latest=dump_latest, report=False)

    def log(self, name: str, val: float, step: int = None):
        if not self.dir:
            if DVCLIVE_PATH in os.environ:
                self._from_env()
            else:
                raise InitializationError()

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


dvclive = DvcLive()


def init(
    directory: str,
    is_continue: bool = False,
    step: int = 0,
    report=True,
    dump_latest=True,
):
    dvclive.init(directory, is_continue, step, report, dump_latest)
