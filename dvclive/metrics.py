import json
import logging
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Union

from .dvc import get_signal_file_path, make_checkpoint
from .error import DvcLiveError
from .serialize import update_tsv, write_json
from .utils import nested_set

logger = logging.getLogger(__name__)


class MetricLogger:
    DEFAULT_DIR = "dvclive"

    def __init__(
        self,
        path: str = "dvclive",
        resume: bool = False,
        step: int = 0,
        summary=True,
        html=True,
        checkpoint=False,
    ):
        self._path: str = path
        self._step: int = step
        self._html: bool = html
        self._summary = summary
        self._metrics: Dict[str, float] = OrderedDict()
        self._checkpoint: bool = checkpoint

        if resume and self.exists:
            if step == 0:
                self._step = self.read_step()
                if self._step != 0:
                    self._step += 1
            else:
                self._step = step
        else:
            self._cleanup()
            try:
                os.makedirs(self.dir, exist_ok=True)
            except Exception as exception:
                raise DvcLiveError(
                    "dvc-live cannot create log dir - '{}'".format(self.dir),
                ) from exception

    def _cleanup(self):

        for dvclive_file in Path(self.dir).rglob("*.tsv"):
            dvclive_file.unlink()

        if os.path.exists(self.summary_path):
            os.remove(self.summary_path)

        if os.path.exists(self.html_path):
            os.remove(self.html_path)

    @staticmethod
    def from_env():
        from . import env

        if env.DVCLIVE_PATH in os.environ:
            directory = os.environ[env.DVCLIVE_PATH]
            env_config = {
                "summary": bool(int(os.environ.get(env.DVCLIVE_SUMMARY, "0"))),
                "html": bool(int(os.environ.get(env.DVCLIVE_HTML, "0"))),
                "checkpoint": bool(
                    int(os.environ.get(env.DVC_CHECKPOINT, "0"))
                ),
                "resume": bool(int(os.environ.get(env.DVCLIVE_RESUME, "0"))),
            }
            return MetricLogger(directory, **env_config)
        return None

    def matches_env_setup(self):
        from . import env

        if env.DVCLIVE_PATH in os.environ:
            env_dir = os.environ[env.DVCLIVE_PATH]
            return self.dir == env_dir

        return True

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

    @property
    def html_path(self):
        return self.dir + ".html"

    def next_step(self):
        if self._summary:
            metrics = OrderedDict({"step": self._step})
            metrics.update(self._metrics)
            write_json(metrics, self.summary_path)

        if self._html:
            signal_file_path = get_signal_file_path()
            if signal_file_path:
                if not os.path.exists(signal_file_path):
                    with open(signal_file_path, "w"):
                        pass

        self._metrics.clear()

        self._step += 1

        if self._checkpoint:
            make_checkpoint()

    def log(self, name: str, val: Union[int, float], step: int = None):
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

        if step is not None:
            self._step = step

        metric_history_path = os.path.join(self.history_path, name + ".tsv")
        os.makedirs(os.path.dirname(metric_history_path), exist_ok=True)

        nested_set(
            self._metrics, os.path.normpath(name).split(os.path.sep), val,
        )

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
