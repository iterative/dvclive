import json
import logging
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Union

from .dvc import get_signal_file_path, make_checkpoint
from .error import ConfigMismatchError, InvalidMetricTypeError
from .serialize import update_tsv, write_json
from .utils import nested_get, nested_set

logger = logging.getLogger(__name__)


class MetricLogger:
    DEFAULT_DIR = "dvclive"

    def __init__(
        self,
        path: str = "dvclive",
        resume: bool = False,
        summary: bool = True,
        html: bool = True,
        checkpoint: bool = False,
        from_env: bool = True
    ):
        self._path: str = path
        self._summary = summary
        self._html: bool = html
        self._checkpoint: bool = checkpoint

        self._step: int = 0
        self._metrics: Dict[str, Any] = OrderedDict()

        if from_env:
            self.update_from_env()

        if resume and self.exists:
            self._step = self.read_step()
            if self._step != 0:
                self._step += 1

        else:
            self._cleanup()
            os.makedirs(self.dir, exist_ok=True)

    def _cleanup(self):

        for dvclive_file in Path(self.dir).rglob("*.tsv"):
            dvclive_file.unlink()

        if os.path.exists(self.summary_path):
            os.remove(self.summary_path)

        if os.path.exists(self.html_path):
            os.remove(self.html_path)

    def update_from_env(self) -> None:
        from . import env

        if env.DVCLIVE_PATH in os.environ:

            if self.dir != os.environ[env.DVCLIVE_PATH]:
                raise ConfigMismatchError(self)

            env_config = {
                "summary": bool(int(os.environ.get(env.DVCLIVE_SUMMARY, "0"))),
                "html": bool(int(os.environ.get(env.DVCLIVE_HTML, "0"))),
                "checkpoint": bool(
                    int(os.environ.get(env.DVC_CHECKPOINT, "0"))
                ),
                "resume": bool(int(os.environ.get(env.DVCLIVE_RESUME, "0"))),
            }
            for k, v in env_config.items():
                if getattr(self, k) != v:
                    logger.info(f"Overriding {k} with value provided by DVC: {v}")
                    setattr(self, k, v)

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
        return self.dir + "_dvc_plots/index.html"

    def get_step(self) -> int:
        return self._step

    def set_step(self, step: int):
        if self._metrics:
            self.next_step()
        self._step = step

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

    def log(self, name: str, val: Union[int, float]):
        splitted_name = os.path.normpath(name).split(os.path.sep)
        if nested_get(self._metrics, splitted_name) is not None:
            logger.info(
                f"Found {name} in metrics dir, assuming new epoch started"
            )
            self.next_step()

        if not isinstance(val, (int, float)):
            raise InvalidMetricTypeError(name, type(val))

        metric_history_path = os.path.join(self.history_path, name + ".tsv")
        os.makedirs(os.path.dirname(metric_history_path), exist_ok=True)

        nested_set(
            self._metrics,
            splitted_name,
            val,
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
