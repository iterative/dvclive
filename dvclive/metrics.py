import json
import logging
import shutil
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Union

from .data import DATA_TYPES
from .dvc import make_checkpoint, make_html
from .error import InvalidDataTypeError

logger = logging.getLogger(__name__)


class MetricLogger:
    DEFAULT_DIR = "dvclive"

    def __init__(
        self,
        path: str = "dvclive",
        resume: bool = False,
        summary=True,
        html=True,
        checkpoint=False,
    ):
        self._path: str = path
        self._step: int = 0
        self._data: Dict[str, Any] = OrderedDict()

        self._summary: bool = summary

        self._html: bool = html
        self._checkpoint: bool = checkpoint

        if resume and self.exists:
            self._step = self.read_step()
            if self._step != 0:
                self._step += 1
        else:
            self._cleanup()
            self._init_paths()

    def _cleanup(self):

        for data_type in DATA_TYPES:
            subdir = Path(self.dir) / data_type.subdir
            for suffix in data_type.suffixes:
                for data_file in subdir.rglob(f"*{suffix}"):
                    data_file.unlink()

        if os.path.exists(self.summary_path):
            os.remove(self.summary_path)

        if os.path.exists(self.html_path):
            shutil.rmtree(self.html_path, ignore_errors=True)

    def _init_paths(self):
        os.makedirs(self.dir, exist_ok=True)
        for data_type in DATA_TYPES:
            os.makedirs(
                os.path.join(self.dir, data_type.subdir), exist_ok=True
            )

        if self._summary:
            self.make_summary()
        if self._html:
            os.makedirs(self.html_path, exist_ok=True)

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
    def summary_path(self):
        return os.path.join(self.dir, "summary.json")

    @property
    def html_path(self):
        return os.path.join(self.dir, "html")

    def get_step(self) -> int:
        return self._step

    def set_step(self, step: int) -> None:
        if self._html:
            make_html()

        if self._checkpoint:
            make_checkpoint()

        self._step = step

    def next_step(self):
        self.set_step(self.get_step() + 1)

    def log(self, name: str, val: Union[int, float]):

        data = None
        if name in self._data:
            data = self._data[name]
        else:
            for data_type in DATA_TYPES:
                if data_type.could_log(val):
                    data = data_type(name, self.dir)
                    self._data[name] = data
        if data is None:
            raise InvalidDataTypeError(name, type(val))
        data.dump(val, self._step)

        if self._summary:
            self.make_summary()

    def make_summary(self):
        summary_data = {"step": self.get_step()}

        for data in self._data.values():
            summary_data.update(data.summary)

        with open(self.summary_path, "w") as f:
            json.dump(summary_data, f, indent=4)

    def read_step(self):
        if self.exists:
            latest = self.read_latest()
            return int(latest["step"])
        return 0

    def read_latest(self):
        with open(self.summary_path, "r") as fobj:
            return json.load(fobj)
