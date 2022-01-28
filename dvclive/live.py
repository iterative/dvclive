import json
import logging
import os
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .data import DATA_TYPES
from .dvc import make_checkpoint, make_html
from .error import ConfigMismatchError, InvalidDataTypeError
from .utils import nested_update

logger = logging.getLogger(__name__)


class Live:
    DEFAULT_DIR = "dvclive"

    def __init__(
        self, path: Optional[str] = None, resume: bool = False, **kwargs
    ):
        if "summary" in kwargs:
            logger.warning(
                "`summary` is being deprecated in 0.5.0 "
                "and will be removed in 0.6.0. Making the "
                "summary generation no longer optional",
            )
        summary = kwargs.get("summary", True)
        if resume and not summary:
            raise ValueError("`resume` can't be used without `summary`")
        self._path: Optional[str] = path
        self._resume: bool = resume
        self._summary: bool = summary
        self._html: bool = True
        self._checkpoint: bool = False

        self.init_from_env()

        if self._path is None:
            self._path = self.DEFAULT_DIR

        self._step: Optional[int] = None
        self._data: Dict[str, Any] = OrderedDict()

        if self._resume:
            self._step = self.read_step()
            if self._step != 0:
                self._step += 1
        else:
            self._cleanup()
            self._init_paths()

    def _cleanup(self):

        for data_type in DATA_TYPES:
            for suffix in data_type.suffixes:
                for data_file in Path(self.dir).rglob(f"*{suffix}"):
                    data_file.unlink()

        if os.path.exists(self.summary_path):
            os.remove(self.summary_path)

        if os.path.exists(self.html_path):
            shutil.rmtree(Path(self.html_path).parent, ignore_errors=True)

    def _init_paths(self):
        if self._step is not None:
            os.makedirs(self.dir, exist_ok=True)
            if self._html:
                os.makedirs(Path(self.html_path).parent, exist_ok=True)
        if self._summary:
            os.makedirs(Path(self.summary_path).parent, exist_ok=True)
            self.make_summary()

    def init_from_env(self) -> None:
        from . import env

        if env.DVCLIVE_PATH in os.environ:

            if self.dir and self.dir != os.environ[env.DVCLIVE_PATH]:
                raise ConfigMismatchError(self)

            env_config = {
                "_path": os.environ.get(env.DVCLIVE_PATH),
                "_summary": bool(
                    int(os.environ.get(env.DVCLIVE_SUMMARY, "0"))
                ),
                "_html": bool(int(os.environ.get(env.DVCLIVE_HTML, "0"))),
                "_checkpoint": bool(
                    int(os.environ.get(env.DVC_CHECKPOINT, "0"))
                ),
                "_resume": bool(int(os.environ.get(env.DVCLIVE_RESUME, "0"))),
            }
            for k, v in env_config.items():
                if getattr(self, k) != v:
                    logger.info(
                        f"Overriding {k} with value provided by DVC: {v}"
                    )
                    setattr(self, k, v)

    @property
    def dir(self):
        return self._path

    @property
    def exists(self):
        return os.path.isdir(self.dir)

    @property
    def summary_path(self):
        return str(self.dir) + ".json"

    @property
    def html_path(self):
        return str(self.dir) + "_dvc_plots/index.html"

    def get_step(self) -> int:
        return self._step or 0

    def set_step(self, step: int) -> None:
        if self._step is None:
            self._step = 0
            self._init_paths()
            for data in self._data.values():
                data.dump(data.val, self._step)
            if self._summary:
                self.make_summary()

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
        summary_data = {}
        if self._step is not None:
            summary_data["step"] = self.get_step()

        for data in self._data.values():
            summary_data = nested_update(summary_data, data.summary)

        with open(self.summary_path, "w") as f:
            json.dump(summary_data, f, indent=4)

    def read_step(self):
        if Path(self.summary_path).exists():
            latest = self.read_latest()
            return latest.get("step", 0)
        return 0

    def read_latest(self):
        with open(self.summary_path, "r") as fobj:
            return json.load(fobj)
