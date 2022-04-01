import json
import logging
import os
import shutil
from collections import OrderedDict
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .data import DATA_TYPES, PLOTS, Image, Scalar
from .dvc import make_checkpoint
from .error import (
    ConfigMismatchError,
    InvalidDataTypeError,
    InvalidPlotTypeError,
)
from .report import html_report
from .utils import nested_update, open_file_in_browser

logger = logging.getLogger(__name__)


class Live:
    DEFAULT_DIR = "dvclive"

    def __init__(
        self,
        path: Optional[str] = None,
        resume: bool = False,
        report: Optional[str] = "html",
        auto_open: bool = True,
    ):

        self._path: Optional[str] = path
        self._resume: bool = resume
        self._report: str = report
        self._checkpoint: bool = False
        self._auto_open: bool = auto_open

        self.init_from_env()

        if self._path is None:
            self._path = self.DEFAULT_DIR

        if self._report is not None:
            out = Path(self.html_path).resolve()
            print(f"Report will be saved at {out}")

        self._step: Optional[int] = None
        self._scalars: Dict[str, Any] = OrderedDict()
        self._images: Dict[str, Any] = OrderedDict()
        self._plots: Dict[str, Any] = OrderedDict()

        if self._resume:
            self._step = self.read_step()
            if self._step != 0:
                self._step += 1
        else:
            self._cleanup()
            self._init_paths()

    def _cleanup(self):
        for data_type in DATA_TYPES:
            shutil.rmtree(
                Path(self.dir) / data_type.subfolder, ignore_errors=True
            )

        for f in {self.summary_path, self.html_path}:
            if os.path.exists(f):
                os.remove(f)

    def _init_paths(self):
        os.makedirs(self.dir, exist_ok=True)

    def init_from_env(self) -> None:
        from . import env

        if env.DVCLIVE_PATH in os.environ:

            if self.dir and self.dir != os.environ[env.DVCLIVE_PATH]:
                raise ConfigMismatchError(self)

            env_config = {
                "_path": os.environ.get(env.DVCLIVE_PATH),
                "_checkpoint": bool(
                    int(os.environ.get(env.DVC_CHECKPOINT, "0"))
                ),
                "_resume": bool(int(os.environ.get(env.DVCLIVE_RESUME, "0"))),
            }
            if not bool(int(os.environ.get(env.DVCLIVE_HTML, "0"))):
                env_config["_report"] = None

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
        return os.path.join(self.dir, "report.html")

    def get_step(self) -> int:
        return self._step or 0

    def set_step(self, step: int) -> None:
        if self._step is None:
            self._step = 0
            self._init_paths()
            for data in chain(
                self._scalars.values(),
                self._images.values(),
                self._plots.values(),
            ):
                data.dump(data.val, self._step)
            self.make_summary()

        self.make_report()

        if self._checkpoint:
            make_checkpoint()

        self._step = step

    def next_step(self):
        self.set_step(self.get_step() + 1)

    def log(self, name: str, val: Union[int, float]):
        if not Scalar.could_log(val):
            raise InvalidDataTypeError(name, type(val))

        if name in self._scalars:
            data = self._scalars[name]
        else:
            data = Scalar(name, self.dir)
            self._scalars[name] = data

        data.dump(val, self._step)

        self.make_summary()

    def log_image(self, name: str, val):
        if not Image.could_log(val):
            raise InvalidDataTypeError(name, type(val))

        if name in self._images:
            data = self._images[name]
        else:
            data = Image(name, self.dir)
            self._images[name] = data

        data.dump(val, self._step)

    def log_plot(self, name, labels, predictions, **kwargs):
        val = (labels, predictions)

        if name in self._plots:
            data = self._plots[name]
        elif name in PLOTS and PLOTS[name].could_log(val):
            data = PLOTS[name](name, self.dir)
            self._plots[name] = data
        else:
            raise InvalidPlotTypeError(name)

        data.dump(val, self._step, **kwargs)

    def make_summary(self):
        summary_data = {}
        if self._step is not None:
            summary_data["step"] = self.get_step()

        for data in self._scalars.values():
            summary_data = nested_update(summary_data, data.summary)

        with open(self.summary_path, "w") as f:
            json.dump(summary_data, f, indent=4)

    def make_report(self):
        if self._report == "html":
            html_report(self.dir, self.summary_path, self.html_path)
            if self._auto_open:
                open_file_in_browser(self.html_path)
                self._auto_open = False

    def read_step(self):
        if Path(self.summary_path).exists():
            latest = self.read_latest()
            return latest.get("step", 0)
        return 0

    def read_latest(self):
        with open(self.summary_path, "r") as fobj:
            return json.load(fobj)
