import json
import logging
import os
import shutil
from collections import OrderedDict
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ruamel.yaml.representer import RepresenterError

from . import env
from .data import DATA_TYPES, PLOTS, Image, NumpyEncoder, Scalar
from .dvc import make_checkpoint
from .error import (
    ConfigMismatchError,
    InvalidDataTypeError,
    InvalidParameterTypeError,
    InvalidPlotTypeError,
)
from .report import make_report
from .serialize import dump_yaml, load_yaml
from .studio import post_to_studio
from .utils import env2bool, nested_update, open_file_in_browser

logging.basicConfig()
logger = logging.getLogger("dvclive")
logger.setLevel(os.getenv(env.DVCLIVE_LOGLEVEL, "INFO").upper())

ParamLike = Union[
    int, float, str, bool, List["ParamLike"], Dict[str, "ParamLike"]
]


class Live:
    DEFAULT_DIR = "dvclive"

    def __init__(
        self,
        path: Optional[str] = None,
        resume: bool = False,
        report: Optional[str] = "auto",
    ):
        self._path: Optional[str] = path
        self._resume: bool = resume or env2bool(env.DVCLIVE_RESUME)

        self.studio_url = os.getenv(env.STUDIO_REPO_URL, None)
        self.studio_token = os.getenv(env.STUDIO_TOKEN, None)
        self.rev = None

        if report == "auto":
            if self.studio_url and self.studio_token:
                report = "studio"
            elif env2bool("CI"):
                report = "md"
            else:
                report = "html"
        else:
            if report not in {None, "html", "md"}:
                raise ValueError(
                    "`report` can only be `None`, `auto`, `html` or `md`"
                )

        self.report_mode: Optional[str] = report
        self.report_path = ""

        self.init_from_env()

        if self._path is None:
            self._path = self.DEFAULT_DIR

        self._step: Optional[int] = None
        self._scalars: Dict[str, Any] = OrderedDict()
        self._images: Dict[str, Any] = OrderedDict()
        self._plots: Dict[str, Any] = OrderedDict()
        self._params: Dict[str, Any] = OrderedDict()

        self._init_paths()

        if self.report_mode in ("html", "md"):
            if not self.report_path:
                self.report_path = os.path.join(self.dir, f"report.{report}")
            out = Path(self.report_path).resolve()
            logger.info(f"Report path (if generated): {out}")

        if self._resume:
            self._read_params()
            self._step = self.read_step()
            if self._step != 0:
                self._step += 1
            logger.info(f"Resumed from step {self._step}")
        else:
            self._cleanup()

        self._latest_studio_step = self.get_step()

        if self.report_mode == "studio":
            from scmrepo.git import Git

            self.rev = Git().get_rev()

            if not post_to_studio(self, "start", logger):
                logger.warning(
                    "`post_to_studio` `start` event failed. "
                    "`studio` report cancelled."
                )
                self.report_mode = None

    def _cleanup(self):
        for data_type in DATA_TYPES:
            shutil.rmtree(
                Path(self.dir) / data_type.subfolder, ignore_errors=True
            )

        for f in (self.summary_path, self.report_path, self.params_path):
            if os.path.exists(f):
                os.remove(f)

    def _init_paths(self):
        os.makedirs(self.dir, exist_ok=True)

    def init_from_env(self) -> None:
        if os.getenv(env.DVCLIVE_PATH):

            if self.dir and self.dir != os.getenv(env.DVCLIVE_PATH):
                raise ConfigMismatchError(self)

            env_config = {
                "_path": os.getenv(env.DVCLIVE_PATH),
            }

            # Keeping backward compatibility with `live` section
            if not env2bool(env.DVCLIVE_HTML, "0"):
                env_config["report_mode"] = None
            else:
                env_config["report_mode"] = "html"
                path = str(env_config["_path"])
                self.report_path = path + "_dvc_plots/index.html"

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
    def params_path(self):
        return os.path.join(self.dir, "params.yaml")

    @property
    def exists(self):
        return os.path.isdir(self.dir)

    @property
    def summary_path(self):
        return str(self.dir) + ".json"

    def get_step(self) -> int:
        return self._step or 0

    def set_step(self, step: int) -> None:
        if self._step is None:
            self._step = 0
            for data in chain(
                self._scalars.values(),
                self._images.values(),
                self._plots.values(),
            ):
                data.dump(data.val, self._step)
            self.make_summary()

        if self.report_mode == "studio":
            if not post_to_studio(self, "data", logger):
                logger.warning(
                    "`post_to_studio` `data` event failed."
                    " Data will be resent on next call."
                )
            else:
                self._latest_studio_step = step
        else:
            self.make_report()

        self.make_checkpoint()

        self._step = step
        logger.debug(f"Step: {self._step}")

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
        logger.debug(f"Logged {name}: {val}")

    def log_image(self, name: str, val):
        if not Image.could_log(val):
            raise InvalidDataTypeError(name, type(val))

        if name in self._images:
            data = self._images[name]
        else:
            data = Image(name, self.dir)
            self._images[name] = data

        data.dump(val, self._step)
        logger.debug(f"Logged {name}: {val}")

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
        logger.debug(f"Logged {name}")

    def _read_params(self):
        if os.path.isfile(self.params_path):
            params = load_yaml(self.params_path)
            self._params.update(params)

    def _dump_params(self):
        try:
            dump_yaml(self.params_path, self._params)
        except RepresenterError as exc:
            raise InvalidParameterTypeError(exc.args) from exc

    def log_params(self, params: Dict[str, ParamLike]):
        """Saves the given set of parameters (dict) to yaml"""
        self._params.update(params)
        self._dump_params()
        logger.debug(f"Logged {params} parameters to {self.params_path}")

    def log_param(
        self,
        name: str,
        val: ParamLike,
    ):
        """Saves the given parameter value to yaml"""
        self.log_params({name: val})

    def make_summary(self):
        summary_data = {}
        if self._step is not None:
            summary_data["step"] = self.get_step()

        for data in self._scalars.values():
            summary_data = nested_update(summary_data, data.summary)

        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4, cls=NumpyEncoder)

    def make_report(self):
        if self.report_mode is not None:
            make_report(self)
            if self.report_mode == "html" and env2bool(env.DVCLIVE_OPEN):
                open_file_in_browser(self.report_path)

    def end(self):
        if self.report_mode == "studio":
            if not post_to_studio(self, "done", logger):
                logger.warning("`post_to_studio` `done` event failed.")

    def make_checkpoint(self):
        if env2bool(env.DVC_CHECKPOINT):
            make_checkpoint()

    def read_step(self):
        if Path(self.summary_path).exists():
            latest = self.read_latest()
            return latest.get("step", 0)
        return 0

    def read_latest(self):
        with open(self.summary_path, encoding="utf-8") as fobj:
            return json.load(fobj)
