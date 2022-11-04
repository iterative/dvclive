import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ruamel.yaml.representer import RepresenterError

from . import env
from .dvc import make_checkpoint
from .error import (
    InvalidDataTypeError,
    InvalidParameterTypeError,
    InvalidPlotTypeError,
)
from .plots import PLOT_TYPES, SKLEARN_PLOTS, Image, Metric, NumpyEncoder
from .report import make_report
from .serialize import dump_json, dump_yaml, load_yaml
from .studio import post_to_studio
from .utils import (
    env2bool,
    matplotlib_installed,
    nested_update,
    open_file_in_browser,
)

logging.basicConfig()
logger = logging.getLogger("dvclive")
logger.setLevel(os.getenv(env.DVCLIVE_LOGLEVEL, "INFO").upper())

ParamLike = Union[
    int, float, str, bool, List["ParamLike"], Dict[str, "ParamLike"]
]


class Live:
    def __init__(
        self,
        dir: str = "dvclive",  # noqa pylint: disable=redefined-builtin
        resume: bool = False,
        report: Optional[str] = "auto",
    ):
        self._dir: str = dir
        self._resume: bool = resume or env2bool(env.DVCLIVE_RESUME)

        self.studio_url = os.getenv(env.STUDIO_REPO_URL, None)
        self.studio_token = os.getenv(env.STUDIO_TOKEN, None)
        self.rev = None

        if report == "auto":
            if self.studio_url and self.studio_token:
                report = "studio"
            elif env2bool("CI") and matplotlib_installed():
                report = "md"
            else:
                report = "html"
        else:
            if report not in {None, "html", "md"}:
                raise ValueError(
                    "`report` can only be `None`, `auto`, `html` or `md`"
                )

        self.report_mode: Optional[str] = report
        self.report_file = ""

        self.summary: Dict[str, Any] = {}
        self._step: Optional[int] = None
        self._metrics: Dict[str, Any] = {}
        self._images: Dict[str, Any] = {}
        self._plots: Dict[str, Any] = {}
        self._params: Dict[str, Any] = {}

        self._init_paths()

        if self.report_mode in ("html", "md"):
            if not self.report_file:
                self.report_file = os.path.join(self.dir, f"report.{report}")
            out = Path(self.report_file).resolve()
            logger.info(f"Report file (if generated): {out}")

        if self._resume:
            self._read_params()
            self._step = self.read_step()
            if self._step != 0:
                self._step += 1
            logger.info(f"Resumed from step {self._step}")
        else:
            self._cleanup()

        self._latest_studio_step = self.step if resume else -1
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
        for plot_type in PLOT_TYPES:
            shutil.rmtree(
                Path(self.plots_dir) / plot_type.subfolder, ignore_errors=True
            )

        for f in (self.metrics_file, self.report_file, self.params_file):
            if os.path.exists(f):
                os.remove(f)

    def _init_paths(self):
        os.makedirs(self.dir, exist_ok=True)

    @property
    def dir(self):
        return self._dir

    @property
    def params_file(self):
        return os.path.join(self.dir, "params.yaml")

    @property
    def metrics_file(self):
        return os.path.join(self.dir, "metrics.json")

    @property
    def plots_dir(self):
        return os.path.join(self.dir, "plots")

    @property
    def step(self) -> int:
        return self._step or 0

    @step.setter
    def step(self, value: int) -> None:
        self._step = value
        logger.debug(f"Step: {self.step}")

    def next_step(self):
        if self._step is None:
            self._step = 0

        self.make_summary()
        self.make_report()
        self.make_checkpoint()
        self.step += 1

    def log_metric(
        self, name: str, val: Union[int, float], timestamp: bool = False
    ):
        if not Metric.could_log(val):
            raise InvalidDataTypeError(name, type(val))

        if name in self._metrics:
            data = self._metrics[name]
        else:
            data = Metric(name, self.plots_dir)
            self._metrics[name] = data

        data.step = self.step
        data.dump(val, timestamp=timestamp)

        self.summary = nested_update(self.summary, data.to_summary(val))
        logger.debug(f"Logged {name}: {val}")

    def log_image(self, name: str, val):
        if not Image.could_log(val):
            raise InvalidDataTypeError(name, type(val))

        if name in self._images:
            data = self._images[name]
        else:
            data = Image(name, self.plots_dir)
            self._images[name] = data

        data.step = self.step
        data.dump(val)
        logger.debug(f"Logged {name}: {val}")

    def log_sklearn_plot(self, kind, labels, predictions, name=None, **kwargs):
        val = (labels, predictions)

        name = name or kind
        if name in self._plots:
            data = self._plots[name]
        elif kind in SKLEARN_PLOTS and SKLEARN_PLOTS[kind].could_log(val):
            data = SKLEARN_PLOTS[kind](name, self.plots_dir)
            self._plots[name] = data
        else:
            raise InvalidPlotTypeError(name)

        data.step = self.step
        data.dump(val, **kwargs)
        logger.debug(f"Logged {name}")

    def _read_params(self):
        if os.path.isfile(self.params_file):
            params = load_yaml(self.params_file)
            self._params.update(params)

    def _dump_params(self):
        try:
            dump_yaml(self._params, self.params_file)
        except RepresenterError as exc:
            raise InvalidParameterTypeError(exc.args) from exc

    def log_params(self, params: Dict[str, ParamLike]):
        """Saves the given set of parameters (dict) to yaml"""
        self._params.update(params)
        self._dump_params()
        logger.debug(f"Logged {params} parameters to {self.params_file}")

    def log_param(
        self,
        name: str,
        val: ParamLike,
    ):
        """Saves the given parameter value to yaml"""
        self.log_params({name: val})

    def make_summary(self):
        if self._step is not None:
            self.summary["step"] = self.step
        dump_json(self.summary, self.metrics_file, cls=NumpyEncoder)

    def make_report(self):
        if self.report_mode == "studio":
            if not post_to_studio(self, "data", logger):
                logger.warning(
                    "`post_to_studio` `data` event failed."
                    " Data will be resent on next call."
                )
            else:
                self._latest_studio_step = self.step
        elif self.report_mode is not None:
            make_report(self)
            if self.report_mode == "html" and env2bool(env.DVCLIVE_OPEN):
                open_file_in_browser(self.report_file)

    def end(self):
        self.make_summary()
        if self.report_mode == "studio":
            if not post_to_studio(self, "done", logger):
                logger.warning("`post_to_studio` `done` event failed.")
        else:
            self.make_report()

    def make_checkpoint(self):
        if env2bool(env.DVC_CHECKPOINT):
            make_checkpoint()

    def read_step(self):
        if Path(self.metrics_file).exists():
            latest = self.read_latest()
            return latest.get("step", 0)
        return 0

    def read_latest(self):
        with open(self.metrics_file, encoding="utf-8") as fobj:
            return json.load(fobj)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()
