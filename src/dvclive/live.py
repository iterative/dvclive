import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from dvc_studio_client.post_live_metrics import post_live_metrics
from funcy import set_in
from pathspec import PathSpec
from ruamel.yaml.representer import RepresenterError

from . import env
from .dvc import (
    get_dvc_repo,
    get_random_exp_name,
    make_checkpoint,
    make_dvcyaml,
    mark_dvclive_only_ended,
    mark_dvclive_only_started,
)
from .error import (
    InvalidDataTypeError,
    InvalidParameterTypeError,
    InvalidPlotTypeError,
    InvalidReportModeError,
)
from .plots import PLOT_TYPES, SKLEARN_PLOTS, CustomPlot, Image, Metric, NumpyEncoder
from .report import BLANK_NOTEBOOK_REPORT, make_report
from .serialize import dump_json, dump_yaml, load_yaml
from .studio import get_dvc_studio_config, get_studio_updates
from .utils import (
    StrPath,
    clean_and_copy_into,
    env2bool,
    inside_notebook,
    matplotlib_installed,
    open_file_in_browser,
)

logger = logging.getLogger("dvclive")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(os.getenv(env.DVCLIVE_LOGLEVEL, "INFO").upper())

ParamLike = Union[int, float, str, bool, List["ParamLike"], Dict[str, "ParamLike"]]


class Live:
    def __init__(
        self,
        dir: str = "dvclive",  # noqa: A002
        resume: bool = False,
        report: Optional[str] = "auto",
        save_dvc_exp: bool = False,
        dvcyaml: bool = True,
        exp_message: Optional[str] = None,
    ):
        self.summary: Dict[str, Any] = {}

        self._dir: str = dir
        self._resume: bool = resume or env2bool(env.DVCLIVE_RESUME)
        self._save_dvc_exp: bool = save_dvc_exp
        self._step: Optional[int] = None
        self._metrics: Dict[str, Any] = {}
        self._images: Dict[str, Any] = {}
        self._params: Dict[str, Any] = {}
        self._plots: Dict[str, Any] = {}
        self._artifacts: Dict[str, Dict] = {}
        self._inside_with = False
        self._dvcyaml = dvcyaml

        os.makedirs(self.dir, exist_ok=True)

        self._report_mode: Optional[str] = report
        self._report_notebook = None
        self._init_report()

        if self._resume:
            self._init_resume()
        else:
            self._init_cleanup()

        self._baseline_rev: Optional[str] = None
        self._exp_name: Optional[str] = None
        self._exp_message: Optional[str] = exp_message
        self._experiment_rev: Optional[str] = None
        self._inside_dvc_exp: bool = False
        self._dvc_repo = None
        self._include_untracked: List[str] = []
        self._init_dvc()

        self._latest_studio_step = self.step if resume else -1
        self._studio_events_to_skip: Set[str] = set()
        self._dvc_studio_config: Dict[str, Any] = {}
        self._init_studio()

    def _init_resume(self):
        self._read_params()
        self._step = self.read_step()
        if self._step != 0:
            logger.info(f"Resuming from step {self._step}")
            self._step += 1
        logger.debug(f"{self._step=}")

    def _init_cleanup(self):
        for plot_type in PLOT_TYPES:
            shutil.rmtree(
                Path(self.plots_dir) / plot_type.subfolder, ignore_errors=True
            )

        for f in (
            self.metrics_file,
            self.report_file,
            self.params_file,
        ):
            if f and os.path.exists(f):
                os.remove(f)

    def _init_dvc(self):
        from dvc.scm import NoSCM

        if os.getenv(env.DVC_EXP_BASELINE_REV, None):
            # `dvc exp` execution
            self._baseline_rev = os.getenv(env.DVC_EXP_BASELINE_REV, "")
            self._exp_name = os.getenv(env.DVC_EXP_NAME, "")
            self._inside_dvc_exp = True
            if self._save_dvc_exp:
                logger.warning(
                    "Ignoring `_save_dvc_exp` because `dvc exp run` is running"
                )
                self._save_dvc_exp = False

        self._dvc_repo = get_dvc_repo()

        dvc_logger = logging.getLogger("dvc")
        dvc_logger.setLevel(os.getenv(env.DVCLIVE_LOGLEVEL, "WARNING").upper())

        if (self._dvc_repo is None) or isinstance(self._dvc_repo.scm, NoSCM):
            if self._save_dvc_exp:
                logger.warning(
                    "Can't save experiment without a Git Repo."
                    "\nCreate a Git repo (`git init`) and commit (`git commit`)."
                )
                self._save_dvc_exp = False
            return
        if self._dvc_repo.scm.no_commits:
            if self._save_dvc_exp:
                logger.warning(
                    "Can't save experiment to an empty Git Repo."
                    "\nAdd a commit (`git commit`) to save experiments."
                )
                self._save_dvc_exp = False
            return

        if self._inside_dvc_exp:
            return

        self._baseline_rev = self._dvc_repo.scm.get_rev()
        if self._save_dvc_exp:
            self._exp_name = get_random_exp_name(self._dvc_repo.scm, self._baseline_rev)
            mark_dvclive_only_started()
            self._include_untracked.append(self.dir)

    def _init_studio(self):
        self._dvc_studio_config = get_dvc_studio_config(self)
        if not self._dvc_studio_config:
            logger.debug("Skipping `studio` report.")
            self._studio_events_to_skip.add("start")
            self._studio_events_to_skip.add("data")
            self._studio_events_to_skip.add("done")
        elif self._inside_dvc_exp:
            logger.debug("Skipping `studio` report `start` and `done` events.")
            self._studio_events_to_skip.add("start")
            self._studio_events_to_skip.add("done")
        elif self._dvc_repo is None:
            logger.warning(
                "Can't connect to Studio without a DVC Repo."
                "\nYou can create a DVC Repo by calling `dvc init`."
            )
            self._studio_events_to_skip.add("start")
            self._studio_events_to_skip.add("data")
            self._studio_events_to_skip.add("done")
        elif not self._save_dvc_exp:
            logger.warning(
                "Can't connect to Studio without creating a DVC experiment."
                "\nIf you have a DVC Pipeline, run it with `dvc exp run`."
                "\nIf you are using DVCLive alone, use `save_dvc_exp=True`."
            )
            self._studio_events_to_skip.add("start")
            self._studio_events_to_skip.add("data")
            self._studio_events_to_skip.add("done")
        else:
            response = post_live_metrics(
                "start",
                self._baseline_rev,
                self._exp_name,
                "dvclive",
                dvc_studio_config=self._dvc_studio_config,
                message=self._exp_message,
            )
            if not response:
                logger.debug(
                    "`studio` report `start` event failed. "
                    "`studio` report cancelled."
                )
                self._studio_events_to_skip.add("start")
                self._studio_events_to_skip.add("data")
                self._studio_events_to_skip.add("done")

    def _init_report(self):
        if self._report_mode == "auto":
            if env2bool("CI") and matplotlib_installed():
                self._report_mode = "md"
            else:
                self._report_mode = "html"
        elif self._report_mode == "notebook":
            if inside_notebook():
                from IPython.display import Markdown, display

                self._report_mode = "notebook"
                self._report_notebook = display(
                    Markdown(BLANK_NOTEBOOK_REPORT), display_id=True
                )
            else:
                self._report_mode = "html"
        elif self._report_mode not in {None, "html", "notebook", "md"}:
            raise InvalidReportModeError(self._report_mode)
        logger.debug(f"{self._report_mode=}")

    @property
    def dir(self) -> str:  # noqa: A003
        return self._dir

    @property
    def params_file(self) -> str:
        return os.path.join(self.dir, "params.yaml")

    @property
    def metrics_file(self) -> str:
        return os.path.join(self.dir, "metrics.json")

    @property
    def dvc_file(self) -> str:
        return os.path.join(self.dir, "dvc.yaml")

    @property
    def plots_dir(self) -> str:
        return os.path.join(self.dir, "plots")

    @property
    def artifacts_dir(self) -> str:
        return os.path.join(self.dir, "artifacts")

    @property
    def report_file(self) -> Optional[str]:
        if self._report_mode in ("html", "md"):
            suffix = self._report_mode
            return os.path.join(self.dir, f"report.{suffix}")
        return None

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

        if self._dvcyaml:
            self.make_dvcyaml()

        self.make_report()
        self.make_checkpoint()
        self.step += 1

    def log_metric(
        self,
        name: str,
        val: Union[int, float],
        timestamp: bool = False,
        plot: bool = True,
    ):
        if not Metric.could_log(val):
            raise InvalidDataTypeError(name, type(val))

        if name in self._metrics:
            metric = self._metrics[name]
        else:
            metric = Metric(name, self.plots_dir)
            self._metrics[name] = metric

        metric.step = self.step
        if plot:
            metric.dump(val, timestamp=timestamp)

        self.summary = set_in(self.summary, metric.summary_keys, val)
        logger.debug(f"Logged {name}: {val}")

    def log_image(self, name: str, val):
        if not Image.could_log(val):
            raise InvalidDataTypeError(name, type(val))

        if isinstance(val, (str, Path)):
            from PIL import Image as ImagePIL

            val = ImagePIL.open(val)

        if name in self._images:
            image = self._images[name]
        else:
            image = Image(name, self.plots_dir)
            self._images[name] = image

        image.step = self.step
        image.dump(val)
        logger.debug(f"Logged {name}: {val}")

    def log_plot(
        self,
        name: str,
        datapoints: List[Dict],
        x: str,
        y: str,
        template: Optional[str] = None,
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
    ):
        if not CustomPlot.could_log(datapoints):
            raise InvalidDataTypeError(name, type(datapoints))

        if name in self._plots:
            plot = self._plots[name]
        else:
            plot = CustomPlot(
                name,
                self.plots_dir,
                x=x,
                y=y,
                template=template,
                title=title,
                x_label=x_label,
                y_label=y_label,
            )
            self._plots[name] = plot

        plot.step = self.step
        plot.dump(datapoints)
        logger.debug(f"Logged {name}")

    def log_sklearn_plot(self, kind, labels, predictions, name=None, **kwargs):
        val = (labels, predictions)

        plot_config = {
            k: v
            for k, v in kwargs.items()
            if k in ("title", "x_label", "y_label", "normalized")
        }
        name = name or kind
        if name in self._plots:
            plot = self._plots[name]
        elif kind in SKLEARN_PLOTS and SKLEARN_PLOTS[kind].could_log(val):
            plot = SKLEARN_PLOTS[kind](name, self.plots_dir, **plot_config)
            self._plots[plot.name] = plot
        else:
            raise InvalidPlotTypeError(name)

        sklearn_kwargs = {
            k: v for k, v in kwargs.items() if k not in plot_config or k != "normalized"
        }
        plot.step = self.step
        plot.dump(val, **sklearn_kwargs)
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

    def log_param(self, name: str, val: ParamLike):
        """Saves the given parameter value to yaml"""
        self.log_params({name: val})

    def log_artifact(
        self,
        path: StrPath,
        type: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        desc: Optional[str] = None,  # noqa: ARG002
        labels: Optional[List[str]] = None,  # noqa: ARG002
        meta: Optional[Dict[str, Any]] = None,  # noqa: ARG002
        copy: bool = False,
    ):
        """Tracks a local file or directory with DVC"""
        if not isinstance(path, (str, Path)):
            raise InvalidDataTypeError(path, type(path))

        if self._dvc_repo is not None:
            from dvc.repo.artifacts import name_is_compatible

            if copy:
                path = clean_and_copy_into(path, self.artifacts_dir)

            try:
                stage = self._dvc_repo.add(path)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to dvc add {path}: {e}")
                return

            name = name or Path(path).stem
            if name_is_compatible(name):
                self._artifacts[name] = {
                    k: v
                    for k, v in locals().items()
                    if k in ("path", "type", "desc", "labels", "meta") and v is not None
                }
            else:
                logger.warning(
                    "Can't use '%s' as artifact name (ID)."
                    " It will not be included in the `artifacts` section.",
                    name,
                )

            dvc_file = stage[0].addressing

            if self._save_dvc_exp:
                self._include_untracked.append(dvc_file)
                self._include_untracked.append(
                    str(Path(dvc_file).parent / ".gitignore")
                )

    def make_summary(self, update_step: bool = True):
        if self._step is not None and update_step:
            self.summary["step"] = self.step
        dump_json(self.summary, self.metrics_file, cls=NumpyEncoder)

    def make_report(self):
        if "data" not in self._studio_events_to_skip:
            response = False
            if post_live_metrics is not None:
                metrics, params, plots = get_studio_updates(self)
                response = post_live_metrics(
                    "data",
                    self._baseline_rev,
                    self._exp_name,
                    "dvclive",
                    step=self.step,
                    metrics=metrics,
                    params=params,
                    plots=plots,
                    dvc_studio_config=self._dvc_studio_config,
                )
            if not response:
                logger.warning(
                    "`post_to_studio` `data` event failed."
                    " Data will be resent on next call."
                )
            else:
                self._latest_studio_step = self.step

        if self._report_mode is not None:
            make_report(self)
            if self._report_mode == "html" and env2bool(env.DVCLIVE_OPEN):
                open_file_in_browser(self.report_file)

    def make_dvcyaml(self):
        make_dvcyaml(self)

    def end(self):
        if self._inside_with:
            # Prevent `live.end` calls inside context manager
            return
        self.make_summary(update_step=False)
        if self._dvcyaml:
            self.make_dvcyaml()

        self._ensure_paths_are_tracked_in_dvc_exp()

        self.save_dvc_exp()

        if "done" not in self._studio_events_to_skip:
            response = False
            if post_live_metrics is not None:
                kwargs = {}
                if self._experiment_rev:
                    kwargs["experiment_rev"] = self._experiment_rev
                response = post_live_metrics(
                    "done",
                    self._baseline_rev,
                    self._exp_name,
                    "dvclive",
                    dvc_studio_config=self._dvc_studio_config,
                    **kwargs,
                )
            if not response:
                logger.warning("`post_to_studio` `done` event failed.")
            self._studio_events_to_skip.add("done")
            self._studio_events_to_skip.add("data")
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
        self._inside_with = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._inside_with = False
        self.end()

    def _ensure_paths_are_tracked_in_dvc_exp(self):
        if self._inside_dvc_exp and self._dvc_repo:
            dir_spec = PathSpec.from_lines("gitwildmatch", [self.dir])
            outs_spec = PathSpec.from_lines(
                "gitwildmatch", [str(o) for o in self._dvc_repo.index.outs]
            )
            try:
                paths_to_track = [
                    f
                    for f in self._dvc_repo.scm.untracked_files()
                    if (dir_spec.match_file(f) and not outs_spec.match_file(f))
                ]
                if paths_to_track:
                    self._dvc_repo.scm.add(paths_to_track)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to git add paths:\n{e}")

    def save_dvc_exp(self):
        if self._save_dvc_exp:
            from dvc.exceptions import DvcException

            try:
                self._experiment_rev = self._dvc_repo.experiments.save(
                    name=self._exp_name,
                    include_untracked=self._include_untracked,
                    force=True,
                    message=self._exp_message,
                )
            except DvcException as e:
                logger.warning(f"Failed to save experiment:\n{e}")
            finally:
                mark_dvclive_only_ended()
