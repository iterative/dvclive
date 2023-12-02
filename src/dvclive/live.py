import glob
import json
import logging
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from dvc.exceptions import DvcException
from funcy import set_in
from ruamel.yaml.representer import RepresenterError

from . import env
from .dvc import (
    ensure_dir_is_tracked,
    find_overlapping_stage,
    get_dvc_repo,
    get_exp_name,
    make_dvcyaml,
)
from .error import (
    InvalidDataTypeError,
    InvalidDvcyamlError,
    InvalidParameterTypeError,
    InvalidPlotTypeError,
    InvalidReportModeError,
)
from .plots import PLOT_TYPES, SKLEARN_PLOTS, CustomPlot, Image, Metric, NumpyEncoder
from .report import BLANK_NOTEBOOK_REPORT, make_report
from .serialize import dump_json, dump_yaml, load_yaml
from .studio import get_dvc_studio_config, post_to_studio
from .utils import (
    StrPath,
    catch_and_warn,
    clean_and_copy_into,
    env2bool,
    inside_notebook,
    matplotlib_installed,
    open_file_in_browser,
)
from .vscode import (
    cleanup_dvclive_step_completed,
    mark_dvclive_only_ended,
    mark_dvclive_only_started,
    mark_dvclive_step_completed,
)

logger = logging.getLogger("dvclive")
logger.setLevel(os.getenv(env.DVCLIVE_LOGLEVEL, "WARNING").upper())
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

ParamLike = Union[int, float, str, bool, List["ParamLike"], Dict[str, "ParamLike"]]


class Live:
    def __init__(
        self,
        dir: str = "dvclive",  # noqa: A002
        resume: bool = False,
        report: Optional[str] = None,
        save_dvc_exp: bool = True,
        dvcyaml: Union[str, bool] = "dvc.yaml",
        cache_images: bool = False,
        exp_name: Optional[str] = None,
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
        self._cache_images = cache_images

        self._report_mode: Optional[str] = report
        self._report_notebook = None
        self._init_report()

        self._baseline_rev: Optional[str] = None
        self._exp_name: Optional[str] = exp_name
        self._exp_message: Optional[str] = exp_message
        self._experiment_rev: Optional[str] = None
        self._inside_dvc_exp: bool = False
        self._inside_dvc_pipeline: bool = False
        self._dvc_repo = None
        self._include_untracked: List[str] = []
        if env2bool(env.DVCLIVE_TEST):
            self._init_test()
        else:
            self._init_dvc()

        os.makedirs(self.dir, exist_ok=True)

        if self._resume:
            self._init_resume()
        else:
            self._init_cleanup()

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
            self.params_file,
            os.path.join(self.dir, "report.html"),
            os.path.join(self.dir, "report.md"),
        ):
            if f and os.path.exists(f):
                os.remove(f)

        for dvc_file in glob.glob(os.path.join(self.dir, "**dvc.yaml")):
            os.remove(dvc_file)

    @catch_and_warn(DvcException, logger)
    def _init_dvc(self):
        from dvc.scm import NoSCM

        if os.getenv(env.DVC_ROOT, None):
            self._inside_dvc_pipeline = True
            self._init_dvc_pipeline()
        self._dvc_repo = get_dvc_repo()

        dvc_logger = logging.getLogger("dvc")
        dvc_logger.setLevel(os.getenv(env.DVCLIVE_LOGLEVEL, "WARNING").upper())

        self._dvc_file = self._init_dvc_file()

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

        if self._dvcyaml and (
            stage := find_overlapping_stage(self._dvc_repo, self.dvc_file)
        ):
            logger.warning(
                f"'{self.dvc_file}' is in outputs of stage '{stage.addressing}'."
                "\nRemove it from outputs to make DVCLive work as expected."
            )

        if self._inside_dvc_pipeline:
            return

        self._baseline_rev = self._dvc_repo.scm.get_rev()
        if self._save_dvc_exp:
            self._exp_name = get_exp_name(
                self._exp_name, self._dvc_repo.scm, self._baseline_rev
            )
            logger.info(f"Logging to experiment '{self._exp_name}'")
            mark_dvclive_only_started(self._exp_name)
            self._include_untracked.append(self.dir)

    def _init_dvc_file(self) -> str:
        if isinstance(self._dvcyaml, str):
            if os.path.basename(self._dvcyaml) == "dvc.yaml":
                return self._dvcyaml
            raise InvalidDvcyamlError
        return "dvc.yaml"

    def _init_dvc_pipeline(self):
        if os.getenv(env.DVC_EXP_BASELINE_REV, None):
            # `dvc exp` execution
            self._baseline_rev = os.getenv(env.DVC_EXP_BASELINE_REV, "")
            self._exp_name = os.getenv(env.DVC_EXP_NAME, "")
            self._inside_dvc_exp = True
            if self._save_dvc_exp:
                logger.info("Ignoring `save_dvc_exp` because `dvc exp run` is running")
        else:
            # `dvc repro` execution
            if self._save_dvc_exp:
                logger.info("Ignoring `save_dvc_exp` because `dvc repro` is running")
            logger.warning(
                "Some DVCLive features are unsupported in `dvc repro`."
                "\nTo use DVCLive with a DVC Pipeline, run it with `dvc exp run`."
            )
        self._save_dvc_exp = False

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
            )
            self._studio_events_to_skip.add("start")
            self._studio_events_to_skip.add("data")
            self._studio_events_to_skip.add("done")
        else:
            self.post_to_studio("start")

    def _init_report(self):
        if self._report_mode not in {None, "html", "notebook", "md"}:
            raise InvalidReportModeError(self._report_mode)
        if self._report_mode == "notebook":
            if inside_notebook():
                from IPython.display import Markdown, display

                self._report_mode = "notebook"
                self._report_notebook = display(
                    Markdown(BLANK_NOTEBOOK_REPORT), display_id=True
                )
            else:
                logger.warning(
                    "Report mode 'notebook' requires to be"
                    " inside a notebook. Disabling report."
                )
                self._report_mode = None
        if self._report_mode in ("notebook", "md") and not matplotlib_installed():
            logger.warning(
                f"Report mode '{self._report_mode}' requires 'matplotlib'"
                " to be installed. Disabling report."
            )
            self._report_mode = None
        logger.debug(f"{self._report_mode=}")

    def _init_test(self):
        """
        Enables test mode that writes to temp paths and doesn't depend on repo.

        Needed to run integration tests in external libraries like huggingface
        accelerate.
        """
        with tempfile.TemporaryDirectory() as dirpath:
            self._dir = os.path.join(dirpath, self._dir)
            if isinstance(self._dvcyaml, str):
                self._dvc_file = os.path.join(dirpath, self._dvcyaml)
            self._save_dvc_exp = False
            logger.warning(
                "DVCLive testing mode enabled."
                f"Repo will be ignored and output will be written to {dirpath}."
            )

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
        return self._dvc_file

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

        self.post_to_studio("data")

        mark_dvclive_step_completed(self.step)
        self.step += 1

    def log_metric(
        self,
        name: str,
        val: Union[int, float, str],
        timestamp: bool = False,
        plot: bool = True,
    ):
        if not Metric.could_log(val):
            raise InvalidDataTypeError(name, type(val))

        if not isinstance(val, str) and (math.isnan(val) or math.isinf(val)):
            val = str(val)

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
            raise InvalidParameterTypeError(exc.args[0]) from exc

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
        desc: Optional[str] = None,
        labels: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
        copy: bool = False,
        cache: bool = True,
    ):
        """Tracks a local file or directory with DVC"""
        if not isinstance(path, (str, Path)):
            raise InvalidDataTypeError(path, type(path))

        if self._dvc_repo is not None:
            from gto.constants import assert_name_is_valid
            from gto.exceptions import ValidationError

            if copy:
                path = clean_and_copy_into(path, self.artifacts_dir)

            if cache:
                self.cache(path)

            if any((type, name, desc, labels, meta)):
                name = name or Path(path).stem
                try:
                    assert_name_is_valid(name)
                    self._artifacts[name] = {
                        k: v
                        for k, v in locals().items()
                        if k in ("path", "type", "desc", "labels", "meta")
                        and v is not None
                    }
                except ValidationError:
                    logger.warning(
                        "Can't use '%s' as artifact name (ID)."
                        " It will not be included in the `artifacts` section.",
                        name,
                    )
        else:
            logger.warning(
                "A DVC repo is required to log artifacts. "
                f"Skipping `log_artifact({path})`."
            )

    @catch_and_warn(DvcException, logger)
    def cache(self, path):
        if self._inside_dvc_pipeline:
            existing_stage = find_overlapping_stage(self._dvc_repo, path)

            if existing_stage:
                if existing_stage.cmd:
                    logger.info(
                        f"Skipping `dvc add {path}` because it is already being"
                        " tracked automatically as an output of the DVC pipeline."
                    )
                    return  # skip caching
                logger.warning(
                    f"To track '{path}' automatically in the DVC pipeline:"
                    f"\n1. Run `dvc remove {existing_stage.addressing}` "
                    "to stop tracking it outside the pipeline."
                    "\n2. Add it as an output of the pipeline stage."
                )
            else:
                logger.warning(
                    f"To track '{path}' automatically in the DVC pipeline, "
                    "add it as an output of the pipeline stage."
                )

        stage = self._dvc_repo.add(str(path))

        dvc_file = stage[0].addressing

        if self._save_dvc_exp:
            self._include_untracked.append(dvc_file)
            self._include_untracked.append(str(Path(dvc_file).parent / ".gitignore"))

    def make_summary(self, update_step: bool = True):
        if self._step is not None and update_step:
            self.summary["step"] = self.step
        dump_json(self.summary, self.metrics_file, cls=NumpyEncoder)

    def make_report(self):
        if self._report_mode is not None:
            make_report(self)
            if self._report_mode == "html" and env2bool(env.DVCLIVE_OPEN):
                open_file_in_browser(self.report_file)

    @catch_and_warn(DvcException, logger)
    def make_dvcyaml(self):
        make_dvcyaml(self)

    @catch_and_warn(DvcException, logger)
    def post_to_studio(self, event):
        post_to_studio(self, event)

    def end(self):
        if self._inside_with:
            # Prevent `live.end` calls inside context manager
            return

        if self._images and self._cache_images:
            images_path = Path(self.plots_dir) / Image.subfolder
            self.cache(images_path)

        self.make_summary(update_step=False)
        if self._dvcyaml:
            self.make_dvcyaml()

        if self._inside_dvc_exp and self._dvc_repo:
            catch_and_warn(DvcException, logger)(ensure_dir_is_tracked)(
                self.dir, self._dvc_repo
            )
            if self._dvcyaml:
                catch_and_warn(DvcException, logger)(self._dvc_repo.scm.add)(
                    self.dvc_file
                )

        self.make_report()

        self.save_dvc_exp()

        # Post any data that hasn't been sent
        self.post_to_studio("data")
        # Mark experiment as done
        self.post_to_studio("done")

        cleanup_dvclive_step_completed()

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

    @catch_and_warn(DvcException, logger, mark_dvclive_only_ended)
    def save_dvc_exp(self):
        if self._save_dvc_exp:
            if self._dvcyaml:
                self._include_untracked.append(self.dvc_file)
            self._experiment_rev = self._dvc_repo.experiments.save(
                name=self._exp_name,
                include_untracked=self._include_untracked,
                force=True,
                message=self._exp_message,
            )
