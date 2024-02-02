from __future__ import annotations
import builtins
import glob
import json
import logging
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

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
    convert_datapoints_to_list_of_dicts,
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
        dvcyaml: Union[str, None] = "dvc.yaml",
        cache_images: bool = False,
        exp_name: Optional[str] = None,
        exp_message: Optional[str] = None,
    ):
        """
        Initializes a DVCLive logger. A `Live` instance is required to log machine
        learning parameters, metrics, and other metadata.
        Warning: Calling `Live` with an existing `dir` overwrites the existing
        metrics and reports. Use `resume=True` to continue logging to an existing DVC
        experiment.

        Args:
            dir (str | Path): The location of the directory to store outputs, such as
                the metrics file and report. Defaults to "dvclive".
            resume (bool): Whether to resume from the last step of the previous run.
                Defaults to False.
            report ("html", "md", "notebook", None): The format of the report
                summarizing the metrics. Defaults to None.
            save_dvc_exp (bool): Allows `Live` to create a new DVC experiment. If you're
                running `Live` with `dvc exp run`, this will be ignored.
                Defaults to True.
            dvcyaml (str | None): The location to write the `dvc.yaml` file, which is a
                configuration file. If `None`, no `dvc.yaml` will be created.
                Defaults to "dvc.yaml".
            cache_images (bool): Determines whether `Live` should cache the images saved
                with `log_image` in the DVC cache. If you're using `Live` with a DVC
                pipeline, this will be ignored, and you should instead cache the images
                as pipeline outputs. Defaults to False.
            exp_name (str | None): The name of your DVC experiment if `save_dvc_exp` is
                `True`. If you're running `Live` with `dvc exp run`, it will be ignored,
                and you should instead use the `--name` CLI argument.
                Defaults to None.
            exp_message (str | None): The message associated with your DVC experiment if
                `save_dvc_exp` is `True`. If you're running `Live` with `dvc exp run`,
                it will be ignored, and you should instead use the `--message` CLI
                argument. Defaults to None.
        """
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
        self.summary = self.read_latest()
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
        Enables a test mode that writes to temporary paths and doesn't depend on the
        repository.

        This is needed to run integration tests in external libraries, such as
        HuggingFace Accelerate.
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

    def sync(self):
        """
        Updates the report and summary files. Also synchronizes data with DVCstudio
        """
        self.make_summary()

        if self._dvcyaml:
            self.make_dvcyaml()

        self.make_report()

        self.post_to_studio("data")

    def next_step(self):
        """
        Signals that the current iteration has ended and increases the step value by
        one. This function also calls `sync` to update the report and summary files.
        Note: The `step` property can be accessed directly in `Live` if you want to use
        custom step values. If you choose this option, you're also responsible for
        calling the `sync` method.
        """
        if self._step is None:
            self._step = 0

        self.sync()
        mark_dvclive_step_completed(self.step)
        self.step += 1

    def log_metric(
        self,
        name: str,
        val: Union[int, float, str],
        timestamp: bool = False,
        plot: bool = True,
    ):
        """
        Tracks a metric. Call this function multiple times with the same `name` to
        track the evolution of a metric.

        Args:
            name (str): The name of the metric being logged.
            val (int | float | str): The value to be logged.
            timestamp (bool): Whether to automatically log the timestamp alongside the
                metric value.
                Defaults to False.
            plot (bool): Whether to add the metric file for plotting. If `False`, the
                metric will only be saved to the metrics summary. Defaults to True.

        Raises:
            `InvalidDataTypeError`: If the `path` argument is neither a string nor a
                Path.
        """
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
        """
        Tracks an image file or content with DVC.

        Args:
            name (str): The name of the image file that this command will output.
            val (np.ndarray | matplotlib.figure.Figure | PIL.Image | StrPath):
                The image to be saved. If you're using a `np.ndarray`, the array
                should be convertible via `PIL.Image.fromarray`.

        Raises:
            `InvalidDataTypeError`: If the `path` argument is neither a string nor a
                Path.
        """
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
        datapoints: pd.DataFrame | np.ndarray | List[Dict],
        x: str,
        y: str,
        template: Optional[str] = None,
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
    ):
        """
        Tracks plot content with DVC.

        Args:
            name (StrPath): The name of the output file.
            datapoints (pd.DataFrame | np.ndarray | List[Dict]): The content for the
                plot.
            x (str): The name of the key (present in the dictionaries) to use as the x
                axis. If you're using `np.ndarray` or `pd.DataFrame`, it should be the
                name of the column. If your column doesn't have a name, you can refer to
                the index as a str, for example `"0"` for the first column.
            y (str): The name of the key (present in the dictionaries) to use as the y
                axis. If you're using `np.ndarray` or `pd.DataFrame`, it should be the
                name of the column. If your column doesn't have a name, you can refer to
                the index as a str, for example `"0"` for the first column.
            template (str): The name of the DVC plots template to use.
                Defaults to 'linear'.
            title (str): The title of the plot. Defaults to None.
            x_label (str): The label for the x axis. If not provided, the name passed
                for `x` will be used as the label. Defaults to None.
            y_label (str): The label for the y axis. If not provided, the name passed
                for `y` will be used as the label. Defaults to None.

        Raises:
            InvalidDataTypeError: `datapoints` must be a pd.DataFrame, np.ndarray, or
                List[Dict].
        """
        # Convert the given datapoints to List[Dict]
        datapoints = convert_datapoints_to_list_of_dicts(datapoints=datapoints)

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
        """
        Tracks plot content with DVC using sklearn plots.

        Args:
            kind ("calibration" | "confusion_matrix" | "det" | "precision_recall" |
                "roc"): A supported plot type.
            labels (List | np.ndarray): An array of ground truth labels.
            predictions (List | np.ndarray): An array of predicted labels for the
                "confusion_matrix" `kind` or predicted probabilities for other plots.
            name (str): The name of the output file. If not provided, `kind` will be
                used as the name. Defaults to None.
            kwargs: Additional arguments to be passed to the scikit-learn function.
                Extra arguments supported are:
                - `normalized` for "confusion_matrix", which defaults to False.

        Raises:
            InvalidPlotTypeError: `kind` must be one of "calibration",
                "confusion_matrix", "det", "precision_recall", or "roc".
        """
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
        """
        Saves the given set of parameters (dict) to the metrics file.

        Args:
            params (Dict[str, ParamLike]): A dictionary with name/value
                pairs of parameters to be logged.

        Raises:
            `InvalidParameterTypeError`: If the parameter value does not match the
            supported types. Supported types for value are int, float, str, bool.
            This also includes lists and dicts containing these types, if the dict keys
            are str.
        """
        self._params.update(params)
        self._dump_params()
        logger.debug(f"Logged {params} parameters to {self.params_file}")

    def log_param(self, name: str, val: ParamLike):
        """
        Saves the given parameter value to the metrics file.

        Args:
            name (str): The name of the parameter being logged.
            val (ParamLike): The value to be logged.
                Generally, `val` is an int, float, str, bool, or a list containing
                these elements. Dicts are also supported if the keys are str.

        Raises:
            `InvalidParameterTypeError`: If the parameter value does not match the
                supported types. Supported types for value are int, float, str, bool.
                This also includes lists and dicts containing these types, if the dict
                keys are str.
        """
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
        """
        Tracks a local file or directory with DVC.

        Args:
            path (StrPath): An existing directory or file to log.
            type (Optional[str]): An optional type of the artifact.
                Common ones are "model" or "dataset".
                Defaults to None.
            name (Optional[str]): An optional custom name of an artifact.
                If not provided, the path stem will be used as the name.
                Defaults to None.
            desc (Optional[str]): An optional description of an artifact.
                Defaults to None.
            labels (Optional[List[str]]): Optional labels describing the artifact.
                Defaults to None.
            meta (Optional[Dict[str, Any]]): Optional metainformation.
                Should be in `key: value` format.
                Defaults to None.
            copy (bool): Copy a directory or file at `path` into the
                `dvclive/artifacts` location before tracking it.
                Useful if you don't want to track the original path in your repo
                (for example, it is outside the repo or in a Git-ignored directory).
                Defaults to False.
            cache (bool): Cache the files with DVC to track them outside of Git.
                Use `false` for small files you want to track with Git.
                Defaults to True.

        Raises:
            `InvalidDataTypeError`: If the `path` argument is not a string nor a Path.
            `ValidationError`: If the `path` or `name` arguments have already been
            logged as an artifact within the same `step`.
        """
        if not isinstance(path, (str, Path)):
            raise InvalidDataTypeError(path, builtins.type(path))

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

    def make_summary(self):
        """
        Serializes a summary of the logged metrics to `Live.metrics_file`.

        The default path for the metrics file is `metrics.json` in the directory
        specified by the `dir` argument passed to `Live`. The summary will contain the
        latest value of each metric logged.
        """
        if self._step is not None:
            self.summary["step"] = self.step
        dump_json(self.summary, self.metrics_file, cls=NumpyEncoder)

    def make_report(self):
        """
        Generates a report from the logged data.

        On each call, this function collects all the data logged in the directory
        specified by `Live`'s `dir` argument, generates a report, and saves it under
        the name "report.{format}". The format can be HTML or Markdown, depending on
        the value of the `report` argument passed to `Live`.

        Note: If you're using a Notebook, the report will be displayed in the notebook
        if you provide the `report` argument as "notebook".
        """
        if self._report_mode is not None:
            make_report(self)
            if self._report_mode == "html" and env2bool(env.DVCLIVE_OPEN):
                open_file_in_browser(self.report_file)

    @catch_and_warn(DvcException, logger)
    def make_dvcyaml(self):
        """
        Creates a `dvc.yaml` file that describes and configures metrics, plots, and
        parameters.

        DVC tools use this file to display reports and experiment tables. This function
        is automatically invoked by `Live.next_step` and `Live.end`.
        """
        make_dvcyaml(self)

    @catch_and_warn(DvcException, logger)
    def post_to_studio(self, event):
        post_to_studio(self, event)

    def end(self):
        """Signals that the current experiment has ended.

        This function is called automatically when exiting the `Live` context manager or
        when the training of a supported framework ends.
        """
        if self._inside_with:
            # Prevent `live.end` calls inside context manager
            return

        if self._images and self._cache_images:
            images_path = Path(self.plots_dir) / Image.subfolder
            self.cache(images_path)

        # If next_step called before end, don't want to update step number
        if "step" in self.summary:
            self.step = self.summary["step"]
        self.sync()

        if self._inside_dvc_exp and self._dvc_repo:
            catch_and_warn(DvcException, logger)(ensure_dir_is_tracked)(
                self.dir, self._dvc_repo
            )
            if self._dvcyaml:
                catch_and_warn(DvcException, logger)(self._dvc_repo.scm.add)(
                    self.dvc_file
                )

        self.save_dvc_exp()

        # Mark experiment as done
        self.post_to_studio("done")

        cleanup_dvclive_step_completed()

    def read_step(self):
        """Reads the latest step from the metrics file.

        This function is useful when resuming an experiment.

        Returns:
            int: The step value from the metrics file.
        """
        latest = self.read_latest()
        return latest.get("step", 0)

    def read_latest(self):
        """
        Reads the latest metrics file.

        This function is useful when resuming an experiment.

        Returns:
            Dict[str, ParamLike]: The content from the metrics file.
        """
        if Path(self.metrics_file).exists():
            with open(self.metrics_file, encoding="utf-8") as fobj:
                return json.load(fobj)
        return {}

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
