# ruff: noqa: ARG002
import inspect
from typing import Any, Dict, Optional

from lightning_fabric.utilities.logger import (
    _convert_params,
    _sanitize_callable_params,
    _sanitize_params,
)
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
from torch import is_tensor

from dvclive import Live
from dvclive.utils import standardize_metric_name


def _should_call_next_step():
    """
    Find out if pytorch_lightning is calling `log_metrics` from the functions
    where we actually want to call `next_step`.
    For example, prevents calling next_step when external callbacks call
    `log_metrics` or during the multiple `update_eval_step_metrics`.
    """
    return any(
        frame.function
        in (
            "update_train_step_metrics",
            "update_train_epoch_metrics",
            "log_eval_end_metrics",
        )
        for frame in inspect.stack()
    )


class DVCLiveLogger(Logger):
    def __init__(
        self,
        run_name: Optional[str] = "dvclive_run",
        prefix="",
        experiment=None,
        dir: Optional[str] = None,  # noqa: A002
        resume: bool = False,
        report: Optional[str] = "auto",
        save_dvc_exp: bool = False,
        dvcyaml: bool = True,
    ):
        super().__init__()
        self._prefix = prefix
        self._live_init: Dict[str, Any] = {
            "resume": resume,
            "report": report,
            "save_dvc_exp": save_dvc_exp,
            "dvcyaml": dvcyaml,
        }
        if dir is not None:
            self._live_init["dir"] = dir
        self._experiment = experiment
        self._version = run_name
        if report == "notebook":
            # Force Live instantiation
            self.experiment  # noqa: B018

    @property
    def name(self):
        return "DvcLiveLogger"

    @rank_zero_only
    def log_hyperparams(self, params, *args, **kwargs):
        def sanitize_dict(params):
            dict_values = {}
            non_dict_values = {}
            for k, v in params.items():
                if isinstance(v, dict):
                    dict_values[k] = sanitize_dict(v)
                else:
                    non_dict_values[k] = v
            non_dict_values = _sanitize_params(non_dict_values)
            return {**dict_values, **non_dict_values}

        params = _convert_params(params)
        params = _sanitize_callable_params(params)
        params = sanitize_dict(params)
        self.experiment.log_params(params)

    @property  # type: ignore
    @rank_zero_experiment
    def experiment(self):
        r"""
        Actual DVCLive object. To use DVCLive features in your
        :class:`~LightningModule` do the following.
        Example::
            self.logger.experiment.some_dvclive_function()
        """
        if self._experiment is not None:
            return self._experiment
        self._experiment = Live(**self._live_init)

        return self._experiment

    @property
    def version(self):
        return self._version

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        self.experiment.step = step
        for metric_name, metric_val in metrics.items():
            val = metric_val
            if is_tensor(val):
                val = val.cpu().detach().item()
            name = standardize_metric_name(metric_name, __name__)
            self.experiment.log_metric(name=name, val=val)
        if _should_call_next_step():
            if step == self.experiment._latest_studio_step:  # noqa: SLF001
                # We are in log_eval_end_metrics but there has been already
                # a studio request sent with `step`.
                # We decrease the number to bypass `live.studio._get_unsent_datapoints`
                self.experiment._latest_studio_step -= 1  # noqa: SLF001
            self.experiment.next_step()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.experiment.end()
