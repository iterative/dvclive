# ruff: noqa: ARG002
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from lightning.fabric.utilities.logger import (
        _convert_params,
        _sanitize_callable_params,
    )
    from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
    from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
    from lightning.pytorch.loggers.utilities import _scan_checkpoints
    from lightning.pytorch.utilities import rank_zero_only
except ImportError:
    from lightning_fabric.utilities.logger import (
        _convert_params,
        _sanitize_callable_params,
    )
    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
    from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
    from pytorch_lightning.utilities import rank_zero_only

    try:
        from pytorch_lightning.utilities.logger import _scan_checkpoints
    except ImportError:
        from pytorch_lightning.loggers.utilities import _scan_checkpoints
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
        log_model: Union[str, bool] = False,
        experiment=None,
        **kwargs,
    ):
        super().__init__()
        self._prefix = prefix
        self._live_init: Dict[str, Any] = kwargs
        self._experiment = experiment
        self._version = run_name
        self._log_model = log_model
        self._logged_model_time: Dict[str, float] = {}
        self._checkpoint_callback: Optional[ModelCheckpoint] = None
        self._all_checkpoint_paths: List[str] = []

    @property
    def name(self):
        return "DvcLiveLogger"

    @rank_zero_only
    def log_hyperparams(self, params, *args, **kwargs):
        params = _convert_params(params)
        params = _sanitize_callable_params(params)
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

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        if self._log_model in [True, "all"]:
            self._checkpoint_callback = checkpoint_callback
            self._scan_checkpoints(checkpoint_callback)
        if self._log_model == "all" or (
            self._log_model is True and checkpoint_callback.save_top_k == -1
        ):
            self._save_checkpoints(checkpoint_callback)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        # Log best model.
        if self._checkpoint_callback:
            self._scan_checkpoints(self._checkpoint_callback)
            self._save_checkpoints(self._checkpoint_callback)
            best_model_path = self._checkpoint_callback.best_model_path
            self.experiment.log_artifact(
                best_model_path, name="best", type="model", copy=True
            )
        self.experiment.end()

    def _scan_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        # get checkpoints to be saved with associated score
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)

        # update model time and append path to list of all checkpoints
        for t, p, _, _ in checkpoints:
            self._logged_model_time[p] = t
            self._all_checkpoint_paths.append(p)

    def _save_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        # drop unused checkpoints
        if not self._experiment._resume:  # noqa: SLF001
            for p in Path(checkpoint_callback.dirpath).iterdir():
                if str(p) not in self._all_checkpoint_paths:
                    p.unlink(missing_ok=True)

        # save directory
        self.experiment.log_artifact(checkpoint_callback.dirpath)
