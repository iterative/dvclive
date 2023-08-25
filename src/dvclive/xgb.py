# ruff: noqa: ARG002
from typing import Optional
from warnings import warn

from xgboost.callback import TrainingCallback

from dvclive import Live


class DVCLiveCallback(TrainingCallback):
    def __init__(
        self,
        metric_data: Optional[str] = None,
        live: Optional[Live] = None,
        **kwargs,
    ):
        super().__init__()
        if metric_data is not None:
            warn(
                "`metric_data` is deprecated and will be removed",
                category=DeprecationWarning,
                stacklevel=2,
            )
        self._metric_data = metric_data
        self.live = live if live is not None else Live(**kwargs)

    def after_iteration(self, model, epoch, evals_log):
        if self._metric_data:
            evals_log = {"": evals_log[self._metric_data]}
        for subdir, data in evals_log.items():
            for key, values in data.items():
                self.live.log_metric(f"{subdir}/{key}" if subdir else key, values[-1])
        self.live.next_step()

    def after_training(self, model):
        self.live.end()
        return model
