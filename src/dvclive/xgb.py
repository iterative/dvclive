# ruff: noqa: ARG002
from typing import Optional, Sequence, Union

from xgboost.callback import TrainingCallback

from dvclive import Live


class DVCLiveCallback(TrainingCallback):
    def __init__(
        self,
        metric_data: Union[str, Sequence[str]],
        model_file=None,
        live: Optional[Live] = None,
        **kwargs,
    ):
        super().__init__()
        if isinstance(metric_data, str):
            metric_data = (metric_data,)
        self._metric_spec = {
            name: name if len(metric_data) > 1 else "" for name in metric_data
        }
        self.model_file = model_file
        self.live = live if live is not None else Live(**kwargs)

    def after_iteration(self, model, epoch, evals_log):
        for name, subdir in self._metric_spec.items():
            for key, values in evals_log[name].items():
                if values:
                    latest_metric = values[-1]
                self.live.log_metric(
                    f"{subdir}/{key}" if subdir else key, latest_metric
                )
        if self.model_file:
            model.save_model(self.model_file)
        self.live.next_step()

    def after_training(self, model):
        self.live.end()
        return model
