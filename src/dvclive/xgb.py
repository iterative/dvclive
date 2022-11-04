from typing import Optional

from xgboost.callback import TrainingCallback

from dvclive import Live


class DVCLiveCallback(TrainingCallback):
    def __init__(
        self,
        metric_data,
        model_file=None,
        live: Optional[Live] = None,
        **kwargs
    ):
        super().__init__()
        self._metric_data = metric_data
        self.model_file = model_file
        self.live = live if live is not None else Live(**kwargs)

    def after_iteration(self, model, epoch, evals_log):
        for key, values in evals_log[self._metric_data].items():
            if values:
                latest_metric = values[-1]
            self.live.log_metric(key, latest_metric)
        if self.model_file:
            model.save_model(self.model_file)
        self.live.next_step()
