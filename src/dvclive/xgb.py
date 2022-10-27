from typing import Optional

from xgboost.callback import TrainingCallback

from dvclive import Live


class DvcLiveCallback(TrainingCallback):
    def __init__(
        self,
        metric_data,
        model_file=None,
        dvclive: Optional[Live] = None,
        **kwargs
    ):
        super().__init__()
        self._metric_data = metric_data
        self.model_file = model_file
        self.dvclive = dvclive if dvclive is not None else Live(**kwargs)

    def after_iteration(self, model, epoch, evals_log):
        for key, values in evals_log[self._metric_data].items():
            if values:
                latest_metric = values[-1]
            self.dvclive.log_metric(key, latest_metric)
        if self.model_file:
            model.save_model(self.model_file)
        self.dvclive.make_report()
        self.dvclive.next_step()
