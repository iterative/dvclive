from xgboost.callback import TrainingCallback

import dvclive


class DvcLiveCallback(TrainingCallback):
    def __init__(self, metric_data):
        super().__init__()
        self._metric_data = metric_data

    def after_iteration(self, model, epoch, evals_log):
        for key, values in evals_log[self._metric_data].items():
            if values:
                latest_metric = values[-1]
            dvclive.log(key, latest_metric)
        dvclive.next_step()
