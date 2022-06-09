from fastai.callback.core import Callback

from dvclive import Live
from dvclive.utils import standardize_metric_name


class DvcLiveCallback(Callback):
    def __init__(self, model_file=None, **kwargs):
        super().__init__()
        self.model_file = model_file
        self.dvclive = Live(**kwargs)

    def after_epoch(self):
        for key, value in zip(
            self.learn.recorder.metric_names, self.learn.recorder.log
        ):
            self.dvclive.log(
                standardize_metric_name(key, __name__), float(value)
            )

        if self.model_file:
            self.learn.save(self.model_file)
        self.dvclive.next_step()
