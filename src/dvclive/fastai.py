from typing import Optional

from fastai.callback.core import Callback

from dvclive import Live
from dvclive.utils import standardize_metric_name


class DVCLiveCallback(Callback):
    def __init__(self, model_file=None, live: Optional[Live] = None, **kwargs):
        super().__init__()
        self.model_file = model_file
        self.live = live if live is not None else Live(**kwargs)

    def after_epoch(self):
        for key, value in zip(
            self.learn.recorder.metric_names, self.learn.recorder.log
        ):
            self.live.log_metric(
                standardize_metric_name(key, __name__), float(value)
            )

        if self.model_file:
            self.learn.save(self.model_file)
        self.live.next_step()
