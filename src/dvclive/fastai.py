from typing import Optional

from fastai.callback.core import Callback

from dvclive import Live
from dvclive.utils import standardize_metric_name


class DVCLiveCallback(Callback):
    def __init__(
        self,
        model_file: Optional[str] = None,
        with_opt: bool = False,
        live: Optional[Live] = None,
        **kwargs
    ):
        super().__init__()
        self.model_file = model_file
        self.with_opt = with_opt
        self.live = live if live is not None else Live(**kwargs)

    def after_epoch(self):
        logged_metrics = False
        for key, value in zip(
            self.learn.recorder.metric_names, self.learn.recorder.log
        ):
            if key == "epoch":
                continue
            self.live.log_metric(
                standardize_metric_name(key, __name__), float(value)
            )
            logged_metrics = True

        # When resuming (i.e. passing `start_epoch` to learner)
        # fast.ai calls after_epoch but we don't want to increase the step.
        if logged_metrics:
            if self.model_file:
                self.learn.save(self.model_file, with_opt=self.with_opt)
            self.live.next_step()

    def after_fit(self):
        self.live.end()
