import inspect
from typing import Optional

from fastai.callback.core import Callback

from dvclive import Live
from dvclive.utils import standardize_metric_name


def _inside_fine_tune():
    """
    Hack to find out if fastai is calling `after_fit` at the end of the
    "freeze" stage part of `learn.fine_tune` .
    """
    fine_tune = False
    fit_one_cycle = False
    for frame in inspect.stack():
        if frame.function == "fine_tune":
            fine_tune = True
        if frame.function == "fit_one_cycle":
            fit_one_cycle = True
        if fine_tune and fit_one_cycle:
            return True
    return False


class DVCLiveCallback(Callback):
    def __init__(
        self,
        model_file: Optional[str] = None,
        with_opt: bool = False,
        live: Optional[Live] = None,
        **kwargs,
    ):
        super().__init__()
        self.model_file = model_file
        self.with_opt = with_opt
        self.live = live if live is not None else Live(**kwargs)
        self.freeze_stage_ended = False

    def before_fit(self):
        if hasattr(self, "lr_finder") or hasattr(self, "gather_preds"):
            return
        params = {
            "model": type(self.learn.model).__qualname__,
            "batch_size": getattr(self.dls, "bs", None),
            "batch_per_epoch": len(getattr(self.dls, "train", [])),
            "frozen": bool(getattr(self.opt, "frozen_idx", -1)),
            "frozen_idx": getattr(self.opt, "frozen_idx", -1),
            "transforms": f"{getattr(self.dls, 'tfms', None)}",
        }
        self.live.log_params(params)

    def after_epoch(self):
        if hasattr(self, "lr_finder") or hasattr(self, "gather_preds"):
            return
        logged_metrics = False
        for key, value in zip(
            self.learn.recorder.metric_names, self.learn.recorder.log
        ):
            if key == "epoch":
                continue
            self.live.log_metric(standardize_metric_name(key, __name__), float(value))
            logged_metrics = True

        # When resuming (i.e. passing `start_epoch` to learner)
        # fast.ai calls after_epoch but we don't want to increase the step.
        if logged_metrics:
            if self.model_file:
                self.learn.save(self.model_file, with_opt=self.with_opt)
            self.live.next_step()

    def after_fit(self):
        if hasattr(self, "lr_finder") or hasattr(self, "gather_preds"):
            return
        if _inside_fine_tune() and not self.freeze_stage_ended:
            self.freeze_stage_ended = True
        else:
            self.live.end()
