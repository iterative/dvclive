# ruff: noqa: ARG002
import os
from typing import Dict, Optional

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model

from dvclive import Live
from dvclive.utils import standardize_metric_name


class DVCLiveCallback(Callback):
    def __init__(
        self,
        model_file=None,
        save_weights_only: bool = False,
        live: Optional[Live] = None,
        **kwargs
    ):
        super().__init__()
        self.model_file = model_file
        self.save_weights_only = save_weights_only
        self.live = live if live is not None else Live(**kwargs)

    def on_train_begin(self, logs=None):
        if (
            self.live._resume  # noqa: SLF001
            and self.model_file is not None
            and os.path.exists(self.model_file)
        ):
            if self.save_weights_only:
                self.model.load_weights(self.model_file)
            else:
                self.model = load_model(self.model_file)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        logs = logs or {}
        for metric, value in logs.items():
            self.live.log_metric(standardize_metric_name(metric, __name__), value)
        if self.model_file:
            if self.save_weights_only:
                self.model.save_weights(self.model_file)
            else:
                self.model.save(self.model_file)
            self.live.log_artifact(self.model_file)
        self.live.next_step()

    def on_train_end(self, logs: Optional[Dict] = None):
        self.live.end()
