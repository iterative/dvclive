import os

from tensorflow.keras.callbacks import (  # pylint: disable=no-name-in-module
    Callback,
)
from tensorflow.keras.models import (  # pylint: disable=no-name-in-module
    load_model,
)

from dvclive import Live


class DvcLiveCallback(Callback):
    def __init__(
        self, model_file=None, save_weights_only: bool = False, **kwargs
    ):
        super().__init__()
        self.model_file = model_file
        self.save_weights_only = save_weights_only
        self.dvclive = Live(**kwargs)

    def on_train_begin(self, logs=None):
        if (
            self.dvclive._resume
            and self.model_file is not None
            and os.path.exists(self.model_file)
        ):
            if self.save_weights_only:
                self.model.load_weights(self.model_file)
            else:
                self.model = load_model(self.model_file)

    def on_epoch_end(self, epoch: int, logs: dict = None):
        logs = logs or {}
        for metric, value in logs.items():
            self.dvclive.log(metric, value)
        if self.model_file:
            if self.save_weights_only:
                self.model.save_weights(self.model_file)
            else:
                self.model.save(self.model_file)
        self.dvclive.next_step()
