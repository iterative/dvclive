from tensorflow.keras.callbacks import (  # pylint: disable=no-name-in-module
    Callback,
)

import dvclive


class DvcLiveCallback(Callback):
    def __init__(self, model_file=None, save_weights_only: bool = False):
        super().__init__()
        self.model_file = model_file
        self.save_weights_only = save_weights_only

    def on_epoch_end(self, epoch: int, logs: dict = None):
        logs = logs or {}
        for metric, value in logs.items():
            dvclive.log(metric, value)
        if self.model_file:
            if self.save_weights_only:
                self.model.save_weights(self.model_file)
            else:
                self.model.save(self.model_file)
        dvclive.next_step()
