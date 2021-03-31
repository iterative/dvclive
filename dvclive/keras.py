from tensorflow.keras.callbacks import Callback

import dvclive


class DvcLiveCallback(Callback):
    def __init__(self, model_file = None):
        self.model_file = model_file

    def on_epoch_end(self, epoch: int, logs: dict = None):
        logs = logs or {}
        for metric, value in logs.items():
            dvclive.log(metric, value)
        if self.model_file:
            self.model.save(self.model_file)
        dvclive.next_step()
