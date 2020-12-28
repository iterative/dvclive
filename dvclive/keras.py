from tensorflow.keras.callbacks import Callback

from dvclive import dvclive
from dvclive.dvc import make_checkpoint


class DvcLiveCallback(Callback):
    def on_epoch_end(self, epoch: int, logs: dict = None):
        logs = logs or {}
        for metric, value in logs.items():
            dvclive.log(metric, value)
        dvclive.next_step()
        make_checkpoint()
