from tensorflow.keras.callbacks import Callback

from dvclive import dvclive


class DvcLiveCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric, value in logs.items():
            dvclive.log(metric, value)
        dvclive.next_epoch()
