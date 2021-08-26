from fastai.callback.core import Callback

import dvclive


class DvcLiveCallback(Callback):
    def __init__(self, model_file=None):
        super().__init__()
        self.model_file = model_file

    def after_epoch(self):
        for key, value in zip(
            self.learn.recorder.metric_names, self.learn.recorder.log
        ):
            key = key.replace("_", "/")
            dvclive.log(f"{key}", float(value), self.learn.epoch)

        if self.model_file:
            self.learn.save(self.model_file)
        dvclive.next_step()
