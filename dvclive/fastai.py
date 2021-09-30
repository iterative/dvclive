from fastai.callback.core import Callback

from dvclive import DVCLive


class DvcLiveCallback(Callback):
    def __init__(self, model_file=None, **kwargs):
        super().__init__()
        self.model_file = model_file
        self.dvclive = DVCLive(**kwargs)

    def after_epoch(self):
        for key, value in zip(
            self.learn.recorder.metric_names, self.learn.recorder.log
        ):
            key = key.replace("_", "/")
            self.dvclive.log(f"{key}", float(value))

        if self.model_file:
            self.learn.save(self.model_file)
        self.dvclive.next_step()
