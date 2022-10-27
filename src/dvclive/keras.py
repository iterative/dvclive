import os
from typing import Dict, Optional

from tensorflow.keras.callbacks import (  # noqa pylint: disable=import-error, no-name-in-module
    Callback,
)
from tensorflow.keras.models import (  # noqa pylint: disable=import-error, no-name-in-module
    load_model,
)

from dvclive import Live
from dvclive.utils import standardize_metric_name


class DvcLiveCallback(Callback):
    def __init__(
        self,
        model_file=None,
        save_weights_only: bool = False,
        dvclive: Optional[Live] = None,
        **kwargs
    ):
        super().__init__()
        self.model_file = model_file
        self.save_weights_only = save_weights_only
        self.dvclive = dvclive if dvclive is not None else Live(**kwargs)

    def on_train_begin(self, logs=None):  # pylint: disable=unused-argument
        if (
            self.dvclive._resume  # pylint: disable=protected-access
            and self.model_file is not None
            and os.path.exists(self.model_file)
        ):
            if self.save_weights_only:
                self.model.load_weights(  # noqa pylint: disable=access-member-before-definition
                    self.model_file
                )
            else:
                self.model = load_model(  # noqa pylint: disable=attribute-defined-outside-init
                    self.model_file
                )

    def on_epoch_end(
        self, epoch: int, logs: Optional[Dict] = None
    ):  # pylint: disable=unused-argument
        logs = logs or {}
        for metric, value in logs.items():
            self.dvclive.log(standardize_metric_name(metric, __name__), value)
        if self.model_file:
            if self.save_weights_only:
                self.model.save_weights(self.model_file)
            else:
                self.model.save(self.model_file)
        self.dvclive.make_report()
        self.dvclive.next_step()
