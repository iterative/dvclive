import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .metrics import MetricLogger


class DvcLiveError(Exception):
    pass


class InitializationError(DvcLiveError):
    def __init__(self):
        super().__init__(
            "Initialization error - no call was made to `dvclive.init()` "
            " or `dvclive.log()` "
        )


class ConfigMismatchError(DvcLiveError):
    def __init__(self, ml: "MetricLogger"):
        from . import env

        super().__init__(
            f"Dvclive initialized in '{ml.dir}' conflicts "
            f"with '{os.environ[env.DVCLIVE_PATH]}' provided by DVC."
        )


class InvalidMetricTypeError(DvcLiveError):
    def __init__(self, name, val):
        self.name = name
        self.val = val
        super().__init__(f"Metrics '{name}' has not supported type {val}")


class CannotCreateLogDirError(DvcLiveError):
    def __init__(self, directory):
        self.directory = directory
        super().__init__(f"dvc-live cannot create log dir - '{directory}'")
