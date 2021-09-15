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
            fr"Dvclive initialized in '{ml.dir}' conflicts "
            fr"with '{os.environ[env.DVCLIVE_PATH]}' provided by DVC."
        )


class InvalidDataTypeError(DvcLiveError):
    def __init__(self, name, val):
        self.name = name
        self.val = val
        super().__init__(fr"Data '{name}' has not supported type {val}")


class DataAlreadyLoggedError(DvcLiveError):
    def __init__(self, name, step):
        self.name = name
        self.val = step
        super().__init__(
            fr"Data '{name}' has already being logged whith step '{step}'"
        )
