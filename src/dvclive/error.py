import os
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .live import Live


class DvcLiveError(Exception):
    pass


class ConfigMismatchError(DvcLiveError):
    def __init__(self, ml: "Live"):
        from . import env

        super().__init__(
            f"Dvclive initialized in '{ml.dir}' conflicts "
            f"with '{os.environ[env.DVCLIVE_PATH]}' provided by DVC."
        )


class InvalidDataTypeError(DvcLiveError):
    def __init__(self, name, val):
        self.name = name
        self.val = val
        super().__init__(f"Data '{name}' has not supported type {val}")


class InvalidPlotTypeError(DvcLiveError):
    def __init__(self, name):
        from .data import PLOTS

        self.name = name
        super().__init__(
            f"Plot type '{name}' is not supported."
            f"\nSupported types are: {list(PLOTS)}"
        )


class DataAlreadyLoggedError(DvcLiveError):
    def __init__(self, name, step):
        self.name = name
        self.val = step
        super().__init__(
            f"Data '{name}' has already been logged with step '{step}'"
        )


class ParameterAlreadyLoggedError(DvcLiveError):
    def __init__(
        self, name: str, val: Any, previous_val: Optional[Any] = None
    ):
        self.name = name
        self.val = val
        self.previous_val = previous_val
        super().__init__(
            f"Parameter '{name}={val}' has already been logged"
            + (
                f" (previous value={self.previous_val})."
                if self.previous_val is not None
                and self.val != self.previous_val
                else "."
            )
        )


class InvalidParameterTypeError(DvcLiveError):
    def __init__(self, val: Any):
        self.val = val
        super().__init__(f"Parameter type {type(val)} is not supported.")
