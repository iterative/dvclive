from typing import Any


class DvcLiveError(Exception):
    pass


class InvalidDataTypeError(DvcLiveError):
    def __init__(self, name, val):
        self.name = name
        self.val = val
        super().__init__(f"Data '{name}' has not supported type {val}")


class InvalidPlotTypeError(DvcLiveError):
    def __init__(self, name):
        from .plots import SKLEARN_PLOTS

        self.name = name
        super().__init__(
            f"Plot type '{name}' is not supported."
            f"\nSupported types are: {list(SKLEARN_PLOTS)}"
        )


class DataAlreadyLoggedError(DvcLiveError):
    def __init__(self, name, step):
        self.name = name
        self.val = step
        super().__init__(
            f"Data '{name}' has already been logged with step '{step}'"
        )


class InvalidParameterTypeError(DvcLiveError):
    def __init__(self, val: Any):
        self.val = val
        super().__init__(f"Parameter type {type(val)} is not supported.")
