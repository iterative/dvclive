from typing import Any


class DvcLiveError(Exception):
    pass


class InvalidDataTypeError(DvcLiveError):
    def __init__(self, name, val, subfield=None):
        self.name = name
        self.val = val
        self.subfield = subfield
        if subfield:
            super().__init__(
                f"Data '{name}' has not supported type {val} for subfield {subfield}"
            )
        else:
            super().__init__(f"Data '{name}' has not supported type {val}")


class InvalidDvcyamlError(DvcLiveError):
    def __init__(self):
        super().__init__("`dvcyaml` path must have filename 'dvc.yaml'")


class InvalidPlotTypeError(DvcLiveError):
    def __init__(self, name):
        from .plots import SKLEARN_PLOTS

        self.name = name
        super().__init__(
            f"Plot type '{name}' is not supported."
            f"\nSupported types are: {list(SKLEARN_PLOTS)}"
        )


class InvalidParameterTypeError(DvcLiveError):
    def __init__(self, msg: Any):
        super().__init__(msg)


class InvalidReportModeError(DvcLiveError):
    def __init__(self, val):
        super().__init__(
            f"`report` can only be `None`, `auto`, `html`, `notebook` or `md`. "
            f"Got {val} instead."
        )


class InvalidSameSizeError(DvcLiveError):
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y
        super().__init__(f"Data '{name}': '{x}' and '{y}' must have the same length")


class MissingFieldError(DvcLiveError):
    def __init__(self, name, dictionary, field):
        self.name = name
        self.dictionary = dictionary
        self.field = field
        super().__init__(
            f"Data '{name}': {dictionary} does not contain the '{field}' field"
        )
