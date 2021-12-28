import abc
from pathlib import Path
from typing import Optional

from dvclive.error import DataAlreadyLoggedError


class Data(abc.ABC):
    def __init__(self, name: str, output_folder: str) -> None:
        self.name = name
        self.output_folder: Path = Path(output_folder)
        self._step: Optional[int] = None
        self.val = None
        self._step_none_logged: bool = False

    @property
    def step(self) -> int:
        return self._step

    @step.setter
    def step(self, val: int) -> None:
        if not self._step_none_logged and val is None:
            self._step_none_logged = True
        elif val == self._step:
            raise DataAlreadyLoggedError(self.name, val)

        self._step = val

    @property
    @abc.abstractmethod
    def output_path(self) -> Path:
        pass

    @property
    @abc.abstractmethod
    def summary(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def could_log(val: object) -> bool:
        pass

    def dump(self, val, step):
        self.val = val
        self.step = step
