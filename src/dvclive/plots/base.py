import abc
from pathlib import Path

from dvclive.error import DataAlreadyLoggedError


class Data(abc.ABC):
    def __init__(self, name: str, output_folder: str) -> None:
        self.name = name
        self.output_folder: Path = Path(output_folder) / self.subfolder
        self._step: int = -1

    @property
    def step(self) -> int:
        return self._step

    @step.setter
    def step(self, val: int) -> None:
        if val == self._step:
            raise DataAlreadyLoggedError(self.name, val)
        self._step = val

    @property
    @abc.abstractmethod
    def output_path(self) -> Path:
        pass

    @property
    @abc.abstractmethod
    def subfolder(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def could_log(val) -> bool:
        pass

    @abc.abstractmethod
    def dump(self, val, **kwargs):
        pass
