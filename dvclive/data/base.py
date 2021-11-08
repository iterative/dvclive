import abc
from pathlib import Path
from typing import Optional

from dvclive.error import DataAlreadyLoggedError


class Data(abc.ABC):
    def __init__(self, name: str, output_folder: str) -> None:
        self.name = name
        self.output_folder: Path = Path(output_folder) / self.subfolder
        self._step: Optional[int] = None
        self.val = None
        self._step_none_logged: bool = False
        self._dump_kwargs = None

    @property
    def step(self) -> int:
        return self._step

    @step.setter
    def step(self, val: int) -> None:
        if self._step_none_logged and val == self._step:
            raise DataAlreadyLoggedError(self.name, val)

        self._step = val

    @property
    @abc.abstractmethod
    def output_path(self) -> Path:
        pass

    @property
    @abc.abstractmethod
    def no_step_output_path(self) -> Path:
        pass

    @property
    @abc.abstractmethod
    def subfolder(self):
        pass

    @property
    @abc.abstractmethod
    def summary(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def could_log(val: object) -> bool:
        pass

    def dump(self, val, step, **kwargs):
        self.val = val
        self.step = step
        self._dump_kwargs = kwargs
        if not self._step_none_logged and step is None:
            self._step_none_logged = True
            self.no_step_dump()
        elif step == 0:
            self.first_step_dump()
        else:
            self.step_dump()

    @abc.abstractmethod
    def first_step_dump(self) -> None:
        pass

    @abc.abstractmethod
    def no_step_dump(self) -> None:
        pass

    @abc.abstractmethod
    def step_dump(self) -> None:
        pass
