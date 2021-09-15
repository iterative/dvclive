import csv
import json
import os
import time

from collections import OrderedDict
from pathlib import Path
from typing import Optional

from dvclive.error import AlreadyLoggedError
from dvclive.utils import nested_set


class Scalar:

    def __init__(self, name: str, output_folder: str) -> None:
        self.name = name
        self.output_folder: Path = Path(output_folder)
        self._step: Optional[int] = None
        self.val = None

    @staticmethod
    def could_log(val: object) -> bool:
        if isinstance(val, (int, float)):
            return True
        return False

    @property
    def step(self) -> int:
        return self._step
    
    @step.setter
    def step(self, val: int) -> None:
        if val == self._step:
            raise AlreadyLoggedError(self.name, val)
        self._step = val

    @property
    def output_path(self) -> Path:
        _path = self.output_folder / self.name
        _path.parent.mkdir(exist_ok=True, parents=True)
        return _path.with_suffix(".tsv")

    def dump(self, val, step):
        self.val = val
        self.step = step

        ts = int(time.time() * 1000)
        d = OrderedDict([("timestamp", ts), ("step", self.step), (self.name, val)])

        existed = os.path.exists(self.output_path)
        with open(self.output_path, "a") as fobj:
            writer = csv.DictWriter(fobj, d.keys(), delimiter="\t")

            if not existed:
                writer.writeheader()

            writer.writerow(d)
    
    @property
    def summary(self):
        d = {}
        nested_set(
            d,
            os.path.normpath(self.name).split(os.path.sep),
            self.val
        )
        return d
