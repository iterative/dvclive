import csv
import os
import time
from pathlib import Path

from dvclive.utils import nested_set

from .base import Data
from .utils import NUMPY_SCALARS


class Metric(Data):
    suffixes = [".csv", ".tsv"]
    subfolder = "metrics"

    @staticmethod
    def could_log(val: object) -> bool:
        if isinstance(val, (int, float)):
            return True
        if val.__class__.__module__ == "numpy":
            if val.__class__.__name__ in NUMPY_SCALARS:
                return True
        return False

    @property
    def output_path(self) -> Path:
        _path = Path(f"{self.output_folder / self.name}.tsv")
        _path.parent.mkdir(exist_ok=True, parents=True)
        return _path

    def dump(self, val, **kwargs) -> None:
        row = {}
        if kwargs.get("timestamp", False):
            row["timestamp"] = int(time.time() * 1000)
        row["step"] = self.step
        row[self.name] = val

        existed = self.output_path.exists()
        with open(self.output_path, "a", encoding="utf-8", newline="") as fobj:
            writer = csv.DictWriter(
                fobj, row.keys(), delimiter="\t", lineterminator="\n"
            )
            if not existed:
                writer.writeheader()
            writer.writerow(row)

    def to_summary(self, val):
        d = {}
        nested_set(d, os.path.normpath(self.name).split(os.path.sep), val)
        return d
