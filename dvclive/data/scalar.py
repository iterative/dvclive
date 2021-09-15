import csv
import json
import os
import time

from collections import OrderedDict

from dvclive.error import AlreadyLoggedError
from dvclive.utils import nested_set


class Scalar:

    def __init__(self, name: str, output_folder: str) -> None:
        self.name = name
        self.output_folder = output_folder
        self._step: int = -1

        os.makedirs(os.path.dirname(self.output_plot_path), exist_ok=True)

    @staticmethod
    def could_log(o: object) -> bool:
        if isinstance(o, (int, float)):
            return True
        return False

    @property
    def step(self) -> int:
        return self._step
    
    @step.setter
    def step(self, val: int) -> None:
        if val == self._step:
            raise AlreadyLoggedError(self.name, val)

    @property
    def output_plot_path(self) -> str:
        return os.path.join(self.output_folder, self.name + ".tsv")

    @property
    def output_summary_path(self) -> str:
        return self.output_folder + ".json"

    def dump(self, val, step: int, summary: bool = False):
        self.step = step

        ts = int(time.time() * 1000)
        d = OrderedDict([("timestamp", ts), ("step", self.step), (self.name, val)])

        existed = os.path.exists(self.output_plot_path)
        with open(self.output_plot_path, "a") as fobj:
            writer = csv.DictWriter(fobj, d.keys(), delimiter="\t")

            if not existed:
                writer.writeheader()

            writer.writerow(d)
        
        if summary:
            splitted_name = os.path.normpath(self.name).split(os.path.sep)

            with open(self.output_summary_path) as f:
                summary_data = json.load(f)

            nested_set(
                summary_data,
                splitted_name,
                val
            )

            with open(self.output_summary_path, "w") as f:
                json.dump(summary_data, f, indent=4)
