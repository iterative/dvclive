import csv
import os
import time

from collections import OrderedDict

from dvclive.error import AlreadyLoggedError


class Scalar:

    def __init__(self, name, output_folder) -> None:
        self.name = name
        self.output_folder = output_folder
        self._step = None

        os.makedirs(os.path.dirname(self.output_plot_path), exist_ok=True)

    @staticmethod
    def could_log(o: object):
        if isinstance(o, (int, float)):
            return True
        return False

    @property
    def step(self):
        return self._step
    
    @step.setter
    def step(self, val):
        if val == self._step:
            raise AlreadyLoggedError(self.name, val)

    @property
    def output_plot_path(self):
        return os.path.join(self.output_folder, self.name + ".tsv")

    @property
    def output_summary_path(self):
        return self.output_folder + ".json"

    def dump(self, val, step):
        self.step = step
        ts = int(time.time() * 1000)
        d = OrderedDict([("timestamp", ts), ("step", self.step), (self.name, val)])

        existed = os.path.exists(self.output_plot_path)
        with open(self.output_plot_path, "a") as fobj:
            writer = csv.DictWriter(fobj, d.keys(), delimiter="\t")

            if not existed:
                writer.writeheader()

            writer.writerow(d)
        

        splitted_name = os.path.normpath(self.name).split(os.path.sep)
