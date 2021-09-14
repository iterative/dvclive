import csv
import os
import time

from collections import OrderedDict


class Scalar:

    def __init__(self, name, val, step, output_folder) -> None:
        self.name = name
        self.val = val
        self.step = step
        self.output_folder = output_folder

    def __eq__(self, o: object) -> bool:
        if isinstance(o, (int, float)):
            return True
        return False

    @property
    def output_plot_path(self):
        return os.path.join(self.output_folder, self.name + ".tsv")

    @property
    def output_summary_path(self):
        return self.output_folder + ".json"

    def dump(self, output_folder, step):
        output_path = os.path.join(output_folder, self.output_plot_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        ts = int(time.time() * 1000)
        d = OrderedDict([("timestamp", ts), ("step", step), (self.name, self.val)])

        existed = os.path.exists(path)
        with open(path, "a") as fobj:
            writer = csv.DictWriter(fobj, d.keys(), delimiter="\t")

            if not existed:
                writer.writeheader()

            writer.writerow(d)
        
        splitted_name = os.path.normpath(self.name).split(os.path.sep)



