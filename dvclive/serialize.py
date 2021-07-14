import csv
import json
import os
from collections import OrderedDict


def update_tsv(d: OrderedDict, path: str):
    existed = os.path.exists(path)
    with open(path, "a") as fobj:
        writer = csv.DictWriter(fobj, d.keys(), delimiter="\t")

        if not existed:
            writer.writeheader()

        writer.writerow(d)


def write_json(d: dict, path: str):
    with open(path, "w") as fobj:
        json.dump(d, fobj, indent=4)
