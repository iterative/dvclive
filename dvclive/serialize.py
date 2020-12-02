import csv
import json
import os
from collections import OrderedDict

from ruamel import yaml


def update_tsv(d: OrderedDict, path: str):
    existed = os.path.exists(path)
    with open(path, "a") as fd:
        writer = csv.DictWriter(fd, d.keys(), delimiter="\t")

        if not existed:
            writer.writeheader()

        writer.writerow(d)


def write_json(d: dict, path: str):
    with open(path, "w") as fd:
        json.dump(d, fd)


def write_yaml(d: dict, path: str):
    with open(path, "w") as fd:
        yaml.safe_dump(d, fd)
