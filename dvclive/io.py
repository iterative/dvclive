import csv
import os
from collections import OrderedDict


def update_tsv(d: dict, path: str):
    # TODO what if d keys are not in order?
    assert isinstance(d, OrderedDict)

    exists = os.path.exists(path)

    with open(path, "a") as fd:
        writer = csv.writer(fd, delimiter="\t")
        if not exists:
            writer.writerow(list(d.keys()))

        writer.writerow(list(d.values()))
