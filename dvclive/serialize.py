import csv
import json
import os
from collections import OrderedDict

from ruamel import yaml


def update_tsv(d: dict, path: str):
    _write_tsv(d, path, "a")


def write_tsv(d: dict, path: str):
    _write_tsv(d, path, "w")


def _write_tsv(d: dict, path: str, mode: str):
    assert isinstance(d, OrderedDict)

    exists = os.path.exists(path)

    with open(path, mode) as fd:
        writer = csv.writer(fd, delimiter="\t")
        if not exists or mode == "w":
            writer.writerow(list(d.keys()))

        writer.writerow(list(d.values()))


def write_json(d: dict, path: str):
    with open(path, "w") as fd:
        json.dump(d, fd)


def write_yaml(d: dict, path: str):
    with open(path, "w") as fd:
        yaml.safe_dump(d, fd)


def parse(path):
    _, extension = os.path.splitext(path)
    extension = extension.lower()
    if extension == ".tsv":
        with open(path, "r") as fd:
            reader = csv.DictReader(fd, delimiter="\t")
            return list(reader)
    elif extension == ".json":
        with open(path, "r") as fd:
            return json.load(fd)
    raise NotImplementedError


def read_logs(path):
    assert os.path.isdir(path)
    history = {}
    for p in os.listdir(path):
        history[p] = parse(os.path.join(path, p))
    latest = parse(path + ".json")
    return history, latest
