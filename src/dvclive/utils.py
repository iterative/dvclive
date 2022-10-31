import csv
import json
import os
import re
import webbrowser
from collections.abc import Mapping
from pathlib import Path
from platform import uname


def nested_set(d, keys, value):
    """Set d[keys[0]]...[keys[-1]] to `value`.

    Example:
    >>> d = {}
    >>> nested_set(d, ['person', 'address', 'city'], 'New York')
    >>> d
    {'person': {'address': {'city': 'New York'}}}

    From:
    https://stackoverflow.com/questions/13687924/setting-a-value-in-a-nested-python-dictionary-given-a-list-of-indices-and-value
    """
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value
    return d


def nested_update(d, u):
    """Update values of a nested dictionary of varying depth"""
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = nested_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


@run_once
def open_file_in_browser(file) -> bool:
    path = Path(file)
    url = (
        str(path)
        if "Microsoft" in uname().release
        else path.resolve().as_uri()
    )

    return webbrowser.open(url)


def env2bool(var, undefined=False):
    """
    undefined: return value if env var is unset
    """
    var = os.getenv(var, None)
    if var is None:
        return undefined
    return bool(re.search("1|y|yes|true", var, flags=re.I))


def standardize_metric_name(metric_name: str, framework: str) -> str:
    """Map framework-specific format to DVCLive standard.

    Use `{split}/` as prefix in order to separate by subfolders.
    Use `{train|eval}` as split name.
    """
    if framework == "dvclive.fastai":
        metric_name = metric_name.replace("train_", "train/")
        metric_name = metric_name.replace("valid_", "eval/")

    elif framework == "dvclive.huggingface":
        for split in {"train", "eval"}:
            metric_name = metric_name.replace(f"{split}_", f"{split}/")

    elif framework == "dvclive.keras":
        if "val_" in metric_name:
            metric_name = metric_name.replace("val_", "eval/")
        else:
            metric_name = f"train/{metric_name}"

    elif framework == "dvclive.lightning":
        parts = metric_name.split("_")
        if len(parts) > 2:
            split, *rest, freq = parts
            metric_name = f"{split}/{freq}/{'_'.join(rest)}"

    return metric_name


def parse_tsv(path):
    with open(path, encoding="utf-8", newline="") as fd:
        reader = csv.DictReader(fd, delimiter="\t")
        return list(reader)


def parse_json(path):
    with open(path, encoding="utf-8") as fd:
        return json.load(fd)


def parse_metrics(live):
    from .plots import Metric

    plots_path = Path(live.plots_dir)
    history = {}
    for suffix in Metric.suffixes:
        for scalar_file in plots_path.rglob(f"*{suffix}"):
            history[str(scalar_file)] = parse_tsv(scalar_file)
    latest = parse_json(live.metrics_file)
    return history, latest


def matplotlib_installed() -> bool:
    # noqa pylint: disable=unused-import
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True
