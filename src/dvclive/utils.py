import csv
import json
import os
import re
import shutil
import webbrowser
from pathlib import Path
from platform import uname
from typing import Union

StrPath = Union[str, Path]


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
        return None

    wrapper.has_run = False
    return wrapper


@run_once
def open_file_in_browser(file) -> bool:
    path = Path(file)
    url = str(path) if "Microsoft" in uname().release else path.resolve().as_uri()

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
        for split in ("train", "eval"):
            metric_name = metric_name.replace(f"{split}_", f"{split}/")

    elif framework == "dvclive.keras":
        if "val_" in metric_name:
            metric_name = metric_name.replace("val_", "eval/")
        else:
            metric_name = f"train/{metric_name}"

    elif framework == "dvclive.lightning":
        parts = metric_name.split("_")
        split, freq, rest = None, None, None
        if parts[0] in ["train", "val", "test"]:
            split = parts.pop(0)
            # Only set freq if split was also found.
            # Otherwise we end up conflicting with out internal `step` property.
            if parts[-1] in ["step", "epoch"]:
                freq = parts.pop()
        rest = "_".join(parts)
        parts = [part for part in (split, freq, rest) if part]
        metric_name = "/".join(parts)

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

    metrics_path = Path(live.plots_dir) / Metric.subfolder
    history = {}
    for suffix in Metric.suffixes:
        for scalar_file in metrics_path.rglob(f"*{suffix}"):
            history[str(scalar_file)] = parse_tsv(scalar_file)
    latest = parse_json(live.metrics_file)
    return history, latest


def matplotlib_installed() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def inside_colab() -> bool:
    try:
        from google import colab  # noqa: F401
    except ImportError:
        return False
    return True


def inside_notebook() -> bool:
    if inside_colab():
        return True

    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
    except NameError:
        return False

    if shell == "ZMQInteractiveShell":
        import IPython

        return IPython.__version__ >= "6.0.0"
    return False


def clean_and_copy_into(src: StrPath, dst: StrPath) -> str:
    Path(dst).mkdir(exist_ok=True)

    basename = os.path.basename(os.path.normpath(src))
    dst_path = Path(os.path.join(dst, basename))

    if dst_path.is_file() or dst_path.is_symlink():
        dst_path.unlink()
    elif dst_path.is_dir():
        shutil.rmtree(dst_path)

    if os.path.isdir(src):
        shutil.copytree(src, dst_path)
    else:
        shutil.copy2(src, dst_path)

    return str(dst_path)
