import base64
import csv
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


def nested_update(d, u):
    """Update values of a nested dictionnary of varying depth"""
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = nested_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def parse_tsv(path):
    with open(path, "r") as fd:
        reader = csv.DictReader(fd, delimiter="\t")
        return list(reader)


def to_base64_url(image_file):
    image_bytes = Path(image_file).read_bytes()
    base64_str = base64.b64encode(image_bytes).decode()
    return f"data:image;base64,{base64_str}"


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
    url = path if "Microsoft" in uname().release else path.resolve().as_uri()

    return webbrowser.open(url)


def env2bool(var, undefined=False):
    """
    undefined: return value if env var is unset
    """
    var = os.getenv(var, None)
    if var is None:
        return undefined
    return bool(re.search("1|y|yes|true", var, flags=re.I))
