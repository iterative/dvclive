import json
from collections import OrderedDict

from dvclive.error import DvcLiveError


class YAMLError(DvcLiveError):
    pass


class YAMLFileCorruptedError(YAMLError):
    def __init__(self, path):
        super().__init__(path, "YAML file structure is corrupted")


def load_yaml(path, typ="safe"):
    from ruamel.yaml import YAML
    from ruamel.yaml import YAMLError as _YAMLError

    yaml = YAML(typ=typ)
    with open(path, encoding="utf-8") as fd:
        try:
            return yaml.load(fd.read())
        except _YAMLError:
            raise YAMLFileCorruptedError(path)


def _get_yaml():
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.default_flow_style = False

    # tell Dumper to represent OrderedDict as normal dict
    yaml_repr_cls = yaml.Representer
    yaml_repr_cls.add_representer(OrderedDict, yaml_repr_cls.represent_dict)
    return yaml


def dump_yaml(content, output_file):
    yaml = _get_yaml()
    with open(output_file, "w", encoding="utf-8") as fd:
        yaml.dump(content, fd)


def dump_json(content, output_file, indent=4, **kwargs):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=indent, **kwargs)
