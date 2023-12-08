import os

import pytest

from dvclive import Live
from dvclive.error import InvalidParameterTypeError
from dvclive.serialize import load_yaml


def test_cleanup_params(tmp_dir):
    dvclive = Live("logs")
    dvclive.log_param("param", 42)

    assert os.path.isfile(dvclive.params_file)

    dvclive = Live("logs")
    assert not os.path.exists(dvclive.params_file)


@pytest.mark.parametrize(
    ("param_name", "param_value"),
    [
        ("param_string", "value"),
        ("param_int", 42),
        ("param_float", 42.0),
        ("param_bool_true", True),
        ("param_bool_false", False),
        ("param_list", [1, 2, 3]),
        (
            "param_dict_simple",
            {"str": "value", "int": 42, "bool": True, "list": [1, 2, 3]},
        ),
        (
            "param_dict_nested",
            {
                "str": "value",
                "int": 42,
                "bool": True,
                "list": [1, 2, 3],
                "dict": {"nested-str": "value", "nested-int": 42},
            },
        ),
    ],
)
def test_log_param(tmp_dir, param_name, param_value):
    dvclive = Live()

    dvclive.log_param(param_name, param_value)

    s = load_yaml(dvclive.params_file)
    assert s[param_name] == param_value


def test_log_params(tmp_dir):
    dvclive = Live()
    params = {
        "param_string": "value",
        "param_int": 42,
        "param_float": 42.0,
        "param_bool_true": True,
        "param_bool_false": False,
    }

    dvclive.log_params(params)

    s = load_yaml(dvclive.params_file)
    assert s == params


@pytest.mark.parametrize("resume", [False, True])
def test_log_params_resume(tmp_dir, resume):
    dvclive = Live(resume=resume)
    dvclive.log_param("param", 42)

    dvclive = Live(resume=resume)
    assert ("param" in dvclive._params) == resume


def test_log_param_custom_obj(tmp_dir):
    dvclive = Live("logs")

    class Dummy:
        val = 42

    param_value = Dummy()

    with pytest.raises(InvalidParameterTypeError) as excinfo:
        dvclive.log_param("param_complex", param_value)
    assert "Dummy" in excinfo.value.args[0]
