import math

import numpy as np
import pytest

from dvclive import Live
from dvclive.error import InvalidDataTypeError


@pytest.mark.parametrize("invalid_type", [{0: 1}, [0, 1], (0, 1)])
def test_invalid_metric_type(tmp_dir, invalid_type):
    dvclive = Live()

    with pytest.raises(
        InvalidDataTypeError,
        match=f"Data 'm' has not supported type {type(invalid_type)}",
    ):
        dvclive.log_metric("m", invalid_type)


@pytest.mark.parametrize(
    ("val"),
    [math.inf, math.nan, np.nan, np.inf],
)
def test_log_metric_inf_nan(tmp_dir, val):
    with Live() as live:
        live.log_metric("metric", val)
    assert live.summary["metric"] == str(val)


def test_log_metic_str(tmp_dir):
    with Live() as live:
        live.log_metric("metric", "foo")
    assert live.summary["metric"] == "foo"
