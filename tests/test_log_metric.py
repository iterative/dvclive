import math

import numpy as np
import pytest

from dvclive import Live


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
