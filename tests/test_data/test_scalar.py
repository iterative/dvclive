import json

import numpy as np
import pytest

# pylint: disable=unused-argument
from dvclive import Live
from dvclive.data.scalar import Scalar
from dvclive.data.utils import NUMPY_INTS, NUMPY_SCALARS
from dvclive.utils import parse_tsv


@pytest.mark.parametrize("dtype", NUMPY_SCALARS)
def test_numpy(tmp_dir, dtype):
    scalar = np.random.rand(1).astype(dtype)[0]
    live = Live()

    live.log("scalar", scalar)
    live.next_step()

    parsed = json.loads((tmp_dir / live.summary_path).read_text())
    assert isinstance(parsed["scalar"], int if dtype in NUMPY_INTS else float)
    tsv_file = tmp_dir / live.dir / Scalar.subfolder / "scalar.tsv"
    tsv_val = parse_tsv(tsv_file)[0]["scalar"]
    assert tsv_val == str(scalar)


def test_name_with_dot(tmp_dir):
    """Regression test for #284"""
    live = Live()

    live.log("scalar.foo.bar", 1.0)
    live.next_step()

    tsv_file = tmp_dir / live.dir / Scalar.subfolder / "scalar.foo.bar.tsv"
    assert tsv_file.exists()
    tsv_val = parse_tsv(tsv_file)[0]["scalar.foo.bar"]
    assert tsv_val == "1.0"
