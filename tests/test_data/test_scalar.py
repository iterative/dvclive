import tensorflow as tf

from dvclive import Live
from tests.test_main import _parse_json

# pylint: disable=unused-argument


def test_tensorflow(tmp_dir):
    dvclive = Live()
    dvclive.log("int", tf.constant(1))
    dvclive.log("float", tf.constant(1.5))

    summary = _parse_json("dvclive.json")

    assert isinstance(summary["int"], int)
    assert summary["int"] == 1
    assert isinstance(summary["float"], float)
    assert summary["float"] == 1.5
