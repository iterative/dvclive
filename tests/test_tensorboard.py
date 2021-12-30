import os

import gorilla
import tensorflow as tf

from dvclive.tensorboard import patch_tensorboard
from tests.test_main import _parse_json

# pylint: disable=unused-argument, no-value-for-parameter


def test_patch_tensorboard(tmp_dir, mocker):
    scalar = mocker.spy(tf.summary, "scalar")
    image = mocker.spy(tf.summary, "image")

    patches = patch_tensorboard()

    tf.summary.scalar("m", 0.5)
    tf.summary.image("image", [tf.zeros(shape=[8, 8, 1], dtype=tf.uint8)])

    assert not scalar.call_count
    assert not image.call_count

    summary = _parse_json("dvclive.json")
    image_path = os.path.join("dvclive", "images", "image.png")
    assert summary["m"] == 0.5
    assert os.path.exists(image_path)

    for patch in patches:
        gorilla.revert(patch)


def test_patch_tensorboard_no_override(tmp_dir, mocker):
    scalar = mocker.spy(tf.summary, "scalar")
    image = mocker.spy(tf.summary, "image")

    patches = patch_tensorboard(override=False)

    tf.summary.scalar("m", 0.5)
    tf.summary.image("image", [tf.zeros(shape=[8, 8, 1], dtype=tf.uint8)])

    assert scalar.call_count
    assert image.call_count

    summary = _parse_json("dvclive.json")
    image_path = os.path.join("dvclive", "images", "image.png")
    assert summary["m"] == 0.5
    assert os.path.exists(image_path)

    for patch in patches:
        gorilla.revert(patch)


def test_patch_tensorboard_live_args(tmp_dir, mocker):
    scalar = mocker.spy(tf.summary, "scalar")
    image = mocker.spy(tf.summary, "image")

    patches = patch_tensorboard(path="logs")

    tf.summary.scalar("m", 0.5)
    tf.summary.image("image", [tf.zeros(shape=[8, 8, 1], dtype=tf.uint8)])

    assert not scalar.call_count
    assert not image.call_count

    summary = _parse_json("logs.json")
    image_path = os.path.join("logs", "images", "image.png")
    assert summary["m"] == 0.5
    assert os.path.exists(image_path)

    for patch in patches:
        gorilla.revert(patch)
