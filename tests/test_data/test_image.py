import os

import numpy as np
import pytest
from PIL import Image

# pylint: disable=unused-argument
from dvclive import Live
from tests.test_main import _parse_json


def test_PIL(tmp_dir):
    dvclive = Live()
    img = Image.new("RGB", (500, 500), (250, 250, 250))
    dvclive.log("image.png", img)

    assert (tmp_dir / dvclive.dir / "image.png").exists()
    summary = _parse_json("dvclive.json")

    assert summary["image.png"] == os.path.join(dvclive.dir, "image.png")


def test_invalid_extension(tmp_dir):
    dvclive = Live()
    img = Image.new("RGB", (500, 500), (250, 250, 250))
    with pytest.raises(ValueError):
        dvclive.log("image.foo", img)


@pytest.mark.parametrize("shape", [(500, 500), (500, 500, 3), (500, 500, 4)])
def test_numpy(tmp_dir, shape):
    dvclive = Live()
    img = np.ones(shape, np.uint8) * 255
    dvclive.log("image.png", img)

    assert (tmp_dir / dvclive.dir / "image.png").exists()


@pytest.mark.parametrize(
    "pattern", ["image_{step}.png", str(os.path.join("{step}", "image.png"))]
)
def test_step_formatting(tmp_dir, pattern):
    dvclive = Live()
    img = np.ones((500, 500, 3), np.uint8)
    for _ in range(3):
        dvclive.log(pattern, img)
        dvclive.next_step()

    for step in range(3):
        assert (tmp_dir / dvclive.dir / pattern.format(step=step)).exists()

    summary = _parse_json("dvclive.json")

    assert summary[pattern] == os.path.join(
        dvclive.dir, pattern.format(step=step)
    )
