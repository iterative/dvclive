import os

import numpy as np
import pytest
from PIL import Image

import dvclive
from dvclive.data import ImageNumpy, ImagePIL


def test_PIL(tmp_dir):
    logger = dvclive.init()
    img = Image.new("RGB", (500, 500), (250, 250, 250))
    dvclive.log("image.png", img)

    assert (tmp_dir / logger.dir / ImagePIL.subdir / "image.png").exists()


def test_invalid_extension(tmp_dir):
    dvclive.init()
    img = Image.new("RGB", (500, 500), (250, 250, 250))
    with pytest.raises(ValueError):
        dvclive.log("image.foo", img)


@pytest.mark.parametrize("shape", [(500, 500), (500, 500, 3), (500, 500, 4)])
def test_numpy(tmp_dir, shape):
    logger = dvclive.init()
    img = np.ones(shape, np.uint8) * 255
    dvclive.log("image.png", img)

    assert (tmp_dir / logger.dir / ImageNumpy.subdir / "image.png").exists()


@pytest.mark.parametrize(
    "pattern", ["image_{step}.png", str(os.path.join("{step}", "image.png"))]
)
def test_step_formatting(tmp_dir, pattern):
    logger = dvclive.init()
    img = np.ones((500, 500, 3), np.uint8)
    for _ in range(3):
        dvclive.log(pattern, img)
        dvclive.next_step()

    for step in range(3):
        assert (
            tmp_dir / logger.dir / ImagePIL.subdir / pattern.format(step=step)
        ).exists()
