import numpy as np
import pytest
from PIL import Image

# pylint: disable=unused-argument
from dvclive import Live
from dvclive.plots import Image as LiveImage


def test_PIL(tmp_dir):
    live = Live()
    img = Image.new("RGB", (10, 10), (250, 250, 250))
    live.log_image("image.png", img)

    assert (
        tmp_dir / live.plots_dir / LiveImage.subfolder / "image.png"
    ).exists()


def test_invalid_extension(tmp_dir):
    live = Live()
    img = Image.new("RGB", (10, 10), (250, 250, 250))
    with pytest.raises(ValueError):
        live.log_image("image.foo", img)


@pytest.mark.parametrize("shape", [(10, 10), (10, 10, 3), (10, 10, 4)])
def test_numpy(tmp_dir, shape):
    live = Live()
    img = np.ones(shape, np.uint8) * 255
    live.log_image("image.png", img)

    assert (
        tmp_dir / live.plots_dir / LiveImage.subfolder / "image.png"
    ).exists()


def test_override_on_step(tmp_dir):
    live = Live()

    zeros = np.zeros((2, 2, 3), np.uint8)
    live.log_image("image.png", zeros)

    live.next_step()

    ones = np.ones((2, 2, 3), np.uint8)
    live.log_image("image.png", ones)

    img_path = tmp_dir / live.plots_dir / LiveImage.subfolder / "image.png"
    assert np.array_equal(np.array(Image.open(img_path)), ones)


def test_cleanup(tmp_dir):
    live = Live()
    img = np.ones((10, 10, 3), np.uint8)
    live.log_image("image.png", img)

    assert (
        tmp_dir / live.plots_dir / LiveImage.subfolder / "image.png"
    ).exists()

    Live()

    assert not (tmp_dir / live.plots_dir / LiveImage.subfolder).exists()
