import numpy as np
import pytest
from PIL import Image

# pylint: disable=unused-argument
from dvclive import Live
from dvclive.data import Image as LiveImage


def test_PIL(tmp_dir):
    live = Live()
    img = Image.new("RGB", (500, 500), (250, 250, 250))
    live.log_image("image.png", img)

    assert (
        tmp_dir / live.plots_path / LiveImage.subfolder / "image.png"
    ).exists()


def test_invalid_extension(tmp_dir):
    live = Live()
    img = Image.new("RGB", (500, 500), (250, 250, 250))
    with pytest.raises(ValueError):
        live.log_image("image.foo", img)


@pytest.mark.parametrize("shape", [(500, 500), (500, 500, 3), (500, 500, 4)])
def test_numpy(tmp_dir, shape):
    live = Live()
    img = np.ones(shape, np.uint8) * 255
    live.log_image("image.png", img)

    assert (
        tmp_dir / live.plots_path / LiveImage.subfolder / "image.png"
    ).exists()


def test_step_formatting(tmp_dir):
    live = Live()
    img = np.ones((500, 500, 3), np.uint8)
    for _ in range(3):
        live.log_image("image.png", img)
        live.next_step()

    for step in range(3):
        assert (
            tmp_dir
            / live.plots_path
            / LiveImage.subfolder
            / str(step)
            / "image.png"
        ).exists()


def test_step_rename(tmp_dir, mocker):
    from pathlib import Path

    rename = mocker.spy(Path, "rename")
    live = Live()
    img = np.ones((500, 500, 3), np.uint8)
    live.log_image("image.png", img)
    assert (
        tmp_dir / live.plots_path / LiveImage.subfolder / "image.png"
    ).exists()

    live.next_step()

    assert not (
        tmp_dir / live.plots_path / LiveImage.subfolder / "image.png"
    ).exists()
    assert (
        tmp_dir / live.plots_path / LiveImage.subfolder / "0" / "image.png"
    ).exists()
    rename.assert_called_once_with(
        Path(live.plots_path) / LiveImage.subfolder / "image.png",
        Path(live.plots_path) / LiveImage.subfolder / "0" / "image.png",
    )


def test_cleanup(tmp_dir):
    live = Live()
    img = np.ones((500, 500, 3), np.uint8)
    live.log_image("image.png", img)

    assert (
        tmp_dir / live.plots_path / LiveImage.subfolder / "image.png"
    ).exists()

    Live()

    assert not (tmp_dir / live.plots_path / LiveImage.subfolder).exists()
