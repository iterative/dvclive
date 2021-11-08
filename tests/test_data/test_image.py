import numpy as np
import pytest
from PIL import Image

# pylint: disable=unused-argument
from dvclive import Live
from dvclive.data import Image as LiveImage


def test_PIL(tmp_dir):
    dvclive = Live()
    img = Image.new("RGB", (500, 500), (250, 250, 250))
    dvclive.log_image("image.png", img)

    assert (tmp_dir / dvclive.dir / LiveImage.subfolder / "image.png").exists()


def test_invalid_extension(tmp_dir):
    dvclive = Live()
    img = Image.new("RGB", (500, 500), (250, 250, 250))
    with pytest.raises(ValueError):
        dvclive.log_image("image.foo", img)


@pytest.mark.parametrize("shape", [(500, 500), (500, 500, 3), (500, 500, 4)])
def test_numpy(tmp_dir, shape):
    dvclive = Live()
    img = np.ones(shape, np.uint8) * 255
    dvclive.log_image("image.png", img)

    assert (tmp_dir / dvclive.dir / LiveImage.subfolder / "image.png").exists()


def test_step_formatting(tmp_dir):
    dvclive = Live()
    img = np.ones((500, 500, 3), np.uint8)
    for _ in range(3):
        dvclive.log_image("image.png", img)
        dvclive.next_step()

    for step in range(3):
        assert (
            tmp_dir
            / dvclive.dir
            / LiveImage.subfolder
            / str(step)
            / "image.png"
        ).exists()


def test_step_rename(tmp_dir, mocker):
    from pathlib import Path

    rename = mocker.spy(Path, "rename")
    dvclive = Live()
    img = np.ones((500, 500, 3), np.uint8)
    dvclive.log_image("image.png", img)
    assert (tmp_dir / dvclive.dir / LiveImage.subfolder / "image.png").exists()

    dvclive.next_step()

    assert not (
        tmp_dir / dvclive.dir / LiveImage.subfolder / "image.png"
    ).exists()
    assert (
        tmp_dir / dvclive.dir / LiveImage.subfolder / "0" / "image.png"
    ).exists()
    rename.assert_called_once_with(
        Path(dvclive.dir) / LiveImage.subfolder / "image.png",
        Path(dvclive.dir) / LiveImage.subfolder / "0" / "image.png",
    )


def test_cleanup(tmp_dir):
    dvclive = Live()
    img = np.ones((500, 500, 3), np.uint8)
    dvclive.log_image("image.png", img)

    assert (tmp_dir / dvclive.dir / LiveImage.subfolder / "image.png").exists()

    Live()

    assert not (tmp_dir / dvclive.dir / LiveImage.subfolder).exists()
