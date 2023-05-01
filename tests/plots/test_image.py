import numpy as np
import pytest
from PIL import Image

from dvclive import Live
from dvclive.plots import Image as LiveImage


# From https://stackoverflow.com/questions/5165317/how-can-i-extend-image-class
class ExtendedImage(Image.Image):
    def __init__(self, img):
        self._img = img

    def __getattr__(self, key):
        return getattr(self._img, key)


def test_pil(tmp_dir):
    live = Live()
    img = Image.new("RGB", (10, 10), (250, 250, 250))
    live.log_image("image.png", img)

    assert (tmp_dir / live.plots_dir / LiveImage.subfolder / "image.png").exists()


def test_invalid_extension(tmp_dir):
    live = Live()
    img = Image.new("RGB", (10, 10), (250, 250, 250))
    with pytest.raises(ValueError, match="unknown file extension"):
        live.log_image("image.foo", img)


@pytest.mark.parametrize("shape", [(10, 10), (10, 10, 3), (10, 10, 4)])
def test_numpy(tmp_dir, shape):
    from PIL import Image as ImagePIL

    live = Live()
    img = np.ones(shape, np.uint8) * 255
    live.log_image("image.png", img)

    img_path = tmp_dir / live.plots_dir / LiveImage.subfolder / "image.png"
    assert img_path.exists()

    val = np.asarray(ImagePIL.open(img_path))
    assert np.array_equal(val, img)


def test_path(tmp_dir):
    import numpy as np
    from PIL import Image as ImagePIL

    live = Live()
    image_data = np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8)
    pil_image = ImagePIL.fromarray(image_data)
    image_path = tmp_dir / "temp.png"
    pil_image.save(image_path)

    live = Live()
    live.log_image("foo.png", image_path)
    live.end()

    plot_file = tmp_dir / live.plots_dir / "images" / "foo.png"
    assert plot_file.exists()

    val = np.asarray(ImagePIL.open(plot_file))
    assert np.array_equal(val, image_data)


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

    assert (tmp_dir / live.plots_dir / LiveImage.subfolder / "image.png").exists()

    Live()

    assert not (tmp_dir / live.plots_dir / LiveImage.subfolder).exists()


def test_custom_class(tmp_dir):
    live = Live()
    img = Image.new("RGB", (10, 10), (250, 250, 250))
    extended_img = ExtendedImage(img)
    live.log_image("image.png", extended_img)

    assert (tmp_dir / live.plots_dir / LiveImage.subfolder / "image.png").exists()
