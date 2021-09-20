from PIL import Image

from .image_pil import ImagePIL


class ImageNumpy(ImagePIL):
    @staticmethod
    def could_log(val: object) -> bool:
        if val.__class__.__module__ == "numpy":
            return True
        return False

    def dump(self, val, step) -> None:
        val = Image.fromarray(val)
        super().dump(val, step)

    @property
    def summary(self):
        return {}
