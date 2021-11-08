from dvclive.error import DvcLiveError

from .image_pil import ImagePIL


class ImageNumpy(ImagePIL):
    @staticmethod
    def could_log(val: object) -> bool:
        if val.__class__.__module__ == "numpy":
            return True
        return False

    def dump(self, val, step) -> None:
        try:
            from PIL import Image
        except ImportError as e:
            raise DvcLiveError(
                "'pillow' is required for logging images."
                " You can install it by running"
                " 'pip install pillow'"
            ) from e

        val = Image.fromarray(val)
        super().dump(val, step)
