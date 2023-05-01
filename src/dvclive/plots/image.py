from pathlib import Path, PurePath

from .base import Data


class Image(Data):
    suffixes = [".jpg", ".jpeg", ".gif", ".png"]
    subfolder = "images"

    @property
    def output_path(self) -> Path:
        _path = self.output_folder / self.name
        _path.parent.mkdir(exist_ok=True, parents=True)
        return _path

    @staticmethod
    def could_log(val: object) -> bool:
        acceptable = {
            ("numpy", "ndarray"),
            ("PIL.Image", "Image"),
        }
        for cls in type(val).mro():
            if (cls.__module__, cls.__name__) in acceptable:
                return True
        if isinstance(val, (PurePath, str)):
            return True
        return False

    def dump(self, val, **kwargs) -> None:  # noqa: ARG002
        if val.__class__.__module__ == "numpy":
            from PIL import Image as ImagePIL

            pil_image = ImagePIL.fromarray(val)
        else:
            pil_image = val
        pil_image.save(self.output_path)
