from pathlib import Path

from PIL.Image import Image
from .base import Data


class ImagePIL(Data):
    subdir = "images"
    suffixes = [".jpg", ".jpeg", ".gif", ".png"]

    @staticmethod
    def could_log(val: object) -> bool:
        if isinstance(val, Image.Image):
            return True
        return False

    @property
    def output_path(self) -> Path:
        if self.name.suffix not in self.suffixes:
            raise ValueError(
                f"Invalid image suffix {self.name.suffix}."
                f" Must be one of {self.suffixes}"
            )
        return self.output_folder / self.subdir / self.name

    def dump(self, val, step) -> None:
        super().dump(val, step)
        output_path = Path(str(self.output_path).format(step=step))
        output_path.parent.mkdir(exist_ok=True, parents=True)

        val.save(output_path)

    @property
    def summary(self):
        return {}
