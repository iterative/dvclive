from pathlib import Path

from .base import Data


class Image(Data):
    suffixes = [".jpg", ".jpeg", ".gif", ".png"]

    @staticmethod
    def could_log(val: object) -> bool:
        if val.__class__.__module__ == "PIL.Image":
            return True
        if val.__class__.__module__ == "numpy":
            return True
        return False

    @property
    def output_path(self) -> Path:
        if Path(self.name).suffix not in self.suffixes:
            raise ValueError(
                f"Invalid image suffix '{Path(self.name).suffix}'"
                f" Must be one of {self.suffixes}"
            )
        if self._step is None:
            output_path = self.output_folder / self.name
        else:
            output_path = self.output_folder / f"{self._step}" / self.name
        output_path.parent.mkdir(exist_ok=True, parents=True)
        return output_path

    def dump(self, val, step) -> None:
        if self._step_none_logged and self._step is None:
            super().dump(val, step)
            step_none_path = self.output_folder / self.name
            if step_none_path.exists():
                step_none_path.rename(self.output_path)
        else:
            super().dump(val, step)
            if val.__class__.__module__ == "numpy":
                from PIL import Image as ImagePIL

                val = ImagePIL.fromarray(val)

            val.save(self.output_path)

    @property
    def summary(self):
        return {self.name: str(self.output_path)}
