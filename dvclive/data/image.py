from pathlib import Path

from .base import Data


class Image(Data):
    suffixes = [".jpg", ".jpeg", ".gif", ".png"]
    subfolder = "images"

    @property
    def no_step_output_path(self) -> Path:
        return self.output_folder / self.name

    @property
    def output_path(self) -> Path:
        if self._step is None:
            output_path = self.no_step_output_path
        else:
            output_path = self.output_folder / f"{self._step}" / self.name
        output_path.parent.mkdir(exist_ok=True, parents=True)
        return output_path

    @staticmethod
    def could_log(val: object) -> bool:
        if val.__class__.__module__ == "PIL.Image":
            return True
        if val.__class__.__module__ == "numpy":
            return True
        return False

    def first_step_dump(self) -> None:
        if self.no_step_output_path.exists():
            self.no_step_output_path.rename(self.output_path)

    def no_step_dump(self) -> None:
        self.step_dump()

    def step_dump(self) -> None:
        if self.val.__class__.__module__ == "numpy":
            from PIL import Image as ImagePIL

            _val = ImagePIL.fromarray(self.val)
        else:
            _val = self.val

        _val.save(self.output_path)

    @property
    def summary(self):
        return {self.name: str(self.output_path)}
