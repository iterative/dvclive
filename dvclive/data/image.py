from pathlib import Path

from .base import Data, _is_np, _is_tf


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
        if _is_np(val):
            return True
        if _is_tf(val) and val.ndim in [2, 3]:
            return True
        return False

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, x):
        if _is_np(x) or _is_tf(x):
            from PIL import Image as ImagePIL

            if _is_tf(x):
                x = x.numpy().squeeze()
            self._val = ImagePIL.fromarray(x)
        else:
            self._val = x

    def first_step_dump(self) -> None:
        if self.no_step_output_path.exists():
            self.no_step_output_path.rename(self.output_path)

    def no_step_dump(self) -> None:
        self.step_dump()

    def step_dump(self) -> None:
        self.val.save(self.output_path)

    @property
    def summary(self):
        return {self.name: str(self.output_path)}
