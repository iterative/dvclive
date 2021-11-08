from pathlib import Path

from .base import Data


class ImagePIL(Data):
    suffixes = [".jpg", ".jpeg", ".gif", ".png"]

    @staticmethod
    def could_log(val: object) -> bool:
        if val.__class__.__module__ == "PIL.Image":
            return True
        return False

    @property
    def output_path(self) -> Path:
        if Path(self.name).suffix not in self.suffixes:
            raise ValueError(
                f"Invalid image suffix '{Path(self.name).suffix}'"
                f" Must be one of {self.suffixes}"
            )
        return self.output_folder / "{step}" / self.name

    def dump(self, val, step) -> None:
        super().dump(val, step)
        output_path = Path(str(self.output_path).format(step=step))
        output_path.parent.mkdir(exist_ok=True, parents=True)

        val.save(output_path)

    @property
    def summary(self):
        return {self.name: str(self.output_path).format(step=self.step)}
