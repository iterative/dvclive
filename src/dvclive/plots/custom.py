from pathlib import Path
from typing import Optional

from dvclive.serialize import dump_json

from .base import Data


class CustomPlot(Data):
    suffixes = (".json",)
    subfolder = "custom"

    def __init__(
        self,
        name: str,
        output_folder: str,
        x: str,
        y: str,
        template: Optional[str],
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
    ) -> None:
        super().__init__(name, output_folder)
        self.name = self.name.replace(".json", "")
        config = {
            "template": template,
            "x": x,
            "y": y,
            "title": title,
            "x_label": x_label,
            "y_label": y_label,
        }
        self._plot_config = {k: v for k, v in config.items() if v is not None}

    @property
    def output_path(self) -> Path:
        _path = Path(f"{self.output_folder / self.name}.json")
        _path.parent.mkdir(exist_ok=True, parents=True)
        return _path

    @staticmethod
    def could_log(val: object) -> bool:
        if isinstance(val, list) and all(isinstance(x, dict) for x in val):
            return True
        return False

    @property
    def plot_config(self):
        return self._plot_config

    def dump(self, val, **kwargs) -> None:  # noqa: ARG002
        dump_json(val, self.output_path)
