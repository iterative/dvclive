from typing import List, Literal, Union
import math


import numpy as np
from pathlib import Path

from dvclive.plots.utils import NumpyEncoder
from dvclive.serialize import dump_json
from dvclive.error import InvalidSameSizeError, InvalidDataTypeError

from .base import Data

BboxFormatKind = Literal["tlbr", "tlhw", "xywh"]


class BoundingBoxes(Data):
    suffixes = (".json",)
    subfolder = "images"

    @property
    def output_path(self) -> Path:
        _path = self.output_folder / self.name
        _path.parent.mkdir(exist_ok=True, parents=True)
        return _path

    @staticmethod
    def could_log(
        name: str,
        bounding_boxes: Union[List[List[int]], np.ndarray],
        labels: List[str],
        scores: List[float],
        format: BboxFormatKind,  # noqa: A002
    ) -> bool:
        if not any(name.endswith(suffix) for suffix in BoundingBoxes.suffixes):
            raise InvalidDataTypeError(name, type(name))

        if len(bounding_boxes) != len(labels):
            raise InvalidSameSizeError(name, "bounding_boxes", "labels")

        if len(bounding_boxes) != len(scores):
            raise InvalidSameSizeError(name, "bounding_boxes", "scores")

        if format not in ["tlbr", "tlhw", "xywh"]:
            raise InvalidDataTypeError(name, type(format))

        if not all(isinstance(score, float) for score in scores):
            raise InvalidDataTypeError(name, type(scores))

        if not all(
            isinstance(x, (int, np.int_)) for bbox in bounding_boxes for x in bbox
        ):
            raise InvalidDataTypeError(name, type(bounding_boxes))

        if not all(isinstance(label, str) for label in labels):
            raise InvalidDataTypeError(name, type(labels))
        return

    def dump(
        self,
        bboxes: Union[List[List[int]], np.ndarray],
        labels: List[str],
        scores: List[float],
        format: BboxFormatKind,  # noqa: A002
    ) -> None:
        bboxes = self.convert_to_tlbr(bboxes, format)
        json_content = {
            "boxes": [
                {
                    "label": label,
                    "box": {
                        "top": tlbr[0],
                        "left": tlbr[1],
                        "bottom": tlbr[2],
                        "right": tlbr[3],
                    },
                    "score": score,
                }
                for tlbr, label, score in zip(bboxes, labels, scores)
            ]
        }

        dump_json(json_content, self.output_path, cls=NumpyEncoder)

    @staticmethod
    def convert_to_tlbr(
        bboxes: Union[List[List[int]], "np.ndarray"],
        format: BboxFormatKind,  # noqa: A002
    ) -> List[List[int]]:
        """
        Converts bounding boxes from different formats to the top, left, bottom, right
        format.
        """
        if format == "tlhw":
            return [
                [box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in bboxes
            ]

        if format == "xywh":
            return [
                [
                    box[0] - math.ceil(box[2] / 2),
                    box[1] - math.ceil(box[3] / 2),
                    box[0] + math.floor(box[2] / 2),
                    box[1] + math.floor(box[3] / 2),
                ]
                for box in bboxes
            ]
        return bboxes
