from typing import List, Literal, Union, Dict
import math

import numpy as np
from pathlib import Path

from dvclive.plots.utils import NumpyEncoder
from dvclive.serialize import dump_json
from dvclive.error import InvalidSameSizeError, InvalidDataTypeError, MissingFieldError

from .base import Data

BboxFormatKind = Literal["tlbr", "tlhw", "xywh", "ltrb"]


class Annotations(Data):
    suffixes = (".json",)
    subfolder = "images"

    @property
    def output_path(self) -> Path:
        _path = self.output_folder / self.name
        _path.parent.mkdir(exist_ok=True, parents=True)
        return _path

    @staticmethod
    def could_log(  # noqa: C901
        name: str,
        annotations: Dict[str, List],
    ) -> bool:
        # no missing fields
        for field in ["boxes", "labels", "scores", "format"]:
            if field not in annotations:
                raise MissingFieldError(name, annotations, field)

        # `boxes`, `labels`, and `scores` fields should have the same size
        if len(annotations["boxes"]) != len(annotations["labels"]):
            raise InvalidSameSizeError(name, "boxes", "labels")

        if len(annotations["boxes"]) != len(annotations["scores"]):
            raise InvalidSameSizeError(name, "boxes", "scores")

        # `format` should be one of the supported formats
        if annotations["format"] not in ["tlbr", "tlhw", "xywh", "ltrb"]:
            raise InvalidDataTypeError(name, type(format))

        # `scores` should be a List[float]
        for idx, score in enumerate(annotations["scores"]):
            if not isinstance(score, (float, np.floating)):
                raise InvalidDataTypeError(
                    name, type(score), f"annotations['scores'][{idx}]"
                )

        # `boxes` should be a List[List[int, 4]]
        for idx, boxes in enumerate(annotations["boxes"]):
            if not all(isinstance(x, (int, np.int_)) for x in boxes):
                raise InvalidDataTypeError(
                    name, type(boxes), f"annotations['boxes'][{idx}]"
                )
            if len(boxes) != 4:  # noqa: PLR2004
                raise InvalidDataTypeError(
                    name, len(boxes), f"annotations['boxes'][{idx}]"
                )

        # `labels` should be a List[str]
        for idx, label in enumerate(annotations["labels"]):
            if not isinstance(label, str):
                raise InvalidDataTypeError(
                    name, type(label), f"annotations['labels'][{idx}]"
                )
        return

    def dump(
        self,
        annotations: Dict[str, List],
    ) -> None:
        boxes = self.convert_to_tlbr(annotations["boxes"], annotations["format"])
        labels = annotations["labels"]
        scores = annotations["scores"]
        boxes_info = [
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
            for tlbr, label, score in zip(boxes, labels, scores)
        ]
        # format for VScode and Studio
        boxes_json = {}
        for box in boxes_info:
            label = box.pop("label")
            if label not in boxes_json:
                boxes_json[label] = []
            boxes_json[label].append(box)

        dump_json(
            {"boxes": boxes_json},
            self.output_path.with_suffix(".json"),
            cls=NumpyEncoder,
        )

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

        if format == "ltrb":
            return [[box[1], box[0], box[3], box[2]] for box in bboxes]

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
