from typing import List, Literal, TYPE_CHECKING
import math


from pathlib import Path
from pydantic import BaseModel, ValidationError, field_validator, model_validator
import logging

from dvclive.plots.utils import NumpyEncoder
from dvclive.serialize import dump_json

if TYPE_CHECKING:
    from dvclive.live import BBoxes as UserFacingBBoxes

from .base import Data

LEN_BOX = 4
logger = logging.getLogger("dvclive")


class BBoxes(BaseModel):
    boxes: List[List[int]]
    labels: List[str]
    scores: List[float]
    box_format: Literal["tlbr", "tlhw", "xywh", "ltrb"]

    @field_validator("boxes")
    @classmethod
    def box_contains_4_values(cls, boxes):
        if any(len(box) != LEN_BOX for box in boxes):
            err_msg = "Annotations 'boxes' must contain lists of length 4."
            raise ValueError(err_msg)
        return boxes

    @field_validator("scores")
    @classmethod
    def score_is_between_0_and_1(cls, scores):
        if any(score > 1 or score < 0 for score in scores):
            err_msg = "Annotations 'score' must be between 0 and 1."
            raise ValueError(err_msg)
        return scores

    @model_validator(mode="after")
    def boxes_labels_scores_have_same_length(self):
        if len(self.boxes) != len(self.labels) or len(self.boxes) != len(self.scores):
            err_msg = (
                "Annotations 'boxes', 'labels', and 'scores' should have the same size."
            )
            raise ValueError(err_msg)
        return self

    @model_validator(mode="after")
    def convert_box_to_tlbr(self):
        if self.box_format == "tlhw":
            self.boxes = [
                [box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in self.boxes
            ]

        elif self.box_format == "ltrb":
            self.boxes = [[box[1], box[0], box[3], box[2]] for box in self.boxes]

        elif self.box_format == "xywh":
            self.boxes = [
                [
                    box[0] - math.ceil(box[2] / 2),
                    box[1] - math.ceil(box[3] / 2),
                    box[0] + math.floor(box[2] / 2),
                    box[1] + math.floor(box[3] / 2),
                ]
                for box in self.boxes
            ]
        self.box_format = "tlbr"
        return self


class Annotations(Data):
    suffixes = (".json",)
    subfolder = "images"

    @property
    def output_path(self) -> Path:
        _path = self.output_folder / self.name
        _path.parent.mkdir(exist_ok=True, parents=True)
        return _path

    @staticmethod
    def could_log(
        annotations: "UserFacingBBoxes",
    ) -> bool:
        result = True
        try:
            BBoxes(**annotations)
        except ValidationError as exc:
            logger.warning(exc)
            result = False
        return result

    def dump(
        self,
        val,  # UserFacingBBoxes
    ):
        annotations = BBoxes(**val)
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
            for tlbr, label, score in zip(
                annotations.boxes, annotations.labels, annotations.scores
            )
        ]

        # group by label (https://github.com/iterative/dvc/issues/10198)
        boxes_json = {}
        for box in boxes_info:
            label = box.pop("label")
            if label not in boxes_json:
                boxes_json[label] = []
            boxes_json[label].append(box)

        dump_json(
            {"annotations": boxes_json},
            self.output_path.with_suffix(".json"),
            cls=NumpyEncoder,
        )
