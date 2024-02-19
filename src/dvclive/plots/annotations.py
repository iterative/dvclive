from typing import List, Literal, Union, Dict, get_args
import math

import numpy as np
from pathlib import Path

from dvclive.plots.utils import NumpyEncoder
from dvclive.serialize import dump_json
import logging

from .base import Data


logger = logging.getLogger("dvclive")

BOXES_NAME = "boxes"
LABELS_NAME = "labels"
SCORES_NAME = "scores"
FORMAT_NAME = "format"

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
    def could_log(  # noqa: PLR0911
        annotations: Dict[str, List],
    ) -> bool:
        # no missing fields
        if any(
            field not in annotations
            for field in [BOXES_NAME, LABELS_NAME, SCORES_NAME, FORMAT_NAME]
        ):
            logger.warning(
                f"Missing fields in annotations. Expected: '{BOXES_NAME}',"
                f" '{LABELS_NAME}', '{SCORES_NAME}', and '{FORMAT_NAME}'."
            )
            return False

        # `boxes`, `labels`, and `scores` fields should have the same size
        boxes_and_labels_same_size = len(annotations[BOXES_NAME]) == len(
            annotations[LABELS_NAME]
        )
        boxes_and_scores_same_size = len(annotations[BOXES_NAME]) == len(
            annotations[SCORES_NAME]
        )
        if not boxes_and_labels_same_size or not boxes_and_scores_same_size:
            logger.warning(
                f"'{BOXES_NAME}', '{LABELS_NAME}', and '{SCORES_NAME}' should have the "
                "same size."
            )
            return False

        # `format` should be one of the supported formats
        if annotations[FORMAT_NAME] not in get_args(BboxFormatKind):
            logger.warning(
                f"Annotations format '{annotations['format']}' is not supported."
            )
            return False

        # `scores` should be a List[float]
        if not all(
            isinstance(score, (float, np.floating))
            for score in annotations[SCORES_NAME]
        ):
            logger.warning(
                "Annotations `'scores'` should be a `List[float]`, received "
                f"'{annotations[SCORES_NAME]}'."
            )
            return False

        # `boxes` should be a List[List[int, 4]]
        for boxes in annotations[BOXES_NAME]:
            if not all(isinstance(x, (int, np.int_)) for x in boxes):
                logger.warning(
                    f"Annotations `'{BOXES_NAME}'` should be a `List[int]`, received "
                    f"'{annotations[BOXES_NAME]}'."
                )
                return False

            if len(boxes) != 4:  # noqa: PLR2004
                logger.warning(f"Annotations `'{BOXES_NAME}'` should be of length 4.")
                return False

        # `labels` should be a List[str]
        if not all(
            isinstance(label, (str, np.str_)) for label in annotations[LABELS_NAME]
        ):
            logger.warning(
                f"Annotations `'{LABELS_NAME}'` should be a `List[str]`, received "
                f"'{annotations[LABELS_NAME]}'."
            )
            return False
        return True

    def dump(
        self,
        val,
    ):
        boxes = self.convert_to_tlbr(val[BOXES_NAME], val[FORMAT_NAME])
        labels = val[LABELS_NAME]
        scores = val[SCORES_NAME]
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
            {"annotations": boxes_json},
            self.output_path.with_suffix(".json"),
            cls=NumpyEncoder,
        )

    @staticmethod
    def convert_to_tlbr(
        bboxes: Union[List[List[int]], "np.ndarray"],
        format: BboxFormatKind,  # noqa: A002
    ) -> Union[List[List[int]], "np.ndarray"]:
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
