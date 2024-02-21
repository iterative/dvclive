import numpy as np
import pytest

from dvclive import Live
from dvclive.plots import Annotations, BBoxes
from dvclive.error import InvalidDataTypeError


@pytest.mark.parametrize(
    ("name", "boxes", "labels", "scores"),
    [
        ("image.png", [], [], []),
        ("image.png", [[10, 20, 30, 40]], ["A"], [0.7]),
        ("image.png", [[0, 0, 10, 10], [10, 150, 300, 500]], ["B", "C"], [0.5, 0.8]),
        (
            "image.png",
            np.array([[0, 0, 10, 10], [10, 150, 300, 500]]),
            ["B", "C"],
            [0.5, 0.8],
        ),
        (
            "image.png",
            [[0, 0, 10, 10], [10, 150, 300, 500]],
            np.array(["B", "C"]),
            [0.5, 0.8],
        ),
        (
            "image.png",
            [[0, 0, 10, 10], [10, 150, 300, 500]],
            ["B", "C"],
            np.array([0.5, 0.8]),
        ),
    ],
)
def test_save_annotation_file(name, boxes, labels, scores, tmp_dir):
    live = Live()
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    live.log_image(
        name,
        img,
        annotations={
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
            "box_format": "tlbr",
        },
    )
    assert (tmp_dir / live.plots_dir / Annotations.subfolder / name).exists()
    assert (
        (tmp_dir / live.plots_dir / Annotations.subfolder / name)
        .with_suffix(".json")
        .exists()
    )


@pytest.mark.parametrize(
    "annotations",
    [
        {
            "labels": ["A", "B"],
            "scores": [0.1, 0.2],
            "box_format": "tlbr",
        },
        {
            "boxes": [[10, 20, 30, 40], [10, 20, 30, 40]],
            "scores": [0.1, 0.2],
            "box_format": "tlbr",
        },
        {
            "boxes": [[10, 20, 30, 40], [10, 20, 30, 40]],
            "labels": ["A", "B"],
            "box_format": "tlbr",
        },
        {
            "boxes": [[10, 20, 30, 40], [10, 20, 30, 40]],
            "labels": ["A", "B"],
            "scores": [0.1, 0.2],
        },
    ],
)
def test_invalid_field(tmp_dir, annotations):
    live = Live()
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    with pytest.raises(
        InvalidDataTypeError,
        match="Data 'image.png' has not supported type <class 'dict'>",
    ):
        live.log_image(
            "image.png",
            img,
            annotations=annotations,
        )


@pytest.mark.parametrize(
    "annotations",
    [
        {
            "boxes": [[10, 20, 30, 40]],
            "labels": ["A", "B"],
            "scores": [0.1, 0.2],
            "box_format": "tlbr",
        },
        {
            "boxes": [[10, 20, 30, 40], [10, 20, 30, 40]],
            "labels": ["A"],
            "scores": [0.1, 0.2],
            "box_format": "tlbr",
        },
        {
            "boxes": [[10, 20, 30, 40], [10, 20, 30, 40]],
            "labels": ["A", "B"],
            "scores": [0.1],
            "box_format": "tlbr",
        },
    ],
)
def test_invalid_labels_size(tmp_dir, annotations):
    live = Live()
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    with pytest.raises(
        InvalidDataTypeError,
        match="Data 'image.png' has not supported type <class 'dict'>",
    ):
        live.log_image(
            "image.png",
            img,
            annotations=annotations,
        )


def test_invalid_boxes_type(tmp_dir):
    live = Live()
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    with pytest.raises(
        InvalidDataTypeError,
        match="Data 'image.png' has not supported type <class 'dict'>",
    ):
        live.log_image(
            "image.png",
            img,
            annotations={
                "boxes": [[10, 20, 30, 40.5], [10, 20, 30, 40]],
                "labels": ["A", "B"],
                "scores": [0.1, 0.4],
                "box_format": "tlbr",
            },
        )


def test_invalid_boxes_length(tmp_dir):
    live = Live()
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    with pytest.raises(
        InvalidDataTypeError,
        match="Data 'image.png' has not supported type <class 'dict'>",
    ):
        live.log_image(
            "image.png",
            img,
            annotations={
                "boxes": [[10, 20, 30, 40, 50], [10, 20, 30, 40]],
                "labels": ["A", "B"],
                "scores": [0.1, 0.4],
                "box_format": "tlbr",
            },
        )


def test_invalid_labels_type(tmp_dir):
    live = Live()
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    with pytest.raises(
        InvalidDataTypeError,
        match="Data 'image.png' has not supported type <class 'dict'>",
    ):
        live.log_image(
            "image.png",
            img,
            annotations={
                "boxes": [[10, 20, 30, 40], [10, 20, 30, 40]],
                "labels": ["A", 1],
                "scores": [0.1, 0.5],
                "box_format": "tlbr",
            },
        )


def test_invalid_scores_type(tmp_dir):
    live = Live()
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    with pytest.raises(
        InvalidDataTypeError,
        match="Data 'image.png' has not supported type <class 'dict'>",
    ):
        live.log_image(
            "image.png",
            img,
            annotations={
                "boxes": [[10, 20, 30, 40], [10, 20, 30, 40]],
                "labels": ["A", "B"],
                "scores": [0.1, 4],
                "box_format": "tlbr",
            },
        )


@pytest.mark.parametrize(
    ("boxes", "box_format", "expected_result"),
    [
        ([[10, 20, 30, 40]], "tlbr", [[10, 20, 30, 40]]),
        (
            [[10, 20, 30, 40], [100, 200, 300, 400]],
            "tlbr",
            [[10, 20, 30, 40], [100, 200, 300, 400]],
        ),
        ([[10, 20, 30, 40]], "tlhw", [[10, 20, 40, 60]]),
        (
            [[10, 20, 30, 40], [100, 200, 300, 400]],
            "tlhw",
            [[10, 20, 40, 60], [100, 200, 400, 600]],
        ),
        ([[20, 30, 30, 40]], "xywh", [[5, 10, 35, 50]]),
        ([[20, 30, 31, 41]], "xywh", [[4, 9, 35, 50]]),
        (
            [[20, 30, 30, 40], [200, 300, 300, 400]],
            "xywh",
            [[5, 10, 35, 50], [50, 100, 350, 500]],
        ),
        ([[10, 20, 30, 40]], "ltrb", [[20, 10, 40, 30]]),
        (
            [[10, 20, 30, 40], [100, 200, 300, 400]],
            "ltrb",
            [[20, 10, 40, 30], [200, 100, 400, 300]],
        ),
    ],
)
def test_tlbr_conversion(boxes, box_format, expected_result):
    new_bbox = BBoxes(
        boxes=boxes,
        labels=["A" for _ in boxes],
        scores=[0.1 for _ in boxes],
        box_format=box_format,
    )
    assert expected_result == new_bbox.boxes
    assert new_bbox.box_format == "tlbr"
