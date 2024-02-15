import numpy as np
import pytest

from dvclive import Live
from dvclive.plots import Annotations
from dvclive.error import InvalidDataTypeError, InvalidSameSizeError


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
            "format": "tlbr",
        },
    )
    assert (tmp_dir / live.plots_dir / Annotations.subfolder / name).exists()
    assert (
        (tmp_dir / live.plots_dir / Annotations.subfolder / name)
        .with_suffix(".json")
        .exists()
    )


@pytest.mark.parametrize(
    ("name", "boxes", "labels", "scores"),
    [
        ("image.png", [[10, 20, 30, 40], [10, 20, 30, 40]], ["A"], [0.1, 0.2]),
        ("image.png", [[10, 20, 30, 40]], ["A", "B"], [0.7]),
    ],
)
def test_invalid_labels_size(name, boxes, labels, scores, tmp_dir):
    live = Live()
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    with pytest.raises(
        InvalidSameSizeError,
        match="'boxes' and 'labels' must have the same length",
    ):
        live.log_image(
            name,
            img,
            annotations={
                "boxes": boxes,
                "labels": labels,
                "scores": scores,
                "format": "tlbr",
            },
        )


@pytest.mark.parametrize(
    ("name", "boxes", "labels", "scores"),
    [
        ("image.png", [[10, 20, 30, 40], [10, 20, 30, 40]], ["A", "B"], [0.1]),
        ("image.png", [[10, 20, 30, 40]], ["A"], [0.7, 0.4]),
    ],
)
def test_invalid_scores_size(name, boxes, labels, scores, tmp_dir):
    live = Live()
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    with pytest.raises(
        InvalidSameSizeError,
        match="'boxes' and 'scores' must have the same length",
    ):
        live.log_image(
            name,
            img,
            annotations={
                "boxes": boxes,
                "labels": labels,
                "scores": scores,
                "format": "tlbr",
            },
        )


@pytest.mark.parametrize(
    ("name", "boxes", "labels", "scores"),
    [
        ("image.png", [[10, 20, 30.5, 40], [10, 20, 30, 40]], ["A", "B"], [0.1, 0.2]),
        ("image.png", [[10, 20, 30, 40]], [0], [0.7]),
        ("image.png", [[10, 20, 30, 40]], [0.1], [0.7]),
        ("image.png", [[10, 20, 30, 40]], [["A", "B"]], [0.7]),
        ("image.png", [[10, 20, 30, 40]], ["A"], [1]),
        ("image.png", [[10, 20, 30, 40]], ["A"], [0]),
        ("image.png", [[10, 20, 30, 40]], ["A"], ["score"]),
    ],
)
def test_invalid_inputs_format(name, boxes, labels, scores, tmp_dir):
    live = Live()
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    with pytest.raises(InvalidDataTypeError):
        live.log_image(
            name,
            img,
            annotations={
                "boxes": boxes,
                "labels": labels,
                "scores": scores,
                "format": "tlbr",
            },
        )


@pytest.mark.parametrize(
    ("boxes", "format", "expected_result"),
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
def test_tlbr_conversion(boxes, format, expected_result):  # noqa: A002
    assert expected_result == Annotations.convert_to_tlbr(boxes, format)
