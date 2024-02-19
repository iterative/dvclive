import numpy as np
import pytest

from dvclive import Live
from dvclive.plots import Annotations
from dvclive.plots.annotations import BOXES_NAME, LABELS_NAME, SCORES_NAME, FORMAT_NAME
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
            BOXES_NAME: boxes,
            LABELS_NAME: labels,
            SCORES_NAME: scores,
            FORMAT_NAME: "tlbr",
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
            LABELS_NAME: ["A", "B"],
            SCORES_NAME: [0.1, 0.2],
            FORMAT_NAME: "tlbr",
        },
        {
            BOXES_NAME: [[10, 20, 30, 40], [10, 20, 30, 40]],
            SCORES_NAME: [0.1, 0.2],
            FORMAT_NAME: "tlbr",
        },
        {
            BOXES_NAME: [[10, 20, 30, 40], [10, 20, 30, 40]],
            LABELS_NAME: ["A", "B"],
            FORMAT_NAME: "tlbr",
        },
        {
            BOXES_NAME: [[10, 20, 30, 40], [10, 20, 30, 40]],
            LABELS_NAME: ["A", "B"],
            SCORES_NAME: [0.1, 0.2],
        },
    ],
)
def test_invalid_field(tmp_dir, annotations, caplog):
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
    assert (
        "Missing fields in annotations. Expected: 'boxes', 'labels', 'scores', and "
        "'format'." in caplog.text
    )


@pytest.mark.parametrize(
    "annotations",
    [
        {
            BOXES_NAME: [[10, 20, 30, 40]],
            LABELS_NAME: ["A", "B"],
            SCORES_NAME: [0.1, 0.2],
            FORMAT_NAME: "tlbr",
        },
        {
            BOXES_NAME: [[10, 20, 30, 40], [10, 20, 30, 40]],
            LABELS_NAME: ["A"],
            SCORES_NAME: [0.1, 0.2],
            FORMAT_NAME: "tlbr",
        },
        {
            BOXES_NAME: [[10, 20, 30, 40], [10, 20, 30, 40]],
            LABELS_NAME: ["A", "B"],
            SCORES_NAME: [0.1],
            FORMAT_NAME: "tlbr",
        },
    ],
)
def test_invalid_labels_size(tmp_dir, annotations, caplog):
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
    assert "'boxes', 'labels', and 'scores' should have the same size." in caplog.text


def test_invalid_boxes_type(tmp_dir, caplog):
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
                BOXES_NAME: [[10, 20, 30, 40.5], [10, 20, 30, 40]],
                LABELS_NAME: ["A", "B"],
                SCORES_NAME: [0.1, 0.4],
                FORMAT_NAME: "tlbr",
            },
        )
    assert (
        "Annotations `'boxes'` should be a `List[int]`, received "
        "'[[10, 20, 30, 40.5], [10, 20, 30, 40]]'." in caplog.text
    )


def test_invalid_boxes_length(tmp_dir, caplog):
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
                BOXES_NAME: [[10, 20, 30, 40, 50], [10, 20, 30, 40]],
                LABELS_NAME: ["A", "B"],
                SCORES_NAME: [0.1, 0.4],
                FORMAT_NAME: "tlbr",
            },
        )
    assert "Annotations `'boxes'` should be of length 4." in caplog.text


def test_invalid_labels_type(tmp_dir, caplog):
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
                BOXES_NAME: [[10, 20, 30, 40], [10, 20, 30, 40]],
                LABELS_NAME: ["A", 1],
                SCORES_NAME: [0.1, 0.5],
                FORMAT_NAME: "tlbr",
            },
        )
    assert (
        "Annotations `'labels'` should be a `List[str]`, received '['A', 1]'."
        in caplog.text
    )


def test_invalid_scores_type(tmp_dir, caplog):
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
                BOXES_NAME: [[10, 20, 30, 40], [10, 20, 30, 40]],
                LABELS_NAME: ["A", "B"],
                SCORES_NAME: [0.1, 4],
                FORMAT_NAME: "tlbr",
            },
        )
    assert (
        "Annotations `'scores'` should be a `List[float]`, received '[0.1, 4]'."
        in caplog.text
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
