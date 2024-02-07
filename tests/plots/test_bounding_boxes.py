import numpy as np
import pytest

from dvclive import Live
from dvclive.plots import BoundingBoxes
from dvclive.error import InvalidDataTypeError, InvalidSameSizeError


@pytest.mark.parametrize(
    ("name", "bounding_boxes", "labels", "scores"),
    [
        ("image.json", [], [], []),
        ("image.json", [[10, 20, 30, 40]], ["A"], [0.7]),
        ("image.json", [[0, 0, 10, 10], [10, 150, 300, 500]], ["B", "C"], [0.5, 0.8]),
        (
            "image.json",
            np.array([[0, 0, 10, 10], [10, 150, 300, 500]]),
            ["B", "C"],
            [0.5, 0.8],
        ),
        (
            "image.json",
            [[0, 0, 10, 10], [10, 150, 300, 500]],
            np.array(["B", "C"]),
            [0.5, 0.8],
        ),
        (
            "image.json",
            [[0, 0, 10, 10], [10, 150, 300, 500]],
            ["B", "C"],
            np.array([0.5, 0.8]),
        ),
    ],
)
def test_save_json_file(name, bounding_boxes, labels, scores, tmp_dir):
    live = Live()
    live.log_bounding_boxes(name, bounding_boxes, labels, scores, format="tlbr")
    assert (tmp_dir / live.plots_dir / BoundingBoxes.subfolder / name).exists()


@pytest.mark.parametrize(
    ("name", "bounding_boxes", "labels", "scores"),
    [
        ("image.png", [], [], []),
        ("image.jpg", [[10, 20, 30, 40]], ["A"], [0.7]),
        ("image.yaml", [[0, 0, 10, 10], [10, 150, 300, 500]], ["B", "C"], [0.5, 0.8]),
    ],
)
def test_invalid_extension(name, bounding_boxes, labels, scores, tmp_dir):
    live = Live()
    with pytest.raises(InvalidDataTypeError):
        live.log_bounding_boxes(
            name,
            bounding_boxes,
            labels,
            scores,
            format="tlbr",
        )


@pytest.mark.parametrize(
    ("name", "bounding_boxes", "labels", "scores"),
    [
        ("image.json", [[10, 20, 30, 40], [10, 20, 30, 40]], ["A"], [0.1, 0.2]),
        ("image.json", [[10, 20, 30, 40]], ["A", "B"], [0.7]),
    ],
)
def test_invalid_labels_size(name, bounding_boxes, labels, scores, tmp_dir):
    live = Live()
    with pytest.raises(
        InvalidSameSizeError,
        match="Data 'image.json': "
        "'bounding_boxes' and 'labels' must have the same length",
    ):
        live.log_bounding_boxes(
            name,
            bounding_boxes,
            labels,
            scores,
            format="tlbr",
        )


@pytest.mark.parametrize(
    ("name", "bounding_boxes", "labels", "scores"),
    [
        ("image.json", [[10, 20, 30, 40], [10, 20, 30, 40]], ["A", "B"], [0.1]),
        ("image.json", [[10, 20, 30, 40]], ["A"], [0.7, 0.4]),
    ],
)
def test_invalid_scores_size(name, bounding_boxes, labels, scores, tmp_dir):
    live = Live()
    with pytest.raises(
        InvalidSameSizeError,
        match="Data 'image.json': "
        "'bounding_boxes' and 'scores' must have the same length",
    ):
        live.log_bounding_boxes(
            name,
            bounding_boxes,
            labels,
            scores,
            format="tlbr",
        )


@pytest.mark.parametrize(
    ("name", "bounding_boxes", "labels", "scores"),
    [
        ("image.json", [[10, 20, 30.5, 40], [10, 20, 30, 40]], ["A", "B"], [0.1, 0.2]),
        ("image.json", [[10, 20, 30, 40]], [0], [0.7]),
        ("image.json", [[10, 20, 30, 40]], [0.1], [0.7]),
        ("image.json", [[10, 20, 30, 40]], [["A", "B"]], [0.7]),
        ("image.json", [[10, 20, 30, 40]], ["A"], [1]),
        ("image.json", [[10, 20, 30, 40]], ["A"], [0]),
        ("image.json", [[10, 20, 30, 40]], ["A"], ["score"]),
    ],
)
def test_invalid_inputs_format(name, bounding_boxes, labels, scores, tmp_dir):
    live = Live()
    with pytest.raises(InvalidDataTypeError):
        live.log_bounding_boxes(
            name,
            bounding_boxes,
            labels,
            scores,
            format="tlbr",
        )


def test_path(tmp_dir):
    import json

    json_path = tmp_dir / "image.json"
    with open(json_path, "w") as json_file:
        json.dump({}, json_file)

    live = Live()
    live.log_bounding_boxes(
        "image.json",
        [[0, 0, 10, 10]],
        ["cat"],
        [0.2],
        format="tlbr",
    )
    live.end()

    file_path = tmp_dir / live.plots_dir / "images" / "image.json"
    assert file_path.exists()

    with open(file_path) as json_file:
        file_content = json.load(json_file)

    assert file_content == {
        "boxes": [
            {
                "box": {"top": 0, "left": 0, "bottom": 10, "right": 10},
                "label": "cat",
                "score": 0.2,
            }
        ]
    }


def test_override_on_step(tmp_dir):
    import json

    live = Live()

    live.log_bounding_boxes(
        "image.json",
        [[0, 0, 10, 10]],
        ["cat"],
        [0.2],
        format="tlbr",
    )

    live.next_step()

    bboxes_step_2 = [[10, 20, 30, 40]]
    labels_step_2 = ["dog"]
    scores_step_2 = [0.5]
    live.log_bounding_boxes(
        "image.json",
        bboxes_step_2,
        labels_step_2,
        scores_step_2,
        format="tlbr",
    )

    file_path = tmp_dir / live.plots_dir / BoundingBoxes.subfolder / "image.json"

    with open(file_path) as json_file:
        file_content = json.load(json_file)

    expected_bboxes = dict(zip(["top", "left", "bottom", "right"], bboxes_step_2[0]))
    assert file_content["boxes"][0]["box"] == expected_bboxes
    assert file_content["boxes"][0]["label"] == labels_step_2[0]
    assert file_content["boxes"][0]["score"] == scores_step_2[0]


def test_cleanup(tmp_dir):
    live = Live()
    live.log_bounding_boxes(
        "image.json",
        [[0, 0, 10, 10]],
        ["cat"],
        [0.2],
        format="tlbr",
    )
    assert (tmp_dir / live.plots_dir / BoundingBoxes.subfolder / "image.json").exists()

    Live()

    assert not (tmp_dir / live.plots_dir / BoundingBoxes.subfolder).exists()


@pytest.mark.parametrize("cache", [False, True])
def test_cache_images(tmp_dir, dvc_repo, cache):
    live = Live(save_dvc_exp=False, cache_images=cache)
    live.log_bounding_boxes(
        "image.json",
        [[0, 0, 10, 10]],
        ["cat"],
        [0.2],
        format="tlbr",
    )
    live.end()
    assert (tmp_dir / "dvclive" / "plots" / "images.dvc").exists() == cache


@pytest.mark.parametrize(
    ("bounding_boxes", "format", "expected_result"),
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
    ],
)
def test_tlbr_conversion(bounding_boxes, format, expected_result):  # noqa: A002
    assert expected_result == BoundingBoxes.convert_to_tlbr(bounding_boxes, format)
