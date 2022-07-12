# pylint: disable=unused-argument
import os

import pytest
from PIL import Image

from dvclive import Live
from dvclive.data import Image as LiveImage
from dvclive.data import Scalar
from dvclive.data.plot import ConfusionMatrix, Plot
from dvclive.env import DVCLIVE_OPEN
from dvclive.report import (
    get_image_renderers,
    get_metrics_renderers,
    get_plot_renderers,
    get_scalar_renderers,
)


def test_get_renderers(tmp_dir, mocker):
    live = Live()

    for i in range(2):
        live.log("foo/bar", i)
        img = Image.new("RGB", (10, 10), (i, i, i))
        live.log_image("image.png", img)
        live.next_step()

    live.set_step(None)
    live.log_plot("confusion_matrix", [0, 0, 1, 1], [1, 0, 0, 1])

    image_renderers = get_image_renderers(
        tmp_dir / live.dir / LiveImage.subfolder
    )
    assert len(image_renderers) == 2
    image_renderers = sorted(
        image_renderers, key=lambda x: x.datapoints[0]["rev"]
    )
    for n, renderer in enumerate(image_renderers):
        assert renderer.datapoints == [
            {"src": mocker.ANY, "rev": os.path.join(str(n), "image.png")}
        ]

    scalar_renderers = get_scalar_renderers(
        tmp_dir / live.dir / Scalar.subfolder
    )
    assert len(scalar_renderers) == 1
    assert scalar_renderers[0].datapoints == [
        {
            "foo/bar": "0",
            "rev": "workspace",
            "step": "0",
            "timestamp": mocker.ANY,
        },
        {
            "foo/bar": "1",
            "rev": "workspace",
            "step": "1",
            "timestamp": mocker.ANY,
        },
    ]
    assert scalar_renderers[0].properties["y"] == "foo/bar"
    assert scalar_renderers[0].name == "static/foo/bar"

    plot_renderers = get_plot_renderers(tmp_dir / live.dir / Plot.subfolder)
    assert len(plot_renderers) == 1
    assert plot_renderers[0].datapoints == [
        {"actual": "0", "rev": "workspace", "predicted": "1"},
        {"actual": "0", "rev": "workspace", "predicted": "0"},
        {"actual": "1", "rev": "workspace", "predicted": "0"},
        {"actual": "1", "rev": "workspace", "predicted": "1"},
    ]
    assert plot_renderers[0].properties == ConfusionMatrix.get_properties()

    metrics_renderer = get_metrics_renderers(live.summary_path)[0]
    assert metrics_renderer.datapoints == [{"step": 1, "foo": {"bar": 1}}]


def test_report_init(monkeypatch):
    monkeypatch.setenv("CI", "false")
    live = Live()
    assert live._report == "html"

    monkeypatch.setenv("CI", "true")
    live = Live()
    assert live._report == "md"

    for report in {None, "html", "md"}:
        live = Live(report=report)
        assert live._report == report

    with pytest.raises(ValueError):
        Live(report="foo")


@pytest.mark.parametrize("mode", ["html", "md"])
def test_make_report(tmp_dir, mode):
    live = Live(report=mode)
    for i in range(3):
        live.log("foobar", i)
        live.log("foo/bar", i)
        live.next_step()

    # Format of the report is tested in `dvc-render`
    assert (tmp_dir / live.report_path).exists()
    assert (tmp_dir / live.dir / "static").exists() == (mode == "md")


@pytest.mark.vscode
def test_make_report_open(tmp_dir, mocker, monkeypatch):
    mocked_open = mocker.patch("webbrowser.open")
    live = Live()
    live.log_plot("confusion_matrix", [0, 0, 1, 1], [1, 0, 0, 1])
    live.make_report()
    live.make_report()

    assert not mocked_open.called

    live = Live(report=None)
    live.log("foo", 1)
    live.next_step()

    assert not mocked_open.called

    monkeypatch.setenv(DVCLIVE_OPEN, True)

    live = Live()
    live.log_plot("confusion_matrix", [0, 0, 1, 1], [1, 0, 0, 1])
    live.make_report()

    mocked_open.assert_called_once()
