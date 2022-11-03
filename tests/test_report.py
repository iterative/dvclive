# pylint: disable=unused-argument,protected-access
import os

import pytest
from PIL import Image

from dvclive import Live
from dvclive.env import DVCLIVE_OPEN
from dvclive.plots import Image as LiveImage
from dvclive.plots import Metric
from dvclive.plots.sklearn import ConfusionMatrix, SKLearnPlot
from dvclive.report import (
    get_image_renderers,
    get_metrics_renderers,
    get_params_renderers,
    get_plot_renderers,
    get_scalar_renderers,
)


def test_get_renderers(tmp_dir, mocker):
    live = Live()

    live.log_param("string", "goo")
    live.log_param("number", 2)

    for i in range(2):
        live.log_metric("foo/bar", i)
        img = Image.new("RGB", (10, 10), (i, i, i))
        live.log_image("image.png", img)
        live.next_step()

    live.log_sklearn_plot("confusion_matrix", [0, 0, 1, 1], [1, 0, 0, 1])

    image_renderers = get_image_renderers(
        tmp_dir / live.plots_dir / LiveImage.subfolder
    )
    assert len(image_renderers) == 1
    assert image_renderers[0].datapoints == [
        {
            "src": os.path.join("plots", LiveImage.subfolder, "image.png"),
            "rev": "image.png",
        }
    ]

    scalar_renderers = get_scalar_renderers(
        tmp_dir / live.plots_dir / Metric.subfolder
    )
    assert len(scalar_renderers) == 1
    assert scalar_renderers[0].datapoints == [
        {
            "foo/bar": "0",
            "rev": "workspace",
            "step": "0",
        },
        {
            "foo/bar": "1",
            "rev": "workspace",
            "step": "1",
        },
    ]
    assert scalar_renderers[0].properties["y"] == "foo/bar"
    assert scalar_renderers[0].name == "static/foo/bar"

    plot_renderers = get_plot_renderers(
        tmp_dir / live.plots_dir / SKLearnPlot.subfolder
    )
    assert len(plot_renderers) == 1
    assert plot_renderers[0].datapoints == [
        {"actual": "0", "rev": "workspace", "predicted": "1"},
        {"actual": "0", "rev": "workspace", "predicted": "0"},
        {"actual": "1", "rev": "workspace", "predicted": "0"},
        {"actual": "1", "rev": "workspace", "predicted": "1"},
    ]
    assert plot_renderers[0].properties == ConfusionMatrix.get_properties()

    metrics_renderer = get_metrics_renderers(live.metrics_file)[0]
    assert metrics_renderer.datapoints == [{"step": 1, "foo": {"bar": 1}}]

    params_renderer = get_params_renderers(live.params_file)[0]
    assert params_renderer.datapoints == [{"string": "goo", "number": 2}]


def test_report_init(monkeypatch, mocker):
    monkeypatch.setenv("CI", "false")
    live = Live()
    assert live.report_mode == "html"

    monkeypatch.setenv("CI", "true")
    live = Live()
    assert live.report_mode == "md"

    mocker.patch("dvclive.live.matplotlib_installed", return_value=False)
    live = Live()
    assert live.report_mode == "html"

    for report in {None, "html", "md"}:
        live = Live(report=report)
        assert live.report_mode == report

    with pytest.raises(ValueError):
        Live(report="foo")


@pytest.mark.parametrize("mode", ["html", "md"])
def test_make_report(tmp_dir, mode):
    last_report = ""
    live = Live(report=mode)
    for i in range(3):
        live.log_metric("foobar", i)
        live.log_metric("foo/bar", i)
        live.make_report()
        live.next_step()
        assert (tmp_dir / live.report_file).exists()
        current_report = (tmp_dir / live.report_file).read_text()
        assert last_report != current_report
        last_report = current_report


@pytest.mark.vscode
def test_make_report_open(tmp_dir, mocker, monkeypatch):
    mocked_open = mocker.patch("webbrowser.open")
    live = Live()
    live.log_sklearn_plot("confusion_matrix", [0, 0, 1, 1], [1, 0, 0, 1])
    live.make_report()
    live.make_report()

    assert not mocked_open.called

    live = Live(report=None)
    live.log_metric("foo", 1)
    live.next_step()

    assert not mocked_open.called

    monkeypatch.setenv(DVCLIVE_OPEN, True)

    live = Live()
    live.log_sklearn_plot("confusion_matrix", [0, 0, 1, 1], [1, 0, 0, 1])
    live.make_report()

    mocked_open.assert_called_once()
