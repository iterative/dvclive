# pylint: disable=unused-argument
import os

from PIL import Image

from dvclive import Live
from dvclive.data import Image as LiveImage
from dvclive.data import Scalar
from dvclive.data.plot import ConfusionMatrix, Plot
from dvclive.report import (
    get_image_renderers,
    get_plot_renderers,
    get_scalar_renderers,
)


def test_get_renderers(tmp_dir, mocker):
    live = Live()

    for i in range(2):
        live.log("foo", i)
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
        {"foo": "0", "rev": "workspace", "step": "0", "timestamp": mocker.ANY},
        {"foo": "1", "rev": "workspace", "step": "1", "timestamp": mocker.ANY},
    ]

    plot_renderers = get_plot_renderers(tmp_dir / live.dir / Plot.subfolder)
    assert len(plot_renderers) == 1
    assert plot_renderers[0].datapoints == [
        {"actual": "0", "rev": "workspace", "predicted": "1"},
        {"actual": "0", "rev": "workspace", "predicted": "0"},
        {"actual": "1", "rev": "workspace", "predicted": "0"},
        {"actual": "1", "rev": "workspace", "predicted": "1"},
    ]
    assert plot_renderers[0].properties == ConfusionMatrix.get_properties()


def test_make_report_open(tmp_dir, mocker):
    mocked_open = mocker.patch("webbrowser.open")
    live = Live()
    live.log_plot("confusion_matrix", [0, 0, 1, 1], [1, 0, 0, 1])
    live.make_report()
    live.make_report()

    mocked_open.assert_called_once()

    mocked_open = mocker.patch("webbrowser.open")
    live = Live(auto_open=False)
    live.log_plot("confusion_matrix", [0, 0, 1, 1], [1, 0, 0, 1])
    live.make_report()

    assert not mocked_open.called

    mocked_open = mocker.patch("webbrowser.open")
    live = Live(report=None)
    live.log("foo", 1)
    live.next_step()

    assert not mocked_open.called
