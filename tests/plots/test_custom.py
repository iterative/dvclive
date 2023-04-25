import json

from dvclive import Live
from dvclive.plots.custom import CustomPlot


def test_log_custom_plot(tmp_dir):
    live = Live()
    out = tmp_dir / live.plots_dir / CustomPlot.subfolder

    datapoints = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
    live.log_plot(
        "custom_linear",
        datapoints,
        x="x",
        y="y",
        template="linear",
        title="custom_title",
        x_label="x_label",
        y_label="y_label",
    )

    assert json.loads((out / "custom_linear.json").read_text()) == datapoints
    assert live._plots["custom_linear"].plot_config == {
        "template": "linear",
        "title": "custom_title",
        "x": "x",
        "y": "y",
        "x_label": "x_label",
        "y_label": "y_label",
    }
