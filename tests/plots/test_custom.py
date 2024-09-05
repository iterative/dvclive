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


def test_log_custom_plot_multi_y(tmp_dir):
    live = Live()
    out = tmp_dir / live.plots_dir / CustomPlot.subfolder

    datapoints = [{"x": 1, "y1": 2, "y2": 3}, {"x": 4, "y1": 5, "y2": 6}]
    live.log_plot(
        "custom_linear",
        datapoints,
        x="x",
        y=["y1", "y2"],
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
        "y": ["y1", "y2"],
        "x_label": "x_label",
        "y_label": "y_label",
    }


def test_log_custom_plot_with_template_as_empty_string(tmp_dir):
    live = Live()
    out = tmp_dir / live.plots_dir / CustomPlot.subfolder

    datapoints = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
    live.log_plot(
        "custom_linear",
        datapoints,
        x="x",
        y="y",
        template="",
        title="custom_title",
        x_label="x_label",
        y_label="y_label",
    )

    assert json.loads((out / "custom_linear.json").read_text()) == datapoints
    # 'template' should not be in plot_config. Default template will be assigned later.
    assert live._plots["custom_linear"].plot_config == {
        "title": "custom_title",
        "x": "x",
        "y": "y",
        "x_label": "x_label",
        "y_label": "y_label",
    }
