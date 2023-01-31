import json
from pathlib import Path
from typing import TYPE_CHECKING

from dvc_render.html import render_html
from dvc_render.image import ImageRenderer
from dvc_render.markdown import render_markdown
from dvc_render.table import TableRenderer
from dvc_render.vega import VegaRenderer

from dvclive.plots import SKLEARN_PLOTS, Image, Metric
from dvclive.plots.sklearn import SKLearnPlot
from dvclive.serialize import load_yaml
from dvclive.utils import parse_tsv

if TYPE_CHECKING:
    from dvclive import Live


# noqa pylint: disable=protected-access


def get_scalar_renderers(metrics_path):
    renderers = []
    for suffix in Metric.suffixes:
        for file in metrics_path.rglob(f"*{suffix}"):
            data = parse_tsv(file)
            for row in data:
                row["rev"] = "workspace"

            y = file.relative_to(metrics_path).with_suffix("")
            y = y.as_posix()

            name = file.relative_to(metrics_path.parent).with_suffix("")
            name = name.as_posix()
            name = name.replace(metrics_path.name, "static")

            properties = {"x": "step", "y": y}
            renderers.append(VegaRenderer(data, name, **properties))
    return renderers


def get_image_renderers(images_folder):
    plots_path = images_folder.parent.parent
    renderers = []
    for suffix in Image.suffixes:
        all_images = Path(images_folder).rglob(f"*{suffix}")
        for file in sorted(all_images):
            src = str(file.relative_to(plots_path))
            name = str(file.relative_to(images_folder))
            data = [
                {
                    ImageRenderer.SRC_FIELD: src,
                    ImageRenderer.TITLE_FIELD: name,
                }
            ]
            renderers.append(ImageRenderer(data, name))
    return renderers


def get_plot_renderers(plots_folder, live):
    renderers = []
    for suffix in SKLearnPlot.suffixes:
        for file in Path(plots_folder).rglob(f"*{suffix}"):
            name = file.relative_to(plots_folder).with_suffix("").as_posix()
            properties = {}

            if name in SKLEARN_PLOTS:
                properties = SKLEARN_PLOTS[name].get_properties()
                data_field = name
            else:
                # Plot with custom name
                logged_plot = live._plots[name]
                for default_name, plot_class in SKLEARN_PLOTS.items():
                    if isinstance(logged_plot, plot_class):
                        properties = plot_class.get_properties()
                        data_field = default_name
                        break

            data = json.loads(file.read_text())

            if data_field in data:
                data = data[data_field]

            for row in data:
                row["rev"] = "workspace"

            renderers.append(VegaRenderer(data, name, **properties))
    return renderers


def get_metrics_renderers(dvclive_summary):
    metrics_path = Path(dvclive_summary)
    if metrics_path.exists():
        return [
            TableRenderer(
                [json.loads(metrics_path.read_text(encoding="utf-8"))],
                metrics_path.name,
            )
        ]
    return []


def get_params_renderers(dvclive_params):
    params_path = Path(dvclive_params)
    if params_path.exists():
        return [
            TableRenderer(
                [load_yaml(params_path)],
                params_path.name,
            )
        ]
    return []


def make_report(live: "Live"):
    plots_path = Path(live.plots_dir)

    renderers = []
    renderers.extend(get_params_renderers(live.params_file))
    renderers.extend(get_metrics_renderers(live.metrics_file))
    renderers.extend(get_scalar_renderers(plots_path / Metric.subfolder))
    renderers.extend(get_image_renderers(plots_path / Image.subfolder))
    renderers.extend(get_plot_renderers(plots_path / SKLearnPlot.subfolder, live))

    if live._report_mode == "html":
        render_html(renderers, live.report_file, refresh_seconds=5)
    elif live._report_mode == "md":
        render_markdown(renderers, live.report_file)
    else:
        raise ValueError(f"Invalid `mode` {live._report_mode}.")
