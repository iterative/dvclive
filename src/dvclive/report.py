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


def get_plot_renderers(plots_folder):
    renderers = []
    for suffix in SKLearnPlot.suffixes:
        for file in Path(plots_folder).rglob(f"*{suffix}"):
            name = file.stem
            data = json.loads(file.read_text())
            if name in data:
                data = data[name]
            for row in data:
                row["rev"] = "workspace"
            properties = SKLEARN_PLOTS[name].get_properties()
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


def make_report(dvclive: "Live"):
    plots_path = Path(dvclive.plots_dir)

    renderers = []
    renderers.extend(get_params_renderers(dvclive.params_file))
    renderers.extend(get_metrics_renderers(dvclive.metrics_file))
    renderers.extend(get_scalar_renderers(plots_path / Metric.subfolder))
    renderers.extend(get_image_renderers(plots_path / Image.subfolder))
    renderers.extend(get_plot_renderers(plots_path / SKLearnPlot.subfolder))

    if dvclive.report_mode == "html":
        render_html(renderers, dvclive.report_file, refresh_seconds=5)
    elif dvclive.report_mode == "md":
        render_markdown(renderers, dvclive.report_file)
    else:
        raise ValueError(f"Invalid `mode` {dvclive.report_mode}.")
