import json
from pathlib import Path

from dvc_render.html import render_html
from dvc_render.image import ImageRenderer
from dvc_render.vega import VegaRenderer

from dvclive.data import PLOTS, Image, Scalar
from dvclive.data.plot import Plot
from dvclive.utils import parse_tsv, to_base64_url


def get_scalar_renderers(scalars_folder):
    renderers = []
    for suffix in Scalar.suffixes:
        for file in Path(scalars_folder).rglob(f"*{suffix}"):
            data = parse_tsv(file)
            for row in data:
                row["rev"] = "workspace"
            rel = file.relative_to(scalars_folder).with_suffix("")
            name = str(rel.as_posix())
            properties = {"x": "step", "y": name}
            renderers.append(VegaRenderer(data, name, **properties))
    return renderers


def get_image_renderers(images_folder):
    renderers = []
    for suffix in Image.suffixes:
        for file in Path(images_folder).rglob(f"*{suffix}"):
            name = str(file.relative_to(images_folder))
            data = [
                {
                    ImageRenderer.SRC_FIELD: to_base64_url(file),
                    ImageRenderer.TITLE_FIELD: name,
                }
            ]
            renderers.append(ImageRenderer(data, name))
    return renderers


def get_plot_renderers(plots_folder):
    renderers = []
    for suffix in Plot.suffixes:
        for file in Path(plots_folder).rglob(f"*{suffix}"):
            name = file.stem
            data = json.loads(file.read_text())
            if name in data:
                data = data[name]
            for row in data:
                row["rev"] = "workspace"
            properties = PLOTS[name].get_properties()
            renderers.append(VegaRenderer(data, name, **properties))
    return renderers


def get_metrics(dvclive_summary):
    summary_path = Path(dvclive_summary)
    if summary_path.exists():
        return {
            "": {
                "data": {
                    summary_path.name: {
                        "data": json.loads(summary_path.read_text())
                    }
                }
            }
        }
    return {}


def html_report(dvclive_folder, dvclive_summary, output_html_path):
    dvclive_path = Path(dvclive_folder)
    renderers = []
    renderers.extend(get_scalar_renderers(dvclive_path / Scalar.subfolder))
    renderers.extend(get_image_renderers(dvclive_path / Image.subfolder))
    renderers.extend(get_plot_renderers(dvclive_path / Plot.subfolder))

    metrics = get_metrics(dvclive_summary)
    render_html(
        renderers, output_html_path, metrics=metrics, refresh_seconds=5
    )
