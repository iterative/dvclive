import json
from pathlib import Path

from dvc_render.html import render_html
from dvc_render.image import ImageRenderer
from dvc_render.markdown import render_markdown
from dvc_render.table import TableRenderer
from dvc_render.vega import VegaRenderer

from dvclive.data import PLOTS, Image, Scalar
from dvclive.data.plot import Plot
from dvclive.utils import parse_tsv


def get_scalar_renderers(scalars_folder):
    renderers = []
    for suffix in Scalar.suffixes:
        for file in Path(scalars_folder).rglob(f"*{suffix}"):
            data = parse_tsv(file)
            for row in data:
                row["rev"] = "workspace"

            y = file.relative_to(scalars_folder).with_suffix("")
            y = y.as_posix()

            name = file.relative_to(scalars_folder.parent).with_suffix("")
            name = name.as_posix()
            name = name.replace(scalars_folder.name, "static")

            properties = {"x": "step", "y": y}
            renderers.append(VegaRenderer(data, name, **properties))
    return renderers


def get_image_renderers(images_folder):
    dvclive_path = images_folder.parent
    renderers = []
    for suffix in Image.suffixes:
        all_images = Path(images_folder).rglob(f"*{suffix}")
        for file in sorted(all_images):
            src = str(file.relative_to(dvclive_path))
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


def get_metrics_renderers(dvclive_summary):
    summary_path = Path(dvclive_summary)
    if summary_path.exists():
        return [
            TableRenderer(
                [json.loads(summary_path.read_text())], summary_path.name
            )
        ]
    return []


def make_report(dvclive_folder, dvclive_summary, output_path, mode):
    dvclive_path = Path(dvclive_folder)

    renderers = []
    renderers.extend(get_metrics_renderers(dvclive_summary))
    renderers.extend(get_scalar_renderers(dvclive_path / Scalar.subfolder))
    renderers.extend(get_image_renderers(dvclive_path / Image.subfolder))
    renderers.extend(get_plot_renderers(dvclive_path / Plot.subfolder))
    renderers.extend(get_metrics_renderers(dvclive_summary))

    if mode == "html":
        render_html(renderers, output_path, refresh_seconds=5)
    elif mode == "md":
        render_markdown(renderers, output_path)
    else:
        raise ValueError(f"Invalid `mode` {mode}.")
