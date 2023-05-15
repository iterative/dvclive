# ruff: noqa: SLF001
import base64
import os
from pathlib import Path

from dvc_studio_client.post_live_metrics import get_studio_config

from dvclive.serialize import load_yaml
from dvclive.utils import parse_metrics


def _get_unsent_datapoints(plot, latest_step):
    return [x for x in plot if int(x["step"]) > latest_step]


def _cast_to_numbers(datapoints):
    for datapoint in datapoints:
        for k, v in datapoint.items():
            if k == "step":
                datapoint[k] = int(v)
            elif k == "timestamp":
                continue
            else:
                datapoint[k] = float(v)
    return datapoints


def _rel_path(path, dvc_root_path):
    absolute_path = Path(path).resolve()
    return str(absolute_path.relative_to(dvc_root_path).as_posix())


def _adapt_plot_name(live, name):
    if live._dvc_repo is not None:
        name = _rel_path(name, live._dvc_repo.root_dir)
    if os.path.isfile(live.dvc_file):
        dvc_file = live.dvc_file
        if live._dvc_repo is not None:
            dvc_file = _rel_path(live.dvc_file, live._dvc_repo.root_dir)
        name = f"{dvc_file}::{name}"
    return name


def _adapt_plot_datapoints(live, plot):
    datapoints = _get_unsent_datapoints(plot, live._latest_studio_step)
    return _cast_to_numbers(datapoints)


def _adapt_image(image_path):
    with open(image_path, "rb") as fobj:
        return base64.b64encode(fobj.read()).decode("utf-8")


def _adapt_images(live):
    return {
        _adapt_plot_name(live, image.output_path): {
            "image": _adapt_image(image.output_path)
        }
        for image in live._images.values()
        if image.step > live._latest_studio_step
    }


def get_studio_updates(live):
    if os.path.isfile(live.params_file):
        params_file = live.params_file
        if live._dvc_repo is not None:
            params_file = _rel_path(params_file, live._dvc_repo.root_dir)
        params = {params_file: load_yaml(live.params_file)}
    else:
        params = {}

    plots, metrics = parse_metrics(live)

    metrics_file = live.metrics_file
    if live._dvc_repo is not None:
        metrics_file = _rel_path(metrics_file, live._dvc_repo.root_dir)
    metrics = {metrics_file: {"data": metrics}}

    plots = {
        _adapt_plot_name(live, name): _adapt_plot_datapoints(live, plot)
        for name, plot in plots.items()
    }
    plots = {k: {"data": v} for k, v in plots.items()}

    plots.update(_adapt_images(live))

    return metrics, params, plots


def get_dvc_studio_config(live):
    config = {}
    if live._dvc_repo:
        config = live._dvc_repo.config.get("studio")
    return get_studio_config(dvc_studio_config=config)
