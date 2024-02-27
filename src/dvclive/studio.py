# ruff: noqa: SLF001
from __future__ import annotations
import base64
import logging
import math
import os
from pathlib import PureWindowsPath
from typing import TYPE_CHECKING, Literal, Mapping

from dvc_studio_client.config import get_studio_config
from dvc_studio_client.post_live_metrics import post_live_metrics

if TYPE_CHECKING:
    from dvclive.live import Live
from dvclive.serialize import load_yaml
from dvclive.utils import parse_metrics, rel_path, StrPath

logger = logging.getLogger("dvclive")


def _get_unsent_datapoints(plot: Mapping, latest_step: int):
    return [x for x in plot if int(x["step"]) > latest_step]


def _cast_to_numbers(datapoints: Mapping):
    for datapoint in datapoints:
        for k, v in datapoint.items():
            if k == "step":
                datapoint[k] = int(v)
            elif k == "timestamp":
                continue
            else:
                float_v = float(v)
                if math.isnan(float_v) or math.isinf(float_v):
                    datapoint[k] = str(v)
                else:
                    datapoint[k] = float_v
    return datapoints


def _adapt_path(live: Live, name: StrPath):
    if live._dvc_repo is not None:
        name = rel_path(name, live._dvc_repo.root_dir)
    if os.name == "nt":
        name = str(PureWindowsPath(name).as_posix())
    return name


def _adapt_plot_datapoints(live: Live, plot: Mapping):
    datapoints = _get_unsent_datapoints(plot, live._latest_studio_step)
    return _cast_to_numbers(datapoints)


def _adapt_image(image_path: StrPath):
    with open(image_path, "rb") as fobj:
        return base64.b64encode(fobj.read()).decode("utf-8")


def _adapt_images(live: Live):
    return {
        _adapt_path(live, image.output_path): {"image": _adapt_image(image.output_path)}
        for image in live._images.values()
        if image.step > live._latest_studio_step
    }


def get_studio_updates(live: Live):
    if os.path.isfile(live.params_file):
        params_file = live.params_file
        params_file = _adapt_path(live, params_file)
        params = {params_file: load_yaml(live.params_file)}
    else:
        params = {}

    plots, metrics = parse_metrics(live)

    metrics_file = live.metrics_file
    metrics_file = _adapt_path(live, metrics_file)
    metrics = {metrics_file: {"data": metrics}}

    plots = {
        _adapt_path(live, name): _adapt_plot_datapoints(live, plot)
        for name, plot in plots.items()
    }
    plots = {k: {"data": v} for k, v in plots.items()}

    plots.update(_adapt_images(live))

    return metrics, params, plots


def get_dvc_studio_config(live: Live):
    config = {}
    if live._dvc_repo:
        config = live._dvc_repo.config.get("studio")
    return get_studio_config(dvc_studio_config=config)


def post_to_studio(live: Live, event: Literal["start", "data", "done"]):  # noqa: C901
    if event in live._studio_events_to_skip:
        return

    kwargs = {}
    if event == "start":
        if message := live._exp_message:
            kwargs["message"] = message
        if subdir := live._subdir:
            kwargs["subdir"] = subdir
    elif event == "data":
        metrics, params, plots = get_studio_updates(live)
        kwargs["step"] = live.step  # type: ignore
        kwargs["metrics"] = metrics
        kwargs["params"] = params
        kwargs["plots"] = plots
    elif event == "done" and live._experiment_rev:
        kwargs["experiment_rev"] = live._experiment_rev

    response = post_live_metrics(
        event,
        live._baseline_rev,
        live._exp_name,  # type: ignore
        "dvclive",
        dvc_studio_config=live._dvc_studio_config,
        **kwargs,  # type: ignore
    )
    if not response:
        logger.warning(f"`post_to_studio` `{event}` failed.")
        if event == "start":
            live._studio_events_to_skip.add("start")
            live._studio_events_to_skip.add("data")
            live._studio_events_to_skip.add("done")
    elif event == "data":
        live._latest_studio_step = live.step

    if event == "done":
        live._studio_events_to_skip.add("done")
        live._studio_events_to_skip.add("data")
