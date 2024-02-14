# ruff: noqa: SLF001
import base64
import logging
import math
import os

from dvc_studio_client.config import get_studio_config
from dvc_studio_client.post_live_metrics import post_live_metrics

from dvclive.serialize import load_yaml
from dvclive.utils import parse_metrics, rel_path

logger = logging.getLogger("dvclive")


def _cast_to_numbers(datapoints):
    for datapoint in datapoints:
        for k, v in datapoint.items():
            if k in ["step", "timestamp"]:
                datapoint[k] = int(v)
            else:
                float_v = float(v)
                if math.isnan(float_v) or math.isinf(float_v):
                    datapoint[k] = str(v)
                else:
                    datapoint[k] = float_v
    return datapoints


def _adapt_path(live, name):
    if live._dvc_repo is not None:
        name = rel_path(name, live._dvc_repo.root_dir)
    return name


def _adapt_image(image_path):
    with open(image_path, "rb") as fobj:
        return base64.b64encode(fobj.read()).decode("utf-8")


def _adapt_images(live):
    return {
        _adapt_path(live, image.output_path): {"image": _adapt_image(image.output_path)}
        for image in live._images.values()
        if image.step > live._latest_studio_step
    }


def get_studio_updates(live):
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

    plots_to_send = {}
    live._num_points_read_from_file = {}
    for name, plot in plots.items():
        path = _adapt_path(live, name)
        nb_points_already_sent = live._num_points_sent_to_studio.get(path, 0)
        plots_to_send[path] = _cast_to_numbers(plot[nb_points_already_sent:])
        live._num_points_read_from_file.update({path: len(plot)})

    plots_to_send = {k: {"data": v} for k, v in plots_to_send.items()}
    plots_to_send.update(_adapt_images(live))

    return metrics, params, plots_to_send


def get_dvc_studio_config(live):
    config = {}
    if live._dvc_repo:
        config = live._dvc_repo.config.get("studio")
    return get_studio_config(dvc_studio_config=config)


def post_to_studio(live, event):
    if event in live._studio_events_to_skip:
        return

    kwargs = {}
    if event == "start" and live._exp_message:
        kwargs["message"] = live._exp_message
    elif event == "data":
        metrics, params, plots = get_studio_updates(live)
        kwargs["step"] = live.step
        kwargs["metrics"] = metrics
        kwargs["params"] = params
        kwargs["plots"] = plots
    elif event == "done" and live._experiment_rev:
        kwargs["experiment_rev"] = live._experiment_rev

    response = post_live_metrics(
        event,
        live._baseline_rev,
        live._exp_name,
        "dvclive",
        dvc_studio_config=live._dvc_studio_config,
        **kwargs,
    )
    if not response:
        logger.warning(f"`post_to_studio` `{event}` failed.")
        if event == "start":
            live._studio_events_to_skip.add("start")
            live._studio_events_to_skip.add("data")
            live._studio_events_to_skip.add("done")
    elif event == "data":
        for path, num_points in live._num_points_read_from_file.items():
            live._num_points_sent_to_studio[path] = num_points
        live._latest_studio_step = live.step

    if event == "done":
        live._studio_events_to_skip.add("done")
        live._studio_events_to_skip.add("data")
