# pylint: disable=protected-access
import logging
import os

from dvclive.serialize import load_yaml
from dvclive.utils import parse_metrics

logger = logging.getLogger(__name__)


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


def _to_dvc_format(plots):
    formatted = {}
    for k, v in plots.items():
        formatted[k] = {"data": v}
    return formatted


def get_studio_updates(live):
    plots, metrics = parse_metrics(live)
    latest_step = live._latest_studio_step

    if os.path.isfile(live.dvc_file):
        # Add prefix to match DVC's `repo.plos.show`.
        # See https://github.com/iterative/studio/issues/4981
        plots = {f"{live.dvc_file}::{name}": plot for name, plot in plots.items()}
    for name, plot in plots.items():
        datapoints = _get_unsent_datapoints(plot, latest_step)
        plots[name] = _cast_to_numbers(datapoints)

    metrics = {live.metrics_file: {"data": metrics}}

    if os.path.isfile(live.params_file):
        params = {live.params_file: load_yaml(live.params_file)}
    else:
        params = {}

    plots = _to_dvc_format(plots)

    return metrics, params, plots
