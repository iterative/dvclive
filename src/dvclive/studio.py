from os import getenv

from dvclive.env import STUDIO_ENDPOINT
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


def _to_dvc_format(plots):
    formatted = {}
    for k, v in plots.items():
        formatted[k] = {"data": v}
    return formatted


def _get_updates(live):
    plots, metrics = parse_metrics(live)
    latest_step = live._latest_studio_step  # pylint: disable=protected-access

    for name, plot in plots.items():
        datapoints = _get_unsent_datapoints(plot, latest_step)
        plots[name] = _cast_to_numbers(datapoints)

    metrics = {live.metrics_file: {"data": metrics}}
    plots = _to_dvc_format(plots)
    return metrics, plots


def post_to_studio(live, event_type, logger) -> bool:
    import requests
    from requests.exceptions import RequestException

    data = {
        "type": event_type,
        "repo_url": live.studio_url,
        "rev": live.rev,
        "client": "dvclive",
    }

    if event_type == "data":
        metrics, plots = _get_updates(live)
        data["metrics"] = metrics
        data["plots"] = plots
        data["step"] = live.step

    logger.debug(f"post_to_studio `{event_type=}`")

    try:
        response = requests.post(
            getenv(STUDIO_ENDPOINT, "https://studio.iterative.ai/api/live"),
            json=data,
            headers={
                "Content-type": "application/json",
                "Authorization": f"token {live.studio_token}",
            },
            timeout=5,
        )
    except RequestException:
        return False

    message = response.content.decode()
    logger.debug(
        f"post_to_studio: {response.status_code=}" f", {message=}"
        if message
        else ""
    )

    return response.status_code == 200
