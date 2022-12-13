# pylint: disable=protected-access
import logging
from os import getenv

from dvclive.env import STUDIO_ENDPOINT
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


def _get_updates(live):
    plots, metrics = parse_metrics(live)
    latest_step = live._latest_studio_step  # pylint: disable=protected-access

    for name, plot in plots.items():
        datapoints = _get_unsent_datapoints(plot, latest_step)
        plots[name] = _cast_to_numbers(datapoints)

    metrics = {live.metrics_file: {"data": metrics}}
    plots = _to_dvc_format(plots)
    return metrics, plots


def post_to_studio(live, event_type) -> bool:
    import requests
    from requests.exceptions import RequestException

    data = {
        "type": event_type,
        "repo_url": live._studio_url,
        "rev": live._baseline_rev,
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
                "Authorization": f"token {live._studio_token}",
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


def _get_remote_url(git_repo):
    return git_repo.git.ls_remote("--get-url")


VALID_PREFIXES = ("https://", "git@")
VALID_PROVIDERS = ("github.com", "gitlab.com", "bitbucket.org")
VALID_URLS = [
    f"{prefix}{provider}"
    for prefix in VALID_PREFIXES
    for provider in VALID_PROVIDERS
]


def _convert_to_studio_url(remote_url):
    studio_url = ""
    for prefix in VALID_PREFIXES:
        for provider in VALID_PROVIDERS:
            if remote_url.startswith(f"{prefix}{provider}"):
                repo = remote_url.split(provider)[-1]
                repo = repo.rstrip(".git")
                repo = repo.lstrip("/")
                repo = repo.lstrip(":")
                studio_url = f"{provider.split('.')[0]}:{repo}"
    if not studio_url:
        raise ValueError
    return studio_url


def get_studio_repo_url(git_repo) -> str:
    from git.exc import GitError

    studio_url = ""
    try:
        remote_url = _get_remote_url(git_repo)
        studio_url = _convert_to_studio_url(remote_url)
    except GitError:
        logger.warning(
            "Tried to find remote url for the active branch but failed.\n"
        )
    except ValueError:
        logger.warning(
            "Found invalid remote url for the active branch.\n"
            f" Supported urls must start with any of {VALID_URLS}"
        )
    finally:
        if not studio_url:
            logger.warning(
                "You can try manually setting the Studio repo url using the"
                " environment variable `STUDIO_REPO_URL`."
            )
        return studio_url  # noqa: B012  # pylint:disable=lost-exception
