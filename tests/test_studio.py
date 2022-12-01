# pylint: disable=protected-access
# pylint: disable=unused-argument
import os

import pytest

from dvclive import Live, env
from dvclive.plots import Metric


def test_post_to_studio(tmp_dir, mocker, monkeypatch):
    mocker.patch("dvclive.live.get_dvc_repo")
    mocked_response = mocker.MagicMock()
    mocked_response.status_code = 200
    mocked_post = mocker.patch("requests.post", return_value=mocked_response)
    monkeypatch.setenv(env.STUDIO_ENDPOINT, "https://0.0.0.0")
    monkeypatch.setenv(env.STUDIO_REPO_URL, "STUDIO_REPO_URL")
    monkeypatch.setenv(env.STUDIO_TOKEN, "STUDIO_TOKEN")

    live = Live()

    scalar_path = os.path.join(live.plots_dir, Metric.subfolder, "foo.tsv")

    mocked_post.assert_called_with(
        "https://0.0.0.0",
        json={
            "type": "start",
            "repo_url": "STUDIO_REPO_URL",
            "rev": mocker.ANY,
            "client": "dvclive",
        },
        headers={
            "Authorization": "token STUDIO_TOKEN",
            "Content-type": "application/json",
        },
        timeout=5,
    )

    live.log_metric("foo", 1)

    live.next_step()
    mocked_post.assert_called_with(
        "https://0.0.0.0",
        json={
            "type": "data",
            "repo_url": "STUDIO_REPO_URL",
            "rev": mocker.ANY,
            "step": 0,
            "metrics": {live.metrics_file: {"data": {"step": 0, "foo": 1}}},
            "plots": {scalar_path: {"data": [{"step": 0, "foo": 1.0}]}},
            "client": "dvclive",
        },
        headers={
            "Authorization": "token STUDIO_TOKEN",
            "Content-type": "application/json",
        },
        timeout=5,
    )

    live.log_metric("foo", 2)

    live.next_step()
    mocked_post.assert_called_with(
        "https://0.0.0.0",
        json={
            "type": "data",
            "repo_url": "STUDIO_REPO_URL",
            "rev": mocker.ANY,
            "step": 1,
            "metrics": {live.metrics_file: {"data": {"step": 1, "foo": 2}}},
            "plots": {scalar_path: {"data": [{"step": 1, "foo": 2.0}]}},
            "client": "dvclive",
        },
        headers={
            "Authorization": "token STUDIO_TOKEN",
            "Content-type": "application/json",
        },
        timeout=5,
    )

    live.end()
    mocked_post.assert_called_with(
        "https://0.0.0.0",
        json={
            "type": "done",
            "repo_url": "STUDIO_REPO_URL",
            "rev": mocker.ANY,
            "client": "dvclive",
        },
        headers={
            "Authorization": "token STUDIO_TOKEN",
            "Content-type": "application/json",
        },
        timeout=5,
    )


def test_post_to_studio_failed_data_request(tmp_dir, mocker, monkeypatch):
    mocker.patch("dvclive.live.get_dvc_repo")
    valid_response = mocker.MagicMock()
    valid_response.status_code = 200
    mocker.patch("requests.post", return_value=valid_response)
    monkeypatch.setenv(env.STUDIO_ENDPOINT, "https://0.0.0.0")
    monkeypatch.setenv(env.STUDIO_REPO_URL, "STUDIO_REPO_URL")
    monkeypatch.setenv(env.STUDIO_TOKEN, "STUDIO_TOKEN")

    live = Live()

    scalar_path = os.path.join(live.plots_dir, Metric.subfolder, "foo.tsv")

    error_response = mocker.MagicMock()
    error_response.status_code = 400
    mocker.patch("requests.post", return_value=error_response)
    live.log_metric("foo", 1)
    live.next_step()

    mocked_post = mocker.patch("requests.post", return_value=valid_response)
    live.log_metric("foo", 2)
    live.next_step()
    mocked_post.assert_called_with(
        "https://0.0.0.0",
        json={
            "type": "data",
            "repo_url": "STUDIO_REPO_URL",
            "rev": mocker.ANY,
            "step": 1,
            "metrics": {live.metrics_file: {"data": {"step": 1, "foo": 2}}},
            "plots": {
                scalar_path: {
                    "data": [
                        {"step": 0, "foo": 1.0},
                        {"step": 1, "foo": 2.0},
                    ]
                }
            },
            "client": "dvclive",
        },
        headers={
            "Authorization": "token STUDIO_TOKEN",
            "Content-type": "application/json",
        },
        timeout=5,
    )


def test_post_to_studio_failed_start_request(tmp_dir, mocker, monkeypatch):
    mocker.patch("dvclive.live.get_dvc_repo")
    mocked_response = mocker.MagicMock()
    mocked_response.status_code = 400
    mocked_post = mocker.patch("requests.post", return_value=mocked_response)

    monkeypatch.setenv(env.STUDIO_ENDPOINT, "https://0.0.0.0")
    monkeypatch.setenv(env.STUDIO_REPO_URL, "STUDIO_REPO_URL")
    monkeypatch.setenv(env.STUDIO_TOKEN, "STUDIO_TOKEN")

    live = Live()

    live.log_metric("foo", 1)
    live.next_step()

    live.log_metric("foo", 2)
    live.next_step()

    assert mocked_post.call_count == 1


def test_post_to_studio_end_only_once(tmp_dir, mocker, monkeypatch):
    mocker.patch("dvclive.live.get_dvc_repo")
    valid_response = mocker.MagicMock()
    valid_response.status_code = 200
    mocked_post = mocker.patch("requests.post", return_value=valid_response)
    monkeypatch.setenv(env.STUDIO_ENDPOINT, "https://0.0.0.0")
    monkeypatch.setenv(env.STUDIO_REPO_URL, "STUDIO_REPO_URL")
    monkeypatch.setenv(env.STUDIO_TOKEN, "STUDIO_TOKEN")

    with Live() as live:
        live.log_metric("foo", 1)
        live.next_step()

    assert mocked_post.call_count == 3
    live.end()
    assert mocked_post.call_count == 3


@pytest.mark.studio
def test_post_to_studio_skip_on_env_var(tmp_dir, mocker, monkeypatch):
    mocker.patch("dvclive.live.get_dvc_repo")
    valid_response = mocker.MagicMock()
    valid_response.status_code = 200
    mocked_post = mocker.patch("requests.post", return_value=valid_response)
    monkeypatch.setenv(env.STUDIO_ENDPOINT, "https://0.0.0.0")
    monkeypatch.setenv(env.STUDIO_REPO_URL, "STUDIO_REPO_URL")
    monkeypatch.setenv(env.STUDIO_TOKEN, "STUDIO_TOKEN")

    monkeypatch.setenv(env.DVC_EXP_BASELINE_REV, "foo")
    monkeypatch.setenv(env.DVC_EXP_NAME, "bar")

    with Live() as live:
        live.log_metric("foo", 1)
        live.next_step()

    assert mocked_post.call_count == 1
