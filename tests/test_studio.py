# pylint: disable=protected-access
# pylint: disable=unused-argument
import os

import pytest
from dvc_studio_client.env import STUDIO_ENDPOINT, STUDIO_REPO_URL, STUDIO_TOKEN

from dvclive import Live
from dvclive.env import DVC_EXP_BASELINE_REV, DVC_EXP_NAME
from dvclive.plots import Metric


def test_post_to_studio(tmp_dir, mocker, monkeypatch):
    dvc_repo = mocker.MagicMock()
    dvc_repo.scm.get_rev.return_value = "f" * 40
    mocker.patch("dvclive.live.get_dvc_repo", return_value=dvc_repo)
    mocked_response = mocker.MagicMock()
    mocked_response.status_code = 200
    mocked_post = mocker.patch("requests.post", return_value=mocked_response)
    monkeypatch.setenv(STUDIO_ENDPOINT, "https://0.0.0.0")
    monkeypatch.setenv(STUDIO_REPO_URL, "STUDIO_REPO_URL")
    monkeypatch.setenv(STUDIO_TOKEN, "STUDIO_TOKEN")

    live = Live()
    live.log_param("fooparam", 1)
    scalar_path = os.path.join(live.plots_dir, Metric.subfolder, "foo.tsv")

    mocked_post.assert_called_with(
        "https://0.0.0.0",
        json={
            "type": "start",
            "repo_url": "STUDIO_REPO_URL",
            "baseline_sha": "f" * 40,
            "name": "dvclive-exp",
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
            "baseline_sha": "f" * 40,
            "name": "dvclive-exp",
            "step": 0,
            "metrics": {live.metrics_file: {"data": {"step": 0, "foo": 1}}},
            "params": {live.params_file: {"fooparam": 1}},
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
            "baseline_sha": "f" * 40,
            "name": "dvclive-exp",
            "step": 1,
            "metrics": {live.metrics_file: {"data": {"step": 1, "foo": 2}}},
            "params": {live.params_file: {"fooparam": 1}},
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
            "baseline_sha": "f" * 40,
            "name": "dvclive-exp",
            "client": "dvclive",
        },
        headers={
            "Authorization": "token STUDIO_TOKEN",
            "Content-type": "application/json",
        },
        timeout=5,
    )


def test_post_to_studio_failed_data_request(tmp_dir, mocker, monkeypatch):
    dvc_repo = mocker.MagicMock()
    dvc_repo.scm.get_rev.return_value = "f" * 40
    mocker.patch("dvclive.live.get_dvc_repo", return_value=dvc_repo)
    valid_response = mocker.MagicMock()
    valid_response.status_code = 200
    mocker.patch("requests.post", return_value=valid_response)
    monkeypatch.setenv(STUDIO_ENDPOINT, "https://0.0.0.0")
    monkeypatch.setenv(STUDIO_REPO_URL, "STUDIO_REPO_URL")
    monkeypatch.setenv(STUDIO_TOKEN, "STUDIO_TOKEN")

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
            "baseline_sha": "f" * 40,
            "name": "dvclive-exp",
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
    dvc_repo = mocker.MagicMock()
    dvc_repo.scm.get_rev.return_value = "f" * 40
    mocker.patch("dvclive.live.get_dvc_repo", return_value=dvc_repo)
    mocked_response = mocker.MagicMock()
    mocked_response.status_code = 400
    mocked_post = mocker.patch("requests.post", return_value=mocked_response)

    monkeypatch.setenv(STUDIO_ENDPOINT, "https://0.0.0.0")
    monkeypatch.setenv(STUDIO_REPO_URL, "STUDIO_REPO_URL")
    monkeypatch.setenv(STUDIO_TOKEN, "STUDIO_TOKEN")

    live = Live()

    live.log_metric("foo", 1)
    live.next_step()

    live.log_metric("foo", 2)
    live.next_step()

    assert mocked_post.call_count == 1


def test_post_to_studio_end_only_once(tmp_dir, mocker, monkeypatch):
    dvc_repo = mocker.MagicMock()
    dvc_repo.scm.get_rev.return_value = "f" * 40
    mocker.patch("dvclive.live.get_dvc_repo", return_value=dvc_repo)
    valid_response = mocker.MagicMock()
    valid_response.status_code = 200
    mocked_post = mocker.patch("requests.post", return_value=valid_response)
    monkeypatch.setenv(STUDIO_ENDPOINT, "https://0.0.0.0")
    monkeypatch.setenv(STUDIO_REPO_URL, "STUDIO_REPO_URL")
    monkeypatch.setenv(STUDIO_TOKEN, "STUDIO_TOKEN")

    with Live() as live:
        live.log_metric("foo", 1)
        live.next_step()

    assert mocked_post.call_count == 3
    live.end()
    assert mocked_post.call_count == 3


@pytest.mark.studio
def test_post_to_studio_skip_on_env_var(tmp_dir, mocker, monkeypatch):
    dvc_repo = mocker.MagicMock()
    dvc_repo.scm.get_rev.return_value = "f" * 40
    mocker.patch("dvclive.live.get_dvc_repo", return_value=dvc_repo)
    valid_response = mocker.MagicMock()
    valid_response.status_code = 200
    mocked_post = mocker.patch("requests.post", return_value=valid_response)
    monkeypatch.setenv(STUDIO_ENDPOINT, "https://0.0.0.0")
    monkeypatch.setenv(STUDIO_REPO_URL, "STUDIO_REPO_URL")
    monkeypatch.setenv(STUDIO_TOKEN, "STUDIO_TOKEN")

    monkeypatch.setenv(DVC_EXP_BASELINE_REV, "foo")
    monkeypatch.setenv(DVC_EXP_NAME, "bar")

    with Live() as live:
        live.log_metric("foo", 1)
        live.next_step()

    assert mocked_post.call_count == 0


@pytest.mark.studio
def test_post_to_studio_skip_if_no_token(tmp_dir, mocker, monkeypatch):
    dvc_repo = mocker.MagicMock()
    dvc_repo.scm.get_rev.return_value = "f" * 40
    mocker.patch("dvclive.live.get_dvc_repo", return_value=dvc_repo)

    mocked_post = mocker.patch("dvclive.live.post_live_metrics", return_value=None)

    monkeypatch.setenv(DVC_EXP_BASELINE_REV, "foo")
    monkeypatch.setenv(DVC_EXP_NAME, "bar")

    with Live() as live:
        live.log_metric("foo", 1)
        live.next_step()

    assert mocked_post.call_count == 0


def test_post_to_studio_include_prefix_if_needed(tmp_dir, mocker, monkeypatch):
    dvc_repo = mocker.MagicMock()
    dvc_repo.scm.get_rev.return_value = "f" * 40
    dvc_repo.index.stages = []
    dvc_repo.scm.get_ref.return_value = None
    mocker.patch("dvclive.live.get_dvc_repo", return_value=dvc_repo)

    mocked_response = mocker.MagicMock()
    mocked_response.status_code = 200
    mocked_post = mocker.patch("requests.post", return_value=mocked_response)

    monkeypatch.setenv(STUDIO_ENDPOINT, "https://0.0.0.0")
    monkeypatch.setenv(STUDIO_REPO_URL, "STUDIO_REPO_URL")
    monkeypatch.setenv(STUDIO_TOKEN, "STUDIO_TOKEN")

    # Create dvclive/dvc.yaml
    live = Live("custom_dir", save_dvc_exp=True)
    live.log_metric("foo", 1)
    live.next_step()

    scalar_path = os.path.join(live.plots_dir, Metric.subfolder, "foo.tsv")
    scalar_name = f"{live.dvc_file}::{scalar_path}"
    mocked_post.assert_called_with(
        "https://0.0.0.0",
        json={
            "type": "data",
            "repo_url": "STUDIO_REPO_URL",
            "baseline_sha": "f" * 40,
            "name": live._exp_name,
            "step": 0,
            "metrics": {live.metrics_file: {"data": {"step": 0, "foo": 1}}},
            "plots": {scalar_name: {"data": [{"step": 0, "foo": 1.0}]}},
            "client": "dvclive",
        },
        headers={
            "Authorization": "token STUDIO_TOKEN",
            "Content-type": "application/json",
        },
        timeout=5,
    )
