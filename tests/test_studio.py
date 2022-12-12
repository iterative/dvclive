# pylint: disable=protected-access
# pylint: disable=unused-argument
import os

import pytest

from dvclive import Live, env
from dvclive.plots import Metric
from dvclive.studio import (
    VALID_URLS,
    _convert_to_studio_url,
    _get_remote_url,
    get_studio_repo_url,
)


def test_post_to_studio(tmp_dir, mocker, monkeypatch):
    dvc_repo = mocker.MagicMock()
    dvc_repo.scm.get_rev.return_value = "current_rev"
    mocker.patch("dvclive.live.get_dvc_repo", return_value=dvc_repo)
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
            "rev": "current_rev",
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
            "rev": "current_rev",
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
            "rev": "current_rev",
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


@pytest.mark.studio
@pytest.mark.parametrize(
    "url,expected",
    [
        (
            "https://github.com/USERNAME/REPOSITORY.git",
            "github:USERNAME/REPOSITORY",
        ),
        (
            "https://gitlab.com/USERNAME/REPOSITORY",
            "gitlab:USERNAME/REPOSITORY",
        ),
        (
            "https://bitbucket.org/USERNAME/REPOSITORY",
            "bitbucket:USERNAME/REPOSITORY",
        ),
        (
            "git@github.com:USERNAME/REPOSITORY.git",
            "github:USERNAME/REPOSITORY",
        ),
        (
            "git@gitlab.com:USERNAME/REPOSITORY.git",
            "gitlab:USERNAME/REPOSITORY",
        ),
        (
            "git@bitbucket.org:USERNAME/REPOSITORY.git",
            "bitbucket:USERNAME/REPOSITORY",
        ),
    ],
)
def test_convert_to_studio_url(url, expected, mocker):
    assert _convert_to_studio_url(url) == expected


def test_get_remote_url(tmp_dir):
    from git import Repo

    repo = Repo.clone_from("https://github.com/iterative/dvclive.git", tmp_dir)
    assert _get_remote_url(repo) == "https://github.com/iterative/dvclive.git"


def test_get_studio_repo_url_warnings(caplog, mocker):
    from git.exc import GitError

    mocker.patch("dvclive.studio._get_remote_url", side_effect=GitError())
    caplog.clear()
    get_studio_repo_url(None)
    assert caplog.records[0].message == (
        "Tried to find remote url for the active branch but failed.\n"
    )
    assert caplog.records[1].message == (
        "You can try manually setting the Studio repo url using the"
        " environment variable `STUDIO_REPO_URL`."
    )

    mocker.patch("dvclive.studio._get_remote_url", return_value="bad@repo:url")
    caplog.clear()
    get_studio_repo_url(None)
    assert caplog.records[0].message == (
        "Found invalid remote url for the active branch.\n"
        f" Supported urls must start with any of {VALID_URLS}"
    )
    assert caplog.records[1].message == (
        "You can try manually setting the Studio repo url using the"
        " environment variable `STUDIO_REPO_URL`."
    )
