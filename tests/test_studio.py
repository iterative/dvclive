from pathlib import Path

import pytest
from dvc_studio_client.env import DVC_STUDIO_REPO_URL, DVC_STUDIO_TOKEN
from dvc_studio_client.post_live_metrics import STUDIO_URL
from PIL import Image as ImagePIL

from dvclive import Live
from dvclive.env import DVC_EXP_BASELINE_REV, DVC_EXP_NAME
from dvclive.plots import Image, Metric
from dvclive.studio import _adapt_image, get_dvc_studio_config


def test_post_to_studio(tmp_dir, mocked_dvc_repo, mocked_studio_post):
    live = Live(save_dvc_exp=True)
    live.log_param("fooparam", 1)

    dvc_path = Path(live.dvc_file).as_posix()
    metrics_path = Path(live.metrics_file).as_posix()
    params_path = Path(live.params_file).as_posix()
    foo_path = (Path(live.plots_dir) / Metric.subfolder / "foo.tsv").as_posix()

    mocked_post, _ = mocked_studio_post

    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        json={
            "type": "start",
            "repo_url": "STUDIO_REPO_URL",
            "baseline_sha": "f" * 40,
            "name": live._exp_name,
            "client": "dvclive",
        },
        headers={
            "Authorization": "token STUDIO_TOKEN",
            "Content-type": "application/json",
        },
        timeout=(30, 5),
    )

    live.log_metric("foo", 1)

    live.next_step()
    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        json={
            "type": "data",
            "repo_url": "STUDIO_REPO_URL",
            "baseline_sha": "f" * 40,
            "name": live._exp_name,
            "step": 0,
            "metrics": {metrics_path: {"data": {"step": 0, "foo": 1}}},
            "params": {params_path: {"fooparam": 1}},
            "plots": {f"{dvc_path}::{foo_path}": {"data": [{"step": 0, "foo": 1.0}]}},
            "client": "dvclive",
        },
        headers={
            "Authorization": "token STUDIO_TOKEN",
            "Content-type": "application/json",
        },
        timeout=(30, 5),
    )

    live.log_metric("foo", 2)

    live.next_step()
    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        json={
            "type": "data",
            "repo_url": "STUDIO_REPO_URL",
            "baseline_sha": "f" * 40,
            "name": live._exp_name,
            "step": 1,
            "metrics": {metrics_path: {"data": {"step": 1, "foo": 2}}},
            "params": {params_path: {"fooparam": 1}},
            "plots": {f"{dvc_path}::{foo_path}": {"data": [{"step": 1, "foo": 2.0}]}},
            "client": "dvclive",
        },
        headers={
            "Authorization": "token STUDIO_TOKEN",
            "Content-type": "application/json",
        },
        timeout=(30, 5),
    )

    live.end()
    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        json={
            "type": "done",
            "repo_url": "STUDIO_REPO_URL",
            "baseline_sha": "f" * 40,
            "experiment_rev": live._experiment_rev,
            "name": live._exp_name,
            "client": "dvclive",
        },
        headers={
            "Authorization": "token STUDIO_TOKEN",
            "Content-type": "application/json",
        },
        timeout=(30, 5),
    )


def test_post_to_studio_failed_data_request(
    tmp_dir, mocker, mocked_dvc_repo, mocked_studio_post
):
    mocked_post, valid_response = mocked_studio_post

    live = Live(save_dvc_exp=True)

    dvc_path = Path(live.dvc_file).as_posix()
    metrics_path = Path(live.metrics_file).as_posix()
    foo_path = (Path(live.plots_dir) / Metric.subfolder / "foo.tsv").as_posix()

    error_response = mocker.MagicMock()
    error_response.status_code = 400
    mocker.patch("requests.post", return_value=error_response)
    live.log_metric("foo", 1)
    live.next_step()

    mocked_post = mocker.patch("requests.post", return_value=valid_response)
    live.log_metric("foo", 2)
    live.next_step()
    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        json={
            "type": "data",
            "repo_url": "STUDIO_REPO_URL",
            "baseline_sha": "f" * 40,
            "name": live._exp_name,
            "step": 1,
            "metrics": {metrics_path: {"data": {"step": 1, "foo": 2}}},
            "plots": {
                f"{dvc_path}::{foo_path}": {
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
        timeout=(30, 5),
    )


def test_post_to_studio_failed_start_request(
    tmp_dir, mocker, mocked_dvc_repo, mocked_studio_post
):
    mocked_response = mocker.MagicMock()
    mocked_response.status_code = 400
    mocked_post = mocker.patch("requests.post", return_value=mocked_response)

    live = Live(save_dvc_exp=True)

    live.log_metric("foo", 1)
    live.next_step()

    live.log_metric("foo", 2)
    live.next_step()

    assert mocked_post.call_count == 1


def test_post_to_studio_end_only_once(tmp_dir, mocked_dvc_repo, mocked_studio_post):
    mocked_post, _ = mocked_studio_post
    with Live(save_dvc_exp=True) as live:
        live.log_metric("foo", 1)
        live.next_step()

    assert mocked_post.call_count == 3
    live.end()
    assert mocked_post.call_count == 3


@pytest.mark.studio()
def test_post_to_studio_skip_on_env_var(
    tmp_dir, mocked_dvc_repo, mocked_studio_post, monkeypatch
):
    mocked_post, _ = mocked_studio_post

    monkeypatch.setenv(DVC_EXP_BASELINE_REV, "f" * 40)
    monkeypatch.setenv(DVC_EXP_NAME, "bar")

    with Live() as live:
        live.log_metric("foo", 1)

    assert mocked_post.call_count == 1


@pytest.mark.studio()
def test_post_to_studio_dvc_studio_config(
    tmp_dir, mocker, mocked_dvc_repo, mocked_studio_post, monkeypatch
):
    mocked_post, _ = mocked_studio_post

    monkeypatch.setenv(DVC_EXP_BASELINE_REV, "f" * 40)
    monkeypatch.setenv(DVC_EXP_NAME, "bar")

    mocked_dvc_repo.config = {"studio": {"token": "token"}}

    with Live() as live:
        live.log_metric("foo", 1)

    assert mocked_post.call_count == 1


@pytest.mark.studio()
def test_post_to_studio_skip_if_no_token(
    tmp_dir,
    mocker,
    monkeypatch,
    mocked_dvc_repo,
):
    mocked_post = mocker.patch("dvclive.live.post_live_metrics", return_value=None)

    monkeypatch.setenv(DVC_EXP_BASELINE_REV, "f" * 40)
    monkeypatch.setenv(DVC_EXP_NAME, "bar")

    mocked_dvc_repo.config = {}

    with Live() as live:
        live.log_metric("foo", 1)
        live.next_step()

    assert mocked_post.call_count == 0


def test_post_to_studio_include_prefix_if_needed(
    tmp_dir, mocked_dvc_repo, mocked_studio_post
):
    mocked_post, _ = mocked_studio_post
    # Create dvclive/dvc.yaml
    live = Live("custom_dir", save_dvc_exp=True)
    live.log_metric("foo", 1)
    live.next_step()

    dvc_path = Path(live.dvc_file).as_posix()
    metrics_path = Path(live.metrics_file).as_posix()
    foo_path = (Path(live.plots_dir) / Metric.subfolder / "foo.tsv").as_posix()

    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        json={
            "type": "data",
            "repo_url": "STUDIO_REPO_URL",
            "baseline_sha": "f" * 40,
            "name": live._exp_name,
            "step": 0,
            "metrics": {metrics_path: {"data": {"step": 0, "foo": 1}}},
            "plots": {f"{dvc_path}::{foo_path}": {"data": [{"step": 0, "foo": 1.0}]}},
            "client": "dvclive",
        },
        headers={
            "Authorization": "token STUDIO_TOKEN",
            "Content-type": "application/json",
        },
        timeout=(30, 5),
    )


def test_post_to_studio_shorten_names(tmp_dir, mocked_dvc_repo, mocked_studio_post):
    mocked_post, _ = mocked_studio_post

    live = Live(save_dvc_exp=True)
    live.log_metric("eval/loss", 1)
    live.next_step()

    dvc_path = Path(live.dvc_file).as_posix()
    metrics_path = Path(live.metrics_file).as_posix()
    plots_path = Path(live.plots_dir)
    loss_path = (plots_path / Metric.subfolder / "eval/loss.tsv").as_posix()

    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        json={
            "type": "data",
            "repo_url": "STUDIO_REPO_URL",
            "baseline_sha": "f" * 40,
            "name": live._exp_name,
            "step": 0,
            "metrics": {metrics_path: {"data": {"step": 0, "eval": {"loss": 1}}}},
            "plots": {f"{dvc_path}::{loss_path}": {"data": [{"step": 0, "loss": 1.0}]}},
            "client": "dvclive",
        },
        headers={
            "Authorization": "token STUDIO_TOKEN",
            "Content-type": "application/json",
        },
        timeout=(30, 5),
    )


@pytest.mark.studio()
def test_post_to_studio_inside_dvc_exp(
    tmp_dir, mocker, monkeypatch, mocked_studio_post
):
    mocked_post, _ = mocked_studio_post
    mocker.patch("dvclive.live.get_dvc_repo", return_value=None)

    monkeypatch.setenv(DVC_EXP_BASELINE_REV, "f" * 40)
    monkeypatch.setenv(DVC_EXP_NAME, "bar")

    with Live() as live:
        live.log_metric("foo", 1)

    assert mocked_post.call_count == 1


@pytest.mark.studio()
def test_post_to_studio_inside_subdir(
    tmp_dir, dvc_repo, mocker, monkeypatch, mocked_studio_post
):
    mocked_post, _ = mocked_studio_post
    subdir = tmp_dir / "subdir"
    subdir.mkdir()
    monkeypatch.chdir(subdir)

    live = Live(save_dvc_exp=True)
    live.log_metric("foo", 1)
    live.next_step()

    dvc_path = Path(live.dvc_file).as_posix()
    metrics_path = Path(live.metrics_file).as_posix()
    foo_path = (Path(live.plots_dir) / Metric.subfolder / "foo.tsv").as_posix()

    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        json={
            "type": "data",
            "repo_url": "STUDIO_REPO_URL",
            "baseline_sha": live._baseline_rev,
            "name": live._exp_name,
            "step": 0,
            "metrics": {f"subdir/{metrics_path}": {"data": {"step": 0, "foo": 1}}},
            "plots": {
                f"subdir/{dvc_path}::subdir/{foo_path}": {
                    "data": [{"step": 0, "foo": 1.0}]
                }
            },
            "client": "dvclive",
        },
        headers={
            "Authorization": "token STUDIO_TOKEN",
            "Content-type": "application/json",
        },
        timeout=(30, 5),
    )


@pytest.mark.studio()
def test_post_to_studio_inside_subdir_dvc_exp(
    tmp_dir, dvc_repo, monkeypatch, mocked_studio_post
):
    mocked_post, _ = mocked_studio_post
    subdir = tmp_dir / "subdir"
    subdir.mkdir()
    monkeypatch.chdir(subdir)

    monkeypatch.setenv(DVC_EXP_BASELINE_REV, "f" * 40)
    monkeypatch.setenv(DVC_EXP_NAME, "bar")

    live = Live()
    live.log_metric("foo", 1)
    live.next_step()

    dvc_path = Path(live.dvc_file).as_posix()
    metrics_path = Path(live.metrics_file).as_posix()
    foo_path = (Path(live.plots_dir) / Metric.subfolder / "foo.tsv").as_posix()

    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        json={
            "type": "data",
            "repo_url": "STUDIO_REPO_URL",
            "baseline_sha": live._baseline_rev,
            "name": live._exp_name,
            "step": 0,
            "metrics": {f"subdir/{metrics_path}": {"data": {"step": 0, "foo": 1}}},
            "plots": {
                f"subdir/{dvc_path}::subdir/{foo_path}": {
                    "data": [{"step": 0, "foo": 1.0}]
                }
            },
            "client": "dvclive",
        },
        headers={
            "Authorization": "token STUDIO_TOKEN",
            "Content-type": "application/json",
        },
        timeout=(30, 5),
    )


def test_post_to_studio_requires_exp(tmp_dir, mocked_dvc_repo, mocked_studio_post):
    assert Live()._studio_events_to_skip == {"start", "data", "done"}
    assert not Live(save_dvc_exp=True)._studio_events_to_skip


def test_get_dvc_studio_config_none(mocker):
    mocker.patch("dvclive.live.get_dvc_repo", return_value=None)
    live = Live()
    assert get_dvc_studio_config(live) == {}


def test_get_dvc_studio_config_env_var(monkeypatch, mocker):
    monkeypatch.setenv(DVC_STUDIO_TOKEN, "token")
    monkeypatch.setenv(DVC_STUDIO_REPO_URL, "repo_url")
    mocker.patch("dvclive.live.get_dvc_repo", return_value=None)
    live = Live()
    assert get_dvc_studio_config(live) == {
        "token": "token",
        "repo_url": "repo_url",
        "url": STUDIO_URL,
    }


def test_get_dvc_studio_config_dvc_repo(mocked_dvc_repo):
    mocked_dvc_repo.config = {"studio": {"token": "token", "repo_url": "repo_url"}}
    live = Live()
    assert get_dvc_studio_config(live) == {
        "token": "token",
        "repo_url": "repo_url",
        "url": STUDIO_URL,
    }


def test_post_to_studio_images(tmp_dir, mocked_dvc_repo, mocked_studio_post):
    mocked_post, _ = mocked_studio_post

    live = Live(save_dvc_exp=True)
    live.log_image("foo.png", ImagePIL.new("RGB", (10, 10), (0, 0, 0)))
    live.next_step()

    dvc_path = Path(live.dvc_file).as_posix()
    metrics_path = Path(live.metrics_file).as_posix()
    foo_path = (Path(live.plots_dir) / Image.subfolder / "foo.png").as_posix()

    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        json={
            "type": "data",
            "repo_url": "STUDIO_REPO_URL",
            "baseline_sha": live._baseline_rev,
            "name": live._exp_name,
            "step": 0,
            "metrics": {f"{metrics_path}": {"data": {"step": 0}}},
            "plots": {
                f"{dvc_path}::{foo_path}": {"image": _adapt_image(foo_path)},
            },
            "client": "dvclive",
        },
        headers={
            "Authorization": "token STUDIO_TOKEN",
            "Content-type": "application/json",
        },
        timeout=(30, 5),
    )


def test_post_to_studio_message(tmp_dir, mocked_dvc_repo, mocked_studio_post):
    live = Live(save_dvc_exp=True, exp_message="Custom message")

    mocked_post, _ = mocked_studio_post

    mocked_post.assert_called_with(
        "https://0.0.0.0/api/live",
        json={
            "type": "start",
            "repo_url": "STUDIO_REPO_URL",
            "baseline_sha": "f" * 40,
            "name": live._exp_name,
            "client": "dvclive",
            "message": "Custom message",
        },
        headers={
            "Authorization": "token STUDIO_TOKEN",
            "Content-type": "application/json",
        },
        timeout=(30, 5),
    )
