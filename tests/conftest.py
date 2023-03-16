# pylint: disable=redefined-outer-name
import sys

import pytest
from dvc_studio_client.env import STUDIO_ENDPOINT, STUDIO_REPO_URL, STUDIO_TOKEN


@pytest.fixture
def tmp_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield tmp_path


@pytest.fixture
def mocked_dvc_repo(tmp_dir, mocker):
    _dvc_repo = mocker.MagicMock()
    _dvc_repo.index.stages = []
    _dvc_repo.scm.get_rev.return_value = "f" * 40
    _dvc_repo.scm.get_ref.return_value = None
    _dvc_repo.root_dir = tmp_dir
    mocker.patch("dvclive.live.get_dvc_repo", return_value=_dvc_repo)
    return _dvc_repo


@pytest.fixture
def dvc_repo(tmp_dir):  # pylint: disable=redefined-outer-name
    from dvc.repo import Repo
    from scmrepo.git import Git

    Git.init(tmp_dir)
    repo = Repo.init(tmp_dir)
    repo.scm.add_commit(".", "init")
    return repo


@pytest.fixture(autouse=True)
def capture_wrap():
    # https://github.com/pytest-dev/pytest/issues/5502#issuecomment-678368525
    sys.stderr.close = lambda *args: None
    sys.stdout.close = lambda *args: None
    yield


@pytest.fixture(autouse=True)
def mocked_webbrowser_open(mocker):
    mocker.patch("webbrowser.open")


@pytest.fixture(autouse=True)
def mocked_CI(monkeypatch):
    monkeypatch.setenv("CI", "false")


@pytest.fixture
def mocked_studio_post(mocker, monkeypatch):
    valid_response = mocker.MagicMock()
    valid_response.status_code = 200
    mocked_post = mocker.patch("requests.post", return_value=valid_response)
    monkeypatch.setenv(STUDIO_ENDPOINT, "https://0.0.0.0")
    monkeypatch.setenv(STUDIO_REPO_URL, "STUDIO_REPO_URL")
    monkeypatch.setenv(STUDIO_TOKEN, "STUDIO_TOKEN")
    return mocked_post, valid_response
