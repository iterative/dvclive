import sys

import pytest


@pytest.fixture
def tmp_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield tmp_path


@pytest.fixture
def mocked_dvc_repo(mocker):
    _dvc_repo = mocker.MagicMock()
    _dvc_repo.index.stages = []
    _dvc_repo.scm.get_rev.return_value = "current_rev"
    _dvc_repo.scm.get_ref.return_value = None
    mocker.patch("dvclive.live.get_dvc_repo", return_value=_dvc_repo)
    return _dvc_repo


@pytest.fixture
def dvc_repo(tmp_dir):
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
