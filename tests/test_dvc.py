import os

import pytest
from dvc.exceptions import DvcException
from dvc.repo import Repo
from dvc.scm import NoSCM
from scmrepo.git import Git

from dvclive import Live
from dvclive.dvc import get_dvc_repo
from dvclive.env import DVC_EXP_BASELINE_REV, DVC_EXP_NAME, DVC_ROOT


def test_get_dvc_repo(tmp_dir):
    assert get_dvc_repo() is None
    Git.init(tmp_dir)
    assert isinstance(get_dvc_repo(), Repo)


def test_get_dvc_repo_subdir(tmp_dir):
    Git.init(tmp_dir)
    subdir = tmp_dir / "sub"
    subdir.mkdir()
    os.chdir(subdir)
    assert get_dvc_repo().root_dir == str(tmp_dir)


@pytest.mark.parametrize("save", [True, False])
def test_exp_save_on_end(tmp_dir, save, mocked_dvc_repo):
    live = Live(save_dvc_exp=save)
    live.end()
    if save:
        assert live._baseline_rev is not None
        assert live._exp_name is not None
        mocked_dvc_repo.experiments.save.assert_called_with(
            name=live._exp_name,
            include_untracked=[live.dir, str(tmp_dir / "dvc.yaml")],
            force=True,
            message=None,
        )
    else:
        assert live._baseline_rev is not None
        assert live._exp_name is None
        mocked_dvc_repo.experiments.save.assert_not_called()


def test_exp_save_skip_on_env_vars(tmp_dir, monkeypatch, mocker):
    monkeypatch.setenv(DVC_EXP_BASELINE_REV, "foo")
    monkeypatch.setenv(DVC_EXP_NAME, "bar")

    mocker.patch("dvclive.live.get_dvc_repo", return_value=None)
    live = Live()
    live.end()

    assert live._dvc_repo is None
    assert live._baseline_rev == "foo"
    assert live._exp_name == "bar"
    assert live._inside_dvc_exp


def test_exp_save_run_on_dvc_repro(tmp_dir, mocker):
    dvc_repo = mocker.MagicMock()
    dvc_stage = mocker.MagicMock()
    dvc_file = mocker.MagicMock()
    dvc_repo.index.stages = [dvc_stage, dvc_file]
    dvc_repo.scm.get_rev.return_value = "current_rev"
    dvc_repo.scm.get_ref.return_value = None
    dvc_repo.scm.no_commits = False
    dvc_repo.config = {}
    dvc_repo.root_dir = tmp_dir
    mocker.patch("dvclive.live.get_dvc_repo", return_value=dvc_repo)
    live = Live()
    assert live._save_dvc_exp
    assert live._baseline_rev is not None
    assert live._exp_name is not None
    live.end()

    dvc_repo.experiments.save.assert_called_with(
        name=live._exp_name,
        include_untracked=[live.dir, str(tmp_dir / "dvc.yaml")],
        force=True,
        message=None,
    )


def test_exp_save_with_dvc_files(tmp_dir, mocker):
    dvc_repo = mocker.MagicMock()
    dvc_file = mocker.MagicMock()
    dvc_file.is_data_source = True
    dvc_repo.index.stages = [dvc_file]
    dvc_repo.scm.get_rev.return_value = "current_rev"
    dvc_repo.scm.get_ref.return_value = None
    dvc_repo.scm.no_commits = False
    dvc_repo.root_dir = tmp_dir
    dvc_repo.config = {}

    mocker.patch("dvclive.live.get_dvc_repo", return_value=dvc_repo)
    live = Live()
    live.end()

    dvc_repo.experiments.save.assert_called_with(
        name=live._exp_name,
        include_untracked=[live.dir, str(tmp_dir / "dvc.yaml")],
        force=True,
        message=None,
    )


def test_exp_save_dvcexception_is_ignored(tmp_dir, mocker):
    from dvc.exceptions import DvcException

    dvc_repo = mocker.MagicMock()
    dvc_repo.index.stages = []
    dvc_repo.scm.get_rev.return_value = "current_rev"
    dvc_repo.scm.get_ref.return_value = None
    dvc_repo.config = {}
    dvc_repo.experiments.save.side_effect = DvcException("foo")
    mocker.patch("dvclive.live.get_dvc_repo", return_value=dvc_repo)

    with Live():
        pass


def test_untracked_dvclive_files_inside_dvc_exp_run_are_added(
    tmp_dir, mocked_dvc_repo, monkeypatch
):
    monkeypatch.setenv(DVC_EXP_BASELINE_REV, "foo")
    monkeypatch.setenv(DVC_EXP_NAME, "bar")
    plot_file = os.path.join("dvclive", "plots", "metrics", "foo.tsv")
    mocked_dvc_repo.scm.untracked_files.return_value = [
        "dvclive/metrics.json",
        plot_file,
    ]
    with Live() as live:
        live.log_metric("foo", 1)
        live.next_step()
    live._dvc_repo.scm.add.assert_any_call(["dvclive/metrics.json", plot_file])
    live._dvc_repo.scm.add.assert_any_call(live.dvc_file)


def test_dvc_outs_are_not_added(tmp_dir, mocked_dvc_repo, monkeypatch):
    """Regression test for https://github.com/iterative/dvclive/issues/516"""
    monkeypatch.setenv(DVC_EXP_BASELINE_REV, "foo")
    monkeypatch.setenv(DVC_EXP_NAME, "bar")
    mocked_dvc_repo.index.outs = ["dvclive/plots"]
    plot_file = os.path.join("dvclive", "plots", "metrics", "foo.tsv")
    mocked_dvc_repo.scm.untracked_files.return_value = [
        "dvclive/metrics.json",
        plot_file,
    ]

    with Live() as live:
        live.log_metric("foo", 1)
        live.next_step()

    live._dvc_repo.scm.add.assert_any_call(["dvclive/metrics.json"])


def test_errors_on_git_add_are_catched(tmp_dir, mocked_dvc_repo, monkeypatch):
    monkeypatch.setenv(DVC_EXP_BASELINE_REV, "foo")
    monkeypatch.setenv(DVC_EXP_NAME, "bar")
    mocked_dvc_repo.scm.untracked_files.return_value = ["dvclive/metrics.json"]
    mocked_dvc_repo.scm.add.side_effect = DvcException("foo")

    with Live(dvcyaml=False) as live:
        live.summary["foo"] = 1


def test_exp_save_message(tmp_dir, mocked_dvc_repo):
    live = Live(exp_message="Custom message")
    live.end()
    mocked_dvc_repo.experiments.save.assert_called_with(
        name=live._exp_name,
        include_untracked=[live.dir, str(tmp_dir / "dvc.yaml")],
        force=True,
        message="Custom message",
    )


def test_no_scm_repo(tmp_dir, mocker):
    dvc_repo = mocker.MagicMock()
    dvc_repo.scm = NoSCM()

    mocker.patch("dvclive.live.get_dvc_repo", return_value=dvc_repo)
    live = Live()
    assert live._dvc_repo == dvc_repo

    live = Live()
    assert live._save_dvc_exp is False


def test_dvc_repro(tmp_dir, monkeypatch, mocker):
    monkeypatch.setenv(DVC_ROOT, "root")
    mocker.patch("dvclive.live.get_dvc_repo", return_value=None)
    live = Live(save_dvc_exp=True)
    assert not live._save_dvc_exp
