import os

import pytest
from dvc.repo import Repo
from dvc.scm import NoSCM
from PIL import Image
from ruamel.yaml import YAML
from scmrepo.git import Git

from dvclive import Live
from dvclive.dvc import get_dvc_repo, make_dvcyaml
from dvclive.env import DVC_EXP_BASELINE_REV, DVC_EXP_NAME
from dvclive.serialize import load_yaml

YAML_LOADER = YAML(typ="safe")


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


def test_make_dvcyaml_empty(tmp_dir):
    live = Live()
    make_dvcyaml(live)

    assert load_yaml(live.dvc_file) == {}


def test_make_dvcyaml_param(tmp_dir):
    live = Live()
    live.log_param("foo", 1)
    make_dvcyaml(live)

    assert load_yaml(live.dvc_file) == {
        "params": ["params.yaml"],
    }


def test_make_dvcyaml_metrics(tmp_dir):
    live = Live()
    live.log_metric("bar", 2)
    make_dvcyaml(live)

    assert load_yaml(live.dvc_file) == {
        "metrics": ["metrics.json"],
        "plots": [{"plots/metrics": {"x": "step"}}],
    }


def test_make_dvcyaml_summary(tmp_dir):
    live = Live()
    live.summary["bar"] = 2
    make_dvcyaml(live)

    assert load_yaml(live.dvc_file) == {
        "metrics": ["metrics.json"],
    }


def test_make_dvcyaml_all_plots(tmp_dir):
    live = Live()
    live.log_param("foo", 1)
    live.log_metric("bar", 2)
    live.log_image("img.png", Image.new("RGB", (10, 10), (250, 250, 250)))
    live.log_sklearn_plot("confusion_matrix", [0, 0, 1, 1], [0, 1, 1, 0])
    live.log_sklearn_plot(
        "confusion_matrix",
        [0, 0, 1, 1],
        [0, 1, 1, 0],
        name="confusion_matrix_normalized",
        normalized=True,
    )
    live.log_sklearn_plot("roc", [0, 0, 1, 1], [0.0, 0.5, 0.5, 0.0], "custom_name_roc")
    make_dvcyaml(live)

    assert load_yaml(live.dvc_file) == {
        "metrics": ["metrics.json"],
        "params": ["params.yaml"],
        "plots": [
            {"plots/metrics": {"x": "step"}},
            "plots/images",
            {
                "plots/sklearn/confusion_matrix.json": {
                    "template": "confusion",
                    "x": "actual",
                    "y": "predicted",
                    "title": "Confusion Matrix",
                    "x_label": "True Label",
                    "y_label": "Predicted Label",
                },
            },
            {
                "plots/sklearn/confusion_matrix_normalized.json": {
                    "template": "confusion_normalized",
                    "title": "Confusion Matrix",
                    "x": "actual",
                    "x_label": "True Label",
                    "y": "predicted",
                    "y_label": "Predicted Label",
                }
            },
            {
                "plots/sklearn/custom_name_roc.json": {
                    "template": "simple",
                    "x": "fpr",
                    "y": "tpr",
                    "title": "Receiver operating characteristic (ROC)",
                    "x_label": "False Positive Rate",
                    "y_label": "True Positive Rate",
                }
            },
        ],
    }


@pytest.mark.parametrize("save", [True, False])
def test_exp_save_on_end(tmp_dir, save, mocked_dvc_repo):
    live = Live(save_dvc_exp=save)
    live.end()
    if save:
        assert live._baseline_rev is not None
        assert live._exp_name is not None
        mocked_dvc_repo.experiments.save.assert_called_with(
            name=live._exp_name,
            include_untracked=[live.dir],
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

    with mocker.patch("dvclive.live.get_dvc_repo", return_value=None):
        live = Live(save_dvc_exp=True)
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
    with mocker.patch("dvclive.live.get_dvc_repo", return_value=dvc_repo):
        live = Live(save_dvc_exp=True)
        assert live._save_dvc_exp
        assert live._baseline_rev is not None
        assert live._exp_name is not None
        live.end()

    dvc_repo.experiments.save.assert_called_with(
        name=live._exp_name, include_untracked=[live.dir], force=True, message=None
    )


@pytest.mark.parametrize("dvcyaml", [True, False])
def test_dvcyaml_on_next_step(tmp_dir, dvcyaml, mocked_dvc_repo):
    live = Live(dvcyaml=dvcyaml)
    live.next_step()
    if dvcyaml:
        assert (tmp_dir / live.dvc_file).exists()
    else:
        assert not (tmp_dir / live.dvc_file).exists()


@pytest.mark.parametrize("dvcyaml", [True, False])
def test_dvcyaml_on_end(tmp_dir, dvcyaml, mocked_dvc_repo):
    live = Live(dvcyaml=dvcyaml)
    live.end()
    if dvcyaml:
        assert (tmp_dir / live.dvc_file).exists()
    else:
        assert not (tmp_dir / live.dvc_file).exists()


def test_exp_save_with_dvc_files(tmp_dir, mocker):
    dvc_repo = mocker.MagicMock()
    dvc_file = mocker.MagicMock()
    dvc_file.is_data_source = True
    dvc_repo.index.stages = [dvc_file]
    dvc_repo.scm.get_rev.return_value = "current_rev"
    dvc_repo.scm.get_ref.return_value = None
    dvc_repo.scm.no_commits = False
    dvc_repo.config = {}

    with mocker.patch("dvclive.live.get_dvc_repo", return_value=dvc_repo):
        live = Live(save_dvc_exp=True)
        live.end()

    dvc_repo.experiments.save.assert_called_with(
        name=live._exp_name, include_untracked=[live.dir], force=True, message=None
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

    with Live(save_dvc_exp=True):
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
    with Live(report=None) as live:
        live.log_metric("foo", 1)
        live.next_step()
    live._dvc_repo.scm.add.assert_called_with(["dvclive/metrics.json", plot_file])


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

    with Live(report=None) as live:
        live.log_metric("foo", 1)
        live.next_step()

    live._dvc_repo.scm.add.assert_called_with(["dvclive/metrics.json"])


def test_errors_on_git_add_are_catched(tmp_dir, mocked_dvc_repo, monkeypatch):
    monkeypatch.setenv(DVC_EXP_BASELINE_REV, "foo")
    monkeypatch.setenv(DVC_EXP_NAME, "bar")
    mocked_dvc_repo.scm.untracked_files.return_value = ["dvclive/metrics.json"]
    mocked_dvc_repo.scm.add.side_effect = Exception("foo")

    with Live(report=None) as live:
        live.summary["foo"] = 1


def test_make_dvcyaml_idempotent(tmp_dir, mocked_dvc_repo):
    (tmp_dir / "model.pth").touch()

    with Live() as live:
        live.log_artifact("model.pth", type="model")

    live.make_dvcyaml()

    assert load_yaml(live.dvc_file) == {
        "artifacts": {
            "model": {"path": "../model.pth", "type": "model"},
        }
    }


def test_exp_save_message(tmp_dir, mocked_dvc_repo):
    live = Live(save_dvc_exp=True, exp_message="Custom message")
    live.end()
    mocked_dvc_repo.experiments.save.assert_called_with(
        name=live._exp_name,
        include_untracked=[live.dir],
        force=True,
        message="Custom message",
    )


def test_no_scm_repo(tmp_dir, mocker):
    dvc_repo = mocker.MagicMock()
    dvc_repo.scm = NoSCM()

    with mocker.patch("dvclive.live.get_dvc_repo", return_value=dvc_repo):
        live = Live()
        assert live._dvc_repo == dvc_repo

        live = Live(save_dvc_exp=True)
        assert live._save_dvc_exp is False
