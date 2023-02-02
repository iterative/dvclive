# pylint: disable=unused-argument,protected-access
import pytest
from dvc.repo import Repo
from PIL import Image
from scmrepo.git import Git

from dvclive import Live
from dvclive.dvc import get_dvc_repo, make_dvcyaml
from dvclive.env import DVC_EXP_BASELINE_REV, DVC_EXP_NAME
from dvclive.serialize import load_yaml


def test_get_dvc_repo(tmp_dir):
    assert get_dvc_repo() is None
    Git.init(tmp_dir)
    Repo.init(tmp_dir)
    assert isinstance(get_dvc_repo(), Repo)


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
        "plots": ["plots/metrics"],
    }


def test_make_dvcyaml_all_plots(tmp_dir):
    live = Live()
    live.log_param("foo", 1)
    live.log_metric("bar", 2)
    live.log_image("img.png", Image.new("RGB", (10, 10), (250, 250, 250)))
    live.log_sklearn_plot("confusion_matrix", [0, 0, 1, 1], [0, 1, 1, 0])
    live.log_sklearn_plot("roc", [0, 0, 1, 1], [0.0, 0.5, 0.5, 0.0], "custom_name_roc")
    make_dvcyaml(live)

    assert load_yaml(live.dvc_file) == {
        "metrics": ["metrics.json"],
        "params": ["params.yaml"],
        "plots": [
            "plots/metrics",
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
                "plots/sklearn/custom_name_roc.json": {
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
def test_exp_save_on_end(tmp_dir, mocker, save):
    dvc_repo = mocker.MagicMock()
    dvc_repo.index.stages = []
    dvc_repo.scm.get_rev.return_value = "current_rev"
    dvc_repo.scm.get_ref.return_value = None
    with mocker.patch("dvclive.live.get_dvc_repo", return_value=dvc_repo):
        live = Live(save_dvc_exp=save)
        live.end()
    if save:
        assert live._baseline_rev is not None
        assert live._exp_name != "dvclive-exp"
        dvc_repo.experiments.save.assert_called_with(
            name=live._exp_name, include_untracked=live.dir, force=True
        )
        assert (tmp_dir / live.dvc_file).exists()
    else:
        assert live._baseline_rev is not None
        assert live._exp_name == "dvclive-exp"
        dvc_repo.experiments.save.assert_not_called()
        assert not (tmp_dir / live.dvc_file).exists()


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
    assert not (tmp_dir / live.dvc_file).exists()


def test_exp_save_skip_on_dvc_repro(tmp_dir, mocker):
    dvc_repo = mocker.MagicMock()
    dvc_stage = mocker.MagicMock()
    dvc_stage.is_data_source = False
    dvc_file = mocker.MagicMock()
    dvc_file.is_data_source = False
    dvc_repo.index.stages = [dvc_stage, dvc_file]
    with mocker.patch("dvclive.live.get_dvc_repo", return_value=dvc_repo):
        live = Live(save_dvc_exp=True)
        assert not live._save_dvc_exp
        assert live._baseline_rev is not None
        assert live._exp_name == "dvclive-exp"
        live.end()

    dvc_repo.experiments.save.assert_not_called()
    assert not (tmp_dir / live.dvc_file).exists()


@pytest.mark.parametrize("save", [True, False])
def test_dvcyaml_on_next_step(tmp_dir, mocker, save):
    dvc_repo = mocker.MagicMock()
    dvc_repo.index.stages = []
    dvc_repo.scm.get_rev.return_value = "current_rev"
    dvc_repo.scm.get_ref.return_value = None
    with mocker.patch("dvclive.live.get_dvc_repo", return_value=dvc_repo):
        live = Live(save_dvc_exp=save)
        live.next_step()
    if save:
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

    with mocker.patch("dvclive.live.get_dvc_repo", return_value=dvc_repo):
        live = Live(save_dvc_exp=True)
        live.end()

    dvc_repo.experiments.save.assert_called_with(
        name=live._exp_name, include_untracked=live.dir, force=True
    )
