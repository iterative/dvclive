# pylint: disable=unused-argument,protected-access
import os

import pytest
from dvc.repo import Repo
from dvc.repo.experiments.exceptions import ExperimentExistsError
from PIL import Image
from scmrepo.git import Git

from dvclive import Live
from dvclive.dvc import get_dvc_repo, make_dvcyaml, random_exp_name
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
        "plots": [os.path.join("plots", "metrics")],
    }


def test_make_dvcyaml_all_plots(tmp_dir):
    live = Live()
    live.log_param("foo", 1)
    live.log_metric("bar", 2)
    live.log_image("img.png", Image.new("RGB", (10, 10), (250, 250, 250)))
    live.log_sklearn_plot("confusion_matrix", [0, 0, 1, 1], [0, 1, 1, 0])
    live.log_sklearn_plot(
        "roc", [0, 0, 1, 1], [0.0, 0.5, 0.5, 0.0], "custom_name_roc"
    )
    make_dvcyaml(live)

    assert load_yaml(live.dvc_file) == {
        "metrics": ["metrics.json"],
        "params": ["params.yaml"],
        "plots": [
            os.path.join("plots", "metrics"),
            os.path.join("plots", "images"),
            {
                os.path.join("plots", "sklearn", "confusion_matrix.json"): {
                    "template": "confusion",
                    "x": "actual",
                    "y": "predicted",
                    "title": "Confusion Matrix",
                    "x_label": "True Label",
                    "y_label": "Predicted Label",
                },
            },
            {
                os.path.join("plots", "sklearn", "custom_name_roc.json"): {
                    "x": "fpr",
                    "y": "tpr",
                    "title": "Receiver operating characteristic (ROC)",
                    "x_label": "False Positive Rate",
                    "y_label": "True Positive Rate",
                }
            },
        ],
    }


def test_random_exp_name(mocker):
    dvc_repo = mocker.MagicMock()

    class Validate:
        exists = set()
        n_calls = 0

        def __call__(self, exp_ref):
            self.n_calls += 1
            exp_ref = str(exp_ref)
            if exp_ref not in self.exists:
                self.exists.add(exp_ref)
            else:
                raise ExperimentExistsError(exp_ref)

    validate = Validate()
    dvc_repo.experiments._validate_new_ref.side_effect = validate

    with mocker.patch(
        "dvclive.dvc.choice", side_effect=[0, 0, 0, 0, 1, 1, 0, 0]
    ):
        name = random_exp_name(dvc_repo, "foo")
        assert name == "0-0"
        assert validate.n_calls == 1

        # First try fails with exists error
        # So 2 calls are needed
        name = random_exp_name(dvc_repo, "foo")
        assert name == "1-1"
        assert validate.n_calls == 3

        # Doesn't fail because has a different baseline_rev
        name = random_exp_name(dvc_repo, "bar")
        assert name == "0-0"
        assert validate.n_calls == 4


@pytest.mark.parametrize("save", [True, False])
def test_exp_save_on_end(tmp_dir, mocker, save):
    dvc_repo = mocker.MagicMock()
    dvc_repo.index.stages = []
    with mocker.patch("dvclive.live.get_dvc_repo", return_value=dvc_repo):
        live = Live(save_dvc_exp=save)
        live.end()
    if save:
        assert live._baseline_rev is not None
        assert live._exp_name is not None
        dvc_repo.experiments.save.assert_called_with(
            name=live._exp_name, include_untracked=live.dir
        )
        assert (tmp_dir / live.dvc_file).exists()
    else:
        assert live._baseline_rev is None
        assert live._exp_name is None
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
    dvc_repo.index.stages = ["foo", "bar"]
    with mocker.patch("dvclive.live.get_dvc_repo", return_value=dvc_repo):
        live = Live(save_dvc_exp=True)
        assert not live._save_dvc_exp
        assert live._baseline_rev is None
        assert live._exp_name is None
        live.end()

    dvc_repo.experiments.save.assert_not_called()
    assert not (tmp_dir / live.dvc_file).exists()


@pytest.mark.parametrize("save", [True, False])
def test_dvcyaml_on_next_step(tmp_dir, mocker, save):
    dvc_repo = mocker.MagicMock()
    dvc_repo.index.stages = []
    with mocker.patch("dvclive.live.get_dvc_repo", return_value=dvc_repo):
        live = Live(save_dvc_exp=save)
        live.next_step()
    if save:
        assert (tmp_dir / live.dvc_file).exists()
    else:
        assert not (tmp_dir / live.dvc_file).exists()
