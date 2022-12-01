# pylint: disable=unused-argument,protected-access
import pytest
from dvc.repo import Repo
from dvc.repo.experiments.exceptions import ExperimentExistsError
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


def test_make_dvcyaml(tmp_dir):
    live = Live()
    make_dvcyaml(live)

    assert load_yaml(live.dvc_file) == {
        "metrics": ["metrics.json"],
        "params": ["params.yaml"],
        "plots": ["plots"],
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
    with mocker.patch("dvclive.live.get_dvc_repo", return_value=dvc_repo):
        live = Live(save_dvc_exp=save)
        live.end()
    if save:
        dvc_repo.experiments.save.assert_called_with(
            name=live._exp_name, include_untracked=live.dir
        )
    else:
        dvc_repo.assert_not_called()


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
