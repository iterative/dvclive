from dvc.repo import Repo
from dvc.repo.experiments.exceptions import ExperimentExistsError
from scmrepo.git import Git

from dvclive import Live
from dvclive.dvc import get_dvc_repo, make_dvcyaml, random_exp_name
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
        "stages": {"empty": {"cmd": "empty"}},
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
