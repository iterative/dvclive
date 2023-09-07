import pytest

from dvclive.serialize import load_yaml
from dvclive.utils import parse_json

try:
    import optuna

    from dvclive.optuna import DVCLiveCallback
except ImportError:
    pytest.skip("skipping optuna tests", allow_module_level=True)


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


def test_optuna_(tmp_dir, mocked_dvc_repo):
    n_trials = 5
    metric_name = "custom_name"
    callback = DVCLiveCallback(metric_name=metric_name)
    study = optuna.create_study()

    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    assert mocked_dvc_repo.experiments.save.call_count == n_trials

    metrics = parse_json("dvclive-optuna/metrics.json")
    assert metric_name in metrics
    params = load_yaml("dvclive-optuna/params.yaml")
    assert "x" in params

    assert not (tmp_dir / "dvclive-optuna" / "plots").exists()
