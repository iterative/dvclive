import optuna

from dvclive.optuna import DVCLiveCallback
from dvclive.utils import parse_metrics


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


def test_optuna_callback():
    n_trials = 5
    metric_name = "metric"
    callback = DVCLiveCallback(metric_name=metric_name, multirun=False)
    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    logs, latest = parse_metrics(callback.live)
    metric = next(iter(logs.values()))
    assert len(metric) == n_trials
    assert study.trials[-1].values[0] == latest[metric_name]
