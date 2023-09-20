# ruff: noqa: ARG002
from dvclive import Live


class DVCLiveCallback:
    def __init__(self, metric_name="metric", **kwargs) -> None:
        kwargs["dir"] = kwargs.get("dir", "dvclive-optuna")
        kwargs.pop("save_dvc_exp", None)
        self.metric_name = metric_name
        self.live_kwargs = kwargs

    def __call__(self, study, trial) -> None:
        with Live(**self.live_kwargs) as live:
            self._log_metrics(trial.values, live)
            live.log_params(trial.params)

    def _log_metrics(self, values, live):
        if values is None:
            return

        if isinstance(self.metric_name, str):
            if len(values) > 1:
                # Broadcast default name for multi-objective optimization.
                names = [f"{self.metric_name}_{i}" for i in range(len(values))]

            else:
                names = [self.metric_name]

        elif len(self.metric_name) != len(values):
            raise ValueError(
                "Running multi-objective optimization "
                "with {} objective values, but {} names specified. "
                "Match objective values and names,"
                "or use default broadcasting.".format(
                    len(values), len(self.metric_name)
                )
            )

        else:
            names = [*self.metric_name]

        metrics = dict(zip(names, values))
        for k, v in metrics.items():
            live.summary[k] = v
