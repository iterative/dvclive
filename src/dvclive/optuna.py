from dvclive import Live


def _make_experiment(name):
    # Requires https://github.com/iterative/dvc/pull/8408
    from dvc.repo import Repo

    repo = Repo()
    repo.experiments.save(name=name)


class DVCLiveCallback:
    def __init__(
        self, metric_name="metric", multirun: bool = True, **kwargs
    ) -> None:
        self.metric_name = metric_name
        self.live_kwargs = kwargs
        self.multirun = multirun
        if self.multirun:
            self.live = None
        else:
            self.live = Live(**kwargs)

    def __call__(self, study, trial) -> None:
        if self.multirun:
            self.live = Live(**self.live_kwargs)

        self._log_metrics(trial.values, self.live)
        self.live.log_params(trial.params)

        if self.multirun:
            self.live.make_summary()
            _make_experiment(str(trial.number))
        else:
            self.live.next_step()

    def _log_metrics(self, values, live):
        if values is None:
            return

        if isinstance(self.metric_name, str):
            if len(values) > 1:
                # Broadcast default name for multi-objective optimization.
                names = [
                    "{}_{}".format(self.metric_name, i)
                    for i in range(len(values))
                ]

            else:
                names = [self.metric_name]

        else:
            if len(self.metric_name) != len(values):
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

        metrics = {name: val for name, val in zip(names, values)}
        for k, v in metrics.items():
            live.log_metric(k, v)
