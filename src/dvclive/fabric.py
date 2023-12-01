from typing import Any, Dict, Mapping, Optional, Union

try:
    from lightning.fabric.loggers.logger import Logger, rank_zero_experiment
    from lightning.fabric.utilities.logger import (
        _add_prefix,
        _convert_params,
        _sanitize_callable_params,
    )
    from lightning.fabric.utilities.rank_zero import rank_zero_only
except ImportError:
    from lightning_fabric.loggers.logger import Logger, rank_zero_experiment
    from lightning_fabric.utilities.logger import (
        _add_prefix,
        _convert_params,
        _sanitize_callable_params,
    )
    from lightning_fabric.utilities.rank_zero import rank_zero_only

from torch import is_tensor

from dvclive import Live
from dvclive.plots import Metric
from dvclive.utils import standardize_metric_name


class DVCLiveLogger(Logger):
    LOGGER_JOIN_CHAR = "/"

    def __init__(
        self,
        run_name: Optional[str] = None,
        prefix: str = "",
        experiment: Optional[Live] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self._version = run_name
        self._prefix = prefix
        self._experiment = experiment
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        return "DvcLiveLogger"

    @property
    def version(self) -> Union[int, str]:
        return self._version

    @property
    @rank_zero_experiment
    def experiment(self) -> "Live":
        if self._experiment is not None:
            return self._experiment

        assert (  # noqa: S101
            rank_zero_only.rank == 0
        ), "tried to init DVCLive in non global_rank=0"

        self._experiment = Live(**self._kwargs)

        return self._experiment

    @rank_zero_only
    def log_metrics(
        self, metrics: Mapping[str, float], step: Optional[int] = None
    ) -> None:
        assert (  # noqa: S101
            rank_zero_only.rank == 0
        ), "experiment tried to log from global_rank != 0"

        self.experiment.step = step

        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)

        for metric_name, metric_val in metrics.items():
            val = metric_val
            if is_tensor(val):
                val = val.cpu().detach().item()
            name = standardize_metric_name(metric_name, __name__)
            if Metric.could_log(val):
                self.experiment.log_metric(name=name, val=val)
            else:
                raise ValueError(  # noqa: TRY003
                    f"\n you tried to log {val} which is currently not supported."
                    "Try a scalar/tensor."
                )

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any]]) -> None:
        """Record hyperparameters.

        Args:
            params: a dictionary-like container with the hyperparameters
        """
        params = _convert_params(params)
        params = _sanitize_callable_params(params)
        self.experiment.log_params(params)

    @rank_zero_only
    def finalize(self, status: str) -> None:  # noqa: ARG002
        if self._experiment is not None:
            self.experiment.end()

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_experiment"] = None
        return state
