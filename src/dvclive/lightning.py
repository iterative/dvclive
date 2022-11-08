from typing import Any, Dict, Optional

from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
from torch import is_tensor

from dvclive import Live
from dvclive.utils import standardize_metric_name


class DVCLiveLogger(Logger):
    def __init__(
        self,
        run_name: Optional[str] = "dvclive_run",
        prefix="",
        experiment=None,
        dir: Optional[str] = None,  # noqa pylint: disable=redefined-builtin
        resume: bool = False,
    ):

        super().__init__()
        self._prefix = prefix
        self._live_init: Dict[str, Any] = {"resume": resume}
        if dir is not None:
            self._live_init["dir"] = dir
        self._experiment = experiment
        self._version = run_name

    @property
    def name(self):
        return "DvcLiveLogger"

    @rank_zero_only
    def log_hyperparams(self, params, *args, **kwargs):
        pass

    @property  # type: ignore
    @rank_zero_experiment
    def experiment(self):
        r"""
        Actual DVCLive object. To use DVCLive features in your
        :class:`~LightningModule` do the following.
        Example::
            self.logger.experiment.some_dvclive_function()
        """
        if self._experiment is not None:
            return self._experiment
        else:
            assert (
                rank_zero_only.rank == 0
            ), "tried to init log dirs in non global_rank=0"
            self._experiment = Live(**self._live_init)

        return self._experiment

    @property
    def version(self):
        return self._version

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        assert (
            rank_zero_only.rank == 0  # type: ignore
        ), "experiment tried to log from global_rank != 0"

        for metric_name, metric_val in metrics.items():
            if is_tensor(metric_val):
                metric_val = metric_val.cpu().detach().item()
            metric_name = standardize_metric_name(metric_name, __name__)
            self.experiment.log_metric(name=metric_name, val=metric_val)
        self.experiment.next_step()
