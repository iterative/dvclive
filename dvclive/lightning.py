from typing import Dict, Optional

from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
from torch import is_tensor

from dvclive import Live


class DvcLiveLogger(LightningLoggerBase):
    def __init__(
        self,
        run_name: Optional[str] = "dvclive_run",
        prefix="",
        experiment=None,
        path: Optional[str] = None,
        resume: bool = False,
    ):

        super().__init__()
        self._prefix = prefix
        self._dvclive_init = {
            "path": path,
            "resume": resume,
        }
        self._experiment = experiment
        self._version = run_name

    @property
    def name(self):
        return "DvcLiveLogger"

    @rank_zero_only
    def log_hyperparams(self, params, *args, **kwargs):
        pass

    @property
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
            self._experiment = Live(**self._dvclive_init)

        return self._experiment

    @property
    def version(self):
        return self._version

    @rank_zero_only
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ):
        assert (
            rank_zero_only.rank == 0
        ), "experiment tried to log from global_rank != 0"

        metrics = self._add_prefix(metrics)
        for metric_name, metric_val in metrics.items():
            if is_tensor(metric_val):
                metric_val = metric_val.cpu().detach().item()
            self.experiment.log(name=metric_name, val=metric_val)
        self.experiment.next_step()
