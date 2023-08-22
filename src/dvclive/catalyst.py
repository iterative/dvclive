# ruff: noqa: ARG002
from typing import Optional

from catalyst.core.callback import Callback, CallbackOrder

from dvclive import Live


class DVCLiveCallback(Callback):
    def __init__(self, live: Optional[Live] = None, **kwargs):
        super().__init__(order=CallbackOrder.external)
        self.live = live if live is not None else Live(**kwargs)

    def on_epoch_end(self, runner) -> None:
        for loader_key, per_loader_metrics in runner.epoch_metrics.items():
            for key, value in per_loader_metrics.items():
                self.live.log_metric(
                    f"{loader_key}/{key.replace('/', '_')}", float(value)
                )
        self.live.next_step()

    def on_experiment_end(self, runner):
        self.live.end()
