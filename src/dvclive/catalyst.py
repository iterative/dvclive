from typing import Optional

from catalyst import utils
from catalyst.core.callback import Callback, CallbackOrder

from dvclive import Live


class DVCLiveCallback(Callback):
    def __init__(self, model_file=None, live: Optional[Live] = None, **kwargs):
        super().__init__(order=CallbackOrder.external)
        self.model_file = model_file
        self.live = live if live is not None else Live(**kwargs)

    def on_epoch_end(self, runner) -> None:
        for loader_key, per_loader_metrics in runner.epoch_metrics.items():
            for key, value in per_loader_metrics.items():
                key = key.replace("/", "_")
                self.live.log_metric(f"{loader_key}/{key}", float(value))

        if self.model_file:
            checkpoint = utils.pack_checkpoint(
                model=runner.model,
                criterion=runner.criterion,
                optimizer=runner.optimizer,
                scheduler=runner.scheduler,
            )
            utils.save_checkpoint(checkpoint, self.model_file)
        self.live.next_step()
