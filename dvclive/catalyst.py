from catalyst.core.callback import Callback, CallbackOrder

from dvclive import Live


class DvcLiveCallback(Callback):
    def __init__(self, model_file=None, **kwargs):
        super().__init__(order=CallbackOrder.external)
        self.dvclive = Live(**kwargs)
        self.model_file = model_file

    def on_epoch_end(self, runner) -> None:
        for loader_key, per_loader_metrics in runner.epoch_metrics.items():
            for key, value in per_loader_metrics.items():
                key = key.replace("/", "_")
                self.dvclive.log(f"{loader_key}/{key}", float(value))

        if self.model_file:
            checkpoint = runner.engine.pack_checkpoint(
                model=runner.model,
                criterion=runner.criterion,
                optimizer=runner.optimizer,
                scheduler=runner.scheduler,
            )
            runner.engine.save_checkpoint(checkpoint, self.model_file)
        self.dvclive.next_step()
