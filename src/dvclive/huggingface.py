# ruff: noqa: ARG002
import os
from typing import Literal, Optional

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer import Trainer

from dvclive import Live
from dvclive.utils import standardize_metric_name


class DVCLiveCallback(TrainerCallback):
    def __init__(
        self,
        live: Optional[Live] = None,
        log_model: Optional[Literal["all", "last"]] = None,
        **kwargs,
    ):
        super().__init__()
        self._log_model = log_model
        self.live = live if live is not None else Live(**kwargs)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        for key, value in args.to_dict().items():
            if key in (
                "num_train_epochs",
                "weight_decay",
                "max_grad_norm",
                "warmup_ratio",
                "warmup_steps",
            ):
                self.live.log_param(key, value)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        logs = kwargs["logs"]
        for key, value in logs.items():
            self.live.log_metric(standardize_metric_name(key, __name__), value)
        self.live.next_step()

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self._log_model == "all" and state.is_world_process_zero:
            self.live.log_artifact(args.output_dir)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self._log_model == "last" and state.is_world_process_zero:
            fake_trainer = Trainer(
                args=args, model=kwargs.get("model"), tokenizer=kwargs.get("tokenizer")
            )
            output_dir = os.path.join(args.output_dir, "last")
            fake_trainer.save_model(output_dir)
            self.live.log_artifact(output_dir, type="model", copy=True)
        self.live.end()
