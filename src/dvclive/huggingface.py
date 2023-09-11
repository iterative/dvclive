# ruff: noqa: ARG002
import logging
import os
from typing import Literal, Optional, Union

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer import Trainer

from dvclive import Live
from dvclive.utils import standardize_metric_name

logger = logging.getLogger("dvclive")


class DVCLiveCallback(TrainerCallback):
    def __init__(
        self,
        live: Optional[Live] = None,
        log_model: Optional[Union[Literal["all"], bool]] = None,
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
        self.live.log_params(args.to_dict())

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
        if self._log_model is True and state.is_world_process_zero:
            fake_trainer = Trainer(
                args=args, model=kwargs.get("model"), tokenizer=kwargs.get("tokenizer")
            )
            name = "best" if args.load_best_model_at_end else "last"
            output_dir = os.path.join(args.output_dir, name)
            fake_trainer.save_model(output_dir)
            self.live.log_artifact(output_dir, name=name, type="model", copy=True)
        self.live.end()
