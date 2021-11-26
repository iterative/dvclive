from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from dvclive import Live


class DvcLiveCallback(TrainerCallback):
    def __init__(self, model_file=None, **kwargs):
        super().__init__()
        self.model_file = model_file
        self.dvclive = Live(**kwargs)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        logs = kwargs["logs"]
        for key, value in logs.items():
            self.dvclive.log(key, value)
        self.dvclive.next_step()

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        if self.model_file:
            model = kwargs["model"]
            model.save_pretrained(self.model_file)
            tokenizer = kwargs["tokenizer"]
            tokenizer.save_pretrained(self.model_file)
