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

            if self.model_file:
                model = kwargs["model"]
                model.save_pretrained(self.model_file)
        self.dvclive.next_step()
