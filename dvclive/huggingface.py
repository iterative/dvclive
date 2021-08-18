from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

import dvclive


class DvcLiveCallback(TrainerCallback):
    def __init__(self, model_file=None):
        super().__init__()
        self.model_file = model_file

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        logs = kwargs["logs"]
        for key, value in logs.items():
            dvclive.log(key, value)

            if self.model_file:
                model = kwargs["model"]
                model.save_pretrained(self.model_file)
        dvclive.next_step()
