from typing import Optional

from dvclive import Live


class DVCLiveCallback:
    def __init__(self, model_file=None, live: Optional[Live] = None, **kwargs):
        super().__init__()
        self.model_file = model_file
        self.live = live if live is not None else Live(**kwargs)

    def __call__(self, env):
        for eval_result in env.evaluation_result_list:
            metric = eval_result[1]
            value = eval_result[2]
            self.live.log_metric(metric, value)

        if self.model_file:
            env.model.save_model(self.model_file)
        self.live.next_step()
