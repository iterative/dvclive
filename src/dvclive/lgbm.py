from typing import Optional

from dvclive import Live


class DvcLiveCallback:
    def __init__(
        self, model_file=None, dvclive: Optional[Live] = None, **kwargs
    ):
        super().__init__()
        self.model_file = model_file
        self.dvclive = dvclive if dvclive is not None else Live(**kwargs)

    def __call__(self, env):
        for eval_result in env.evaluation_result_list:
            metric = eval_result[1]
            value = eval_result[2]
            self.dvclive.log_metric(metric, value)

        if self.model_file:
            env.model.save_model(self.model_file)
        self.dvclive.make_report()
        self.dvclive.next_step()
