from typing import Optional

from dvclive import Live


class DVCLiveCallback:
    def __init__(self, live: Optional[Live] = None, **kwargs):
        super().__init__()
        self.live = live if live is not None else Live(**kwargs)

    def __call__(self, env):
        multi_eval = len(env.evaluation_result_list) > 1
        for eval_result in env.evaluation_result_list:
            data_name, eval_name, result = eval_result[:3]
            self.live.log_metric(
                f"{data_name}/{eval_name}" if multi_eval else eval_name, result
            )
        self.live.next_step()
