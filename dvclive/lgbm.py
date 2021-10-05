from dvclive import Live


class DvcLiveCallback:
    def __init__(self, model_file=None, **kwargs):
        super().__init__()
        self.model_file = model_file
        self.dvclive = Live(**kwargs)

    def __call__(self, env):
        for eval_result in env.evaluation_result_list:
            metric = eval_result[1]
            value = eval_result[2]
            self.dvclive.log(metric, value)

        if self.model_file:
            env.model.save_model(self.model_file)

        self.dvclive.next_step()
