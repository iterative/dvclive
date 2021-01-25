import dvclive


def dvclive_callback(env):
    for k, v in env.evaluation_result_list:
        dvclive.log(k, v, env.iteration)
    dvclive.next_step()
