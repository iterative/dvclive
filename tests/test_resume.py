import pytest

from dvclive import Live
from dvclive.env import DVCLIVE_RESUME
from dvclive.utils import read_history, read_latest


@pytest.mark.parametrize(
    ("resume", "steps", "metrics"),
    [(True, [0, 1, 2, 3], [0.9, 0.8, 0.7, 0.6]), (False, [0, 1], [0.7, 0.6])],
)
def test_resume(tmp_dir, resume, steps, metrics):
    dvclive = Live("logs")

    for metric in [0.9, 0.8]:
        dvclive.log_metric("metric", metric)
        dvclive.next_step()

    assert read_history(dvclive, "metric") == ([0, 1], [0.9, 0.8])
    assert read_latest(dvclive, "metric") == (1, 0.8)

    dvclive = Live("logs", resume=resume)

    for new_metric in [0.7, 0.6]:
        dvclive.log_metric("metric", new_metric)
        dvclive.next_step()

    assert read_history(dvclive, "metric") == (steps, metrics)
    assert read_latest(dvclive, "metric") == (steps[-1], metrics[-1])


def test_resume_on_first_init(tmp_dir):
    dvclive = Live(resume=True)

    assert dvclive._step == 0


def test_resume_env_var(tmp_dir, monkeypatch):
    assert not Live()._resume

    monkeypatch.setenv(DVCLIVE_RESUME, "true")
    assert Live()._resume
