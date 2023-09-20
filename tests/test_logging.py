import logging

from dvclive import Live


def test_logger(tmp_dir, mocker):
    logger = mocker.patch("dvclive.live.logger")

    live = Live()
    live.log_metric("foo", 0)
    logger.debug.assert_called_with("Logged foo: 0")
    live.next_step()
    logger.debug.assert_called_with("Step: 1")
    live.log_metric("foo", 1)
    live.next_step()

    live = Live(resume=True)
    logger.info.assert_called_with("Resuming from step 1")


def test_suppress_dvc_logs(tmp_dir, mocked_dvc_repo):
    Live()
    assert logging.getLogger("dvc").level == 30
