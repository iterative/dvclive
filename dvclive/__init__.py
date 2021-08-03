from typing import Optional, Union

from dvclive.version import __version__  # noqa: F401

from .metrics import MetricLogger

_metric_logger: Optional[MetricLogger] = None


def init(
    path: str = None,
    resume: bool = False,
    step: int = 0,
    summary: bool = True,
) -> MetricLogger:
    global _metric_logger  # pylint: disable=global-statement
    _metric_logger = MetricLogger(
        path=path or MetricLogger.DEFAULT_DIR,
        resume=resume,
        step=step,
        summary=summary,
    )
    return _metric_logger


def log(name: str, val: Union[int, float], step: int = None) -> None:
    global _metric_logger  # pylint: disable=global-statement
    if _metric_logger:
        if not _metric_logger.matches_env_setup():
            from .error import ConfigMismatchError

            raise ConfigMismatchError(_metric_logger)
    else:
        _metric_logger = MetricLogger.from_env()
    if not _metric_logger:
        _metric_logger = MetricLogger()

    _metric_logger.log(name=name, val=val, step=step)


def next_step() -> None:
    global _metric_logger  # pylint: disable=global-statement
    if not _metric_logger:
        from .error import InitializationError

        raise InitializationError()
    _metric_logger.next_step()
