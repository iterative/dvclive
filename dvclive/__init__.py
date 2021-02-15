from dvclive.version import __version__  # noqa: F401

from .error import InitializationError
from .metrics import MetricLogger

_metric_logger = None


def init(
    path: str = None,
    resume: bool = False,
    step: int = 0,
    summary: bool = True,
    html: bool = True,
) -> MetricLogger:
    global _metric_logger  # pylint: disable=global-statement
    _metric_logger = MetricLogger(
        path=path or MetricLogger.DEFAULT_DIR,
        resume=resume,
        step=step,
        summary=summary,
        html=html,
    )
    return _metric_logger


def log(name: str, val: float, step: int = None):
    global _metric_logger  # pylint: disable=global-statement
    if not _metric_logger:
        _metric_logger = MetricLogger.from_env()
    if not _metric_logger:
        raise InitializationError()

    _metric_logger.log(name=name, val=val, step=step)


def next_step():
    global _metric_logger  # pylint: disable=global-statement
    if not _metric_logger:
        raise InitializationError()
    _metric_logger.next_step()
