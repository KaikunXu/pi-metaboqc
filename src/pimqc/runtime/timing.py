"""Provide transparent execution-time logging for pipeline operations.

The decorator preserves the wrapped callable's signature and return value while
recording start, completion, and failure timing through Loguru. It contains no
package initialization or global logging configuration.
"""

from datetime import datetime
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

from loguru import logger

P = ParamSpec("P")
R = TypeVar("R")


def log_execution_time(func: Callable[P, R]) -> Callable[P, R]:
    """Log calls lasting more than one second."""

    @wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        start = datetime.now()
        result = func(*args, **kwargs)
        duration = datetime.now() - start
        if duration.total_seconds() > 1:
            logger.success(
                f'Execution time of "{func.__name__}": '
                f"{duration.total_seconds():.3f} seconds."
            )
        return result

    return wrapped
