"""Expose opt-in runtime services without package-import side effects.

The runtime package groups logging, timing, progress integration, notebook
detection, and hardware diagnostics. Services remain dormant until called by
an application or by the public ``pimqc.init`` hook.
"""

from .environment import is_jupyter, print_hardware_diagnostics
from .logging import configure_logging
from .progress import (
    configure_joblib_cpu_limit,
    joblib_execution_context,
    joblib_progress,
    set_progress_enabled,
)
from .timing import log_execution_time

__all__ = [
    "configure_logging",
    "configure_joblib_cpu_limit",
    "is_jupyter",
    "joblib_execution_context",
    "joblib_progress",
    "log_execution_time",
    "print_hardware_diagnostics",
    "set_progress_enabled",
]
