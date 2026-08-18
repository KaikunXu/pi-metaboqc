"""Coordinate optional progress reporting for parallel computations.

This module exposes the worker default, a process-level progress preference,
and a context manager that connects joblib completion callbacks to ``tqdm``.
No joblib classes are patched until the context manager is entered.
"""

import os
from collections.abc import Generator
from contextlib import contextmanager

import joblib
import psutil
from loguru import logger
from tqdm import tqdm

MAX_WORKERS = os.cpu_count() or 1
_PROGRESS_ENABLED = True
_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [Elapsed: {elapsed} | ETA: {remaining}]"
)


def set_progress_enabled(enabled: bool) -> None:
    """Enable or disable package-managed progress displays."""
    global _PROGRESS_ENABLED
    _PROGRESS_ENABLED = bool(enabled)


def configure_joblib_cpu_limit() -> int:
    """Set a stable process-wide Loky worker ceiling and return its value.

    Joblib's Windows physical-core probe can fail and write a traceback
    directly to stderr. Resolve the physical count through psutil instead and
    install Joblib's documented environment constraint once at runtime
    initialization. An embedding application's existing constraint wins.
    """
    existing = os.environ.get("LOKY_MAX_CPU_COUNT")
    if existing is not None:
        try:
            return max(1, int(existing))
        except ValueError:
            logger.warning(
                "Ignoring invalid LOKY_MAX_CPU_COUNT value; resolving a "
                "stable worker limit through psutil."
            )

    logical_cores = max(1, psutil.cpu_count(logical=True) or 1)
    physical_cores = psutil.cpu_count(logical=False)
    if not physical_cores:
        physical_cores = max(1, logical_cores // 2)
        logger.warning(
            "Physical CPU count unavailable via psutil; limiting Loky to "
            f"{physical_cores} worker(s) using a logical-core fallback."
        )

    cpu_limit = max(1, int(physical_cores))
    if logical_cores > 1:
        cpu_limit = min(cpu_limit, logical_cores - 1)
    os.environ["LOKY_MAX_CPU_COUNT"] = str(cpu_limit)
    return cpu_limit


@contextmanager
def joblib_execution_context(backend: str) -> Generator[None, None, None]:
    """Configure one backend without unreliable CPU auto-detection.

    Loky workers receive an explicit one-thread limit for OpenMP/BLAS work.
    They also receive ``LOKY_MAX_CPU_COUNT`` derived from psutil.
    This prevents Joblib from invoking its failing Windows physical-core
    subprocess probe while leaving any user-supplied constraint untouched.
    """
    clean_backend = str(backend).lower()
    config: dict[str, object] = {"backend": clean_backend}

    if clean_backend == "loky":
        config["inner_max_num_threads"] = 1
        configure_joblib_cpu_limit()

    with joblib.parallel_config(**config):
        yield


@contextmanager
def joblib_progress(
    total: int,
    desc: str = "Progress",
    color: str | None = None,
) -> Generator[object | None, None, None]:
    """Temporarily connect joblib batch completion to a tqdm progress bar."""
    if not _PROGRESS_ENABLED:
        yield None
        return

    progress = tqdm(
        total=total,
        desc=desc,
        ncols=120,
        colour=color,
        bar_format=_BAR_FORMAT,
        leave=True,
    )

    class ProgressCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args: object, **kwargs: object) -> object:
            progress.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    # Patch joblib only for the active context and retain the exact callback
    # object supplied by the embedding application.
    previous = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = ProgressCallback
    try:
        yield progress
    finally:
        # Always restore global joblib state, including when a worker fails.
        joblib.parallel.BatchCompletionCallBack = previous
        progress.close()
