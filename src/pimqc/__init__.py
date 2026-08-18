"""Expose the public pi-metaboqc API and explicit runtime initialization.

The package root re-exports the core data model, processing stages, dataset
builder, and pipeline entry point. Importing it is side-effect free; optional
logging, progress, and hardware diagnostics are enabled only through ``init``.
"""

import multiprocessing
from importlib.metadata import PackageNotFoundError, version

from loguru import logger

from .constants import DEFAULT_RANDOM_SEED

# Core Data Structure
from .core import MetaboInt

# Data Ingestion & Pipeline Management
from .dataset.builder import build_dataset
from .pipeline import PipelineResult, run_pipeline

# Processing Modules (Actors)
from .processing.assessment import MetaboIntAssessor
from .processing.correction import MetaboIntCorrector
from .processing.filtering import MetaboIntFilter
from .processing.imputation import MetaboIntImputer
from .processing.normalization import MetaboIntNormalizer
from .runtime import (
    configure_joblib_cpu_limit,
    configure_logging,
    print_hardware_diagnostics,
    set_progress_enabled,
)

# Package Version
try:
    __version__ = version("pi-metaboqc")
except PackageNotFoundError:
    __version__ = "0+unknown"

# Define public API
__all__ = [
    "MetaboInt",
    "MetaboIntAssessor",
    "MetaboIntCorrector",
    "MetaboIntImputer",
    "MetaboIntNormalizer",
    "MetaboIntFilter",
    "build_dataset",
    "PipelineResult",
    "run_pipeline",
    "DEFAULT_RANDOM_SEED",
]

_IS_INITIALIZED = False


def init(
    check_hardware: bool = True,
    log_level: str = "DEBUG",
    show_progress: bool = True,
    preserve_existing_sinks: bool = False,
) -> None:
    """Explicitly initialize the pi-metaboqc runtime environment.

    Usage:
        pimqc.init(
            check_hardware=False,
            log_level="DEBUG",
            show_progress=False,
            preserve_existing_sinks=False,
        )

    Args:
        check_hardware: Emit hardware diagnostics in the main process.
        log_level: Minimum severity emitted by the package console sink.
        show_progress: Enable progress indicators for long calculations.
        preserve_existing_sinks: Retain Loguru sinks set by a host framework.
            The default removes the default sink so notebooks and scripts emit
            each package log record only once.
    """
    global _IS_INITIALIZED

    # Guard: Prevent redundant initializations within the same process.
    if _IS_INITIALIZED:
        logger.debug("pi-metaboqc already initialized. Skipping init().")
        return

    configure_logging(
        level=log_level,
        preserve_existing_sinks=preserve_existing_sinks,
    )
    configure_joblib_cpu_limit()
    set_progress_enabled(show_progress)

    # Guard: Hardware diagnostics can be time-consuming due to system
    # register probes. Execute only in MainProcess if permitted by user.
    if multiprocessing.current_process().name == "MainProcess":
        if check_hardware:
            print_hardware_diagnostics()

    # Mark the global initialization state as complete.
    _IS_INITIALIZED = True
