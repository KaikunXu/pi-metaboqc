# src/pimqc/__init__.py
"""
Script purpose: Define the public package surface for pi-metaboqc.

This module re-exports the core MetaboInt container, dataset builder, full
pipeline runner, and stage-specific processing classes used by applications
and tutorials. It also provides the init() helper that prepares shared
logging, hardware diagnostics, and resource paths before analysis begins.
The import order keeps the user-facing API compact while leaving individual
implementation details inside their dedicated modules.
"""

import sys
import shlex
import subprocess
import multiprocessing
from importlib.resources import files
from loguru import logger

# Initialize Logger and Environment Check
from . import io_utils as iu

# Core Data Structure
from .core_classes import MetaboInt

# Data Ingestion & Pipeline Management
from .dataset_builder import build_dataset
from .pipeline import run_pipeline

# Processing Modules (Actors)
from .assessment import MetaboIntAssessor
from .correction import MetaboIntCorrector
from .imputation import MetaboIntImputer
from .normalization import MetaboIntNormalizer
from .filtering import MetaboIntFilter

# Package Version
__version__ = (files("pimqc") / "VERSION").read_text(encoding="utf-8").strip()

# Define public API
__all__ = [
    "MetaboInt",
    "MetaboIntAssessor",
    "MetaboIntCorrector",
    "MetaboIntImputer",
    "MetaboIntNormalizer",
    "MetaboIntFilter",
    "build_dataset",
    "run_pipeline",
]

# Global initialization state lock
iu.setup_loguru_logger(level="INFO")
_IS_INITIALIZED = False


def init(
    check_hardware: bool = True, log_level: str = "DEBUG", show_progress: bool = True
) -> None:
    """Explicitly initialize the pi-metaboqc runtime environment.

    Usage:
        pimqc.init(
            check_hardware=False,
            log_level="DEBUG",
            show_progress=False
        )
    """
    global _IS_INITIALIZED

    # Guard: Prevent redundant initializations within the same process.
    if _IS_INITIALIZED:
        logger.debug("pi-metaboqc already initialized. Skipping init().")
        return

    # Dynamically update logger level if a custom level is specified.
    if log_level.upper() != "INFO":
        iu.setup_loguru_logger(level=log_level.upper())

    # Dynamically toggle progress bar visibility in the IO utilities.
    iu.SHOW_PROGRESS = show_progress

    # Guard: Hardware diagnostics can be time-consuming due to system
    # register probes. Execute only in MainProcess if permitted by user.
    if multiprocessing.current_process().name == "MainProcess":
        if check_hardware:
            iu.print_hardware_diagnostics()

    # Mark the global initialization state as complete.
    _IS_INITIALIZED = True


# --- Monkey Patch for subprocess on Windows ----------------------------------
if sys.platform == "win32":
    import threading

    # Global toggle to control WinError 2 logging noise from 3rd-party libs.
    # Set to True for deep debugging, False for clean production logs.
    LOG_WINERROR2 = False

    _original_popen = subprocess.Popen
    _popen_patch_lock = threading.local()

    def _safe_popen(*args: object, **kwargs: object) -> subprocess.Popen:
        """Intercept subprocess calls to prevent crashes and infinite loops.

        Includes a thread-local lock to prevent RecursionError when logging
        triggers nested subprocess calls, and filters WinError 2 noise.
        """
        # 1. Recursion Guard: If already patching in this thread, bypass.
        if getattr(_popen_patch_lock, "is_active", False):
            return _original_popen(*args, **kwargs)

        # 2. Activate the recursion lock for the current thread.
        _popen_patch_lock.is_active = True

        try:
            # Extract custom toggle safely without passing it to the OS.
            log_winerror = kwargs.pop("log_winerror2", LOG_WINERROR2)

            # Extract arguments and format for clean logging.
            cmd_args = args[0] if args else kwargs.get("args", [])

            if isinstance(cmd_args, list):
                cmd_str_log = shlex.join([str(x) for x in cmd_args])
                cmd_str_lower = str(cmd_args).lower()
            else:
                cmd_str_log = str(cmd_args)
                cmd_str_lower = cmd_str_log.lower()

            # Define safe probes commonly used by system libraries.
            safe_probes = [
                "--version",
                "--list",
                "powershell",
                "win32_processor",
                "msiexec",
                "where",
            ]
            is_probe = any(p in cmd_str_lower for p in safe_probes)

            # Heuristic to detect blind absolute path hunting.
            is_hunting = is_probe and ("\\" in cmd_str_log or "/" in cmd_str_log)

            # 3. Controlled Logging.
            if is_probe:
                # Hide hunting attempts if the WinError switch is off.
                if log_winerror or not is_hunting:
                    logger.debug(f"Permitted probe: {cmd_str_log}")

                # Prevent flashing command prompt windows on Windows.
                if "powershell" in cmd_str_lower:
                    kwargs.setdefault("creationflags", 0x08000000)
            else:
                # Functional commands logged at DEBUG level.
                logger.debug(f"Functional subprocess: {cmd_str_log}")

            # Ensure text streams do not crash on decode errors.
            err_keys = ("encoding", "text", "universal_newlines")
            if any(k in kwargs for k in err_keys):
                kwargs["errors"] = "ignore"

            # 4. Execute original call with explicit error toggling.
            try:
                return _original_popen(*args, **kwargs)

            except FileNotFoundError as e:
                if getattr(e, "winerror", None) == 2:
                    # Explicitly respect the WinError 2 toggle.
                    if log_winerror:
                        logger.debug("Suppressed: [WinError 2] Not found.")
                else:
                    logger.debug(f"Probe suppressed: {e}")
                raise

            except Exception as e:
                if is_probe and log_winerror:
                    logger.debug(f"Probe suppressed: {e}")
                raise

        finally:
            # 5. Critical: Release the lock before returning or raising.
            _popen_patch_lock.is_active = False

    # Apply the monkey-patch to global subprocess module.
    subprocess.Popen = _safe_popen
    logger.debug("Windows subprocess patch with recursion lock applied.")
# -----------------------------------------------------------------------------
