# src/pimqc/__init__.py
"""
pi-metaboqc: A Python-based high-throughput metabolomics data quality control 
and preprocessing pipeline.
"""
import sys
import shlex
import subprocess
from loguru import logger

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

# Define public API
__all__ = [
    "MetaboInt",
    "build_dataset",
    "run_pipeline",
    "MetaboIntAssessor",
    "MetaboIntCorrector",
    "MetaboIntImputer",
    "MetaboIntNormalizer",
    "MetaboIntFilter",
]

# Package Version
__version__ = "0.0.3.alpha"

# --- Monkey Patch for subprocess on Windows -----------------------------------
if sys.platform == "win32":
    # Global toggle to control WinError 2 logging noise from third-party libs
    # Set to True for deep debugging, False for clean production logs
    LOG_WINERROR2 = False

    _original_popen = subprocess.Popen

    def _safe_popen(*args, **kwargs):
        """Intercepts subprocess calls to filter noise and prevent crashes.

        Includes an explicit switch (log_winerror2) to globally or locally 
        toggle the verbose [WinError 2] missing file logs.
        """
        # 1. Extract custom toggle safely without passing it to the OS
        log_winerror = kwargs.pop("log_winerror2", LOG_WINERROR2)

        # 2. Extract arguments and format for clean logging
        cmd_args = args[0] if args else kwargs.get("args", [])
        
        if isinstance(cmd_args, list):
            cmd_str_for_log = shlex.join([str(x) for x in cmd_args])
            cmd_str_lower = str(cmd_args).lower()
        else:
            cmd_str_for_log = str(cmd_args)
            cmd_str_lower = cmd_str_for_log.lower()

        safe_probes = [
            "--version", "--list", "powershell", "win32_processor", 
            "msiexec", "where"
        ]
        is_probe = any(probe in cmd_str_lower for probe in safe_probes)
        
        # Heuristic to detect blind absolute path hunting
        is_path_hunting = is_probe and (
            "\\" in cmd_str_for_log or "/" in cmd_str_for_log)

        # 3. Controlled Logging
        if is_probe:
            # Hide the hunting attempt entirely if the WinError switch is off
            if log_winerror or not is_path_hunting:
                logger.debug(f"Permitted probe: {cmd_str_for_log}")
                
            if "powershell" in cmd_str_lower:
                kwargs.setdefault("creationflags", 0x08000000)
        else:
            # [UPDATE] Downgraded from WARNING to DEBUG for production.
            # Normal functional commands should not alarm the end-user.
            logger.debug(f"Functional subprocess: {cmd_str_for_log}")

        err_keys = ("encoding", "text", "universal_newlines")
        if any(k in kwargs for k in err_keys):
            kwargs["errors"] = "ignore"

        # 4. Execute original call with explicit error toggling
        try:
            return _original_popen(*args, **kwargs)
            
        except FileNotFoundError as e:
            if getattr(e, "winerror", None) == 2:
                # Explicitly respect the WinError 2 toggle
                if log_winerror:
                    logger.debug(
                        "Probe suppressed: [WinError 2] File not found."
                    )
            else:
                logger.debug(f"Probe suppressed: {e}")
            raise
            
        except Exception as e:
            if is_probe and log_winerror:
                logger.debug(f"Probe suppressed: {e}")
            raise

    # Apply the monkey-patch to global subprocess module
    subprocess.Popen = _safe_popen
    logger.debug("Windows subprocess patch with smart filtering applied.")
# ------------------------------------------------------------------------------