# src/pimqc/__init__.py
"""
pi-metaboqc: A Python-based high-throughput metabolomics data 
quality control and preprocessing pipeline.
"""
import sys
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
__version__ = "0.0.2.alpha"

# --- Monkey Patch for subprocess UnicodeDecodeError on Windows ----------------
if sys.platform == "win32":
    _original_popen = subprocess.Popen

    def _safe_popen(*args, **kwargs):
        # Log the command being executed to identify the culprit
        logger.warning(
            f"Subprocess call intercepted: {args[0] if args else 'Unknown'}")
        
        # Force the subprocess to ignore decoding errors
        if ('encoding' in kwargs) or ('text' in kwargs) or (
            'universal_newlines' in kwargs):
            kwargs['errors'] = 'ignore'
        
        return _original_popen(*args, **kwargs)

    subprocess.Popen = _safe_popen
    logger.debug("Windows subprocess patch applied successfully.")
# ------------------------------------------------------------------------------