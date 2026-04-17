# src/pimqc/__init__.py
"""
pi-metaboqc: A Python-based high-throughput metabolomics data 
quality control and preprocessing pipeline.
"""

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