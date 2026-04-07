# src/pimqc/__init__.py
"""
pi-metaboqc: A comprehensive LC-MS metabolomics data quality control package.

This package provides an object-oriented pipeline for preprocessing metabolomics
data, including dataset building, invalid feature/sample filtering, quality 
assessment, signal correction, and data normalization.
"""

from .core_classes import MetaboInt
from .dataset_builder import build_dataset
from .invalid_feature_sample_filtering import MetaboIntFLTR
from .data_quality_assessment import MetaboIntQA
from .data_signal_correction import MetaboIntSC
from .normalization import MetaboIntNorm
from .imputation import MetaboIntImputer
from .pipeline import run_pipeline

__all__ = [
    "__version__",
    "MetaboInt",
    "build_dataset",
    "MetaboIntFLTR", 
    "MetaboIntQA",
    "MetaboIntSC",
    "MetaboIntNorm",
    "MetaboIntImputer",
    "run_pipeline"
]

__version__ = "0.1.dev"