"""Public dataset-construction and input-validation API.

The package validates input tables, constructs the core ``MetaboInt`` matrix,
and records acquisition metadata required by downstream processing. Dataset
figures are intentionally exposed from :mod:`pimqc.plotting.dataset`.
"""

from .builder import MetaboIntBuilder, build_dataset

__all__ = ["MetaboIntBuilder", "build_dataset"]
