"""Core domain data structures shared by every pipeline stage.

The package exposes MetaboInt, the metadata-preserving dataframe subclass used
to carry matrices, sample labels, feature annotations, metrics, and processing
state through the pi-metaboqc workflow.
"""

from .model import MetaboInt

__all__ = ["MetaboInt"]
