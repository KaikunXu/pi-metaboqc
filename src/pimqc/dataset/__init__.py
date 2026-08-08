"""Dataset construction and entry-level acquisition diagnostics.

The package exports the builder that validates metadata and intensity inputs,
along with the corresponding visualization class used to document the initial
project state before any processing stage changes the dataset.
"""

from .builder import MetaboIntBuilder, build_dataset
from .visualization import MetaboVisualizerBuilder

__all__ = ["MetaboIntBuilder", "MetaboVisualizerBuilder", "build_dataset"]
