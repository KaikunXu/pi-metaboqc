"""Plotting infrastructure and stage-specific plotters for pi-metaboqc.

The package owns the shared figure lifecycle, reusable plotting primitives,
collision-aware annotation layout, dataset diagnostics, and plotters for every
processing stage. Numerical stage logic remains under
:mod:`pimqc.processing`.
"""

from .base import BasePlotter
from .sample_structure import plot_sample_structure_change_map

__all__ = ["BasePlotter", "plot_sample_structure_change_map"]
