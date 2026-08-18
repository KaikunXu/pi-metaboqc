"""Public plotting API for quality-assessment results.

The package composes heatmaps, PCA panels, outlier diagnostics, control charts,
and assessment dashboards into :class:`AssessmentPlotter`. Computation remains
owned by :mod:`pimqc.processing.assessment`.
"""

from .plotter import AssessmentPlotter

__all__ = ["AssessmentPlotter"]
