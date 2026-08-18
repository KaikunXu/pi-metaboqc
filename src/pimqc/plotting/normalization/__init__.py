"""Public plotting API for normalization results.

The package composes distribution diagnostics, preservation scorecards, and
normalization dashboards into :class:`NormalizationPlotter`. Numerical methods
and stage execution remain under :mod:`pimqc.processing.normalization`.
"""

from .plotter import NormalizationPlotter

__all__ = ["NormalizationPlotter"]
