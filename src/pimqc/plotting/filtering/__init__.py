"""Public plotting API for sample- and feature-filtering results.

The package composes filtering diagnostics, decision flowcharts, and dashboard
layouts into :class:`FilteringPlotter`. Filtering calculations remain under
:mod:`pimqc.processing.filtering`.
"""

from .plotter import FilteringPlotter

__all__ = ["FilteringPlotter"]
