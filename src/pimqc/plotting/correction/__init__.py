"""Public plotting API for signal-correction results.

The package composes candidate diagnostics, internal-standard panels, and
correction dashboards into :class:`CorrectionPlotter`. Correction algorithms
and stage orchestration remain under :mod:`pimqc.processing.correction`.
"""

from .plotter import CorrectionPlotter

__all__ = ["CorrectionPlotter"]
