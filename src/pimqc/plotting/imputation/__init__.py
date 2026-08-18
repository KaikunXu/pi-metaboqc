"""Public plotting API for missing-value imputation results.

The package composes reconstruction diagnostics, candidate scorecards, and
imputation dashboards into :class:`ImputationPlotter`. Numerical imputation is
owned by :mod:`pimqc.processing.imputation`.
"""

from .plotter import ImputationPlotter

__all__ = ["ImputationPlotter"]
