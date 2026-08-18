"""Public plotter for missing-value imputation results.

Diagnostics, scorecards, and dashboard composition live in separate modules
and share state through ``ImputationPlotter``.
"""

from __future__ import annotations

from ..base import BasePlotter
from .dashboards import ImputationDashboardMixin
from .diagnostics import ImputationDiagnosticsMixin
from .scorecards import ImputationScorecardMixin


class ImputationPlotter(
    ImputationDiagnosticsMixin,
    ImputationScorecardMixin,
    ImputationDashboardMixin,
    BasePlotter,
):
    """Plotting suite for imputation accuracy and method selection."""

    def __init__(self, raw_obj, imp_obj) -> None:
        """Initialize with pre- and post-imputation matrices."""
        super().__init__(metabo_obj=imp_obj)
        self.raw_obj = raw_obj.astype(float).replace({0: float("nan")})
        self.imp_obj = imp_obj.astype(float)
