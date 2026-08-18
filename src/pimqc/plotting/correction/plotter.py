"""Public plotter for signal-correction results.

Diagnostics, scorecards, internal-standard panels, and dashboard composition
live in separate modules and share state through ``CorrectionPlotter``.
"""

from __future__ import annotations

from ..base import BasePlotter
from .dashboards import CorrectionDashboardMixin
from .diagnostics import CorrectionDiagnosticsMixin
from .internal_standards import CorrectionInternalStandardMixin
from .scorecards import CorrectionScorecardMixin


class CorrectionPlotter(
    CorrectionDiagnosticsMixin,
    CorrectionScorecardMixin,
    CorrectionDashboardMixin,
    CorrectionInternalStandardMixin,
    BasePlotter,
):
    """Plotting suite for correction evaluation and diagnostics."""

    def __init__(self, corr_obj) -> None:
        """Initialize with a computed correction stage."""
        super().__init__(metabo_obj=corr_obj)
        self.corr = corr_obj
