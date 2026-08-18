"""Public plotter for normalization results.

Diagnostics, scorecards, and dashboard composition live in separate modules
and share state through ``NormalizationPlotter``.
"""

from __future__ import annotations

from ..base import BasePlotter
from .. import plot_utils as pu
from .dashboards import NormalizationDashboardMixin
from .diagnostics import NormalizationDiagnosticsMixin
from .scorecards import NormalizationScorecardMixin


class NormalizationPlotter(
    NormalizationScorecardMixin,
    NormalizationDiagnosticsMixin,
    NormalizationDashboardMixin,
    BasePlotter,
):
    """Plotting suite for normalization evaluation and selection."""

    def __init__(self, raw_obj, norm_obj) -> None:
        """Initialize with pre- and post-normalization matrices."""
        super().__init__(metabo_obj=norm_obj)
        self.raw = raw_obj
        self.norm = norm_obj
        self.stages = [("Before Norm", self.raw), ("After Norm", self.norm)]
        self.pal = {
            "Before Norm": pu.NEUTRAL_COLOR,
            "After Norm": pu.PRIMARY_ACCENT_COLOR,
        }
