"""Public plotter for quality-assessment diagnostics.

The concrete panels are grouped by purpose in sibling modules, while
``AssessmentPlotter`` owns their shared ``MetaboInt`` state and public API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..base import BasePlotter
from .control_charts import AssessmentControlChartMixin
from .dashboards import AssessmentDashboardMixin
from .heatmaps import AssessmentHeatmapMixin
from .outliers import AssessmentOutlierMixin
from .pca import AssessmentPcaMixin

if TYPE_CHECKING:
    from ...processing.assessment.analysis import MetaboIntAssessor


class AssessmentPlotter(
    AssessmentHeatmapMixin,
    AssessmentPcaMixin,
    AssessmentOutlierMixin,
    AssessmentControlChartMixin,
    AssessmentDashboardMixin,
    BasePlotter,
):
    """Plotting suite for metabolomics data quality assessment."""

    def __init__(self, qa_obj: MetaboIntAssessor) -> None:
        """Initialize with a computed MetaboIntAssessor object."""
        super().__init__(metabo_obj=qa_obj)
