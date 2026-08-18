"""Public plotter for sample and feature filtering.

Diagnostic panels, the filtering flowchart, and dashboard composition live in
separate modules and share state through ``FilteringPlotter``.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..base import BasePlotter
from .dashboards import FilteringDashboardMixin
from .diagnostics import FilteringDiagnosticsMixin
from .flowchart import FilteringFlowchartMixin


class FilteringPlotter(
    FilteringDiagnosticsMixin,
    FilteringFlowchartMixin,
    FilteringDashboardMixin,
    BasePlotter,
):
    """Plotting suite for sample and feature filtering outcomes."""

    def __init__(
        self,
        engine,
        audit_tables: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize with a computed engine and explicit audit tables."""
        super().__init__(metabo_obj=engine)
        self.engine = engine
        self.audit_tables = dict(audit_tables or {})
