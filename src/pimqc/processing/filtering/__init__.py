"""Public feature- and sample-filtering computation API.

The package implements missing-value and low-quality filtering decisions and
their structured stage execution. Filtering dashboards and diagnostic panels
are provided independently by :mod:`pimqc.plotting.filtering`.
"""

from .analysis import MetaboIntFilter

__all__ = ["MetaboIntFilter"]
