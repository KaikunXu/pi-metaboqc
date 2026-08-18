"""Public normalization computation API.

The package applies fixed or automatically selected normalization strategies
and returns structured stage results. Normalization diagnostics and dashboard
assembly are exposed from :mod:`pimqc.plotting.normalization`.
"""

from .analysis import MetaboIntNormalizer

__all__ = ["MetaboIntNormalizer"]
