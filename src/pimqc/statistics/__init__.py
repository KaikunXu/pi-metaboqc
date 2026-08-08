"""Shared statistical helpers and numerical-engine exports.

PCAEngine and generic candidate-ranking helpers are exported for use across
processing stages. Other metric functions remain internal implementation details
of the statistics package to keep the public surface concise.
"""

from .pca import PCAEngine
from .selection import rank_candidates, selection_margin

__all__ = ["PCAEngine", "rank_candidates", "selection_margin"]
