"""Signal-correction algorithms, orchestration, and visualization exports.

The package exposes correction engines required by the public pipeline and by
method-level tests, while low-level QC-RLSC helpers remain available for
reference validation and specialized analysis workflows.
"""

from .algorithms import _numba_loess_robust, _select_loess_span_oof
from .analysis import (
    MetaboIntCorrector,
    RegressionCorrector,
    RUVCorrector,
    SERRFCorrector,
    WaveICA2Corrector,
)
from .visualization import MetaboVisualizerCorrector

__all__ = [
    "MetaboIntCorrector",
    "MetaboVisualizerCorrector",
    "RegressionCorrector",
    "RUVCorrector",
    "SERRFCorrector",
    "WaveICA2Corrector",
    "_numba_loess_robust",
    "_select_loess_span_oof",
]
