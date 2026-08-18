"""Signal-correction algorithms and orchestration exports.

The package exposes correction engines required by the public pipeline and by
method-level tests, while low-level QC-RLSC helpers remain available for
reference validation and specialized analysis workflows.
"""

from .algorithms import _numba_loess_robust, _select_loess_span_oof
from .analysis import MetaboIntCorrector
from .regression import RegressionCorrector
from .ruv import RUVCorrector
from .serrf import SERRFCorrector
from .waveica import WaveICA2Corrector

__all__ = [
    "MetaboIntCorrector",
    "RegressionCorrector",
    "RUVCorrector",
    "SERRFCorrector",
    "WaveICA2Corrector",
    "_numba_loess_robust",
    "_select_loess_span_oof",
]
