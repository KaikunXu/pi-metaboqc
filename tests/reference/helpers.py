"""Provide numerical and dependency helpers shared by R bridge tests.

The bridge modules intentionally remain separate so failures identify the
affected algorithm. This support module holds only genuinely identical test
infrastructure: optional R package detection and generic matrix-comparison
statistics.
"""

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro


def require_r_package(package: str) -> None:
    """Skip the active bridge test when an optional R package is unavailable."""
    require_namespace = ro.r(
        "function(pkg) requireNamespace(pkg, quietly=TRUE)"
    )
    if not bool(require_namespace(package)[0]):
        pytest.skip(f"System R package '{package}' is not installed.")


def relative_mae(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Return mean absolute error relative to finite reference magnitude."""
    valid = np.isfinite(reference) & np.isfinite(estimate)
    absolute_error = float(np.mean(np.abs(reference[valid] - estimate[valid])))
    denominator = float(np.mean(np.abs(reference[valid])))
    return absolute_error / max(denominator, 1e-12)


def median_qc_rsd(data: pd.DataFrame, qc_mask: np.ndarray) -> float:
    """Return the median feature RSD across the selected QC columns."""
    qc_data = data.iloc[:, qc_mask]
    means = qc_data.mean(axis=1)
    rsd = qc_data.std(axis=1, ddof=1).div(means.replace(0.0, np.nan))
    return float(np.nanmedian(rsd.to_numpy()))
