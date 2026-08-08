"""Focused checks for optional QC-RLSC polynomial and span-selection settings."""

import numpy as np
import pandas as pd

from pimqc.processing.correction import (
    RegressionCorrector,
    _numba_loess_robust,
    _select_loess_span_oof,
)


def test_quadratic_numba_loess_recovers_a_quadratic_signal() -> None:
    """Degree two should preserve an exactly quadratic local trend."""
    x = np.arange(1.0, 16.0)
    y = 12.0 + 1.5 * x + 0.25 * x * x

    predicted = _numba_loess_robust(x, y, x, 0.7, max_iter=0, degree=2)

    np.testing.assert_allclose(predicted, y, rtol=1e-10, atol=1e-10)


def test_default_numba_loess_is_explicit_degree_one() -> None:
    """The new degree option must preserve the existing default path."""
    x = np.arange(1.0, 16.0)
    y = 20.0 + 4.0 * x + 8.0 * np.sin(x / 3.0)
    default = _numba_loess_robust(x, y, x, 0.5, max_iter=3)
    explicit = _numba_loess_robust(x, y, x, 0.5, max_iter=3, degree=1)

    np.testing.assert_allclose(default, explicit, rtol=0.0, atol=0.0)


def test_span_selection_uses_grid_and_falls_back_below_minimum_qcs() -> None:
    """Constrained span selection is QC-only and has a deterministic fallback."""
    x = np.arange(1.0, 10.0)
    y = 10.0 + x + 0.1 * x * x
    grid = (0.3, 0.5, 0.7)

    selected = _select_loess_span_oof(
        x,
        y,
        grid,
        max_iter=0,
        degree=2,
        cv_folds=3,
        fallback_span=0.5,
        min_qc=7,
        random_state=123,
    )
    fallback = _select_loess_span_oof(
        x[:6],
        y[:6],
        grid,
        max_iter=0,
        degree=2,
        cv_folds=3,
        fallback_span=0.5,
        min_qc=7,
        random_state=123,
    )

    assert selected in grid
    assert fallback == 0.5


def test_qc_rlsc_gcv_options_support_standard_and_robust_modes() -> None:
    """Both residual-weighting modes share degree and constrained-span settings."""
    orders = np.arange(1.0, 19.0)
    qc_mask = np.zeros(len(orders), dtype=bool)
    qc_mask[::2] = True
    baseline = 100.0 + 2.0 * orders + 0.15 * orders * orders
    data = pd.DataFrame(
        [baseline, 1.5 * baseline + 5.0],
        index=["feature_1", "feature_2"],
    )
    common_params = {
        "loess_degree": 2,
        "rlsc_span_selection": "gcv",
        "rlsc_span_grid": [0.3, 0.5, 0.7],
        "rlsc_min_qc": 7,
        "cv_folds": 3,
    }

    for robust in (False, True):
        stages = RegressionCorrector(
            "QC-RLSC", robust=robust, robust_iterations=3, **common_params
        ).fit_transform(
            intensity_df=data,
            batch_array=np.repeat("batch_1", len(orders)),
            qc_mask=qc_mask,
            order_array=orders,
        )
        full, oof = stages["Intra-batch corrected"]
        assert full.shape == data.shape == oof.shape
        assert np.isfinite(full.to_numpy(dtype=float)).all()
        assert np.isfinite(oof.to_numpy(dtype=float)).all()
