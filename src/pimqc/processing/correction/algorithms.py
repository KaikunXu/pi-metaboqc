"""Numerical signal-correction algorithms and QC-RLSC helper routines.

The module implements reusable low-level components for QC-based regression,
SERRF, RUV-III, WaveICA 2.0, LOESS span selection, robust residual weighting,
and correction-method label handling. These functions fit or predict correction
models but leave stage orchestration and file export to the analysis module.
"""

import math
import re
from typing import Callable, Tuple

import numpy as np
from numba import njit, prange
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from ...constants import DEFAULT_RANDOM_SEED
from .methods import CORRECTION_METHODS

FitPredictCallable = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
CorrectionModel = (
    RandomForestRegressor
    | TransformedTargetRegressor
    | Pipeline
    | FitPredictCallable
)


def _split_rlsc_robust_label(method: object) -> tuple[str, bool | None]:
    """Split optional QC-RLSC robustness markers from a method label."""
    method_text = str(method).strip()
    match = re.search(
        r"\s*\(\s*robust\s*=\s*(true|false)\s*\)\s*$", method_text, re.I
    )
    if match is not None:
        return method_text[: match.start()].strip(), match.group(
            1
        ).lower() == "true"

    match = re.match(r"^\s*robust\s+(.+?)\s*$", method_text, re.I)
    if match is not None:
        return match.group(1).strip(), True
    return method_text, None


def _format_correction_method_label(method: object) -> str:
    """Return the concise canonical display label for correction methods."""
    base_text, robust = _split_rlsc_robust_label(method)
    canonical = _normalize_correction_method(base_text)
    if robust is not None and canonical == "QC-RLSC":
        return "robust QC-RLSC" if robust else "QC-RLSC"
    return canonical


def _normalize_correction_method(method: object) -> str:
    """Normalize correction method aliases to canonical public names."""
    method_text, _ = _split_rlsc_robust_label(method)
    return CORRECTION_METHODS.canonicalize(method_text, strict=False)


def _parse_correction_candidate(
    raw_method: object,
) -> tuple[str, str, dict[str, object]]:
    """Return canonical method, display label, and candidate-specific params."""
    if isinstance(raw_method, dict):
        candidate_params = dict(raw_method.get("params", {}) or {})
        raw_label = raw_method.get("label")
        raw_method_name = raw_method.get("method", raw_label)
        method = _normalize_correction_method(raw_method_name)
        label = _format_correction_method_label(raw_label or raw_method_name)
    else:
        candidate_params = {}
        method = _normalize_correction_method(raw_method)
        label = _format_correction_method_label(raw_method)
    _, label_robust = _split_rlsc_robust_label(label)
    if method == "QC-RLSC" and label_robust is not None:
        candidate_params.setdefault("robust", label_robust)
    return method, label, candidate_params


def _format_correction_method_file_label(method: str) -> str:
    """Return a filesystem-friendly label for correction method identifiers."""
    return re.sub(r'[<>:"/\\|?*]', "-", _format_correction_method_label(method))


# =============================================================================
# Cross-Validation Engine for Robust Drift Correction to Prevent Overfitting
# =============================================================================
def fit_predict_intra_batch_safely(
    base_model: CorrectionModel,
    x_qc: np.ndarray,
    y_qc: np.ndarray,
    x_all: np.ndarray,
    cv_folds: int = 5,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return both continuous full baseline and OOF baseline for metrics."""
    n_qc = len(y_qc)
    pred_qc_oof = np.zeros(n_qc)
    safe_folds = min(cv_folds, n_qc)

    def _run_model(
        x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
    ) -> np.ndarray:
        if hasattr(base_model, "fit") and hasattr(base_model, "predict"):
            model = clone(base_model)
            model.fit(x_train, y_train)
            return model.predict(x_test)
        return base_model(x_train, y_train, x_test)

    if safe_folds < 3:
        pred_all_full = _run_model(x_qc, y_qc, x_all)
        pred_qc_oof = _run_model(x_qc, y_qc, x_qc)
        return (
            np.clip(pred_all_full, a_min=1e-6, a_max=None),
            np.clip(pred_qc_oof, a_min=1e-6, a_max=None),
        )

    kf = KFold(n_splits=safe_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(x_qc):
        pred_qc_oof[test_idx] = _run_model(
            x_train=x_qc[train_idx],
            y_train=y_qc[train_idx],
            x_test=x_qc[test_idx],
        )

    pred_all_full = _run_model(x_train=x_qc, y_train=y_qc, x_test=x_all)

    return (
        np.clip(pred_all_full, a_min=1e-6, a_max=None),
        np.clip(pred_qc_oof, a_min=1e-6, a_max=None),
    )


# =============================================================================
# Numba JIT Engines for Fast Robust QC-RLSC
# =============================================================================
@njit(fastmath=True)
def _tricube_kernel(x: float) -> float:
    abs_x = abs(x)
    if abs_x >= 1.0:
        return 0.0
    return (1.0 - abs_x**3) ** 3


@njit(fastmath=True)
def _bisquare_weight(res: float, s: float) -> float:
    if s <= 1e-9:
        return 1.0 if abs(res) <= 1e-9 else 0.0
    v = res / (6.0 * s)
    if abs(v) >= 1.0:
        return 0.0
    return (1.0 - v * v) ** 2


@njit(fastmath=True)
def _numba_loess_1d_core(
    x: np.ndarray,
    y: np.ndarray,
    x_pred: np.ndarray,
    loess_span: float,
    delta: np.ndarray,
    degree: int = 1,
) -> np.ndarray:
    n = len(x)
    m = len(x_pred)
    y_pred = np.zeros(m)

    k = int(math.ceil(n * loess_span))
    if k < 2:
        k = 2
    if k > n:
        k = n

    for i in range(m):
        x0 = x_pred[i]
        diffs = np.abs(x - x0)
        sorted_diffs = np.sort(diffs)
        h = sorted_diffs[k - 1]

        if h <= 0.0:
            h = 1e-9

        if degree >= 2:
            s0 = 0.0
            s1 = 0.0
            s2 = 0.0
            s3 = 0.0
            s4 = 0.0
            sy0 = 0.0
            sy1 = 0.0
            sy2 = 0.0
            for j in range(n):
                t = (x[j] - x0) / h
                w = _tricube_kernel(t) * delta[j]
                t2 = t * t
                s0 += w
                s1 += w * t
                s2 += w * t2
                s3 += w * t2 * t
                s4 += w * t2 * t2
                sy0 += w * y[j]
                sy1 += w * t * y[j]
                sy2 += w * t2 * y[j]

            det = (
                s0 * (s2 * s4 - s3 * s3)
                - s1 * (s1 * s4 - s3 * s2)
                + s2 * (s1 * s3 - s2 * s2)
            )
            if abs(det) <= 1e-12 or s0 <= 0.0:
                y_pred[i] = _numba_loess_1d_core(
                    x, y, np.array([x0]), loess_span, delta, 1
                )[0]
                continue

            a0 = (
                sy0 * (s2 * s4 - s3 * s3)
                - s1 * (sy1 * s4 - s3 * sy2)
                + s2 * (sy1 * s3 - s2 * sy2)
            ) / det
            a1 = (
                s0 * (sy1 * s4 - s3 * sy2)
                - sy0 * (s1 * s4 - s3 * s2)
                + s2 * (s1 * sy2 - sy1 * s2)
            ) / det
            a2 = (
                s0 * (s2 * sy2 - sy1 * s3)
                - s1 * (s1 * sy2 - sy1 * s2)
                + sy0 * (s1 * s3 - s2 * s2)
            ) / det
            t_pred = (x_pred[i] - x0) / h
            y_pred[i] = a0 + a1 * t_pred + a2 * t_pred * t_pred
            continue

        sum_w = 0.0
        sum_wx = 0.0
        sum_wy = 0.0

        for j in range(n):
            w = _tricube_kernel((x[j] - x0) / h) * delta[j]
            sum_w += w
            sum_wx += w * x[j]
            sum_wy += w * y[j]

        if sum_w <= 0:
            y_pred[i] = np.mean(y)
            continue

        x_bar = sum_wx / sum_w
        y_bar = sum_wy / sum_w

        numerator = 0.0
        denominator = 0.0

        for j in range(n):
            w = _tricube_kernel((x[j] - x0) / h) * delta[j]
            dev_x = x[j] - x_bar
            numerator += w * dev_x * (y[j] - y_bar)
            denominator += w * dev_x * dev_x

        if denominator <= 0:
            beta1 = 0.0
        else:
            beta1 = numerator / denominator

        beta0 = y_bar - beta1 * x_bar
        y_pred[i] = beta0 + beta1 * x0

    return y_pred


@njit(fastmath=True)
def _numba_loess_robust(
    x: np.ndarray,
    y: np.ndarray,
    x_pred: np.ndarray,
    loess_span: float,
    max_iter: int,
    degree: int = 1,
) -> np.ndarray:
    n = len(x)
    delta = np.ones(n)
    for iteration in range(max_iter):
        y_fit = _numba_loess_1d_core(x, y, x, loess_span, delta, degree)
        residuals = np.abs(y - y_fit)
        s = np.median(residuals)
        for j in range(n):
            delta[j] = _bisquare_weight(residuals[j], s)
    return _numba_loess_1d_core(x, y, x_pred, loess_span, delta, degree)


@njit(parallel=True, fastmath=True)
def _numba_batch_qc_rlsc(
    data: np.ndarray,
    qc_mask: np.ndarray,
    injection_orders: np.ndarray,
    loess_span: float,
    max_iter: int,
    degree: int = 1,
) -> np.ndarray:
    rows, cols = data.shape
    predicted_matrix = np.zeros((rows, cols))
    x_qc_all = injection_orders[qc_mask]

    for i in prange(rows):
        row_data = data[i, :]
        y_qc_all = row_data[qc_mask]

        valid_mask = ~np.isnan(y_qc_all)
        valid_count = np.sum(valid_mask)

        if valid_count < 3:
            mean_val = 0.0
            if valid_count > 0:
                mean_val = np.sum(y_qc_all[valid_mask]) / valid_count
            else:
                mean_val = np.nan
            predicted_matrix[i, :] = mean_val
            continue

        clean_x = x_qc_all[valid_mask]
        clean_y = y_qc_all[valid_mask]

        predicted_matrix[i, :] = _numba_loess_robust(
            clean_x, clean_y, injection_orders, loess_span, max_iter, degree
        )

    return predicted_matrix


@njit(parallel=True, fastmath=True)
def _numba_batch_qc_rlsc_per_feature_spans(
    data: np.ndarray,
    qc_mask: np.ndarray,
    injection_orders: np.ndarray,
    loess_spans: np.ndarray,
    max_iter: int,
    degree: int,
) -> np.ndarray:
    """Fit QC-RLSC with one selected span per feature."""
    rows, cols = data.shape
    predicted_matrix = np.zeros((rows, cols))
    x_qc_all = injection_orders[qc_mask]

    for i in prange(rows):
        row_data = data[i, :]
        y_qc_all = row_data[qc_mask]
        valid_mask = ~np.isnan(y_qc_all)
        valid_count = np.sum(valid_mask)
        if valid_count < 3:
            predicted_matrix[i, :] = np.nanmean(y_qc_all)
            continue
        clean_x = x_qc_all[valid_mask]
        clean_y = y_qc_all[valid_mask]
        predicted_matrix[i, :] = _numba_loess_robust(
            clean_x, clean_y, injection_orders, loess_spans[i], max_iter, degree
        )
    return predicted_matrix


def _select_loess_span_oof(
    x_qc: np.ndarray,
    y_qc: np.ndarray,
    span_grid: tuple[float, ...],
    max_iter: int,
    degree: int,
    cv_folds: int,
    fallback_span: float,
    min_qc: int,
    random_state: int,
) -> float:
    """Select a span using QC-only OOF mean squared error."""
    valid = np.isfinite(x_qc) & np.isfinite(y_qc)
    x_qc = x_qc[valid]
    y_qc = y_qc[valid]
    if len(y_qc) < max(3, min_qc):
        return fallback_span
    n_splits = min(max(2, int(cv_folds)), len(y_qc))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_span = fallback_span
    best_error = np.inf
    for span in span_grid:
        predictions = np.full(len(y_qc), np.nan, dtype=float)
        for train_idx, test_idx in kf.split(x_qc):
            predictions[test_idx] = _numba_loess_robust(
                x_qc[train_idx],
                y_qc[train_idx],
                x_qc[test_idx],
                span,
                max_iter,
                degree,
            )
        valid_pred = np.isfinite(predictions)
        if valid_pred.any():
            error = float(
                np.mean((predictions[valid_pred] - y_qc[valid_pred]) ** 2)
            )
            if error < best_error:
                best_error = error
                best_span = span
    return best_span


def _select_batch_loess_spans(
    data: np.ndarray,
    qc_mask: np.ndarray,
    injection_orders: np.ndarray,
    span_grid: tuple[float, ...],
    max_iter: int,
    degree: int,
    cv_folds: int,
    fallback_span: float,
    min_qc: int,
    random_state: int,
) -> np.ndarray:
    """Select one GCV span per feature for a single batch."""
    selected = np.full(data.shape[0], fallback_span, dtype=float)
    x_qc = injection_orders[qc_mask]
    for i in range(data.shape[0]):
        selected[i] = _select_loess_span_oof(
            x_qc,
            data[i, qc_mask],
            span_grid,
            max_iter,
            degree,
            cv_folds,
            fallback_span,
            min_qc,
            random_state,
        )
    return selected
