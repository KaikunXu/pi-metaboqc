"""Shared numerical transformations and quality-control metrics.

Functions in this module provide robust logging, scaling, QC-RSD summaries,
masked-value fidelity, technical-improvement, and sample-geometry measures.
Processing stages use the same implementations to make AUTO candidate scoring
consistent and to preserve comparable metrics in their audit records.
"""

import math

import numpy as np
import pandas as pd
from numba import njit, prange
from loguru import logger
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, wasserstein_distance


# =============================================================================
# Numba-Compiled Kernels
# =============================================================================
@njit(parallel=True, fastmath=True)
def _numba_gaussian_kde(
    data: np.ndarray, grid: np.ndarray, bandwidth: float
) -> np.ndarray:
    """Extremely fast parallel KDE using Silverman's rule of thumb."""
    n = len(data)
    m = len(grid)
    kde_values = np.zeros(m)
    c = 1.0 / (bandwidth * math.sqrt(2.0 * math.pi))

    for i in prange(m):
        grid_val = grid[i]
        kernel_sum = 0.0
        for j in range(n):
            u = (grid_val - data[j]) / bandwidth
            kernel_sum += math.exp(-0.5 * u * u)
        kde_values[i] = (kernel_sum / n) * c
    return kde_values


# =============================================================================
# Data Transformation & Scaling Operators
# =============================================================================
def robust_log2_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Perform log2 transform with LOD offset (half of min positive)."""
    out_df = df.copy()
    if (out_df <= 0).any().any():
        min_pos = out_df[out_df > 0].min().min()
        offset = min_pos / 2.0
        out_df = out_df.clip(lower=offset)
    return np.log2(out_df)


def calc_auto_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Auto Scaling (Z-score) feature-wise."""
    # Compute mean and std for the provided subset
    row_means = df.mean(axis=1)
    row_stds = df.std(axis=1, ddof=1).replace(0, 1)
    return df.sub(row_means, axis=0).div(row_stds, axis=0)


def calc_pareto_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Pareto Scaling feature-wise."""
    # Divide by square root of standard deviation
    row_means = df.mean(axis=1)
    row_stds = df.std(axis=1, ddof=1).replace(0, 1)
    return df.sub(row_means, axis=0).div(np.sqrt(row_stds), axis=0)


def apply_feature_scaling(
    df: pd.DataFrame, method: str = "None"
) -> pd.DataFrame:
    """Unified entry for applying feature-wise scaling.

    Args:
        df: Feature matrix (Features as rows, Samples as columns).
        method: Scaling strategy ("Auto-scaling", "Pareto-scaling", "None").
    """
    method_upper = str(method).upper()
    if method_upper == "AUTO-SCALING":
        return calc_auto_scaling(df)
    elif method_upper == "PARETO-SCALING":
        return calc_pareto_scaling(df)
    return df


# =============================================================================
# Statistical Similarity Metrics
# =============================================================================
def calc_jsd_similarity(
    arr1: np.ndarray, arr2: np.ndarray, grid_points: int = 200
) -> dict[str, float]:
    """Calculate a stable Jensen-Shannon distance between two distributions.

    KDE bandwidths are bounded by the evaluation-grid resolution so constant
    or nearly constant vectors cannot collapse to an all-zero density. If a
    numerical backend still produces an invalid density, a shared-bin
    histogram provides a deterministic degradation path.
    """
    a = np.asarray(arr1, dtype=float).ravel()
    b = np.asarray(arr2, dtype=float).ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return {"jsd": float("nan")}

    grid_points = max(2, int(grid_points))
    lower = float(min(a.min(), b.min()))
    upper = float(max(a.max(), b.max()))
    value_scale = max(abs(lower), abs(upper), 1.0)
    numeric_tol = np.sqrt(np.finfo(float).eps) * value_scale

    if upper - lower <= numeric_tol:
        return {"jsd": 0.0}

    grid = np.linspace(lower, upper, grid_points)
    bandwidth_floor = max(
        (upper - lower) / (grid_points - 1),
        numeric_tol,
    )

    def _stable_bandwidth(values: np.ndarray) -> float:
        std = float(np.std(values))
        silverman = (
            1.06 * std * (len(values) ** -0.2)
            if np.isfinite(std) and std > numeric_tol
            else 0.0
        )
        return max(silverman, bandwidth_floor)

    p = _numba_gaussian_kde(a, grid, _stable_bandwidth(a))
    q = _numba_gaussian_kde(b, grid, _stable_bandwidth(b))
    p_sum = float(np.sum(p))
    q_sum = float(np.sum(q))

    if (
        not np.isfinite(p_sum)
        or not np.isfinite(q_sum)
        or p_sum <= np.finfo(float).tiny
        or q_sum <= np.finfo(float).tiny
    ):
        logger.warning(
            "KDE density normalization became degenerate while calculating "
            "Jensen-Shannon distance; falling back to shared-bin histograms."
        )
        edges = np.linspace(lower, upper, grid_points + 1)
        p, _ = np.histogram(a, bins=edges)
        q, _ = np.histogram(b, bins=edges)
        p = p.astype(float)
        q = q.astype(float)
        p_sum = float(np.sum(p))
        q_sum = float(np.sum(q))

    if p_sum <= 0.0 or q_sum <= 0.0:
        logger.warning(
            "Jensen-Shannon distance is undefined because both KDE and "
            "histogram normalization failed; returning NaN."
        )
        return {"jsd": float("nan")}

    # Gaussian tails may legitimately underflow to exact zero. They carry no
    # meaningful probability mass, so ignore only that benign IEEE condition
    # while retaining all divide/invalid checks above.
    with np.errstate(under="ignore"):
        p /= p_sum
        q /= q_sum
    probability_floor = np.finfo(float).tiny
    p[p < probability_floor] = 0.0
    q[q < probability_floor] = 0.0
    distance = float(jensenshannon(p, q))
    if not np.isfinite(distance):
        logger.warning(
            "Jensen-Shannon distance was non-finite after normalization; "
            "returning NaN."
        )
        return {"jsd": float("nan")}

    return {"jsd": distance}


def finite_or_nan(value: object) -> float:
    """Convert a scalar metric to float, returning NaN for invalid values."""
    try:
        float_val = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return float_val if np.isfinite(float_val) else float("nan")


def series_iqr(series: pd.Series) -> float:
    """Calculate the interquartile range of a numeric Series."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.quantile(0.75) - clean.quantile(0.25))


def align_paired_matrices(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align paired matrices to the same feature and sample order."""
    common_features = before_df.index.intersection(after_df.index, sort=False)
    common_samples = before_df.columns.intersection(
        after_df.columns, sort=False
    )
    return (
        before_df.loc[common_features, common_samples],
        after_df.loc[common_features, common_samples],
    )


def weighted_mean_score(
    score_weights: list[tuple[float, float]],
    clip_values: bool = True,
) -> float:
    """Return a weighted mean over finite component scores."""
    weighted_sum = 0.0
    weight_sum = 0.0
    for score, weight in score_weights:
        score_val = finite_or_nan(score)
        if not np.isfinite(score_val) or weight <= 0:
            continue
        if clip_values:
            score_val = float(np.clip(score_val, 0.0, 1.0))
        weighted_sum += score_val * weight
        weight_sum += weight
    if weight_sum <= 0:
        return float("nan")
    return float(weighted_sum / weight_sum)


def relative_change_lower_better(before: object, after: object) -> float:
    """Return relative before-to-after change for a lower-is-better metric."""
    before_val = finite_or_nan(before)
    after_val = finite_or_nan(after)
    if not np.isfinite(before_val) or not np.isfinite(after_val):
        return float("nan")
    if before_val <= np.finfo(float).eps:
        return 0.0 if after_val <= np.finfo(float).eps else -1.0
    return float((before_val - after_val) / before_val)


def practical_signed_change_lower_better(
    before: object,
    after: object,
    min_abs_change: float = 0.0,
    min_rel_change: float = 0.0,
) -> float:
    """Return signed relative change after applying practical deadbands."""
    before_val = finite_or_nan(before)
    after_val = finite_or_nan(after)
    if not np.isfinite(before_val) or not np.isfinite(after_val):
        return float("nan")

    signed_change = relative_change_lower_better(before_val, after_val)
    if not np.isfinite(signed_change):
        return float("nan")

    abs_change = abs(before_val - after_val)
    if abs_change < min_abs_change or abs(signed_change) < min_rel_change:
        return 0.0
    return float(signed_change)


def rank_loss_from_distances(
    raw_dist: np.ndarray, transformed_dist: np.ndarray
) -> float:
    """Return 1 - Spearman rho between two pairwise distance vectors."""
    valid_mask = np.isfinite(raw_dist) & np.isfinite(transformed_dist)
    if int(valid_mask.sum()) < 3:
        return float("nan")

    rho_val = spearmanr(raw_dist[valid_mask], transformed_dist[valid_mask])[0]
    rho_val = finite_or_nan(rho_val)
    if not np.isfinite(rho_val):
        return float("nan")
    return float(max(0.0, 1.0 - rho_val))


def robust_feature_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Scale each feature by median/MAD across samples for geometry checks."""
    feature_median = df.median(axis=1, skipna=True)
    centered = df.sub(feature_median, axis=0)
    feature_mad = centered.abs().median(axis=1, skipna=True)
    robust_scale = (feature_mad * 1.4826).replace(0.0, np.nan)

    z_df = centered.div(robust_scale, axis=0).replace([np.inf, -np.inf], np.nan)
    valid_feature = z_df.notna().sum(axis=1) >= 3
    return z_df.loc[valid_feature].fillna(0.0)


def calc_distribution_distance_metrics(
    reference_values: np.ndarray,
    comparison_values: np.ndarray,
) -> dict[str, float]:
    """
    Calculate label-free distribution distances between paired value vectors.
    """
    ref = np.asarray(reference_values, dtype=float)
    comp = np.asarray(comparison_values, dtype=float)
    ref = ref[np.isfinite(ref)]
    comp = comp[np.isfinite(comp)]
    metrics = {
        "jsd": float("nan"),
        "wasserstein": float("nan"),
        "wasserstein_normalized": float("nan"),
    }
    if ref.size < 2 or comp.size < 2:
        return metrics

    jsd_result = calc_jsd_similarity(ref, comp)
    metrics["jsd"] = finite_or_nan(jsd_result.get("jsd"))
    metrics["wasserstein"] = finite_or_nan(wasserstein_distance(ref, comp))

    ref_scale = np.nanpercentile(ref, 75) - np.nanpercentile(ref, 25)
    if not np.isfinite(ref_scale) or ref_scale <= np.finfo(float).eps:
        ref_scale = np.nanmax(ref) - np.nanmin(ref)
    if np.isfinite(ref_scale) and ref_scale > np.finfo(float).eps:
        metrics["wasserstein_normalized"] = finite_or_nan(
            metrics["wasserstein"] / ref_scale
        )

    return metrics


# =============================================================================
# Target Data Extraction for Imputation and Normalization
# =============================================================================
def _extract_log2_target(
    obj: pd.DataFrame | None, auto_log_for_vis: bool = True
) -> pd.DataFrame | None:
    """Extract data and strictly enforce Log2 scale for relative comparisons.

    The function treats already logged matrices and completed VSN outputs as
    log-like data. Other matrices are transformed transiently for comparable
    structure and distribution diagnostics.
    """
    if obj is None:
        return None

    try:
        target_cols = obj.columns.difference(obj._blank.columns)
    except (AttributeError, KeyError):
        target_cols = obj.columns
    data = obj[target_cols].astype(float)

    is_logged = obj.attrs.get("is_logged", False)
    norm_method = str(obj.attrs.get("norm_method", "None")).upper()
    current_stage = str(obj.attrs.get("pipeline_stage", "Unknown"))

    is_post_norm = current_stage == "Normalization"

    # Completed VSN outputs are already in glog-like space.
    if is_logged or (norm_method == "VSN" and is_post_norm):
        return data

    if auto_log_for_vis:
        from . import metrics as su

        return su.robust_log2_transform(data)

    return data
