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
from scipy.spatial.distance import jensenshannon, pdist, squareform
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.manifold import trustworthiness


# ====================================================================
# Numba Compiled Fast Engine
# ====================================================================
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


# ====================================================================
# Data Transformation & Scaling Operators
# ====================================================================
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


# ====================================================================
# Statistical Similarity Metrics
# ====================================================================
def calc_jsd_similarity(
    arr1: np.ndarray, arr2: np.ndarray, grid_points: int = 200
) -> dict[str, float]:
    """Calculate Jensen-Shannon Divergence between two arrays via KDE."""
    a = arr1[~np.isnan(arr1)]
    b = arr2[~np.isnan(arr2)]
    if len(a) == 0 or len(b) == 0:
        return {"jsd": 0.0}

    grid = np.linspace(
        min(a.min(), b.min()), max(a.max(), b.max()), grid_points
    )
    bw_a = 1.06 * np.std(a) * (len(a) ** -0.2) if np.std(a) > 0 else 0.1
    bw_b = 1.06 * np.std(b) * (len(b) ** -0.2) if np.std(b) > 0 else 0.1

    p = _numba_gaussian_kde(a, grid, bw_a)
    q = _numba_gaussian_kde(b, grid, bw_b)
    p /= p.sum() + 1e-10
    q /= q.sum() + 1e-10

    return {"jsd": float(jensenshannon(p, q))}


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


def score_lower_better_series(values: pd.Series) -> pd.Series:
    """Scale finite lower-is-better values to candidate-relative 0-1 scores."""
    numeric = pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    score = pd.Series(np.nan, index=numeric.index, dtype=float)
    valid = numeric.dropna()
    if valid.empty:
        return score

    min_val = float(valid.min())
    max_val = float(valid.max())
    if max_val - min_val <= np.finfo(float).eps:
        score.loc[valid.index] = 1.0
        return score

    score.loc[valid.index] = 1.0 - (valid - min_val) / (max_val - min_val)
    return score.clip(lower=0.0, upper=1.0)


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


def calc_sample_structure_arrays(
    raw_obj: pd.DataFrame,
    transformed_obj: pd.DataFrame,
    sample_cols: pd.Index | None = None,
    max_features: int | None = 5000,
    seed: int = 123,
) -> dict[str, dict[str, object]]:
    """
    Calculate robust actual-sample geometry arrays before and after processing.
    """
    empty = {
        "geometry": {
            "raw_dist": np.array([], dtype=float),
            "norm_dist": np.array([], dtype=float),
            "sample_log2_distance_ratio": pd.Series(dtype=float),
            "sample_distance_rank_rho": pd.Series(dtype=float),
            "sample_neighborhood_trustworthiness": pd.Series(dtype=float),
            "rank_loss": float("nan"),
            "median_relative_delta": float("nan"),
            "median_sample_log2_distance_ratio": float("nan"),
            "neighborhood_trustworthiness": float("nan"),
            "n_neighbors": float("nan"),
        },
    }

    log_raw = _extract_log2_target(raw_obj)
    log_transformed = _extract_log2_target(transformed_obj)
    if log_raw is None or log_transformed is None:
        return empty

    if sample_cols is None:
        try:
            sample_cols = (
                raw_obj._actual_sample.columns.intersection(log_raw.columns)
                .intersection(log_transformed.columns)
                .sort_values()
            )
        except AttributeError:
            sample_cols = log_raw.columns.intersection(log_transformed.columns)
    else:
        sample_cols = (
            pd.Index(sample_cols)
            .intersection(log_raw.columns)
            .intersection(log_transformed.columns)
        )

    if len(sample_cols) < 3:
        return empty

    raw_sample = log_raw[sample_cols].astype(float)
    transformed_sample = log_transformed[sample_cols].astype(float)
    valid_features = raw_sample.index.intersection(transformed_sample.index)
    raw_sample = raw_sample.loc[valid_features]
    transformed_sample = transformed_sample.loc[valid_features]

    finite_rows = np.isfinite(raw_sample.to_numpy()).any(axis=1) & np.isfinite(
        transformed_sample.to_numpy()
    ).any(axis=1)
    raw_sample = raw_sample.loc[finite_rows]
    transformed_sample = transformed_sample.loc[finite_rows]

    if raw_sample.empty:
        return empty

    if max_features is not None and raw_sample.shape[0] > max_features:
        rng = np.random.default_rng(seed)
        keep_idx = rng.choice(
            raw_sample.index, size=max_features, replace=False
        )
        raw_sample = raw_sample.loc[keep_idx]
        transformed_sample = transformed_sample.loc[keep_idx]

    raw_z = robust_feature_zscore(raw_sample)
    transformed_z = robust_feature_zscore(transformed_sample)
    valid_z_features = raw_z.index.intersection(transformed_z.index)
    raw_z = raw_z.loc[valid_z_features]
    transformed_z = transformed_z.loc[valid_z_features]

    if raw_z.shape[0] >= 2:
        scale = np.sqrt(float(raw_z.shape[0]))
        raw_geom_dist = pdist(raw_z.T.to_numpy(dtype=float), metric="euclidean")
        transformed_geom_dist = pdist(
            transformed_z.T.to_numpy(dtype=float), metric="euclidean"
        )
        raw_geom_dist = raw_geom_dist / scale
        transformed_geom_dist = transformed_geom_dist / scale
        raw_geom_dist_full = raw_geom_dist.copy()
        transformed_geom_dist_full = transformed_geom_dist.copy()

        valid_geom = np.isfinite(raw_geom_dist) & np.isfinite(
            transformed_geom_dist
        )
        raw_geom_dist = raw_geom_dist[valid_geom]
        transformed_geom_dist = transformed_geom_dist[valid_geom]
        if raw_geom_dist.size > 0:
            denominator = np.maximum(np.abs(raw_geom_dist), np.finfo(float).eps)
            relative_delta = (
                np.abs(transformed_geom_dist - raw_geom_dist) / denominator
            )
            raw_square = squareform(raw_geom_dist_full)
            transformed_square = squareform(transformed_geom_dist_full)
            ratio_square = np.log2(
                (transformed_square + np.finfo(float).eps)
                / (raw_square + np.finfo(float).eps)
            )
            np.fill_diagonal(ratio_square, np.nan)
            sample_distance_shift = pd.Series(
                np.nanmedian(ratio_square, axis=1),
                index=raw_z.columns,
                dtype=float,
            ).replace([np.inf, -np.inf], np.nan)
            sample_distance_shift = sample_distance_shift.dropna()

            sample_rank_rho = []
            for sample_idx in range(raw_square.shape[0]):
                other_samples = np.arange(raw_square.shape[0]) != sample_idx
                try:
                    rho, _ = spearmanr(
                        raw_square[sample_idx, other_samples],
                        transformed_square[sample_idx, other_samples],
                    )
                except (ValueError, FloatingPointError):
                    rho = float("nan")
                sample_rank_rho.append(finite_or_nan(rho))
            sample_rank_rho = pd.Series(
                sample_rank_rho,
                index=raw_z.columns,
                dtype=float,
            )
            empty["geometry"] = {
                "raw_dist": raw_geom_dist,
                "norm_dist": transformed_geom_dist,
                "sample_log2_distance_ratio": sample_distance_shift,
                "sample_distance_rank_rho": sample_rank_rho,
                "sample_neighborhood_trustworthiness": pd.Series(dtype=float),
                "rank_loss": rank_loss_from_distances(
                    raw_geom_dist,
                    transformed_geom_dist,
                ),
                "median_relative_delta": float(np.median(relative_delta)),
                "median_sample_log2_distance_ratio": finite_or_nan(
                    sample_distance_shift.median()
                ),
                "neighborhood_trustworthiness": float("nan"),
                "n_neighbors": float("nan"),
            }

        n_samples = raw_z.shape[1]
        if n_samples >= 3:
            n_neighbors = max(1, min(5, (n_samples - 1) // 2))
            try:
                trust_value = trustworthiness(
                    X=raw_z.T.to_numpy(dtype=float),
                    X_embedded=transformed_z.T.to_numpy(dtype=float),
                    n_neighbors=n_neighbors,
                    metric="euclidean",
                )
            except (ValueError, FloatingPointError):
                trust_value = float("nan")

            empty["geometry"]["neighborhood_trustworthiness"] = finite_or_nan(
                trust_value
            )
            empty["geometry"]["n_neighbors"] = float(n_neighbors)

            raw_order = np.argsort(raw_square, axis=1, kind="stable")
            transformed_order = np.argsort(
                transformed_square,
                axis=1,
                kind="stable",
            )
            local_trustworthiness = []
            denominator = n_neighbors * (2 * n_samples - 3 * n_neighbors - 1)
            for sample_idx in range(n_samples):
                raw_neighbors = raw_order[sample_idx]
                raw_neighbors = raw_neighbors[raw_neighbors != sample_idx][
                    :n_neighbors
                ]
                transformed_neighbors = transformed_order[sample_idx]
                transformed_neighbors = transformed_neighbors[
                    transformed_neighbors != sample_idx
                ][:n_neighbors]
                ranks = np.empty(n_samples, dtype=int)
                ranks[raw_order[sample_idx]] = np.arange(n_samples)
                intruders = np.setdiff1d(
                    transformed_neighbors,
                    raw_neighbors,
                    assume_unique=False,
                )
                penalty = np.maximum(ranks[intruders] - n_neighbors, 0).sum()
                local_score = 1.0 - (2.0 * penalty / denominator)
                local_trustworthiness.append(
                    float(np.clip(local_score, 0.0, 1.0))
                )
            empty["geometry"]["sample_neighborhood_trustworthiness"] = (
                pd.Series(
                    local_trustworthiness,
                    index=raw_z.columns,
                    dtype=float,
                )
            )

    return empty


def calc_sample_structure_preservation(
    raw_obj: pd.DataFrame,
    transformed_obj: pd.DataFrame,
    sample_cols: pd.Index | None = None,
    max_features: int | None = 5000,
    seed: int = 123,
    scale_log_ratio_tol: float = 0.25,
    scale_rel_delta_tol: float = 0.35,
) -> dict[str, float]:
    """Calculate label-free actual-sample structure preservation metrics."""
    metrics = {
        "robust_distance_rank_loss": float("nan"),
        "robust_distance_relative_delta": float("nan"),
        "median_sample_log2_distance_ratio": float("nan"),
        "sample_structure_trustworthiness": float("nan"),
        "sample_structure_rank_preservation": float("nan"),
        "sample_structure_scale_shift_preservation": float("nan"),
        "sample_structure_scale_delta_preservation": float("nan"),
        "sample_structure_scale_preservation": float("nan"),
        "sample_structure_composite_preservation": float("nan"),
    }

    structure = calc_sample_structure_arrays(
        raw_obj=raw_obj,
        transformed_obj=transformed_obj,
        sample_cols=sample_cols,
        max_features=max_features,
        seed=seed,
    )
    geom_metrics = structure["geometry"]

    metrics["robust_distance_rank_loss"] = finite_or_nan(
        geom_metrics.get("rank_loss")
    )
    metrics["robust_distance_relative_delta"] = finite_or_nan(
        geom_metrics.get("median_relative_delta")
    )
    metrics["median_sample_log2_distance_ratio"] = finite_or_nan(
        geom_metrics.get("median_sample_log2_distance_ratio")
    )
    metrics["sample_structure_trustworthiness"] = finite_or_nan(
        geom_metrics.get("neighborhood_trustworthiness")
    )

    rank_loss = finite_or_nan(metrics["robust_distance_rank_loss"])
    if np.isfinite(rank_loss):
        metrics["sample_structure_rank_preservation"] = float(
            np.clip(1.0 - rank_loss, 0.0, 1.0)
        )

    median_log2_ratio = finite_or_nan(
        metrics["median_sample_log2_distance_ratio"]
    )
    if np.isfinite(median_log2_ratio):
        metrics["sample_structure_scale_shift_preservation"] = float(
            np.exp(-abs(median_log2_ratio) / scale_log_ratio_tol)
        )

    median_relative_delta = finite_or_nan(
        metrics["robust_distance_relative_delta"]
    )
    if np.isfinite(median_relative_delta):
        metrics["sample_structure_scale_delta_preservation"] = float(
            np.exp(-median_relative_delta / scale_rel_delta_tol)
        )

    metrics["sample_structure_scale_preservation"] = weighted_mean_score(
        [
            (metrics["sample_structure_scale_shift_preservation"], 1.0),
            (metrics["sample_structure_scale_delta_preservation"], 1.0),
        ],
    )
    metrics["sample_structure_composite_preservation"] = weighted_mean_score(
        [
            (metrics["sample_structure_trustworthiness"], 0.50),
            (metrics["sample_structure_rank_preservation"], 0.25),
            (metrics["sample_structure_scale_preservation"], 0.25),
        ],
    )

    return metrics


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


# ====================================================================
# Extract target data for imputation and normalization
# ====================================================================
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
