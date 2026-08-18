"""Sample-level geometry and structure-preservation metrics.

These label-free diagnostics compare actual study-sample geometry before and
after a processing stage. They are shared by normalization, imputation, and
correction without depending on any stage-specific implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from ..constants import DEFAULT_RANDOM_SEED
from .metrics import (
    _extract_log2_target,
    finite_or_nan,
    rank_loss_from_distances,
    robust_feature_zscore,
    weighted_mean_score,
)


def calc_sample_structure_arrays(
    raw_obj: pd.DataFrame,
    transformed_obj: pd.DataFrame,
    sample_cols: pd.Index | None = None,
    max_features: int | None = 5000,
    seed: int = DEFAULT_RANDOM_SEED,
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
            # The global trustworthiness is the mean of the per-sample terms
            # above. Computing both from the already available distance ranks
            # avoids a duplicate nearest-neighbor pass and platform-dependent
            # physical-core probing in scikit-learn/joblib.
            trust_value = finite_or_nan(np.mean(local_trustworthiness))
            empty["geometry"]["neighborhood_trustworthiness"] = trust_value
            empty["geometry"]["n_neighbors"] = float(n_neighbors)
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
    seed: int = DEFAULT_RANDOM_SEED,
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
