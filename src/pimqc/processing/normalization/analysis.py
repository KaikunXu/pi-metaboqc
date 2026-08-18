"""Normalization transformations, candidate evaluation, and selection.

MetaboIntNormalizer applies robust-log, TIC, median, PQN, MDFC, quantile, or
VSN transformations and evaluates AUTO candidates with QC RLE, variance,
structure, and sample-preservation metrics. It exports the selected normalized
matrix and its traceable candidate metrics for downstream reporting.
"""

import os
from functools import cached_property
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
from joblib import Parallel, delayed
from loguru import logger
from numba import njit
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

from ...config import resolve_stage_config
from ...constants import DEFAULT_RANDOM_SEED
from ...core import model
from ...runtime import joblib_execution_context
from ...statistics import metrics as su
from ...statistics import sample_structure as structure_stats
from ...statistics import selection as selection_utils
from ..stage import StageResult
from .methods import NORMALIZATION_METHODS
from .runner import NormalizationStageRunner


# =============================================================================
# Private Numba JIT Engines for Normalization
# =============================================================================
@njit(fastmath=True)
def _numba_vsn_nll(params: np.ndarray, fit_data: np.ndarray) -> float:
    """Compute negative log-likelihood for VSN compiled via Numba."""
    rows, cols = fit_data.shape
    a_vec = params[:-1]
    b = params[-1]

    ll_jacobian_sum = 0.0
    total_residuals_sq = 0.0
    total_valid = 0

    for i in range(rows):
        row_sum = 0.0
        row_valid = 0

        # Pass 1: Compute transformed values and accumulate for mean & Jacobian
        for j in range(cols):
            val = fit_data[i, j]
            if not np.isnan(val):
                z = a_vec[j] + b * val
                t_val = np.arcsinh(z)
                row_sum += t_val
                row_valid += 1
                ll_jacobian_sum += np.log(b) - 0.5 * np.log1p(z**2)

        if row_valid > 0:
            row_mean = row_sum / row_valid

            # Pass 2: Compute squared residuals
            for j in range(cols):
                val = fit_data[i, j]
                if not np.isnan(val):
                    z = a_vec[j] + b * val
                    t_val = np.arcsinh(z)
                    total_residuals_sq += (t_val - row_mean) ** 2
            total_valid += row_valid

    if total_valid == 0:
        return 1e10

    sigma_sq = total_residuals_sq / total_valid
    if sigma_sq <= 1e-16:
        return 1e10

    ll = ll_jacobian_sum - (total_valid / 2.0) * np.log(sigma_sq)
    return -ll


# =============================================================================
# Normalization Processor
# =============================================================================
class MetaboIntNormalizer(model.MetaboInt):
    """Normalization engine for global sample-wise preprocessing."""

    _metadata = ["attrs", "stats"]
    _RUNTIME_CONFIG_KEYS = frozenset({"norm_method", "global_seed", "n_jobs"})
    _AUTO_CANDIDATES = (
        "ROBUST_LOG_ONLY",
        "TIC",
        "MEDIAN",
        "PQN",
        "MDFC",
        "QUANTILE",
        "VSN",
    )
    _DEFAULT_EXTERNAL_LOG = {
        "ROBUST_LOG_ONLY": True,
        "TIC": True,
        "MEDIAN": True,
        "PQN": True,
        "MDFC": True,
        "QUANTILE": True,
        "VSN": False,
    }
    _AUTO_SCORE_COMPONENT_WEIGHTS = {
        "rle_alignment_change_score": 0.30,
        "variance_stabilization_score": 0.20,
        "qc_structure_change_score": 0.20,
        "sample_structure_score": 0.30,
    }
    _AUTO_CENTERED_CHANGE_COMPONENTS = {
        "rle_alignment_change_score",
        "variance_stabilization_score",
        "qc_structure_change_score",
    }
    _AUTO_STRUCTURE_SCORE_COL = "sample_structure_score"
    _SAMPLE_SCALE_LOG_RATIO_TOL = 0.25
    _SAMPLE_SCALE_REL_DELTA_TOL = 0.25
    _AUTO_BASELINE_METHOD = "ROBUST_LOG_ONLY"
    _AUTO_CONSERVATIVE_ORDER = {
        "ROBUST_LOG_ONLY": 0,
        "MEDIAN": 1,
        "TIC": 2,
        "PQN": 3,
        "MDFC": 4,
        "VSN": 5,
        "QUANTILE": 6,
    }

    def __init__(
        self,
        *args: object,
        pipeline_params: Optional[Dict[str, Any]] = None,
        norm_method: Optional[str] = None,
        **kwargs: object,
    ) -> None:
        """Initialize MetaboIntNormalizer with parameters and metadata.

        Args:
            *args: Variable length arguments passed to DataFrame.
            pipeline_params: Global configuration dictionary from TOML.
            norm_method: Normalization method (e.g., 'Auto', 'VSN', 'Quantile').
            **kwargs: Keyword arguments passed to DataFrame constructor.
        """
        super().__init__(*args, pipeline_params=pipeline_params, **kwargs)

        norm_configs = resolve_stage_config(
            pipeline_params,
            "MetaboIntNormalizer",
            {"norm_method": "Auto"},
            {"norm_method": norm_method},
        )
        self.attrs.update(norm_configs)

    @property
    def _constructor(self) -> type["MetaboIntNormalizer"]:
        """Override pandas constructor to return current subclass type."""
        return MetaboIntNormalizer

    # =========================================================================
    # Lightweight Auto-normalization Metrics
    # =========================================================================
    @classmethod
    def _uses_external_log_for_method(cls, method: str) -> bool:
        """Return the fixed log-transform policy for a normalization method."""
        method_upper = NORMALIZATION_METHODS.canonicalize(
            method or "ROBUST_LOG_ONLY", strict=False
        )
        return cls._DEFAULT_EXTERNAL_LOG.get(method_upper, True)

    @classmethod
    def _calc_qc_rle_values(
        cls,
        log_df: pd.DataFrame,
        qc_cols: pd.Index,
    ) -> dict[str, float]:
        """Calculate QC-relative log expression center and spread metrics."""
        metrics = {
            "rle_center_offset": float("nan"),
            "rle_spread": float("nan"),
        }
        if len(qc_cols) < 2:
            return metrics

        qc_log = log_df[qc_cols].astype(float)
        global_feature_median = log_df.median(axis=1)
        qc_rle = qc_log.sub(global_feature_median, axis=0)
        qc_rle = qc_rle.replace([np.inf, -np.inf], np.nan)

        qc_rle_medians = qc_rle.median(axis=0).replace(
            [np.inf, -np.inf], np.nan
        )
        qc_rle_iqrs = qc_rle.quantile(0.75, axis=0) - qc_rle.quantile(
            0.25, axis=0
        )
        qc_rle_medians = qc_rle_medians.dropna()
        qc_rle_iqrs = qc_rle_iqrs.replace([np.inf, -np.inf], np.nan).dropna()

        if not qc_rle_medians.empty:
            metrics["rle_center_offset"] = su.finite_or_nan(
                qc_rle_medians.abs().median()
            )
        if not qc_rle_iqrs.empty:
            metrics["rle_spread"] = su.finite_or_nan(qc_rle_iqrs.median())
        return metrics

    @classmethod
    def _calc_qc_structure_values(
        cls,
        log_df: pd.DataFrame,
        qc_cols: pd.Index,
        max_features: Optional[int] = 5000,
        seed: int = DEFAULT_RANDOM_SEED,
    ) -> dict[str, Any]:
        """
        Calculate multivariate QC compactness around the pooled-QC centroid.
        """
        metrics: dict[str, Any] = {
            "qc_centroid_distance": pd.Series(dtype=float),
            "qc_centroid_distance_median": float("nan"),
            "qc_centroid_distance_iqr": float("nan"),
            "qc_pairwise_distance": pd.Series(dtype=float),
            "qc_pairwise_distance_median": float("nan"),
        }
        if len(qc_cols) < 3:
            return metrics

        data_df = log_df.replace([np.inf, -np.inf], np.nan).astype(float)
        finite_rows = np.isfinite(data_df.to_numpy()).any(axis=1)
        data_df = data_df.loc[finite_rows]
        if data_df.empty:
            return metrics

        if max_features is not None and data_df.shape[0] > max_features:
            rng = np.random.default_rng(seed)
            keep_idx = rng.choice(
                data_df.index, size=max_features, replace=False
            )
            data_df = data_df.loc[keep_idx]

        z_df = su.robust_feature_zscore(data_df)
        qc_z = z_df[qc_cols.intersection(z_df.columns)]
        if qc_z.shape[1] < 3 or qc_z.shape[0] < 2:
            return metrics

        qc_centroid = qc_z.median(axis=1, skipna=True)
        qc_residual = qc_z.sub(qc_centroid, axis=0).to_numpy(
            dtype=float, copy=True
        )
        qc_residual[~np.isfinite(qc_residual)] = np.nan
        scale = np.sqrt(float(qc_z.shape[0]))
        centroid_distance = np.sqrt(np.nanmean(np.square(qc_residual), axis=0))
        centroid_distance = centroid_distance / max(scale, np.finfo(float).eps)
        distance_series = pd.Series(
            centroid_distance,
            index=qc_z.columns,
            dtype=float,
        ).replace([np.inf, -np.inf], np.nan)
        distance_series = distance_series.dropna()

        if not distance_series.empty:
            metrics["qc_centroid_distance"] = distance_series
            metrics["qc_centroid_distance_median"] = su.finite_or_nan(
                distance_series.median()
            )
            metrics["qc_centroid_distance_iqr"] = su.series_iqr(distance_series)

        pairwise_dist = pdist(qc_z.T.to_numpy(dtype=float), metric="euclidean")
        pairwise_dist = pairwise_dist / max(scale, np.finfo(float).eps)
        pairwise_dist = pairwise_dist[np.isfinite(pairwise_dist)]
        if pairwise_dist.size > 0:
            metrics["qc_pairwise_distance"] = pd.Series(
                pairwise_dist,
                dtype=float,
            )
            metrics["qc_pairwise_distance_median"] = su.finite_or_nan(
                np.nanmedian(pairwise_dist)
            )
        return metrics

    @classmethod
    def _calc_qc_mean_dispersion_table(
        cls,
        log_df: pd.DataFrame,
        qc_cols: pd.Index,
    ) -> pd.DataFrame:
        """Calculate feature-wise QC mean intensity and robust dispersion."""
        qc_log = log_df[qc_cols.intersection(log_df.columns)].astype(float)
        if qc_log.shape[1] < 3:
            return pd.DataFrame(columns=["mean_intensity", "qc_dispersion"])

        feature_mean = qc_log.mean(axis=1, skipna=True)
        feature_center = qc_log.median(axis=1, skipna=True)
        feature_mad = qc_log.sub(feature_center, axis=0).abs().median(axis=1)
        feature_dispersion = feature_mad * 1.4826
        stats_df = pd.DataFrame(
            {
                "mean_intensity": feature_mean,
                "qc_dispersion": feature_dispersion,
            }
        ).replace([np.inf, -np.inf], np.nan)
        stats_df = stats_df.dropna()
        stats_df = stats_df[stats_df["qc_dispersion"] >= 0]
        return stats_df

    @classmethod
    def _calc_qc_variance_trend(
        cls,
        stats_df: pd.DataFrame,
        n_bins: int = 12,
        min_bin_size: int = 10,
    ) -> pd.DataFrame:
        """Bin QC features by mean intensity and summarize dispersion trend."""
        if stats_df.empty:
            return pd.DataFrame(
                columns=[
                    "mean_intensity",
                    "dispersion_median",
                    "dispersion_q25",
                    "dispersion_q75",
                    "n_features",
                ]
            )

        clean_df = stats_df.replace([np.inf, -np.inf], np.nan).dropna()
        if clean_df.shape[0] < max(3 * min_bin_size, 30):
            return pd.DataFrame()

        bin_count = min(n_bins, max(3, int(clean_df.shape[0] // min_bin_size)))
        if bin_count < 3:
            return pd.DataFrame()

        ordered_index = clean_df["mean_intensity"].sort_values().index
        records = []
        for bin_index in np.array_split(ordered_index.to_numpy(), bin_count):
            bin_df = clean_df.loc[bin_index]
            if bin_df.shape[0] < 3:
                continue
            records.append(
                {
                    "mean_intensity": su.finite_or_nan(
                        bin_df["mean_intensity"].median()
                    ),
                    "dispersion_median": su.finite_or_nan(
                        bin_df["qc_dispersion"].median()
                    ),
                    "dispersion_q25": su.finite_or_nan(
                        bin_df["qc_dispersion"].quantile(0.25)
                    ),
                    "dispersion_q75": su.finite_or_nan(
                        bin_df["qc_dispersion"].quantile(0.75)
                    ),
                    "n_features": int(bin_df.shape[0]),
                }
            )
        return pd.DataFrame(records)

    @classmethod
    def _calc_qc_variance_stabilization_values(
        cls,
        log_df: pd.DataFrame,
        qc_cols: pd.Index,
    ) -> dict[str, Any]:
        """
        Calculate QC mean-dispersion dependence for variance stabilization.
        """
        metrics: dict[str, Any] = {
            "feature_stats": pd.DataFrame(),
            "trend": pd.DataFrame(),
            "qc_dispersion_median": float("nan"),
            "mean_variance_abs_rho": float("nan"),
            "mean_variance_abs_slope": float("nan"),
        }

        stats_df = cls._calc_qc_mean_dispersion_table(
            log_df=log_df, qc_cols=qc_cols
        )
        if stats_df.shape[0] < 3:
            return metrics

        metrics["feature_stats"] = stats_df
        metrics["trend"] = cls._calc_qc_variance_trend(stats_df)
        metrics["qc_dispersion_median"] = su.finite_or_nan(
            stats_df["qc_dispersion"].median()
        )

        rho_val = stats.spearmanr(
            stats_df["mean_intensity"].to_numpy(dtype=float),
            stats_df["qc_dispersion"].to_numpy(dtype=float),
        )[0]
        metrics["mean_variance_abs_rho"] = abs(su.finite_or_nan(rho_val))

        slope_df = stats_df[stats_df["qc_dispersion"] > 0].copy()
        if slope_df.shape[0] >= 3:
            x_vals = slope_df["mean_intensity"].to_numpy(dtype=float)
            y_vals = np.log2(slope_df["qc_dispersion"].to_numpy(dtype=float))
            finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
            if int(finite_mask.sum()) >= 3:
                try:
                    slope_val = stats.theilslopes(
                        y_vals[finite_mask],
                        x_vals[finite_mask],
                    )[0]
                except (ValueError, FloatingPointError):
                    slope_val = float("nan")
                metrics["mean_variance_abs_slope"] = abs(
                    su.finite_or_nan(slope_val)
                )

        return metrics

    def _sample_structure_preservation_metrics(
        self,
        norm_obj: model.MetaboInt,
        max_features: Optional[int] = 5000,
    ) -> dict[str, float]:
        """Quantify local sample structure preservation without labels."""
        return structure_stats.calc_sample_structure_preservation(
            raw_obj=self,
            transformed_obj=norm_obj,
            max_features=max_features,
            seed=int(self.attrs.get("global_seed", DEFAULT_RANDOM_SEED)),
            scale_log_ratio_tol=self._SAMPLE_SCALE_LOG_RATIO_TOL,
            scale_rel_delta_tol=self._SAMPLE_SCALE_REL_DELTA_TOL,
        )

    def calc_auto_norm_candidate_metrics(
        self,
        norm_obj: model.MetaboInt,
    ) -> dict[str, float]:
        """Calculate component scores for Auto normalization."""
        metrics = {
            "rle_center_offset_before": float("nan"),
            "rle_center_offset_after": float("nan"),
            "rle_spread_before": float("nan"),
            "rle_spread_after": float("nan"),
            "qc_dispersion_median_before": float("nan"),
            "qc_dispersion_median_after": float("nan"),
            "mean_variance_abs_rho_before": float("nan"),
            "mean_variance_abs_rho_after": float("nan"),
            "mean_variance_abs_slope_before": float("nan"),
            "mean_variance_abs_slope_after": float("nan"),
            "qc_structure_distance_before": float("nan"),
            "qc_structure_distance_after": float("nan"),
            "qc_pairwise_distance_before": float("nan"),
            "qc_pairwise_distance_after": float("nan"),
            "robust_distance_rank_loss": float("nan"),
            "robust_distance_relative_delta": float("nan"),
            "median_sample_log2_distance_ratio": float("nan"),
            "sample_structure_trustworthiness": float("nan"),
            "sample_structure_rank_preservation": float("nan"),
            "sample_structure_scale_shift_preservation": float("nan"),
            "sample_structure_scale_delta_preservation": float("nan"),
            "sample_structure_scale_preservation": float("nan"),
            "sample_structure_composite_preservation": float("nan"),
            "rle_alignment_change_score": float("nan"),
            "variance_stabilization_score": float("nan"),
            "qc_structure_change_score": float("nan"),
            self._AUTO_STRUCTURE_SCORE_COL: float("nan"),
        }

        # Candidate outputs retain their delivered scale for export, while
        # AUTO scoring compares all candidates through one log-like view.
        log_raw = su._extract_log2_target(self)
        log_norm = su._extract_log2_target(norm_obj)
        if (
            log_raw is None
            or log_raw.empty
            or log_norm is None
            or log_norm.empty
        ):
            return metrics

        log_raw, log_norm = su.align_paired_matrices(log_raw, log_norm)
        if log_raw.empty or log_norm.empty:
            return metrics

        qc_cols = (
            self._qc.columns.intersection(norm_obj._qc.columns)
            .intersection(log_raw.columns)
            .intersection(log_norm.columns)
        )
        if len(qc_cols) >= 2:
            rle_before = self._calc_qc_rle_values(log_raw, qc_cols)
            rle_after = self._calc_qc_rle_values(log_norm, qc_cols)
            metrics["rle_center_offset_before"] = rle_before[
                "rle_center_offset"
            ]
            metrics["rle_center_offset_after"] = rle_after["rle_center_offset"]
            metrics["rle_spread_before"] = rle_before["rle_spread"]
            metrics["rle_spread_after"] = rle_after["rle_spread"]

            metrics["rle_alignment_change_score"] = su.weighted_mean_score(
                [
                    (
                        su.practical_signed_change_lower_better(
                            rle_before["rle_center_offset"],
                            rle_after["rle_center_offset"],
                            min_rel_change=0.01,
                        ),
                        3.0,
                    ),
                    (
                        su.practical_signed_change_lower_better(
                            rle_before["rle_spread"],
                            rle_after["rle_spread"],
                            min_rel_change=0.01,
                        ),
                        2.0,
                    ),
                ],
                clip_values=False,
            )

            qc_log_raw = log_raw[qc_cols].astype(float)
            qc_log_norm = log_norm[qc_cols].astype(float)
            var_before = self._calc_qc_variance_stabilization_values(
                qc_log_raw,
                qc_cols=qc_cols,
            )
            var_after = self._calc_qc_variance_stabilization_values(
                qc_log_norm,
                qc_cols=qc_cols,
            )
            metrics["qc_dispersion_median_before"] = var_before[
                "qc_dispersion_median"
            ]
            metrics["qc_dispersion_median_after"] = var_after[
                "qc_dispersion_median"
            ]
            metrics["mean_variance_abs_rho_before"] = var_before[
                "mean_variance_abs_rho"
            ]
            metrics["mean_variance_abs_rho_after"] = var_after[
                "mean_variance_abs_rho"
            ]
            metrics["mean_variance_abs_slope_before"] = var_before[
                "mean_variance_abs_slope"
            ]
            metrics["mean_variance_abs_slope_after"] = var_after[
                "mean_variance_abs_slope"
            ]
            metrics["variance_stabilization_score"] = su.weighted_mean_score(
                [
                    (
                        su.practical_signed_change_lower_better(
                            var_before["mean_variance_abs_rho"],
                            var_after["mean_variance_abs_rho"],
                            min_abs_change=0.01,
                            min_rel_change=0.02,
                        ),
                        4.0,
                    ),
                    (
                        su.practical_signed_change_lower_better(
                            var_before["mean_variance_abs_slope"],
                            var_after["mean_variance_abs_slope"],
                            min_abs_change=0.005,
                            min_rel_change=0.02,
                        ),
                        4.0,
                    ),
                    (
                        su.practical_signed_change_lower_better(
                            var_before["qc_dispersion_median"],
                            var_after["qc_dispersion_median"],
                            min_abs_change=0.001,
                            min_rel_change=0.02,
                        ),
                        2.0,
                    ),
                ],
                clip_values=False,
            )

        if len(qc_cols) >= 3:
            qc_structure_before = self._calc_qc_structure_values(
                log_raw,
                qc_cols=qc_cols,
                max_features=5000,
                seed=int(
                    self.attrs.get("global_seed", DEFAULT_RANDOM_SEED)
                ),
            )
            qc_structure_after = self._calc_qc_structure_values(
                log_norm,
                qc_cols=qc_cols,
                max_features=5000,
                seed=int(
                    self.attrs.get("global_seed", DEFAULT_RANDOM_SEED)
                ),
            )
            metrics["qc_structure_distance_before"] = qc_structure_before[
                "qc_centroid_distance_median"
            ]
            metrics["qc_structure_distance_after"] = qc_structure_after[
                "qc_centroid_distance_median"
            ]
            metrics["qc_pairwise_distance_before"] = qc_structure_before[
                "qc_pairwise_distance_median"
            ]
            metrics["qc_pairwise_distance_after"] = qc_structure_after[
                "qc_pairwise_distance_median"
            ]
            metrics["qc_structure_change_score"] = su.weighted_mean_score(
                [
                    (
                        su.practical_signed_change_lower_better(
                            qc_structure_before["qc_centroid_distance_median"],
                            qc_structure_after["qc_centroid_distance_median"],
                            min_abs_change=0.005,
                            min_rel_change=0.02,
                        ),
                        2.0,
                    ),
                    (
                        su.practical_signed_change_lower_better(
                            qc_structure_before["qc_pairwise_distance_median"],
                            qc_structure_after["qc_pairwise_distance_median"],
                            min_abs_change=0.005,
                            min_rel_change=0.02,
                        ),
                        1.0,
                    ),
                ],
                clip_values=False,
            )

        metrics.update(self._sample_structure_preservation_metrics(norm_obj))
        sample_structure_preservation = su.finite_or_nan(
            metrics["sample_structure_composite_preservation"]
        )
        if np.isfinite(sample_structure_preservation):
            metrics[self._AUTO_STRUCTURE_SCORE_COL] = float(
                0.5 * np.clip(sample_structure_preservation, 0.0, 1.0)
            )
        return metrics

    @classmethod
    def _score_normalization_candidates(
        cls, records: list[dict[str, object]]
    ) -> pd.DataFrame:
        """Score candidates using weighted Auto-normalization components."""
        score_df = pd.DataFrame(records)
        if score_df.empty:
            raise ValueError("No normalization candidates were evaluated.")

        score_df["selected"] = False
        score_df["auto_score"] = np.nan
        score_df["available_metric_weight"] = 0.0

        ok_mask = score_df["status"].eq("ok")

        overall_weighted_sum = pd.Series(0.0, index=score_df.index)
        overall_weight_sum = pd.Series(0.0, index=score_df.index)

        for (
            score_col,
            score_weight,
        ) in cls._AUTO_SCORE_COMPONENT_WEIGHTS.items():
            score_values = pd.to_numeric(
                score_df.get(score_col), errors="coerce"
            )
            valid_score_mask = ok_mask & np.isfinite(score_values)

            if score_col in cls._AUTO_CENTERED_CHANGE_COMPONENTS:
                baseline_mask = ok_mask & score_df["method"].eq(
                    cls._AUTO_BASELINE_METHOD
                )
                baseline_values = score_values.loc[baseline_mask].dropna()
                baseline_value = (
                    float(baseline_values.iloc[0])
                    if not baseline_values.empty
                    else 0.0
                )

                raw_change = score_values - baseline_value
                centered_scores = pd.Series(
                    np.nan, index=score_df.index, dtype=float
                )

                valid_changes = raw_change.loc[valid_score_mask]
                positive_max = (
                    float(valid_changes[valid_changes > 0].max())
                    if (valid_changes > 0).any()
                    else 0.0
                )
                negative_max = (
                    float(abs(valid_changes[valid_changes < 0].min()))
                    if (valid_changes < 0).any()
                    else 0.0
                )

                centered_scores.loc[valid_score_mask] = 0.5

                positive_mask = valid_score_mask & (raw_change > 0)
                if positive_max > np.finfo(float).eps:
                    centered_scores.loc[positive_mask] = (
                        0.5 + 0.5 * raw_change.loc[positive_mask] / positive_max
                    )

                negative_mask = valid_score_mask & (raw_change < 0)
                if negative_max > np.finfo(float).eps:
                    centered_scores.loc[negative_mask] = (
                        0.5
                        - 0.5
                        * raw_change.loc[negative_mask].abs()
                        / negative_max
                    )

                clipped_scores = centered_scores.clip(lower=0.0, upper=1.0)
            else:
                clipped_scores = score_values.clip(lower=0.0, upper=1.0)
                if score_col == cls._AUTO_STRUCTURE_SCORE_COL:
                    clipped_scores = score_values.clip(lower=0.0, upper=0.5)
                    baseline_mask = ok_mask & score_df["method"].eq(
                        cls._AUTO_BASELINE_METHOD
                    )
                    clipped_scores.loc[baseline_mask] = 0.5

            if score_col in cls._AUTO_CENTERED_CHANGE_COMPONENTS:
                baseline_mask = ok_mask & score_df["method"].eq(
                    cls._AUTO_BASELINE_METHOD
                )
                clipped_scores.loc[baseline_mask] = 0.5
            valid_mask = ok_mask & np.isfinite(clipped_scores)
            score_df.loc[valid_mask, score_col] = clipped_scores.loc[valid_mask]
            overall_weighted_sum.loc[valid_mask] += (
                clipped_scores.loc[valid_mask] * score_weight
            )
            overall_weight_sum.loc[valid_mask] += score_weight

        scoreable_mask = ok_mask & (overall_weight_sum > 0)
        if not scoreable_mask.any():
            raise ValueError(
                "No Auto normalization candidate produced valid metrics."
            )

        score_df.loc[scoreable_mask, "auto_score"] = (
            overall_weighted_sum.loc[scoreable_mask]
            / overall_weight_sum.loc[scoreable_mask]
        )
        score_df.loc[scoreable_mask, "available_metric_weight"] = (
            overall_weight_sum.loc[scoreable_mask]
        )
        baseline_scoreable_mask = scoreable_mask & score_df["method"].eq(
            cls._AUTO_BASELINE_METHOD
        )
        score_df.loc[baseline_scoreable_mask, "auto_score"] = 0.5

        score_df["_conservative_order"] = score_df["method"].map(
            cls._AUTO_CONSERVATIVE_ORDER
        )
        candidate_rank = selection_utils.rank_candidates(
            score_df,
            score_column="auto_score",
            tie_breakers=(("_conservative_order", True), ("method", True)),
            eligible_mask=scoreable_mask,
        )
        selected_idx = candidate_rank.index[0]

        score_df.loc[selected_idx, "selected"] = True

        margin = selection_utils.selection_margin(candidate_rank["auto_score"])

        score_df["selection_margin"] = margin
        return score_df

    @classmethod
    def _summarize_auto_norm_scores(
        cls, score_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Build the concise user-facing Auto normalization summary table."""
        summary_cols = [
            "method",
            "normalization_applied",
            "log_transform_applied",
            "status",
            "selected",
            "auto_score",
            "rle_alignment_change_score",
            "variance_stabilization_score",
            "qc_structure_change_score",
            cls._AUTO_STRUCTURE_SCORE_COL,
            "sample_structure_trustworthiness",
            "sample_structure_rank_preservation",
            "sample_structure_scale_preservation",
            "selection_margin",
        ]
        summary = score_df.reindex(columns=summary_cols).copy()
        summary = summary.rename(columns={"auto_score": "overall_score"})

        for col in [
            "overall_score",
            "rle_alignment_change_score",
            "variance_stabilization_score",
            "qc_structure_change_score",
            cls._AUTO_STRUCTURE_SCORE_COL,
            "sample_structure_trustworthiness",
            "sample_structure_rank_preservation",
            "sample_structure_scale_preservation",
            "selection_margin",
        ]:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")

        robust_log_matches = summary.loc[
            summary["method"].eq(cls._AUTO_BASELINE_METHOD), "overall_score"
        ]
        robust_log_score = (
            su.finite_or_nan(robust_log_matches.iloc[0])
            if not robust_log_matches.empty
            else np.nan
        )
        summary["delta_vs_robust_log_only"] = (
            summary["overall_score"] - robust_log_score
        )

        ordered_cols = [
            "method",
            "normalization_applied",
            "log_transform_applied",
            "status",
            "selected",
            "overall_score",
            "rle_alignment_change_score",
            "variance_stabilization_score",
            "qc_structure_change_score",
            cls._AUTO_STRUCTURE_SCORE_COL,
            "sample_structure_trustworthiness",
            "sample_structure_rank_preservation",
            "sample_structure_scale_preservation",
            "delta_vs_robust_log_only",
            "selection_margin",
        ]
        return summary[ordered_cols]

    # =========================================================================
    # Mathematical Operators (Sample-wise & Global)
    # =========================================================================
    @staticmethod
    def calc_tic_normalization(df: pd.DataFrame) -> pd.DataFrame:
        """Apply Total Ion Current (TIC) normalization sample-wise."""
        col_sums = df.sum(axis="index").replace(0, 1)
        return df.div(col_sums, axis="columns") * col_sums.median()

    @staticmethod
    def calc_median_normalization(df: pd.DataFrame) -> pd.DataFrame:
        """Apply Median normalization sample-wise."""
        col_medians = df.median(axis="index").replace(0, 1)
        return df.div(col_medians, axis="columns") * col_medians.median()

    @staticmethod
    def calc_pqn_normalization(
        df: pd.DataFrame, qc_cols: pd.Index | None = None
    ) -> pd.DataFrame:
        """
        Apply Probabilistic Quotient Normalization sample-wise.

        Ref:
            Probabilistic Quotient Normalization as Robust Method to Account
            for Dilution of Complex Biological Mixtures. Application in 1H
            NMR Metabonomics (Anna Chem, 2006)
        """
        df_safe = df.replace({0: np.nan})

        if qc_cols is not None and not qc_cols.empty:
            ref_spectrum = df_safe[qc_cols].median(axis="columns")
        else:
            logger.warning("No QCs for PQN. Using global median.")
            ref_spectrum = df_safe.median(axis="columns")

        ref_spectrum = ref_spectrum.replace({0: np.nan})
        quotients = df_safe.div(ref_spectrum, axis="index")
        median_quotients = quotients.median(axis="index")

        return df_safe.div(median_quotients, axis="columns").fillna(0)

    @staticmethod
    def calc_vsn_normalization(
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """
        Apply Variance Stabilizing Normalization (VSN).

        Ref:
            Variance stabilization applied to microarray data calibration and
            to the quantification of differential expression (Bioinformatics,
            2002)
        """
        data_arr = df.to_numpy(dtype=np.float64)
        rows, cols = data_arr.shape

        # Stratified subsampling for fast optimization
        max_features = min(rows, 1000)
        row_medians = np.nanmedian(data_arr, axis=1)
        sorted_idx = np.argsort(row_medians)
        best_indices = sorted_idx[
            np.linspace(0, rows - 1, max_features, dtype=int)
        ]
        fit_data = data_arr[best_indices, :]

        # Optimize via L-BFGS-B
        a_init = np.zeros(cols)
        b_init = 1.0 / np.nanmedian(fit_data)
        x0 = np.concatenate([a_init, [b_init]])
        bounds = [(None, None)] * cols + [(1e-12, None)]

        res = minimize(
            _numba_vsn_nll,
            x0=x0,
            args=(fit_data,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-5},
        )
        a_vec, b = res.x[:-1], res.x[-1]

        # Apply glog transformation
        shift_constant = np.log2(2 * b)
        normed_arr = (
            np.arcsinh(a_vec + b * data_arr) / np.log(2)
        ) - shift_constant

        # Correct for global intensity shift
        log2_data = np.log2(np.where(data_arr > 0, data_arr, np.nan))
        valid = ~np.isnan(log2_data) & ~np.isnan(normed_arr)

        pure_shift = 0.0
        if np.any(valid):
            y_val, x_val = normed_arr[valid], log2_data[valid]
            high_mask = x_val > np.percentile(x_val, 50)
            pure_shift = np.median(x_val[high_mask] - y_val[high_mask])
            normed_arr += pure_shift

        res_df = df.copy()
        res_df.iloc[:, :] = normed_arr

        vsn_meta = {"vsn_scale": float(b), "vsn_shift": float(pure_shift)}
        return res_df, vsn_meta

    @staticmethod
    def calc_quantile_normalization(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Quantile normalization ensuring identically distributed.

        Ref:
            A comparison of normalization methods for high density
            oligonucleotide array data based on variance and bias
            (Bioinformatics, 2003)
        """
        origin_arr = df.to_numpy(dtype=np.float64)
        rows, cols = origin_arr.shape

        sorted_arr = np.sort(origin_arr, axis=0)
        non_nas = rows - np.isnan(sorted_arr).sum(axis=0)

        row_means = np.zeros(rows, dtype=np.float64)
        x_target = np.linspace(0, 1, rows)

        # Build reference distribution
        for j in range(cols):
            non_na = non_nas[j]
            if non_na == 0:
                continue
            y = sorted_arr[:non_na, j]
            x = np.linspace(0, 1, non_na)
            row_means += np.interp(x_target, x, y)

        row_means /= float(cols)

        # Map values to reference
        normed_arr = np.full((rows, cols), np.nan, dtype=np.float64)
        for j in range(cols):
            non_na = non_nas[j]
            if non_na < 2:
                if non_na == 1:
                    valid = ~np.isnan(origin_arr[:, j])
                    normed_arr[valid, j] = row_means[rows // 2]
                continue

            col_data = origin_arr[:, j]
            valid = ~np.isnan(col_data)
            ranks = stats.rankdata(col_data[valid], method="average")
            rank_percentiles = (ranks - 1.0) / (non_na - 1.0)
            interp_vals = np.interp(rank_percentiles, x_target, row_means)
            normed_arr[valid, j] = interp_vals

        res_df = df.copy()
        res_df.iloc[:, :] = normed_arr
        return res_df

    @staticmethod
    def calc_mdfc_normalization(
        df: pd.DataFrame,
        qc_cols: pd.Index,
        kde_points: int = 1000,
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        """Apply Maximal Density Fold Change (MDFC) normalization.

        Filters high-quality features against QC/SQC references, calculates
        the Log2 fold change, and extracts the maximal density peak via KDE.
        Accelerated via chunked joblib multi-processing to maximize CPU
        utilization.

        Ref:
            MAFFIN: metabolomics sample normalization using maximal density
            fold change with high-quality metabolic features and corrected
            signal intensities (Bioinformatics, 2022)
        """
        df_safe = df.replace({0: np.nan})

        # Define high-quality reference spectrum
        if qc_cols is not None and not qc_cols.empty:
            ref_spectrum = df_safe[qc_cols].median(axis="columns")
        else:
            logger.warning("No QCs for MDFC. Using global median fallback.")
            ref_spectrum = df_safe.median(axis="columns")

        # Extract strictly decoupled numpy arrays to prevent memory leak
        log2_ref = np.log2(ref_spectrum.replace({0: np.nan})).values
        data_matrix = df_safe.values
        num_samples = data_matrix.shape[1]

        # Pure function: Process a chunk to avoid IPC overhead
        def _process_mdfc_chunk(
            chunk_arr: np.ndarray, ref_arr: np.ndarray, pts: int
        ) -> np.ndarray:
            """Process a chunk of samples independently."""
            out_chunk = np.empty_like(chunk_arr, dtype=np.float64)
            num_cols = chunk_arr.shape[1]

            for i in range(num_cols):
                sample_vals = chunk_arr[:, i]
                valid = ~np.isnan(sample_vals) & ~np.isnan(ref_arr)

                # Fallback 1: Insufficient overlapping features
                if valid.sum() < 10:
                    out_chunk[:, i] = sample_vals
                    continue

                log_fc = np.log2(sample_vals[valid]) - ref_arr[valid]
                clean_log_fc = log_fc[np.isfinite(log_fc)]

                # Fallback 2: Zero variance or extreme data scarcity
                if len(clean_log_fc) < 10 or np.var(clean_log_fc) < 1e-8:
                    if len(clean_log_fc) > 0:
                        shift = np.median(clean_log_fc)
                    else:
                        shift = 0.0
                else:
                    try:
                        # Kernel Density Estimation
                        kde = stats.gaussian_kde(clean_log_fc)
                        grid = np.linspace(
                            np.min(clean_log_fc), np.max(clean_log_fc), pts
                        )
                        density = kde.evaluate(grid)

                        # Fallback 3: KDE produced flat or invalid density
                        if np.max(density) == 0 or np.isnan(density).any():
                            shift = np.median(clean_log_fc)
                        else:
                            shift = grid[np.argmax(density)]
                    except (ValueError, np.linalg.LinAlgError):
                        # Fallback 4: KDE singular matrix failure
                        shift = np.median(clean_log_fc)

                # Apply back-transformation
                norm_factor = 2**shift
                out_chunk[:, i] = sample_vals / norm_factor

            return out_chunk

        # Calculate safe threading and optimal chunking strategy
        actual_cores = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
        safe_n_jobs = max(1, int(actual_cores / 2))

        # Chunk the data matrix to balance the load across processes
        n_chunks = min(num_samples, safe_n_jobs * 4)
        if n_chunks > 0:
            chunks = np.array_split(data_matrix, n_chunks, axis=1)
        else:
            chunks = []

        logger.info(
            f"Executing chunked parallel MDFC "
            f"(backend='loky', cores={safe_n_jobs}, chunks={n_chunks})..."
        )

        # Execute parallel processing via Joblib
        with joblib_execution_context("loky"):
            normed_chunks = Parallel(n_jobs=safe_n_jobs)(
                delayed(_process_mdfc_chunk)(chunk, log2_ref, kde_points)
                for chunk in chunks
            )

        # Reconstruct the output DataFrame
        res_df = df.copy()
        if normed_chunks:
            res_df.iloc[:, :] = np.column_stack(normed_chunks)

        return res_df.fillna(0)

    # =========================================================================
    # Normalization Method Execution
    # =========================================================================
    def _extract_ordered_target_matrix(self) -> pd.DataFrame:
        """Return QC and biological samples in the original injection order."""
        df_target = pd.concat([self._qc, self._actual_sample], axis=1)
        ordered_cols = self.columns.intersection(df_target.columns)
        df_target = df_target[ordered_cols].copy()

        if df_target.empty:
            raise ValueError("No target samples (QC/Actual) available.")
        return df_target

    def _apply_normalization_candidate(
        self,
        df_target: pd.DataFrame,
        method: str,
        apply_external_log: bool,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Apply one fixed normalization strategy to an ordered target matrix.
        """
        method = NORMALIZATION_METHODS.canonicalize(
            method or "ROBUST_LOG_ONLY", strict=False
        )
        df_norm = df_target.copy()

        meta_stamps: dict[str, Any] = {
            "norm_method": method,
            "is_logged": False,
        }

        # ---------------------------------------------------------------------
        # Category A: Linear Scale Methods and log-only baseline
        # Logic: Normalize first, then robust Log2 for variance stabilization.
        # ---------------------------------------------------------------------
        if method in ["TIC", "MEDIAN", "PQN", "MDFC", "ROBUST_LOG_ONLY"]:
            if method == "TIC":
                df_norm = self.calc_tic_normalization(df_norm)
            elif method == "MEDIAN":
                df_norm = self.calc_median_normalization(df_norm)
            elif method == "PQN":
                qc_cols = self._qc.columns
                df_norm = self.calc_pqn_normalization(df_norm, qc_cols)
            elif method == "MDFC":
                qc_cols = self._qc.columns
                n_cores = self.attrs.get("n_jobs", -1)
                df_norm = self.calc_mdfc_normalization(
                    df_norm, qc_cols=qc_cols, n_jobs=n_cores
                )

            meta_stamps["normalization_applied"] = method != "ROBUST_LOG_ONLY"
            if apply_external_log:
                df_norm = su.robust_log2_transform(df_norm)
                meta_stamps["is_logged"] = True

        # ---------------------------------------------------------------------
        # Category B: Distribution Alignment (Quantile)
        # Logic: Robust Log2 first, then align distributions.
        # ---------------------------------------------------------------------
        elif method == "QUANTILE":
            meta_stamps["normalization_applied"] = True
            if apply_external_log:
                df_norm = su.robust_log2_transform(df_norm)
                meta_stamps["is_logged"] = True
            df_norm = self.calc_quantile_normalization(df_norm)

        # ---------------------------------------------------------------------
        # Category C: Variance Stabilizing Normalization (VSN)
        # Logic: Intrinsic glog; no external robust Log2.
        # ---------------------------------------------------------------------
        elif method == "VSN":
            meta_stamps["normalization_applied"] = True
            df_norm, vsn_meta = self.calc_vsn_normalization(df_norm)
            meta_stamps.update(vsn_meta)
            meta_stamps["is_logged"] = True

        else:
            raise ValueError(
                f"Unsupported normalization method: {method}. "
                "Use ROBUST_LOG_ONLY for the log-only baseline."
            )

        return df_norm, meta_stamps

    def _finalize_normalized_object(
        self,
        df_norm: pd.DataFrame,
        meta_stamps: dict[str, Any],
    ) -> "MetaboIntNormalizer":
        """
        Create a finalized normalized object and clear stale scaling stamps.
        """
        clean_obj = self._constructor(df_norm).__finalize__(self)
        clean_obj.attrs.update(meta_stamps)

        clean_obj.attrs.pop("is_scaled", None)
        clean_obj.attrs.pop("scale_method", None)
        return clean_obj

    def _select_auto_normalization(
        self, df_target: pd.DataFrame
    ) -> "MetaboIntNormalizer":
        """
        Evaluate fixed strategies and select the best normalization method.
        """
        logger.info(
            "Auto normalization evaluates fixed strategies on a "
            "common log-like "
            "view using QC RLE, QC variance stabilization, QC structure, and "
            "sample structure criteria."
        )

        records: list[dict[str, object]] = []
        candidate_outputs: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}

        for method in self._AUTO_CANDIDATES:
            apply_external_log = self._uses_external_log_for_method(method)
            logger.info(
                f"Evaluating Auto normalization candidate: {method} "
                f"(log_transform={apply_external_log})."
            )
            try:
                df_norm, meta_stamps = self._apply_normalization_candidate(
                    df_target=df_target,
                    method=method,
                    apply_external_log=apply_external_log,
                )

                arr = df_norm.to_numpy(dtype=float)
                if np.isinf(arr).any() or np.isnan(arr).all():
                    raise ValueError(
                        "Candidate produced invalid normalized values."
                    )

                candidate_obj = self._finalize_normalized_object(
                    df_norm, meta_stamps
                )
                candidate_obj.attrs["pipeline_stage"] = "Normalization"
                auto_metrics = self.calc_auto_norm_candidate_metrics(
                    candidate_obj,
                )

                record = {
                    "method": method,
                    "normalization_applied": meta_stamps.get(
                        "normalization_applied", False
                    ),
                    "log_transform_applied": apply_external_log,
                    "status": "ok",
                    "error": "",
                    **auto_metrics,
                }
                candidate_outputs[method] = (df_norm, meta_stamps)
            except Exception as exc:
                logger.warning(
                    f"Auto normalization candidate {method} failed: {exc}"
                )
                record = {
                    "method": method,
                    "normalization_applied": method != "ROBUST_LOG_ONLY",
                    "log_transform_applied": apply_external_log,
                    "status": "failed",
                    "error": str(exc),
                }
                for metric in self._AUTO_SCORE_COMPONENT_WEIGHTS:
                    record[metric] = float("nan")

            records.append(record)

        scored_candidates = self._score_normalization_candidates(records)
        auto_summary = self._summarize_auto_norm_scores(scored_candidates)
        score_parts = []
        for _, row in auto_summary.iterrows():
            method_name = str(row["method"])
            status = str(row["status"])
            score = su.finite_or_nan(row.get("overall_score"))
            if status == "ok" and np.isfinite(score):
                score_parts.append(f"{method_name}={score:.3f}")
            else:
                score_parts.append(f"{method_name}=failed")
        logger.info(
            "Auto normalization candidate scores: " + ", ".join(score_parts)
        )

        selected_row = auto_summary[auto_summary["selected"]].iloc[0]
        selected_method = str(selected_row["method"])
        selected_score = su.finite_or_nan(selected_row["overall_score"])
        selected_margin = su.finite_or_nan(selected_row.get("selection_margin"))

        if np.isfinite(selected_margin):
            logger.info(
                f"Auto normalization selected {selected_method} "
                f"(score={selected_score:.3f}, margin={selected_margin:.3f})."
            )
        else:
            logger.info(
                f"Auto normalization selected {selected_method} "
                f"(score={selected_score:.3f})."
            )

        df_norm, meta_stamps = candidate_outputs[selected_method]
        meta_stamps = dict(meta_stamps)
        meta_stamps.update(
            {
                "selection": {
                    "requested_method": "Auto",
                    "selected_method": selected_method,
                    "selected_label": selected_method,
                    "is_auto": True,
                    "selected_score": selected_score,
                    "selection_margin": selected_margin,
                    "candidate_summary": auto_summary.to_dict(
                        orient="records"
                    ),
                },
            }
        )

        return self._finalize_normalized_object(df_norm, meta_stamps)

    def apply_normalization(self) -> "MetaboIntNormalizer":
        """Execute normalization workflow to generate a Clean_Dataset.

        Implements different execution orders:
        - ROBUST_LOG_ONLY: Robust Log only
        - TIC/Median/PQN/MDFC: Normalization -> Robust Log
        - Quantile: Robust Log -> Normalization
        - VSN: Intrinsic glog
        """
        df_target = self._extract_ordered_target_matrix()
        method = NORMALIZATION_METHODS.canonicalize(
            self.attrs.get("norm_method", "ROBUST_LOG_ONLY")
            or "ROBUST_LOG_ONLY",
            strict=False,
        )

        if method == "AUTO":
            return self._select_auto_normalization(df_target)

        apply_external_log = self._uses_external_log_for_method(method)
        df_norm, meta_stamps = self._apply_normalization_candidate(
            df_target=df_target,
            method=method,
            apply_external_log=apply_external_log,
        )
        meta_stamps["selection"] = {
            "requested_method": method,
            "selected_method": method,
            "selected_label": method,
            "is_auto": False,
        }
        return self._finalize_normalized_object(df_norm, meta_stamps)

    @cached_property
    def normalization_metrics(self) -> Dict[str, Any]:
        """Extracts configuration and QA metrics from the workflow."""
        curr_stage = self.attrs.get("pipeline_stage", "Unknown")

        selection = self.attrs.get("selection")
        if selection is None:
            requested_method = self.attrs.get(
                "norm_method", "ROBUST_LOG_ONLY"
            )
            selection = {
                "requested_method": requested_method,
                "selected_method": requested_method,
                "selected_label": requested_method,
                "is_auto": str(requested_method).upper() == "AUTO",
            }

        metrics = {
            "current_stage": curr_stage,
            "strategies": {
                "normalization_method": self.attrs.get(
                    "norm_method", "ROBUST_LOG_ONLY"
                ),
                "normalization_applied": self.attrs.get(
                    "normalization_applied", False
                ),
                "log_transform_active": self.attrs.get("is_logged", False),
            },
            "selection": selection,
        }

        if self.attrs.get("norm_method", "ROBUST_LOG_ONLY").upper() == "VSN":
            metrics["vsn_parameters"] = {
                "vsn_scale": self.attrs.get("vsn_scale", float("nan")),
                "vsn_shift": self.attrs.get("vsn_shift", float("nan")),
            }

        return metrics

    def run_normalization(
        self,
        output_dir: str | None = None,
        **runtime_overrides: object,
    ) -> StageResult["MetaboIntNormalizer"]:
        """Return the structured normalization stage result.

        ``norm_method``, ``global_seed``, or ``n_jobs`` supplied here take
        precedence over pipeline configuration and module defaults.
        """
        result = NormalizationStageRunner(
            self,
            output_dir,
            runtime_overrides=runtime_overrides,
            allowed_override_keys=self._RUNTIME_CONFIG_KEYS,
        ).run()
        logger.success("Data normalization completed successfully.")
        return result
