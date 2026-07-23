# src/pimqc/normalization.py
"""
Script purpose: Execute final normalization and diagnostic export.

apply_normalization() selects fixed method-specific transformations:
ROBUST_LOG_ONLY, TIC, Median, PQN, MDFC, log-first distribution alignment for
Quantile, or intrinsic glog transformation for VSN. Auto mode ranks these
log-like strategies with QC RLE alignment, QC variance stabilization, QC
structure distance improvement, and sample structure preservation.
execute_normalization() creates the output folder, drops blanks from the final
target matrix, saves the normalized dataset and Auto summary, and renders a
dashboard aligned with the Auto scoring dimensions.
"""

import os
import copy
import numpy as np
import pandas as pd
from functools import cached_property
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from numba import njit
from joblib import Parallel, delayed
from loguru import logger
from typing import Dict, Any, Optional

from . import core_classes
from . import visualizer_classes
from . import io_utils as iu
from . import plot_utils as pu
from . import stat_utils as su


# =====================================================================
# 1. Private Numba JIT Engines for Normalization
# =====================================================================
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


# =====================================================================
# 2. Main Normalization Class (Calculation Logic Only)
# =====================================================================
class MetaboIntNormalizer(core_classes.MetaboInt):
    """Normalization engine for global sample-wise preprocessing."""

    _metadata = ["attrs", "stats"]
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

        # 1. Base defaults matching pipeline_parameters.toml.
        norm_configs = {"norm_method": "Auto"}

        # 2. TOML global configuration overrides base defaults
        if pipeline_params and "MetaboIntNormalizer" in pipeline_params:
            norm_configs.update(pipeline_params["MetaboIntNormalizer"])

        # 3. Explicit kwargs override TOML (Highest priority)
        if norm_method is not None:
            norm_configs["norm_method"] = norm_method

        # 4. Finalize state strictly into lifecycle attributes
        self.attrs.update(norm_configs)

    @property
    def _constructor(self) -> type["MetaboIntNormalizer"]:
        """Override pandas constructor to return current subclass type."""
        return MetaboIntNormalizer

    def __finalize__(
        self,
        other: object,
        method: Optional[str] = None,
        **kwargs: object,
    ) -> "MetaboIntNormalizer":
        """Deepcopy custom attributes during pandas operations."""
        self = super().__finalize__(other, method=method, **kwargs)
        for name in self._metadata:
            if hasattr(other, name):
                setattr(self, name, copy.deepcopy(getattr(other, name)))
        return self

    # ====================================================================
    # Lightweight Auto-normalization Metrics
    # ====================================================================
    @staticmethod
    def _canonical_norm_method(method: object) -> str:
        """Normalize method names into the internal uppercase representation."""
        method_upper = str(method or "ROBUST_LOG_ONLY").strip().upper()
        compact = method_upper.replace("-", "_").replace(" ", "_")
        aliases = {
            "ROBUSTLOGONLY": "ROBUST_LOG_ONLY",
            "ROBUST_LOG2_ONLY": "ROBUST_LOG_ONLY",
            "LOG2_ONLY": "ROBUST_LOG_ONLY",
            "LOG_ONLY": "ROBUST_LOG_ONLY",
        }
        return aliases.get(compact, compact)

    @classmethod
    def _uses_external_log_for_method(cls, method: str) -> bool:
        """Return the fixed log-transform policy for a normalization method."""
        method_upper = cls._canonical_norm_method(method)
        return cls._DEFAULT_EXTERNAL_LOG.get(method_upper, True)

    @staticmethod
    def _finite_or_nan(value: object) -> float:
        """Convert a scalar metric to float, returning NaN for invalid values."""
        return su.finite_or_nan(value)

    @staticmethod
    def _align_paired_log_matrices(
        before_log: pd.DataFrame,
        after_log: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Align paired log-like matrices to the same feature and sample order."""
        return su.align_paired_matrices(before_log, after_log)

    @staticmethod
    def _series_iqr(series: pd.Series) -> float:
        """Calculate the interquartile range of a numeric Series."""
        return su.series_iqr(series)

    @staticmethod
    def _relative_change_lower_better(before: object, after: object) -> float:
        """Return relative before-to-after change for a lower-is-better metric."""
        return su.relative_change_lower_better(before, after)

    @staticmethod
    def _practical_signed_change_lower_better(
        before: object,
        after: object,
        min_abs_change: float = 0.0,
        min_rel_change: float = 0.0,
    ) -> float:
        """Return signed relative change for a lower-is-better metric."""
        return su.practical_signed_change_lower_better(
            before=before,
            after=after,
            min_abs_change=min_abs_change,
            min_rel_change=min_rel_change,
        )

    @staticmethod
    def _median_signed_change_lower_better(
        before_values: pd.Series,
        after_values: pd.Series,
        min_abs_change: float = 0.0,
        min_rel_change: float = 0.0,
    ) -> float:
        """Return median paired signed change for lower-is-better values."""
        common_index = before_values.index.intersection(after_values.index)
        if len(common_index) == 0:
            return float("nan")

        before_arr = before_values.loc[common_index].to_numpy(dtype=float)
        after_arr = after_values.loc[common_index].to_numpy(dtype=float)
        finite_mask = np.isfinite(before_arr) & np.isfinite(after_arr)
        before_arr = before_arr[finite_mask]
        after_arr = after_arr[finite_mask]
        if before_arr.size == 0:
            return float("nan")

        denominator = np.maximum(before_arr, np.finfo(float).eps)
        signed_change = (before_arr - after_arr) / denominator
        abs_change = np.abs(before_arr - after_arr)
        finite_change = np.isfinite(signed_change) & np.isfinite(abs_change)
        signed_change = signed_change[finite_change]
        abs_change = abs_change[finite_change]
        if signed_change.size == 0:
            return float("nan")

        median_signed_change = float(np.nanmedian(signed_change))
        median_abs_change = float(np.nanmedian(abs_change))
        if (
            median_abs_change < min_abs_change
            or abs(median_signed_change) < min_rel_change
        ):
            return 0.0
        return median_signed_change

    @staticmethod
    def _weighted_mean_score(
        score_weights: list[tuple[float, float]],
        clip_values: bool = True,
    ) -> float:
        """Return a weighted mean over finite component scores."""
        return su.weighted_mean_score(
            score_weights=score_weights,
            clip_values=clip_values,
        )

    @staticmethod
    def _rank_loss_from_distances(raw_dist: np.ndarray, norm_dist: np.ndarray) -> float:
        """Return 1 - Spearman rho between two pairwise distance vectors."""
        return su.rank_loss_from_distances(raw_dist, norm_dist)

    @staticmethod
    def _robust_feature_zscore(df: pd.DataFrame) -> pd.DataFrame:
        """Scale each feature by median/MAD across samples for geometry checks."""
        return su.robust_feature_zscore(df)

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

        qc_rle_medians = qc_rle.median(axis=0).replace([np.inf, -np.inf], np.nan)
        qc_rle_iqrs = qc_rle.quantile(0.75, axis=0) - qc_rle.quantile(0.25, axis=0)
        qc_rle_medians = qc_rle_medians.dropna()
        qc_rle_iqrs = qc_rle_iqrs.replace([np.inf, -np.inf], np.nan).dropna()

        if not qc_rle_medians.empty:
            metrics["rle_center_offset"] = cls._finite_or_nan(
                qc_rle_medians.abs().median()
            )
        if not qc_rle_iqrs.empty:
            metrics["rle_spread"] = cls._finite_or_nan(qc_rle_iqrs.median())
        return metrics

    @classmethod
    def _calc_qc_structure_values(
        cls,
        log_df: pd.DataFrame,
        qc_cols: pd.Index,
        max_features: Optional[int] = 5000,
        seed: int = 123,
    ) -> dict[str, Any]:
        """Calculate multivariate QC compactness around the pooled-QC centroid."""
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
            keep_idx = rng.choice(data_df.index, size=max_features, replace=False)
            data_df = data_df.loc[keep_idx]

        z_df = cls._robust_feature_zscore(data_df)
        qc_z = z_df[qc_cols.intersection(z_df.columns)]
        if qc_z.shape[1] < 3 or qc_z.shape[0] < 2:
            return metrics

        qc_centroid = qc_z.median(axis=1, skipna=True)
        qc_residual = qc_z.sub(qc_centroid, axis=0).to_numpy(dtype=float, copy=True)
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
            metrics["qc_centroid_distance_median"] = cls._finite_or_nan(
                distance_series.median()
            )
            metrics["qc_centroid_distance_iqr"] = cls._series_iqr(distance_series)

        pairwise_dist = pdist(qc_z.T.to_numpy(dtype=float), metric="euclidean")
        pairwise_dist = pairwise_dist / max(scale, np.finfo(float).eps)
        pairwise_dist = pairwise_dist[np.isfinite(pairwise_dist)]
        if pairwise_dist.size > 0:
            metrics["qc_pairwise_distance"] = pd.Series(
                pairwise_dist,
                dtype=float,
            )
            metrics["qc_pairwise_distance_median"] = cls._finite_or_nan(
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
                    "mean_intensity": cls._finite_or_nan(
                        bin_df["mean_intensity"].median()
                    ),
                    "dispersion_median": cls._finite_or_nan(
                        bin_df["qc_dispersion"].median()
                    ),
                    "dispersion_q25": cls._finite_or_nan(
                        bin_df["qc_dispersion"].quantile(0.25)
                    ),
                    "dispersion_q75": cls._finite_or_nan(
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
        """Calculate QC mean-dispersion dependence for variance stabilization."""
        metrics: dict[str, Any] = {
            "feature_stats": pd.DataFrame(),
            "trend": pd.DataFrame(),
            "qc_dispersion_median": float("nan"),
            "mean_variance_abs_rho": float("nan"),
            "mean_variance_abs_slope": float("nan"),
        }

        stats_df = cls._calc_qc_mean_dispersion_table(log_df=log_df, qc_cols=qc_cols)
        if stats_df.shape[0] < 3:
            return metrics

        metrics["feature_stats"] = stats_df
        metrics["trend"] = cls._calc_qc_variance_trend(stats_df)
        metrics["qc_dispersion_median"] = cls._finite_or_nan(
            stats_df["qc_dispersion"].median()
        )

        rho_val = stats.spearmanr(
            stats_df["mean_intensity"].to_numpy(dtype=float),
            stats_df["qc_dispersion"].to_numpy(dtype=float),
        )[0]
        metrics["mean_variance_abs_rho"] = abs(cls._finite_or_nan(rho_val))

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
                metrics["mean_variance_abs_slope"] = abs(cls._finite_or_nan(slope_val))

        return metrics

    @classmethod
    def _calc_sample_structure_arrays(
        cls,
        raw_obj: core_classes.MetaboInt,
        norm_obj: core_classes.MetaboInt,
        max_features: Optional[int] = 5000,
        seed: int = 123,
    ) -> dict[str, dict[str, Any]]:
        """Calculate robust-geometry sample structure arrays."""
        return su.calc_sample_structure_arrays(
            raw_obj=raw_obj,
            transformed_obj=norm_obj,
            max_features=max_features,
            seed=seed,
        )

    def _extract_auto_eval_target(
        self,
        norm_obj: core_classes.MetaboInt,
    ) -> pd.DataFrame | None:
        """Extract a common log-like evaluation view for Auto scoring.

        Candidate outputs keep their delivered scale for export. Auto scoring,
        however, evaluates all candidates through the same log-like view so
        log2/glog-transformed strategies remain comparable.
        """
        return su._extract_log2_target(norm_obj)

    def _sample_structure_preservation_metrics(
        self,
        norm_obj: core_classes.MetaboInt,
        max_features: Optional[int] = 5000,
    ) -> dict[str, float]:
        """Quantify local sample structure preservation without labels."""
        return su.calc_sample_structure_preservation(
            raw_obj=self,
            transformed_obj=norm_obj,
            max_features=max_features,
            seed=int(self.attrs.get("global_seed", 123)),
            scale_log_ratio_tol=self._SAMPLE_SCALE_LOG_RATIO_TOL,
            scale_rel_delta_tol=self._SAMPLE_SCALE_REL_DELTA_TOL,
        )

    def calc_auto_norm_candidate_metrics(
        self,
        norm_obj: core_classes.MetaboInt,
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

        log_raw = self._extract_auto_eval_target(self)
        log_norm = self._extract_auto_eval_target(norm_obj)
        if log_raw is None or log_raw.empty or log_norm is None or log_norm.empty:
            return metrics

        log_raw, log_norm = self._align_paired_log_matrices(log_raw, log_norm)
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
            metrics["rle_center_offset_before"] = rle_before["rle_center_offset"]
            metrics["rle_center_offset_after"] = rle_after["rle_center_offset"]
            metrics["rle_spread_before"] = rle_before["rle_spread"]
            metrics["rle_spread_after"] = rle_after["rle_spread"]

            metrics["rle_alignment_change_score"] = (
                MetaboIntNormalizer._weighted_mean_score(
                    [
                        (
                            MetaboIntNormalizer._practical_signed_change_lower_better(
                                rle_before["rle_center_offset"],
                                rle_after["rle_center_offset"],
                                min_rel_change=0.01,
                            ),
                            3.0,
                        ),
                        (
                            MetaboIntNormalizer._practical_signed_change_lower_better(
                                rle_before["rle_spread"],
                                rle_after["rle_spread"],
                                min_rel_change=0.01,
                            ),
                            2.0,
                        ),
                    ],
                    clip_values=False,
                )
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
            metrics["qc_dispersion_median_before"] = var_before["qc_dispersion_median"]
            metrics["qc_dispersion_median_after"] = var_after["qc_dispersion_median"]
            metrics["mean_variance_abs_rho_before"] = var_before[
                "mean_variance_abs_rho"
            ]
            metrics["mean_variance_abs_rho_after"] = var_after["mean_variance_abs_rho"]
            metrics["mean_variance_abs_slope_before"] = var_before[
                "mean_variance_abs_slope"
            ]
            metrics["mean_variance_abs_slope_after"] = var_after[
                "mean_variance_abs_slope"
            ]
            metrics["variance_stabilization_score"] = (
                MetaboIntNormalizer._weighted_mean_score(
                    [
                        (
                            MetaboIntNormalizer._practical_signed_change_lower_better(
                                var_before["mean_variance_abs_rho"],
                                var_after["mean_variance_abs_rho"],
                                min_abs_change=0.01,
                                min_rel_change=0.02,
                            ),
                            4.0,
                        ),
                        (
                            MetaboIntNormalizer._practical_signed_change_lower_better(
                                var_before["mean_variance_abs_slope"],
                                var_after["mean_variance_abs_slope"],
                                min_abs_change=0.005,
                                min_rel_change=0.02,
                            ),
                            4.0,
                        ),
                        (
                            MetaboIntNormalizer._practical_signed_change_lower_better(
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
            )

        if len(qc_cols) >= 3:
            qc_structure_before = self._calc_qc_structure_values(
                log_raw,
                qc_cols=qc_cols,
                max_features=5000,
                seed=int(self.attrs.get("global_seed", 123)),
            )
            qc_structure_after = self._calc_qc_structure_values(
                log_norm,
                qc_cols=qc_cols,
                max_features=5000,
                seed=int(self.attrs.get("global_seed", 123)),
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
            metrics["qc_structure_change_score"] = (
                MetaboIntNormalizer._weighted_mean_score(
                    [
                        (
                            MetaboIntNormalizer._practical_signed_change_lower_better(
                                qc_structure_before["qc_centroid_distance_median"],
                                qc_structure_after["qc_centroid_distance_median"],
                                min_abs_change=0.005,
                                min_rel_change=0.02,
                            ),
                            2.0,
                        ),
                        (
                            MetaboIntNormalizer._practical_signed_change_lower_better(
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
            )

        metrics.update(self._sample_structure_preservation_metrics(norm_obj))
        sample_structure_preservation = self._finite_or_nan(
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

        for score_col, score_weight in cls._AUTO_SCORE_COMPONENT_WEIGHTS.items():
            score_values = pd.to_numeric(score_df.get(score_col), errors="coerce")
            valid_score_mask = ok_mask & np.isfinite(score_values)

            if score_col in cls._AUTO_CENTERED_CHANGE_COMPONENTS:
                baseline_mask = ok_mask & score_df["method"].eq(
                    cls._AUTO_BASELINE_METHOD
                )
                baseline_values = score_values.loc[baseline_mask].dropna()
                baseline_value = (
                    float(baseline_values.iloc[0]) if not baseline_values.empty else 0.0
                )

                raw_change = score_values - baseline_value
                centered_scores = pd.Series(np.nan, index=score_df.index, dtype=float)

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
                        0.5 - 0.5 * raw_change.loc[negative_mask].abs() / negative_max
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
            raise ValueError("No Auto normalization candidate produced valid metrics.")

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

        candidate_rank = score_df.loc[scoreable_mask].copy()
        candidate_rank["_conservative_order"] = candidate_rank["method"].map(
            cls._AUTO_CONSERVATIVE_ORDER
        )
        selected_idx = candidate_rank.sort_values(
            by=["auto_score", "_conservative_order", "method"],
            ascending=[False, True, True],
        ).index[0]

        score_df.loc[selected_idx, "selected"] = True

        sorted_scores = score_df.loc[scoreable_mask, "auto_score"].sort_values(
            ascending=False
        )
        if len(sorted_scores) > 1:
            margin = float(sorted_scores.iloc[0] - sorted_scores.iloc[1])
        else:
            margin = float("nan")

        score_df["selection_margin"] = margin
        return score_df

    @classmethod
    def _summarize_auto_norm_scores(cls, score_df: pd.DataFrame) -> pd.DataFrame:
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
            cls._finite_or_nan(robust_log_matches.iloc[0])
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

    # ====================================================================
    # Mathematical Operators (Sample-wise & Global)
    # ====================================================================
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
        best_indices = sorted_idx[np.linspace(0, rows - 1, max_features, dtype=int)]
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
        normed_arr = (np.arcsinh(a_vec + b * data_arr) / np.log(2)) - shift_constant

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
        df: pd.DataFrame, qc_cols: pd.Index, kde_points: int = 1000, n_jobs: int = -1
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

        # 1. Define high-quality reference spectrum
        if qc_cols is not None and not qc_cols.empty:
            ref_spectrum = df_safe[qc_cols].median(axis="columns")
        else:
            logger.warning("No QCs for MDFC. Using global median fallback.")
            ref_spectrum = df_safe.median(axis="columns")

        # Extract strictly decoupled numpy arrays to prevent memory leak
        log2_ref = np.log2(ref_spectrum.replace({0: np.nan})).values
        data_matrix = df_safe.values
        num_samples = data_matrix.shape[1]

        # 2. Pure function: Process a chunk to avoid IPC overhead
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
                        # 3. Kernel Density Estimation
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

                # 4. Apply back-transformation
                norm_factor = 2**shift
                out_chunk[:, i] = sample_vals / norm_factor

            return out_chunk

        # 5. Calculate safe threading and optimal chunking strategy
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

        # 6. Execute parallel processing via Joblib
        normed_chunks = Parallel(n_jobs=safe_n_jobs, backend="loky")(
            delayed(_process_mdfc_chunk)(chunk, log2_ref, kde_points)
            for chunk in chunks
        )

        # 7. Reconstruct the output DataFrame
        res_df = df.copy()
        if normed_chunks:
            res_df.iloc[:, :] = np.column_stack(normed_chunks)

        return res_df.fillna(0)

    # ====================================================================
    # Core Execution Logic (Single Lane Refactored)
    # ====================================================================
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
        """Apply one fixed normalization strategy to an ordered target matrix."""
        method = self._canonical_norm_method(method)
        df_norm = df_target.copy()

        meta_stamps: dict[str, Any] = {
            "norm_method": method,
            "is_logged": False,
        }

        # -------------------------------------------------------------
        # Category A: Linear Scale Methods and log-only baseline
        # Logic: Normalize first, then robust Log2 for variance stabilization.
        # -------------------------------------------------------------
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

        # -------------------------------------------------------------
        # Category B: Distribution Alignment (Quantile)
        # Logic: Robust Log2 first, then align distributions.
        # -------------------------------------------------------------
        elif method == "QUANTILE":
            meta_stamps["normalization_applied"] = True
            if apply_external_log:
                df_norm = su.robust_log2_transform(df_norm)
                meta_stamps["is_logged"] = True
            df_norm = self.calc_quantile_normalization(df_norm)

        # -------------------------------------------------------------
        # Category C: Variance Stabilizing Normalization (VSN)
        # Logic: Intrinsic glog; no external robust Log2.
        # -------------------------------------------------------------
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
        """Create a finalized normalized object and clear stale scaling stamps."""
        clean_obj = self._constructor(df_norm).__finalize__(self)
        clean_obj.attrs.update(meta_stamps)

        clean_obj.attrs.pop("is_scaled", None)
        clean_obj.attrs.pop("scale_method", None)
        return clean_obj

    def _select_auto_normalization(
        self, df_target: pd.DataFrame
    ) -> "MetaboIntNormalizer":
        """Evaluate fixed strategies and select the best normalization method."""
        logger.info(
            "Auto normalization evaluates fixed strategies on a common log-like "
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
                    raise ValueError("Candidate produced invalid normalized values.")

                candidate_obj = self._finalize_normalized_object(df_norm, meta_stamps)
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
                logger.warning(f"Auto normalization candidate {method} failed: {exc}")
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
            score = self._finite_or_nan(row.get("overall_score"))
            if status == "ok" and np.isfinite(score):
                score_parts.append(f"{method_name}={score:.3f}")
            else:
                score_parts.append(f"{method_name}=failed")
        logger.info("Auto normalization candidate scores: " + ", ".join(score_parts))

        selected_row = auto_summary[auto_summary["selected"]].iloc[0]
        selected_method = str(selected_row["method"])
        selected_score = self._finite_or_nan(selected_row["overall_score"])
        selected_margin = self._finite_or_nan(selected_row.get("selection_margin"))

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
                "requested_norm_method": "Auto",
                "auto_selected_method": selected_method,
                "auto_selected_score": selected_score,
                "auto_selection_margin": selected_margin,
                "normalization_auto_summary": auto_summary.to_dict(orient="records"),
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
        method = self._canonical_norm_method(
            self.attrs.get("norm_method", "ROBUST_LOG_ONLY")
        )

        if method == "AUTO":
            return self._select_auto_normalization(df_target)

        apply_external_log = self._uses_external_log_for_method(method)
        df_norm, meta_stamps = self._apply_normalization_candidate(
            df_target=df_target,
            method=method,
            apply_external_log=apply_external_log,
        )
        return self._finalize_normalized_object(df_norm, meta_stamps)

    @cached_property
    def normalization_metrics(self) -> Dict[str, Any]:
        """Extracts configuration and QA metrics from the workflow."""
        curr_stage = self.attrs.get("pipeline_stage", "Unknown")

        metrics = {
            "current_stage": curr_stage,
            "strategies": {
                "normalization_method": self.attrs.get(
                    "norm_method", "ROBUST_LOG_ONLY"
                ),
                "normalization_applied": self.attrs.get("normalization_applied", False),
                "log_transform_active": self.attrs.get("is_logged", False),
                "requested_norm_method": self.attrs.get(
                    "requested_norm_method",
                    self.attrs.get("norm_method", "ROBUST_LOG_ONLY"),
                ),
            },
        }

        auto_summary = self.attrs.get("normalization_auto_summary")
        if auto_summary:
            metrics["auto_selection"] = {
                "selected_method": self.attrs.get("auto_selected_method"),
                "selected_score": self.attrs.get("auto_selected_score"),
                "selection_margin": self.attrs.get("auto_selection_margin"),
                "summary": auto_summary,
            }

        if self.attrs.get("norm_method", "ROBUST_LOG_ONLY").upper() == "VSN":
            metrics["vsn_parameters"] = {
                "vsn_scale": self.attrs.get("vsn_scale", float("nan")),
                "vsn_shift": self.attrs.get("vsn_shift", float("nan")),
            }

        return metrics

    @iu._exe_time
    def execute_normalization(self, output_dir: str) -> "MetaboIntNormalizer":
        """Execute workflow, save outputs, and generate plots."""
        iu._check_dir_exists(dir_path=output_dir, handle="makedirs")

        requested_method = self.attrs.get("norm_method", "ROBUST_LOG_ONLY")

        blank_count = len(self._blank.columns)
        if blank_count > 0:
            logger.info(f"Permanently dropping {blank_count} Blank samples.")

        logger.info(f"Applying Normalization | Method: {requested_method}")

        # 1. Execute Core Calculation
        clean_obj = self.apply_normalization()
        method = clean_obj.attrs.get("norm_method", "ROBUST_LOG_ONLY")
        is_log = clean_obj.attrs.get("is_logged", False)

        # 2. Dynamic Suffix and Export
        suffix_parts = [method]
        if is_log and method.upper() not in {"VSN", "ROBUST_LOG_ONLY"}:
            suffix_parts.append("Log2")
        suffix = "_".join(suffix_parts)

        filename = f"Normalized_Data_{suffix}.csv"
        file_path = os.path.join(output_dir, filename)

        clean_obj.attrs["pipeline_stage"] = "Normalization"
        clean_obj.to_csv(path_or_buf=file_path, na_rep="NA", encoding="utf-8-sig")

        auto_summary = clean_obj.attrs.get("normalization_auto_summary")
        if auto_summary:
            summary_path = os.path.join(output_dir, "Normalization_Auto_Summary.csv")
            pd.DataFrame(auto_summary).to_csv(
                path_or_buf=summary_path,
                index=False,
                na_rep="NA",
                encoding="utf-8-sig",
            )
            logger.info(f"Auto normalization summary saved as: {summary_path}")

        # 3. Visualization Phase (2-Stage)
        logger.info("Generating diagnostic plots for normalization...")
        vis = MetaboVisualizerNormalizer(raw_obj=self, norm_obj=clean_obj)

        grid_path = None
        fig_grid = vis.plot_normalization_dashboard()
        if fig_grid:
            grid_path = os.path.join(
                output_dir, f"Normalization_Dashboard_{suffix}.svg"
            )
            vis.save_and_show_pw(pw_obj=fig_grid, file_path=grid_path)

        # if auto_summary:
        #     article_obj = vis.plot_normalization_article_dashboard()
        #     if article_obj is not None:
        #         article_path = os.path.join(
        #             output_dir,
        #             f"Normalization_Article_Dashboard_{suffix}.svg",
        #         )
        #         vis.save_and_show_pw(
        #             pw_obj=article_obj,
        #             width="60%",
        #             file_path=article_path,
        #         )
        #         logger.info(
        #             f"Normalization article dashboard saved as: {article_path}"
        #         )

        if grid_path is not None:
            logger.info(f"Normalization summary dashboard saved as: {grid_path}")
        logger.success("Data normalization completed successfully.")
        return clean_obj


class MetaboVisualizerNormalizer(visualizer_classes.BaseMetaboVisualizer):
    """2-Stage Visualization Suite (Before vs After Normalization).

    Generates high-contrast diagnostic plots evaluating the efficacy of the
    global variance stabilization and normalization preprocessing.
    """

    def __init__(
        self, raw_obj: core_classes.MetaboInt, norm_obj: core_classes.MetaboInt
    ) -> None:
        """Initialize with pre- and post-normalization datasets."""
        super().__init__(metabo_obj=norm_obj)
        self.raw = raw_obj
        self.norm = norm_obj
        self.stages = [("Before Norm", self.raw), ("After Norm", self.norm)]
        self.pal = {
            "Before Norm": pu.NEUTRAL_COLOR,
            "After Norm": pu.PRIMARY_ACCENT_COLOR,
        }

    @staticmethod
    def _normalization_score_component_style() -> tuple[
        list[str],
        dict[str, str],
        dict[str, str],
    ]:
        """Return the score-component ordering, labels, and colors."""
        score_cols = [
            "rle_alignment_change_score",
            "variance_stabilization_score",
            "qc_structure_change_score",
            "sample_structure_score",
        ]
        label_map = {
            "rle_alignment_change_score": "QC RLE alignment change",
            "variance_stabilization_score": "QC variance stabilization",
            "qc_structure_change_score": "QC structure distance change",
            "sample_structure_score": "Sample structure preservation",
        }
        color_map = {
            "rle_alignment_change_score": pu.get_equivalent_hex(
                pu.PRIMARY_ACCENT_COLOR, alpha=1.0
            ),
            "variance_stabilization_score": pu.get_equivalent_hex(
                pu.PRIMARY_ACCENT_COLOR, alpha=0.67
            ),
            "qc_structure_change_score": pu.get_equivalent_hex(
                pu.PRIMARY_ACCENT_COLOR, alpha=0.33
            ),
            "sample_structure_score": pu.get_equivalent_hex("tab:gray", alpha=0.5),
        }
        return score_cols, label_map, color_map

    def plot_normalization_score_summary(
        self,
        auto_summary: list[dict[str, Any]] | pd.DataFrame | None = None,
        figsize: tuple[float, float] = (4.0, 4.0),
        show_legend: bool = True,
    ) -> object | None:
        """Plot Auto normalization weighted score components as stacked bars."""
        if auto_summary is None:
            auto_summary = self.norm.attrs.get("normalization_auto_summary")
        if not auto_summary:
            return None

        try:
            import patchworklib as pw
            import matplotlib.patches as mpatches
        except ImportError as e:
            logger.warning(f"Skipping Auto normalization stacked bar plot: {e}")
            return None

        summary_df = pd.DataFrame(auto_summary).copy()
        if summary_df.empty:
            return None

        score_cols, label_map, color_map = self._normalization_score_component_style()
        contribution_weights = MetaboIntNormalizer._AUTO_SCORE_COMPONENT_WEIGHTS
        bar_edgecolor = "k"
        bar_linewidth = 0.5
        for col in ["overall_score", *score_cols]:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")

        plot_df = summary_df.copy()
        plot_df["status"] = plot_df["status"].fillna("failed")
        scoreable_mask = plot_df["status"].eq("ok") & plot_df["overall_score"].notna()
        if not scoreable_mask.any():
            return None

        def _method_label(method: object) -> str:
            """Create compact display labels for Auto normalization methods."""
            method_str = str(method)
            display_map = {
                "ROBUST_LOG_ONLY": "Robust Log2",
                "TIC": "TIC",
                "MEDIAN": "Median",
                "PQN": "PQN",
                "MDFC": "MDFC",
                "QUANTILE": "Quantile",
                "VSN": "VSN",
            }
            return display_map.get(method_str.upper(), method_str)

        plot_df["_sort_score"] = plot_df["overall_score"].fillna(-1.0)
        plot_df = plot_df.sort_values(
            by=["_sort_score", "method"], ascending=[False, True]
        ).reset_index(drop=True)

        contribution_cols: list[str] = []
        for col in score_cols:
            contribution_col = f"contribution_{col}"
            contribution_cols.append(contribution_col)
            plot_df[contribution_col] = 0.0

        for idx, row in plot_df.iterrows():
            if row["status"] != "ok":
                continue

            available_weight = 0.0
            for col in score_cols:
                value = row[col]
                if np.isfinite(value):
                    available_weight += contribution_weights[col]

            if available_weight <= 0:
                continue

            raw_sum = 0.0
            for col in score_cols:
                value = row[col]
                if np.isfinite(value):
                    contribution = (
                        np.clip(float(value), 0.0, 1.0)
                        * contribution_weights[col]
                        / available_weight
                    )
                    plot_df.loc[idx, f"contribution_{col}"] = contribution
                    raw_sum += contribution

            overall_score = row["overall_score"]
            if np.isfinite(overall_score) and raw_sum > 0:
                scale_factor = float(np.clip(overall_score, 0.0, 1.0)) / raw_sum
                for contribution_col in contribution_cols:
                    plot_df.loc[idx, contribution_col] *= scale_factor

        ax = pw.Brick(figsize=figsize, label="auto_norm_stacked_bar")
        y_pos = np.arange(len(plot_df))
        left = np.zeros(len(plot_df), dtype=float)

        for col in score_cols:
            contribution_col = f"contribution_{col}"
            values = plot_df[contribution_col].to_numpy(dtype=float)
            left_start = left.copy()
            ax.barh(
                y_pos,
                values,
                left=left,
                height=0.62,
                color=color_map[col],
                edgecolor=bar_edgecolor,
                linewidth=bar_linewidth,
                label=label_map[col],
                zorder=3,
            )
            for y_idx, row in enumerate(plot_df.itertuples()):
                score_value = su.finite_or_nan(getattr(row, col))
                if values[y_idx] < 0.09 or not np.isfinite(score_value):
                    continue
                face_color = color_map[col]
                ax.text(
                    left_start[y_idx] + values[y_idx] / 2.0,
                    y_idx,
                    f"{score_value:.2f}",
                    va="center",
                    ha="center",
                    fontsize=9.5,
                    color=pu.get_contrast_color(face_color),
                    clip_on=True,
                    zorder=4,
            )
            left += values

        x_data_max = float(np.nanmax(left)) if left.size else 1.0
        if not np.isfinite(x_data_max) or x_data_max <= 0:
            x_data_max = 1.0
        x_upper = min(1.05, max(0.20, x_data_max * 1.08, x_data_max + 0.04))

        for i, row in plot_df.iterrows():
            total_score = row["overall_score"]
            if row["status"] == "ok" and np.isfinite(total_score):
                label = f"{float(total_score):.3f}"
                ax.text(
                    min(float(left[i]) + x_upper * 0.015, x_upper * 0.965),
                    y_pos[i],
                    label,
                    va="center",
                    ha="left",
                    fontsize=10.5,
                    color="0.15",
                )
            else:
                ax.text(
                    0.015,
                    y_pos[i],
                    "failed",
                    va="center",
                    ha="left",
                    fontsize=10.5,
                    color="0.45",
                    style="italic",
                )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [
                f"* {_method_label(row.method)}"
                if bool(getattr(row, "selected", False))
                else _method_label(row.method)
                for row in plot_df.itertuples()
            ]
        )

        if show_legend:
            handles = [
                mpatches.Patch(
                    facecolor=color_map[col],
                    edgecolor=bar_edgecolor,
                    linewidth=bar_linewidth,
                    label=label_map[col],
                )
                for col in score_cols
            ]
            ax.legend(handles=handles, loc="lower right", bbox_to_anchor=None)
            self._format_single_legend(
                ax,
                loc="lower right",
                bbox_to_anchor=None,
                group_title="AUTO normalization score components",
                max_item_rows=6,
            )

        ax.set_xlim(0, x_upper)
        ax.set_ylim(-0.5, len(plot_df) - 0.5)
        ax.invert_yaxis()
        self._apply_standard_format(
            ax=ax,
            title="Auto Normalization Method Selection",
            xlabel="Weighted contribution to overall score",
            append_stage=False,
        )
        return ax

    def plot_normalization_preservation_scorecard(
        self,
        auto_summary: list[dict[str, Any]] | pd.DataFrame | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Axes | None:
        """Plot candidate-level sample-structure preservation scores."""
        if auto_summary is None:
            auto_summary = self.norm.attrs.get("normalization_auto_summary")
        if not auto_summary:
            return None

        try:
            import patchworklib as pw
        except ImportError as e:
            logger.warning(f"Skipping Auto normalization scorecard plot: {e}")
            return None

        current_ax = (
            pw.Brick(figsize=(4.8, 4.0), label="normalization_preservation_scorecard")
            if ax is None
            else ax
        )
        score_cols = [
            "sample_structure_trustworthiness",
            "sample_structure_rank_preservation",
            "sample_structure_scale_preservation",
        ]
        metric_labels = [
            "Trustworthiness",
            "Distance-rank\npreservation",
            "Distance-scale\npreservation",
        ]

        def _method_label(method: object) -> str:
            """Create compact display labels for Auto normalization methods."""
            method_str = str(method)
            display_map = {
                "ROBUST_LOG_ONLY": "Robust Log2",
                "TIC": "TIC",
                "MEDIAN": "Median",
                "PQN": "PQN",
                "MDFC": "MDFC",
                "QUANTILE": "Quantile",
                "VSN": "VSN",
            }
            return display_map.get(method_str.upper(), method_str)

        score_df = pd.DataFrame(auto_summary).copy()
        if score_df.empty:
            current_ax.axis("off")
            return current_ax

        for col in ["overall_score", *score_cols]:
            score_df[col] = pd.to_numeric(score_df[col], errors="coerce")
        score_df = score_df.loc[score_df["status"].eq("ok")].copy()
        score_df = score_df.dropna(subset=score_cols, how="all")
        score_df = score_df.sort_values(
            by=["overall_score", "method"], ascending=[False, True]
        ).reset_index(drop=True)
        if score_df.empty:
            current_ax.axis("off")
            return current_ax

        matrix = score_df[score_cols].to_numpy(dtype=float)
        cmap = pu.score_heatmap_cmap()
        annot_size = pu.heatmap_annotation_fontsize(
            current_ax,
            n_rows=matrix.shape[0],
            n_cols=matrix.shape[1],
            default_size=11.0,
            max_size=12.0,
            min_size=6.0,
        )
        current_ax.imshow(
            np.ma.masked_invalid(matrix),
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
        )
        current_ax.set_xticks(np.arange(len(score_cols)))
        current_ax.set_xticklabels(metric_labels)
        current_ax.set_yticks(np.arange(len(score_df)))
        current_ax.set_yticklabels(
            [
                f"* {_method_label(row.method)}"
                if bool(getattr(row, "selected", False))
                else _method_label(row.method)
                for row in score_df.itertuples()
            ]
        )
        current_ax.set_xticks(np.arange(-0.5, len(score_cols), 1), minor=True)
        current_ax.set_yticks(np.arange(-0.5, len(score_df), 1), minor=True)
        current_ax.grid(which="minor", color="k", linestyle="-", linewidth=1.0)
        current_ax.tick_params(which="minor", bottom=False, left=False)

        for y_idx in range(matrix.shape[0]):
            for x_idx in range(matrix.shape[1]):
                value = matrix[y_idx, x_idx]
                if not np.isfinite(value):
                    label = "NA"
                    color = "0.35"
                else:
                    label = f"{value:.2f}"
                    color = pu.get_contrast_color(cmap(value))
                current_ax.text(
                    x_idx,
                    y_idx,
                    label,
                    ha="center",
                    va="center",
                    fontsize=annot_size,
                    color=color,
                )

        self._apply_standard_format(
            current_ax,
            title="Candidate Preservation Scorecard",
            xlabel="",
            ylabel="",
            append_stage=False,
        )
        pu.rotate_xticks_if_overlapping(current_ax)
        current_ax.tick_params(axis="both", length=0)
        return current_ax

    @staticmethod
    def _add_metric_note(
        ax: plt.Axes,
        lines: list[str],
        loc: str = "lower right",
        fontsize: float = pu.DEFAULT_ANNOTATION_FONTSIZE,
        pad: float = 0.35,
    ) -> None:
        """Place a compact metric note in a diagnostic panel."""
        clean_lines = [line for line in lines if line]
        if not clean_lines:
            return
        if loc == "upper right":
            x_pos, y_pos = 0.96, 0.98
            verticalalignment = "top"
            horizontalalignment = "right"
        elif loc == "upper left":
            x_pos, y_pos = 0.04, 0.98
            verticalalignment = "top"
            horizontalalignment = "left"
        elif loc == "lower left":
            x_pos, y_pos = 0.04, 0.02
            verticalalignment = "bottom"
            horizontalalignment = "left"
        else:
            x_pos, y_pos = 0.96, 0.02
            verticalalignment = "bottom"
            horizontalalignment = "right"
        ax.text(
            x_pos,
            y_pos,
            "\n".join(clean_lines),
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment=verticalalignment,
            horizontalalignment=horizontalalignment,
            clip_on=False,
            bbox=pu.ai_ready_text_bbox(pad=pad),
            zorder=10,
        )

    def _plot_qc_rle_boxplot(
        self,
        ax: plt.Axes | None = None,
        max_points: int = 50000,
        show_legend: bool = True,
        article_compact: bool = False,
    ) -> plt.Figure | plt.Axes:
        """Plot QC-sample RLE center offset and spread before/after normalization."""
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(4, 4))
        else:
            current_ax = ax
            fig = current_ax.figure

        plot_records = []
        sample_metric_values: dict[str, dict[str, pd.Series]] = {}
        for label, obj in self.stages:
            log_d = su._extract_log2_target(obj)
            if log_d is None or log_d.empty:
                continue

            qc_cols = obj._qc.columns.intersection(log_d.columns)
            if len(qc_cols) < 2:
                continue

            global_feature_median = log_d.median(axis=1)
            qc_rle = log_d[qc_cols].astype(float).sub(global_feature_median, axis=0)
            qc_rle = qc_rle.replace([np.inf, -np.inf], np.nan)

            sample_medians = qc_rle.median(axis=0).replace([np.inf, -np.inf], np.nan)
            sample_q25 = qc_rle.quantile(0.25, axis=0)
            sample_q75 = qc_rle.quantile(0.75, axis=0)
            sample_iqrs = (sample_q75 - sample_q25).replace([np.inf, -np.inf], np.nan)
            sample_iqrs = sample_iqrs.replace([np.inf, -np.inf], np.nan)

            sample_medians = sample_medians.dropna()
            sample_iqrs = sample_iqrs.dropna()
            if sample_medians.empty or sample_iqrs.empty:
                continue

            sample_metric_values[label] = {
                "RLE center offset": sample_medians.abs(),
                "RLE spread": sample_iqrs,
            }
            center_values = sample_metric_values[label]["RLE center offset"]
            spread_values = sample_metric_values[label]["RLE spread"]

            center_offset = MetaboIntNormalizer._finite_or_nan(center_values.median())
            rle_spread = MetaboIntNormalizer._finite_or_nan(spread_values.median())
            if not all(np.isfinite(v) for v in [center_offset, rle_spread]):
                continue

            plot_records.extend(
                [
                    {
                        "Metric": "RLE center offset",
                        "Stage": label,
                        "Value": center_offset,
                        "Q25": MetaboIntNormalizer._finite_or_nan(
                            center_values.quantile(0.25)
                        ),
                        "Q75": MetaboIntNormalizer._finite_or_nan(
                            center_values.quantile(0.75)
                        ),
                    },
                    {
                        "Metric": "RLE spread",
                        "Stage": label,
                        "Value": rle_spread,
                        "Q25": MetaboIntNormalizer._finite_or_nan(
                            spread_values.quantile(0.25)
                        ),
                        "Q75": MetaboIntNormalizer._finite_or_nan(
                            spread_values.quantile(0.75)
                        ),
                    },
                ]
            )

        if plot_records:
            plot_df = pd.DataFrame(plot_records)
            metric_order = ["RLE center offset", "RLE spread"]
            stage_order = ["Before Norm", "After Norm"]
            note_lines = ["Relative change"]
            x_base = np.arange(len(metric_order), dtype=float)
            bar_width = 0.34
            offsets = {
                "Before Norm": -bar_width / 2,
                "After Norm": bar_width / 2,
            }
            bar_label_records = []
            bar_top_lookup: dict[tuple[str, str], float] = {}
            bar_label_top_lookup: dict[str, float] = {}

            for stage in stage_order:
                stage_df = plot_df[plot_df["Stage"].eq(stage)].set_index("Metric")
                values = []
                lower_errors = []
                upper_errors = []
                for metric in metric_order:
                    value = MetaboIntNormalizer._finite_or_nan(
                        stage_df["Value"].get(metric, np.nan)
                    )
                    q25 = MetaboIntNormalizer._finite_or_nan(
                        stage_df["Q25"].get(metric, np.nan)
                    )
                    q75 = MetaboIntNormalizer._finite_or_nan(
                        stage_df["Q75"].get(metric, np.nan)
                    )
                    lower_error = max(value - q25, 0.0) if np.isfinite(q25) else 0.0
                    upper_error = max(q75 - value, 0.0) if np.isfinite(q75) else 0.0
                    values.append(value)
                    lower_errors.append(lower_error)
                    upper_errors.append(upper_error)
                    bar_top_lookup[(metric, stage)] = (
                        value + upper_error if np.isfinite(value) else 0.0
                    )

                bar_container = current_ax.bar(
                    x_base + offsets[stage],
                    values,
                    yerr=np.vstack([lower_errors, upper_errors]),
                    width=bar_width,
                    color=self.pal[stage],
                    edgecolor="k",
                    linewidth=0.5 if article_compact else 1.0,
                    label=stage,
                    zorder=3,
                    error_kw={
                        "ecolor": "0.20",
                        "elinewidth": 0.5 if article_compact else 1.0,
                        "capsize": 2.0 if article_compact else 3.0,
                        "capthick": 0.5 if article_compact else 1.0,
                        "zorder": 4,
                    },
                )
                for patch, value, upper_error in zip(
                    bar_container.patches,
                    values,
                    upper_errors,
                    strict=False,
                ):
                    bar_label_records.append(
                        {
                            "metric": metric,
                            "stage": stage,
                            "x": patch.get_x() + patch.get_width() / 2.0,
                            "value": value,
                            "y": value + upper_error,
                        }
                    )

            if show_legend:
                current_ax.legend(loc="best")
                self._format_single_legend(
                    ax=current_ax,
                    loc="best",
                    bbox_to_anchor=None,
                    group_title="QC RLE alignment stage",
                    max_item_rows=6,
                )

            y_max = float(np.nanmax(plot_df[["Value", "Q75"]].to_numpy(dtype=float)))
            y_upper = y_max * 1.18 if y_max > 0 else 1.0
            current_ax.set_ylim(0, y_upper)
            label_offset = y_upper * 0.018
            for label_record in bar_label_records:
                value = MetaboIntNormalizer._finite_or_nan(label_record["value"])
                if not np.isfinite(value):
                    continue
                metric = str(label_record["metric"])
                inside_bar = value > y_upper * 0.12
                if inside_bar:
                    text_y = max(value - y_upper * 0.035, value * 0.88)
                    va = "top"
                    text_color = "white"
                    label_top = value
                else:
                    text_y = label_record["y"] + label_offset
                    va = "bottom"
                    text_color = "0.15"
                    label_top = text_y + y_upper * 0.025
                current_ax.text(
                    label_record["x"],
                    text_y,
                    f"{value:.3f}",
                    ha="center",
                    va=va,
                    fontsize=4.25 if article_compact else pu.DEFAULT_ANNOTATION_FONTSIZE,
                    color=text_color,
                    zorder=4,
                )
                bar_label_top_lookup[metric] = max(
                    bar_label_top_lookup.get(metric, 0.0),
                    label_top,
                )
            current_ax.set_xlim(-0.5, len(metric_order) - 0.5)
            current_ax.set_xticks(x_base)
            current_ax.set_xticklabels(["RLE center\noffset", "RLE\nspread"])

            metric_note_labels = {
                "RLE center offset": "Center",
                "RLE spread": "Spread",
            }
            for metric in metric_order:
                before_subset = plot_df[
                    plot_df["Metric"].eq(metric) & plot_df["Stage"].eq("Before Norm")
                ]["Value"]
                after_subset = plot_df[
                    plot_df["Metric"].eq(metric) & plot_df["Stage"].eq("After Norm")
                ]["Value"]
                before_value = MetaboIntNormalizer._finite_or_nan(
                    before_subset.iloc[0] if not before_subset.empty else np.nan
                )
                after_value = MetaboIntNormalizer._finite_or_nan(
                    after_subset.iloc[0] if not after_subset.empty else np.nan
                )
                rel_reduction = (
                    100.0
                    * MetaboIntNormalizer._relative_change_lower_better(
                        before_value, after_value
                    )
                )
                if np.isfinite(rel_reduction):
                    note_lines.append(
                        f"{metric_note_labels[metric]}: {rel_reduction:.1f}%"
                    )

            bracket_top = y_upper
            bracket_height = y_upper * 0.018
            bracket_gap = y_upper * 0.045
            for metric_idx, metric in enumerate(metric_order):
                p_val = self._paired_wilcoxon_pvalue(
                    sample_metric_values.get("Before Norm", {}).get(metric),
                    sample_metric_values.get("After Norm", {}).get(metric),
                )
                local_bar_top = max(
                    bar_top_lookup.get((metric, "Before Norm"), 0.0),
                    bar_top_lookup.get((metric, "After Norm"), 0.0),
                    bar_label_top_lookup.get(metric, 0.0),
                )
                y_level = local_bar_top + bracket_gap
                bracket_top = max(
                    bracket_top,
                    y_level + bracket_height + y_upper * 0.05,
                )
                self._add_pairwise_significance(
                    ax=current_ax,
                    x_left=x_base[metric_idx] + offsets["Before Norm"],
                    x_right=x_base[metric_idx] + offsets["After Norm"],
                    y=y_level,
                    text=self._pvalue_to_stars(p_val),
                    height=bracket_height,
                )
            current_ax.set_ylim(0, bracket_top)
            if not article_compact:
                self._add_metric_note(current_ax, note_lines, loc="lower right")
        else:
            current_ax.text(
                0.5,
                0.5,
                "Insufficient QC data",
                transform=current_ax.transAxes,
                ha="center",
                va="center",
                fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                bbox=pu.ai_ready_text_bbox(),
                zorder=10,
            )

        self._apply_standard_format(
            ax=current_ax,
            title="QC RLE Alignment Change",
            xlabel="QC RLE metric",
            ylabel="Median value (IQR)",
            append_stage=False,
        )
        return fig if ax is None else current_ax

    @staticmethod
    def _pvalue_to_stars(p_value: float) -> str:
        """Convert a p-value into compact significance-star text."""
        if not np.isfinite(p_value):
            return "n/a"
        if p_value < 0.001:
            return "***"
        if p_value < 0.01:
            return "**"
        if p_value < 0.05:
            return "*"
        return "ns"

    @staticmethod
    def _paired_wilcoxon_pvalue(
        before_values: pd.Series | None,
        after_values: pd.Series | None,
    ) -> float:
        """Calculate a paired Wilcoxon signed-rank p-value for matched QC samples."""
        if before_values is None or after_values is None:
            return float("nan")

        common_index = before_values.index.intersection(after_values.index)
        if len(common_index) < 3:
            return float("nan")

        before_arr = before_values.loc[common_index].to_numpy(dtype=float)
        after_arr = after_values.loc[common_index].to_numpy(dtype=float)
        finite_mask = np.isfinite(before_arr) & np.isfinite(after_arr)
        before_arr = before_arr[finite_mask]
        after_arr = after_arr[finite_mask]
        if before_arr.size < 3 or np.allclose(before_arr, after_arr):
            return 1.0

        try:
            return float(
                stats.wilcoxon(
                    before_arr,
                    after_arr,
                    zero_method="wilcox",
                    alternative="two-sided",
                ).pvalue
            )
        except ValueError:
            return float("nan")

    @staticmethod
    def _add_pairwise_significance(
        ax: plt.Axes,
        x_left: float,
        x_right: float,
        y: float,
        text: str,
        height: float,
    ) -> None:
        """Draw a paired-comparison bracket with significance text."""
        ax.plot(
            [x_left, x_left, x_right, x_right],
            [y, y + height, y + height, y],
            color="0.20",
            linewidth=1.0,
            clip_on=False,
            zorder=5,
        )
        ax.text(
            (x_left + x_right) / 2.0,
            y + height * 1.20,
            text,
            ha="center",
            va="bottom",
            fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
            color="0.20",
            clip_on=False,
            zorder=6,
        )

    def _plot_density_kde(
        self,
        metrics: dict[str, Any] | None = None,
        ax_qc: plt.Axes | None = None,
        ax_sample: plt.Axes | None = None,
    ) -> plt.Figure | tuple[plt.Axes, plt.Axes]:
        """Plot Log2 intensity density overlay for QC and Samples."""
        return_fig = False
        if ax_qc is None or ax_sample is None:
            fig, (ax_qc, ax_sample) = plt.subplots(1, 2, figsize=(8, 4))
            return_fig = True

        for grp, current_ax in [("QC", ax_qc), ("Sample", ax_sample)]:
            pu.mark_preserve_alpha(current_ax)
            for label, obj in self.stages:
                log_d = su._extract_log2_target(obj)
                if log_d is None or log_d.empty:
                    continue

                if grp == "QC" and hasattr(obj, "_qc"):
                    cols = obj._qc.columns.intersection(log_d.columns)
                elif hasattr(obj, "_actual_sample"):
                    cols = obj._actual_sample.columns.intersection(log_d.columns)
                else:
                    cols = []

                if len(cols) > 0:
                    vals = log_d[cols].values.flatten()
                    vals = vals[~np.isnan(vals)]
                    if len(vals) > 0:
                        sns.kdeplot(
                            vals,
                            ax=current_ax,
                            label=label,
                            color=self.pal[label],
                            linewidth=2,
                            alpha=0.8,
                        )

            self._apply_standard_format(
                ax=current_ax,
                title=f"Density Overlay ({grp})",
                xlabel="Log2 Intensity",
                ylabel="Density",
                append_stage=False,
            )

            if metrics and "JSD" in metrics and grp in metrics["JSD"]:
                jsd_data = metrics["JSD"][grp].get("Before vs After", np.nan)

                if isinstance(jsd_data, dict):
                    jsd_val = jsd_data.get("JSD", jsd_data.get("jsd", np.nan))
                else:
                    jsd_val = jsd_data

                if not pd.isna(jsd_val):
                    annot_text = (
                        "Jensen-Shannon Divergence\n"
                        f"Before Norm vs After Norm: {float(jsd_val):.3f}"
                    )
                    current_ax.text(
                        0.96,
                        0.02,
                        annot_text,
                        transform=current_ax.transAxes,
                        fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                        verticalalignment="bottom",
                        horizontalalignment="right",
                        clip_on=False,
                        bbox=pu.ai_ready_text_bbox(pad=0.4),
                        zorder=10,
                    )

            if current_ax.get_legend_handles_labels()[0]:
                current_ax.legend(loc="best")
                self._format_single_legend(
                    current_ax,
                    group_title="Distribution alignment stage",
                )

        if return_fig:
            plt.tight_layout()
            return fig
        return ax_qc, ax_sample

    def _plot_qc_variance_stabilization(
        self,
        ax: plt.Axes | None = None,
        show_legend: bool = True,
        article_compact: bool = False,
    ) -> plt.Figure | plt.Axes:
        """Plot QC mean-dispersion dependence before/after normalization."""
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(4, 4))
        else:
            current_ax = ax
            fig = current_ax.figure
        pu.mark_preserve_alpha(current_ax)

        stage_records = []
        for label, obj in self.stages:
            log_d = su._extract_log2_target(obj)
            if log_d is None or log_d.empty:
                continue

            qc_cols = obj._qc.columns.intersection(log_d.columns)
            if len(qc_cols) < 3:
                continue

            variance_metrics = (
                MetaboIntNormalizer._calc_qc_variance_stabilization_values(
                    log_d,
                    qc_cols=qc_cols,
                )
            )
            feature_stats = variance_metrics["feature_stats"]
            trend_df = variance_metrics["trend"]
            if feature_stats.empty or trend_df.empty:
                continue

            stage_records.append(
                {
                    "label": label,
                    "trend": trend_df,
                    "metrics": variance_metrics,
                }
            )

        if not stage_records:
            current_ax.text(
                0.5,
                0.5,
                "Insufficient QC data",
                transform=current_ax.transAxes,
                ha="center",
                va="center",
                fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                bbox=pu.ai_ready_text_bbox(),
                zorder=10,
            )
            self._apply_standard_format(
                ax=current_ax,
                title="QC Variance Stabilization",
                xlabel="Feature-wise mean QC log2 intensity",
                ylabel="Feature-wise QC dispersion",
                append_stage=False,
            )
            return fig if ax is None else current_ax

        line_style_map = {
            "Before Norm": {"color": pu.NEUTRAL_COLOR, "linestyle": "--"},
            "After Norm": {"color": pu.PRIMARY_ACCENT_COLOR, "linestyle": "-"},
        }
        ribbon_color_map = {
            "Before Norm": (pu.NEUTRAL_COLOR, 0.25),
            "After Norm": (pu.PRIMARY_ACCENT_COLOR, 0.33),
        }
        for record in stage_records:
            label = record["label"]
            trend_df = record["trend"]
            style = line_style_map.get(label, line_style_map["After Norm"])
            ribbon_color, ribbon_alpha = ribbon_color_map.get(
                label, (pu.PRIMARY_ACCENT_COLOR, 0.33)
            )
            x_vals = trend_df["mean_intensity"].to_numpy(dtype=float)
            y_vals = trend_df["dispersion_median"].to_numpy(dtype=float)
            y_low = trend_df["dispersion_q25"].to_numpy(dtype=float)
            y_high = trend_df["dispersion_q75"].to_numpy(dtype=float)
            current_ax.fill_between(
                x_vals,
                y_low,
                y_high,
                color=ribbon_color,
                alpha=ribbon_alpha,
                linewidth=0,
                zorder=1 if label == "Before Norm" else 2,
            )
            current_ax.plot(
                x_vals,
                y_vals,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.0 if article_compact else 2.0,
                marker="o",
                markersize=2.0 if article_compact else 3.0,
                label=label,
                zorder=4,
            )

        before_metrics = next(
            (
                record["metrics"]
                for record in stage_records
                if record["label"] == "Before Norm"
            ),
            {},
        )
        after_metrics = next(
            (
                record["metrics"]
                for record in stage_records
                if record["label"] == "After Norm"
            ),
            {},
        )
        note_lines = []
        if before_metrics and after_metrics:
            if article_compact:
                note_lines.extend(
                    [
                        (
                            "|rho|: "
                            f"{before_metrics.get('mean_variance_abs_rho', np.nan):.3f} / "
                            f"{after_metrics.get('mean_variance_abs_rho', np.nan):.3f}"
                        ),
                        (
                            "|slope|: "
                            f"{before_metrics.get('mean_variance_abs_slope', np.nan):.3f} / "
                            f"{after_metrics.get('mean_variance_abs_slope', np.nan):.3f}"
                        ),
                    ]
                )
            else:
                note_lines.extend(
                    [
                        "Before / After",
                        (
                            "|rho|: "
                            f"{before_metrics.get('mean_variance_abs_rho', np.nan):.3f} / "
                            f"{after_metrics.get('mean_variance_abs_rho', np.nan):.3f}"
                        ),
                        (
                            "|slope|: "
                            f"{before_metrics.get('mean_variance_abs_slope', np.nan):.3f} / "
                            f"{after_metrics.get('mean_variance_abs_slope', np.nan):.3f}"
                        ),
                        (
                            "Median dispersion: "
                            f"{before_metrics.get('qc_dispersion_median', np.nan):.3f} / "
                            f"{after_metrics.get('qc_dispersion_median', np.nan):.3f}"
                        ),
                    ]
                )
        self._add_metric_note(
            current_ax,
            note_lines,
            loc="upper right",
            fontsize=4.25 if article_compact else pu.DEFAULT_ANNOTATION_FONTSIZE,
            pad=0.25 if article_compact else 0.35,
        )

        if show_legend:
            current_ax.legend(loc="center right")
            self._format_single_legend(
                ax=current_ax,
                loc="center right",
                bbox_to_anchor=None,
                group_title="QC variance stabilization stage",
                max_item_rows=6,
            )

        self._apply_standard_format(
            ax=current_ax,
            title="QC Variance Stabilization",
            xlabel="Feature-wise mean QC log2 intensity",
            ylabel="Feature-wise QC dispersion",
            append_stage=False,
        )
        return fig if ax is None else current_ax

    def _plot_qc_structure_improvement(
        self,
        ax: plt.Axes | None = None,
        max_features: int = 5000,
        max_pair_points: int = 300,
        show_legend: bool = True,
        article_compact: bool = False,
    ) -> plt.Figure | plt.Axes:
        """Plot before/after multivariate QC-distance distributions."""
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(4, 4))
        else:
            current_ax = ax
            fig = current_ax.figure

        summary_by_stage = {}
        for label, obj in self.stages:
            log_d = su._extract_log2_target(obj)
            if log_d is None or log_d.empty:
                continue

            qc_cols = obj._qc.columns.intersection(log_d.columns)
            qc_structure = MetaboIntNormalizer._calc_qc_structure_values(
                log_d,
                qc_cols=qc_cols,
                max_features=max_features,
                seed=int(self.norm.attrs.get("global_seed", 123)),
            )
            distances = qc_structure["qc_centroid_distance"]
            if distances.empty:
                continue

            summary_by_stage[label] = qc_structure

        before_summary = summary_by_stage.get("Before Norm", {})
        after_summary = summary_by_stage.get("After Norm", {})
        if not before_summary or not after_summary:
            current_ax.text(
                0.5,
                0.5,
                "Insufficient QC data",
                transform=current_ax.transAxes,
                ha="center",
                va="center",
                fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                bbox=pu.ai_ready_text_bbox(),
                zorder=10,
            )
            self._apply_standard_format(
                ax=current_ax,
                title="QC Structure Distance Change",
                xlabel="QC distance metric",
                ylabel="Robust QC distance",
                append_stage=False,
            )
            return fig if ax is None else current_ax

        def _clean_distance_series(values: object) -> pd.Series:
            """Convert a stored QC-distance vector to finite numeric values."""
            if not isinstance(values, pd.Series):
                values = pd.Series(values, dtype=float)
            return (
                pd.to_numeric(values, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

        distance_specs = [
            {
                "label": "Distance to\nQC centroid",
                "short_label": "Centroid",
                "before": _clean_distance_series(
                    before_summary.get("qc_centroid_distance", pd.Series(dtype=float))
                ),
                "after": _clean_distance_series(
                    after_summary.get("qc_centroid_distance", pd.Series(dtype=float))
                ),
                "point_limit": None,
            },
            {
                "label": "QC-QC\npairwise distance",
                "short_label": "Pairwise",
                "before": _clean_distance_series(
                    before_summary.get("qc_pairwise_distance", pd.Series(dtype=float))
                ),
                "after": _clean_distance_series(
                    after_summary.get("qc_pairwise_distance", pd.Series(dtype=float))
                ),
                "point_limit": max_pair_points,
            },
        ]
        distance_specs = [
            spec
            for spec in distance_specs
            if not spec["before"].empty and not spec["after"].empty
        ]
        if not distance_specs:
            current_ax.text(
                0.5,
                0.5,
                "Insufficient QC-distance data",
                transform=current_ax.transAxes,
                ha="center",
                va="center",
                fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
            )
            self._apply_standard_format(
                ax=current_ax,
                title="QC Structure Distance Change",
                xlabel="QC distance metric",
                ylabel="Robust QC distance",
                append_stage=False,
            )
            return fig if ax is None else current_ax

        rng = np.random.default_rng(int(self.norm.attrs.get("global_seed", 123)))
        stage_order = ["Before Norm", "After Norm"]
        stage_offsets = {"Before Norm": -0.17, "After Norm": 0.17}
        box_width = 0.22 if article_compact else 0.28
        box_linewidth = 0.5 if article_compact else 1.25
        median_linewidth = 0.65 if article_compact else 1.25
        whisker_linewidth = 0.5 if article_compact else 1.0
        all_values: list[np.ndarray] = []
        note_lines = [] if article_compact else ["Median distance (Before / After)"]

        for metric_idx, spec in enumerate(distance_specs):
            stage_medians: dict[str, float] = {}
            for stage in stage_order:
                values = spec["before"] if stage == "Before Norm" else spec["after"]
                values = values.astype(float).replace([np.inf, -np.inf], np.nan)
                values = values.dropna()
                if values.empty:
                    continue

                value_array = values.to_numpy(dtype=float)
                all_values.append(value_array[np.isfinite(value_array)])
                x_pos = metric_idx + stage_offsets[stage]
                current_ax.boxplot(
                    value_array,
                    positions=[x_pos],
                    widths=box_width,
                    patch_artist=True,
                    showfliers=False,
                    boxprops={
                        "facecolor": "white",
                        "edgecolor": self.pal[stage],
                        "linewidth": box_linewidth,
                    },
                    medianprops={"color": "0.15", "linewidth": median_linewidth},
                    whiskerprops={
                        "color": self.pal[stage], "linewidth": whisker_linewidth
                    },
                    capprops={
                        "color": self.pal[stage], "linewidth": whisker_linewidth
                    },
                )

                point_values = value_array
                point_limit = spec["point_limit"]
                if point_limit is not None and point_values.size > point_limit:
                    keep = rng.choice(
                        point_values.size, size=int(point_limit), replace=False
                    )
                    point_values = point_values[keep]
                jitter = rng.normal(loc=0.0, scale=0.025, size=point_values.size)
                current_ax.scatter(
                    np.full(point_values.size, x_pos, dtype=float) + jitter,
                    point_values,
                    color=self.pal[stage],
                    edgecolor="k",
                    linewidth=0.15 if article_compact else 0.20,
                    s=10 if article_compact else 16,
                    zorder=3,
                )
                stage_medians[stage] = MetaboIntNormalizer._finite_or_nan(
                    values.median()
                )

            if all(
                np.isfinite(stage_medians.get(stage, np.nan)) for stage in stage_order
            ):
                current_ax.plot(
                    [
                        metric_idx + stage_offsets["Before Norm"],
                        metric_idx + stage_offsets["After Norm"],
                    ],
                    [
                        stage_medians["Before Norm"],
                        stage_medians["After Norm"],
                    ],
                    color="0.25",
                    linewidth=0.6 if article_compact else 1.0,
                    zorder=4,
                )

            before_median = stage_medians.get("Before Norm", np.nan)
            after_median = stage_medians.get("After Norm", np.nan)
            if np.isfinite(before_median) and np.isfinite(after_median):
                improvement = MetaboIntNormalizer._relative_change_lower_better(
                    before_median,
                    after_median,
                )
                improvement_text = ""
                if not article_compact and np.isfinite(improvement):
                    improvement_text = f"; improvement {100.0 * improvement:+.1f}%"
                note_lines.append(
                    (
                        f"{spec['short_label']}: "
                        f"{before_median:.3f} / {after_median:.3f}{improvement_text}"
                    )
                )

        if all_values:
            finite_values = np.concatenate([arr for arr in all_values if arr.size > 0])
        else:
            finite_values = np.array([], dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        positive_values = finite_values[finite_values > 0]
        use_log_scale = (
            positive_values.size > 0
            and np.nanmax(positive_values) / np.nanmin(positive_values) > 20
        )
        note_line_count = max(len(note_lines), 1)
        bottom_margin_fraction = min(0.36, 0.10 + 0.035 * note_line_count)
        top_margin_fraction = 0.18
        y_label = "Robust QC distance"
        if use_log_scale:
            current_ax.set_yscale("log")
            y_label = "Robust QC distance (log scale)"
            log_min = np.log10(np.nanmin(positive_values))
            log_max = np.log10(np.nanmax(positive_values))
            log_span = max(log_max - log_min, 0.5)
            y_lower = 10 ** (log_min - log_span * bottom_margin_fraction)
            y_upper = 10 ** (log_max + log_span * top_margin_fraction)
            current_ax.set_ylim(y_lower, y_upper)
        elif finite_values.size > 0:
            data_upper = np.nanpercentile(finite_values, 98)
            data_upper = max(data_upper, np.nanmax(finite_values) * 0.85)
            data_upper = data_upper if data_upper > 0 else 1.0
            y_lower = -data_upper * bottom_margin_fraction
            y_upper = data_upper * (1.0 + top_margin_fraction)
            current_ax.set_ylim(y_lower, y_upper)

        from matplotlib.lines import Line2D

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=self.pal[stage],
                marker="o",
                linestyle="",
                markeredgecolor="k",
                markeredgewidth=0.25,
                markersize=6,
                label=stage,
            )
            for stage in stage_order
        ]
        if show_legend:
            current_ax.legend(handles=legend_handles, loc="best")
            self._format_single_legend(
                ax=current_ax,
                loc="best",
                bbox_to_anchor=None,
                group_title="QC structure distance stage",
                max_item_rows=6,
            )

        current_ax.set_xlim(-0.55, len(distance_specs) - 0.45)
        current_ax.set_xticks(range(len(distance_specs)))
        current_ax.set_xticklabels([spec["label"] for spec in distance_specs])
        self._add_metric_note(
            current_ax,
            note_lines,
            loc="lower right",
            fontsize=4.25 if article_compact else pu.DEFAULT_ANNOTATION_FONTSIZE,
            pad=0.25 if article_compact else 0.35,
        )
        self._apply_standard_format(
            ax=current_ax,
            title="QC Structure Distance Change",
            xlabel="QC distance metric",
            ylabel=y_label,
            append_stage=False,
        )
        return fig if ax is None else current_ax

    def _plot_sample_structure_preservation(
        self,
        ax_geom: plt.Axes | None = None,
        max_features: int = 5000,
    ) -> plt.Axes | plt.Figure:
        """Plot score-aligned sample-structure preservation after normalization."""
        if ax_geom is None:
            created_fig, ax_geom = plt.subplots(figsize=(4, 4))
        else:
            created_fig = None

        structure_metrics: dict[str, float] | None = None
        auto_summary = self.norm.attrs.get("normalization_auto_summary")
        if auto_summary:
            selected_method = str(self.norm.attrs.get("auto_selected_method", ""))
            selected_row = next(
                (
                    row
                    for row in auto_summary
                    if row.get("selected") is True
                    or str(row.get("method", "")) == selected_method
                ),
                None,
            )
            if selected_row is not None:
                structure_metrics = selected_row

        pu.plot_sample_structure_change_map(
            ax=ax_geom,
            raw_obj=self.raw,
            transformed_obj=self.norm,
            structure_metrics=structure_metrics,
            seed=int(self.norm.attrs.get("global_seed", 123)),
            max_features=max_features,
            scale_log_ratio_tol=self.norm._SAMPLE_SCALE_LOG_RATIO_TOL,
            scale_rel_delta_tol=self.norm._SAMPLE_SCALE_REL_DELTA_TOL,
            title="Sample Structure Change Map",
        )
        if created_fig is not None:
            plt.tight_layout()
            return created_fig
        return ax_geom

    def plot_normalization_dashboard_legend(
        self,
        ax: plt.Axes,
        fontsize: float = 9.0,
        title_fontsize: float = 10.0,
        article_compact: bool = False,
        layout_cols: int = 1,
    ) -> plt.Axes:
        """Draw grouped score-component and stage legends for the dashboard."""
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches

        legend_linewidth = 0.5 if article_compact else 1.0
        line_width = 0.75 if article_compact else 2.0
        marker_size = 3.0 if article_compact else 5.5
        score_cols, label_map, color_map = self._normalization_score_component_style()
        score_handles = [
            mpatches.Patch(
                facecolor=color_map[col],
                edgecolor="k",
                linewidth=legend_linewidth,
                label=label_map[col],
            )
            for col in score_cols
        ]
        rle_handles = [
            mpatches.Patch(
                facecolor=self.pal["Before Norm"],
                edgecolor="k",
                linewidth=legend_linewidth,
                label="Before Norm",
            ),
            mpatches.Patch(
                facecolor=self.pal["After Norm"],
                edgecolor="k",
                linewidth=legend_linewidth,
                label="After Norm",
            ),
        ]
        variance_handles = [
            mlines.Line2D(
                [],
                [],
                color=self.pal["Before Norm"],
                linestyle="--",
                marker="o",
                linewidth=line_width,
                markersize=marker_size,
                label="Before Norm",
            ),
            mlines.Line2D(
                [],
                [],
                color=self.pal["After Norm"],
                linestyle="-",
                marker="o",
                linewidth=line_width,
                markersize=marker_size,
                label="After Norm",
            ),
        ]
        distance_handles = [
            mlines.Line2D(
                [0],
                [0],
                color=self.pal["Before Norm"],
                marker="o",
                linestyle="",
                markeredgecolor="k",
                markeredgewidth=0.5 if article_compact else 0.25,
                markersize=marker_size,
                label="Before Norm",
            ),
            mlines.Line2D(
                [0],
                [0],
                color=self.pal["After Norm"],
                marker="o",
                linestyle="",
                markeredgecolor="k",
                markeredgewidth=0.5 if article_compact else 0.25,
                markersize=marker_size,
                label="After Norm",
            ),
        ]

        self._plot_grouped_standalone_legends(
            ax=ax,
            legend_groups=[
                ("AUTO normalization score components", score_handles),
                ("QC RLE alignment stage", rle_handles),
                ("QC variance stabilization stage", variance_handles),
                ("QC structure distance stage", distance_handles),
            ],
            loc="upper left",
            start_bbox=(0.0, 1.0),
            row_gap=0.035,
            layout_cols=layout_cols,
            column_gap=0.12,
            max_item_rows=6,
            borderaxespad=0.0,
            handlelength=1.0 if article_compact else 1.8,
            handletextpad=0.3 if article_compact else 0.8,
            labelspacing=0.25 if article_compact else 0.5,
            borderpad=0.3 if article_compact else 0.4,
            fontsize=fontsize,
            title_fontsize=title_fontsize,
        )
        if article_compact:
            self._apply_article_legend_style(
                ax=ax,
                fontsize=fontsize,
                title_fontsize=title_fontsize,
            )
        return ax

    def plot_normalization_article_legend(self, ax: plt.Axes) -> plt.Axes:
        """Draw right-side grouped legends for the normalization article panel."""
        return self.plot_normalization_dashboard_legend(
            ax=ax,
            fontsize=pu.ARTICLE_LEGEND_FONTSIZE,
            title_fontsize=pu.ARTICLE_LEGEND_TITLE_FONTSIZE,
            article_compact=True,
            layout_cols=1,
        )

    def _plot_ecdf_overlay(
        self, metrics: dict[str, Any] | None = None, ax: plt.Axes | None = None
    ) -> plt.Figure | plt.Axes:
        """Plot Empirical Cumulative Distribution Function (eCDF) overlay.

        Visualizes intensity alignment with a legend in the upper left and
        QA metrics text box in the lower right.
        """
        import matplotlib.lines as mlines

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))
        else:
            fig = ax.figure
        pu.mark_preserve_alpha(ax)

        handles = []
        for label, obj in self.stages:
            log_d = su._extract_log2_target(obj)
            if log_d is None or log_d.empty:
                continue

            for col in log_d.columns:
                vals = log_d[col].dropna().values
                if len(vals) == 0:
                    continue

                vals_sorted = np.sort(vals)
                p = np.linspace(0, 1, len(vals_sorted))
                z = 2 if label == "After Norm" else 1
                ax.plot(
                    vals_sorted,
                    p,
                    color=self.pal[label],
                    alpha=0.2,
                    linewidth=1.0,
                    zorder=z,
                )

            ax.plot([], [], color=self.pal[label], label=label, linewidth=2)
            # Create handles for the legend
            handles.append(
                mlines.Line2D([], [], color=self.pal[label], label=label, linewidth=2)
            )

        self._apply_standard_format(
            ax=ax,
            title="eCDF Distribution Alignment",
            xlabel="Log2 Intensity",
            ylabel="Cumulative Probability",
            append_stage=False,
        )

        # 1. Inject Metrics Text Box (Lower Right)
        if metrics and "eCDF" in metrics:
            lines = ["Dist. Alignment (W / KS)"]
            for label in ["Before Norm", "After Norm"]:
                m_dict = metrics["eCDF"].get(label, {})
                if m_dict:
                    lines.append(
                        f"{label}: {m_dict.get('Wasserstein', 0):.2f} / "
                        f"{m_dict.get('KS', 0):.3f}"
                    )

            ax.text(
                0.96,
                0.02,
                "\n".join(lines),
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="bottom",
                horizontalalignment="right",
                clip_on=False,
                bbox=pu.ai_ready_text_bbox(pad=0.4),
                zorder=10,
            )

        # 2. Force Legend in Upper Left
        if handles:
            ax.legend(handles=handles)
            self._format_single_legend(
                ax=ax,
                group_title="Distribution alignment stage",
                loc="upper left",
                bbox_to_anchor=None,
            )

        return fig if ax is None else ax

    def plot_normalization_dashboard(self) -> object | None:
        """Combine score-aligned normalization diagnostics into a PW dashboard."""
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping summary grid.")
            return None

        pw.clear()

        auto_summary = self.norm.attrs.get("normalization_auto_summary")
        is_auto = bool(auto_summary)

        if is_auto:
            ax_auto = self.plot_normalization_score_summary(
                auto_summary=auto_summary,
                figsize=(5.5, 4.0),
                show_legend=False,
            )
            if ax_auto is None:
                ax_auto = pw.Brick(figsize=(4.5, 4.0), label="Auto_Score_Spacer")
                ax_auto.axis("off")

            ax_scorecard = pw.Brick(
                figsize=(4.8, 4.0), label="Norm_Preservation_Scorecard"
            )
            self.plot_normalization_preservation_scorecard(
                auto_summary=auto_summary,
                ax=ax_scorecard,
            )
            ax_legend = pw.Brick(
                figsize=(3.4, 4.0),
                label="normalization_dashboard_legend",
            )
            self.plot_normalization_dashboard_legend(ax=ax_legend)
            row1 = ax_auto | ax_scorecard | ax_legend

            ax_qc_variance = pw.Brick(figsize=(3.0, 4.0), label="QC_Variance")
            self._plot_qc_variance_stabilization(
                ax=ax_qc_variance,
                show_legend=False,
            )
            ax_qc_structure = pw.Brick(figsize=(3.0, 4.0), label="QC_Structure")
            self._plot_qc_structure_improvement(
                ax=ax_qc_structure,
                show_legend=False,
            )
            ax_sample_structure = pw.Brick(figsize=(3.0, 4.0), label="Sample_Structure")
            self._plot_sample_structure_preservation(ax_geom=ax_sample_structure)
            ax_qc_alignment = pw.Brick(figsize=(3.0, 4.0), label="QC_Alignment")
            self._plot_qc_rle_boxplot(
                ax=ax_qc_alignment,
                show_legend=False,
            )
            row2 = (
                ax_qc_alignment
                | ax_qc_variance
                | ax_qc_structure
                | ax_sample_structure
            )

            return row1 / row2

        ax_qc_alignment = pw.Brick(figsize=(4.0, 4.0), label="QC_Alignment")
        self._plot_qc_rle_boxplot(ax=ax_qc_alignment)
        ax_qc_variance = pw.Brick(figsize=(4.0, 4.0), label="QC_Variance")
        self._plot_qc_variance_stabilization(ax=ax_qc_variance)
        row1 = ax_qc_alignment | ax_qc_variance

        ax_qc_structure = pw.Brick(figsize=(4.0, 4.0), label="QC_Structure")
        self._plot_qc_structure_improvement(ax=ax_qc_structure)
        ax_sample_structure = pw.Brick(figsize=(4.0, 4.0), label="Sample_Structure")
        self._plot_sample_structure_preservation(ax_geom=ax_sample_structure)
        row2 = ax_qc_structure | ax_sample_structure

        return row1 / row2

    def plot_normalization_article_dashboard(self) -> object | None:
        """Create a compact AUTO normalization panel for manuscript figures."""
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping normalization article panel.")
            return None

        auto_summary = self.norm.attrs.get("normalization_auto_summary")
        if not auto_summary:
            return None

        pw.clear()
        panel_height = 1.75

        summary_ax = self.plot_normalization_score_summary(
            auto_summary=auto_summary,
            figsize=(1.42, panel_height),
            show_legend=False,
        )
        if summary_ax is None:
            summary_ax = pw.Brick(
                figsize=(1.42, panel_height), label="article_normalization_summary"
            )
            summary_ax.axis("off")
        self._apply_article_panel_format(
            summary_ax,
            title="Auto Normalization\nMethod Selection",
        )

        rle_ax = pw.Brick(
            figsize=(1.20, panel_height), label="article_normalization_rle"
        )
        self._plot_qc_rle_boxplot(
            ax=rle_ax,
            show_legend=False,
            article_compact=True,
        )
        self._apply_article_panel_format(
            rle_ax,
            title="QC RLE\nAlignment Change",
        )

        variance_ax = pw.Brick(
            figsize=(1.20, panel_height), label="article_normalization_variance"
        )
        self._plot_qc_variance_stabilization(
            ax=variance_ax,
            show_legend=False,
            article_compact=True,
        )
        self._apply_article_panel_format(
            variance_ax,
            title="QC Variance\nStabilization",
        )
        variance_ax.set_xlabel("Mean QC log2 intensity")
        variance_ax.set_ylabel("QC dispersion")

        structure_ax = pw.Brick(
            figsize=(1.20, panel_height), label="article_normalization_structure"
        )
        self._plot_qc_structure_improvement(
            ax=structure_ax,
            show_legend=False,
            article_compact=True,
        )
        self._apply_article_panel_format(
            structure_ax,
            title="QC Structure\nDistance Change",
        )
        structure_ax.set_ylabel("QC distance (log scale)")

        legend_ax = pw.Brick(
            figsize=(1.30, panel_height), label="article_normalization_legend"
        )
        self.plot_normalization_article_legend(ax=legend_ax)
        return summary_ax | rle_ax | variance_ax | structure_ax | legend_ax
