# src/pimqc/normalization.py
"""
Script purpose: Execute final normalization and diagnostic export.

apply_normalization() selects fixed method-specific transformations:
ROBUST_LOG_ONLY, TIC, Median, PQN, MDFC, log-first distribution alignment for
Quantile, or intrinsic glog transformation for VSN. Auto mode ranks these
log-like strategies with QC/RLE/MA change metrics and local
sample-neighborhood preservation.
execute_normalization() creates the output folder, drops blanks from the final
target matrix, saves the normalized dataset, calculates normalization quality
metrics, stores them on the returned MetaboInt object, and renders a dashboard
aligned with the Auto scoring dimensions.
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
from sklearn.manifold import trustworthiness
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
        "qc_precision_change_score": 0.35,
        "rle_alignment_change_score": 0.30,
        "ma_bias_change_score": 0.20,
        "sample_neighborhood_score": 0.15,
    }
    _AUTO_CENTERED_CHANGE_COMPONENTS = {
        "qc_precision_change_score",
        "rle_alignment_change_score",
        "ma_bias_change_score",
    }
    _AUTO_NEIGHBORHOOD_SCORE_COL = "sample_neighborhood_score"
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
    # Statistical Metrics for Normalization
    # ====================================================================
    @staticmethod
    def calc_ma_arrays(
        df_log: pd.DataFrame,
        flat_positions: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate flattened A and M values for MA-plot visualization.

        M is the feature-wise mean-centered log2 intensity, not a biological
        group-level fold change or a sample-to-QC fold change.

        Args:
            df_log: DataFrame of log2 intensities (Features x Samples).
            flat_positions: Optional flattened matrix positions to evaluate.

        Returns:
            a_flat, m_flat: Flattened 1D numpy arrays with NaNs removed.
        """
        # A: The average log2 intensity for each feature (X-axis)
        a_vals = df_log.mean(axis="columns")

        # M: The deviation of each sample from that feature's mean (Y-axis)
        m_df = df_log.sub(a_vals, axis="index")

        if flat_positions is None:
            # Optimized vectorized implementation using numpy repeat.
            a_flat = np.repeat(a_vals.values, m_df.shape[1])
            m_flat = m_df.values.flatten()
        else:
            total_size = m_df.shape[0] * m_df.shape[1]
            selected_positions = np.asarray(flat_positions, dtype=np.int64)
            selected_positions = selected_positions[
                (selected_positions >= 0) & (selected_positions < total_size)
            ]
            row_idx = selected_positions // m_df.shape[1]
            a_flat = a_vals.to_numpy(dtype=float)[row_idx]
            m_flat = m_df.to_numpy(dtype=float, copy=False).ravel()[selected_positions]

        # Filter out non-finite values to ensure metric and plot stability.
        valid = np.isfinite(a_flat) & np.isfinite(m_flat)
        return a_flat[valid], m_flat[valid]

    @classmethod
    def _select_ma_flat_positions(
        cls,
        log_df: pd.DataFrame,
        max_points: Optional[int] = 50000,
    ) -> np.ndarray | None:
        """Select deterministic intensity-stratified flattened MA positions.

        The returned flattened positions can be reused across candidate
        normalization outputs or plotting calls.
        """
        if max_points is None:
            return None

        n_rows, n_cols = log_df.shape
        total_size = n_rows * n_cols
        if total_size <= max_points:
            return None

        a_vals = log_df.mean(axis="columns").to_numpy(dtype=float)
        m_vals = log_df.sub(log_df.mean(axis="columns"), axis="index").to_numpy(
            dtype=float,
            copy=False,
        )
        a_flat = np.repeat(a_vals, n_cols)
        m_flat = m_vals.ravel()
        valid_positions = np.flatnonzero(np.isfinite(a_flat) & np.isfinite(m_flat))
        if valid_positions.size <= max_points:
            return None

        ranked_positions = valid_positions[np.argsort(a_flat[valid_positions])]
        selected_rank = np.linspace(
            0,
            ranked_positions.size - 1,
            num=max_points,
            dtype=np.int64,
        )
        return ranked_positions[selected_rank]

    def calc_norm_quality_metrics(
        self,
        raw_obj: core_classes.MetaboInt,
        norm_obj: core_classes.MetaboInt,
        max_points: Optional[int] = 50000,
    ) -> dict[str, Any]:
        """Calculate normalization-related metrics for technical reporting.

        Computes Jensen-Shannon Divergence (JSD), MA plot statistics
        (MAD and binned trend bias), and eCDF alignment (Wasserstein, KS).
        Supports statistical subsampling to prevent O(N log N) bottlenecks.
        """
        metrics = {
            "JSD": {"QC": {}, "Sample": {}},
            "MA": {"Before Norm": {}, "After Norm": {}},
            "eCDF": {"Before Norm": {}, "After Norm": {}},
        }

        log_raw = su._extract_log2_target(raw_obj)
        log_norm = su._extract_log2_target(norm_obj)

        if log_raw is None or log_norm is None:
            return metrics

        log_raw, log_norm = self._align_paired_log_matrices(log_raw, log_norm)
        if log_raw.empty or log_norm.empty:
            return metrics

        g_seed = self.attrs.get("global_seed", 123)

        # ==========================================
        # Robust inner closure for array subsampling
        # ==========================================
        def _subsample(
            arr: np.ndarray, max_n: Optional[int] = max_points
        ) -> np.ndarray:
            """Safely subsample 1D arrays, bypassed if max_n is None."""
            valid_arr = arr[~np.isnan(arr)]
            # Skip subsampling if max_n is None or array is small enough
            if max_n is not None and len(valid_arr) > max_n:
                np.random.seed(g_seed)
                return np.random.choice(valid_arr, size=max_n, replace=False)
            return valid_arr

        # 1. JSD Metrics (Density Alignment)
        qc_cols = raw_obj._qc.columns.intersection(log_raw.columns)
        sam_cols = raw_obj._actual_sample.columns.intersection(log_raw.columns)

        if not qc_cols.empty:
            raw_vals = _subsample(log_raw[qc_cols].values.flatten())
            norm_vals = _subsample(log_norm[qc_cols].values.flatten())
            qc_jsd = su.calc_jsd_similarity(raw_vals, norm_vals)

            metrics["JSD"]["QC"]["Before vs After"] = (
                float(qc_jsd.get("JSD", qc_jsd.get("jsd", np.nan)))
                if isinstance(qc_jsd, dict)
                else float(qc_jsd)
            )

        if not sam_cols.empty:
            raw_vals = _subsample(log_raw[sam_cols].values.flatten())
            norm_vals = _subsample(log_norm[sam_cols].values.flatten())
            sam_jsd = su.calc_jsd_similarity(raw_vals, norm_vals)

            metrics["JSD"]["Sample"]["Before vs After"] = (
                float(sam_jsd.get("JSD", sam_jsd.get("jsd", np.nan)))
                if isinstance(sam_jsd, dict)
                else float(sam_jsd)
            )

        # 2. MA Plot & 3. eCDF Distribution Metrics
        ma_eval_positions = self._select_ma_flat_positions(
            log_df=log_raw,
            max_points=max_points,
        )
        stages = [("Before Norm", log_raw), ("After Norm", log_norm)]
        for stage, df in stages:
            # ----------------------------------------------------
            # MA Metrics (MAD, binned trend bias, Spearman rho)
            # ----------------------------------------------------
            ma_values = self._calc_ma_metric_values(
                log_df=df,
                max_points=max_points,
                flat_positions=ma_eval_positions,
            )
            if np.isfinite(ma_values["ma_spread"]):
                metrics["MA"][stage] = {
                    "MAD": ma_values["ma_spread"],
                    "TrendBias": ma_values["ma_trend_bias"],
                    "Spearman": ma_values["ma_spearman"],
                }

            # ----------------------------------------------------
            # eCDF Metrics (Wasserstein distance, KS statistic)
            # ----------------------------------------------------
            pooled = df.values.flatten()
            pooled_ref = _subsample(pooled)  # Controlled by max_points

            if len(pooled_ref) > 0:
                w_dists, ks_dists = [], []

                eval_cols = df.columns
                # Subsample samples (columns) for global eCDF estimation
                if max_points is not None and len(eval_cols) > 200:
                    np.random.seed(g_seed)
                    eval_cols = np.random.choice(eval_cols, 200, replace=False)

                for col in eval_cols:
                    vals = df[col].dropna().values
                    if len(vals) > 0:
                        # Subsample intra-sample features if exceptionally huge
                        if max_points is not None and len(vals) > 10000:
                            np.random.seed(g_seed)
                            vals = np.random.choice(vals, 10000, replace=False)

                        w_dists.append(stats.wasserstein_distance(vals, pooled_ref))
                        ks_dists.append(stats.ks_2samp(vals, pooled_ref)[0])

                metrics["eCDF"][stage] = {
                    "Wasserstein": float(np.mean(w_dists)),
                    "KS": float(np.mean(ks_dists)),
                }
        return metrics

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
        try:
            float_val = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return float_val if np.isfinite(float_val) else float("nan")

    @staticmethod
    def _align_paired_log_matrices(
        before_log: pd.DataFrame,
        after_log: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Align paired log-like matrices to the same feature and sample order."""
        common_features = before_log.index.intersection(after_log.index, sort=False)
        common_samples = before_log.columns.intersection(after_log.columns, sort=False)
        return (
            before_log.loc[common_features, common_samples],
            after_log.loc[common_features, common_samples],
        )

    @staticmethod
    def _series_iqr(series: pd.Series) -> float:
        """Calculate the interquartile range of a numeric Series."""
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.empty:
            return float("nan")
        return float(clean.quantile(0.75) - clean.quantile(0.25))

    @staticmethod
    def _relative_change_lower_better(before: object, after: object) -> float:
        """Return relative before-to-after change for a lower-is-better metric."""
        before_val = MetaboIntNormalizer._finite_or_nan(before)
        after_val = MetaboIntNormalizer._finite_or_nan(after)
        if not np.isfinite(before_val) or not np.isfinite(after_val):
            return float("nan")
        if before_val <= np.finfo(float).eps:
            return 0.0 if after_val <= np.finfo(float).eps else -1.0
        return float((before_val - after_val) / before_val)

    @staticmethod
    def _weighted_mean_score(
        score_weights: list[tuple[float, float]],
        clip_values: bool = True,
    ) -> float:
        """Return a weighted mean over finite component scores."""
        weighted_sum = 0.0
        weight_sum = 0.0
        for score, weight in score_weights:
            score_val = MetaboIntNormalizer._finite_or_nan(score)
            if not np.isfinite(score_val) or weight <= 0:
                continue
            if clip_values:
                score_val = float(np.clip(score_val, 0.0, 1.0))
            weighted_sum += score_val * weight
            weight_sum += weight
        if weight_sum <= 0:
            return float("nan")
        return float(weighted_sum / weight_sum)

    @staticmethod
    def _rank_loss_from_distances(raw_dist: np.ndarray, norm_dist: np.ndarray) -> float:
        """Return 1 - Spearman rho between two pairwise distance vectors."""
        valid_mask = np.isfinite(raw_dist) & np.isfinite(norm_dist)
        if int(valid_mask.sum()) < 3:
            return float("nan")

        rho_val = stats.spearmanr(raw_dist[valid_mask], norm_dist[valid_mask])[0]
        rho_val = MetaboIntNormalizer._finite_or_nan(rho_val)
        if not np.isfinite(rho_val):
            return float("nan")
        return float(max(0.0, 1.0 - rho_val))

    @staticmethod
    def _robust_feature_zscore(df: pd.DataFrame) -> pd.DataFrame:
        """Scale each feature by median/MAD across samples for geometry checks."""
        feature_median = df.median(axis=1, skipna=True)
        centered = df.sub(feature_median, axis=0)
        feature_mad = centered.abs().median(axis=1, skipna=True)
        robust_scale = (feature_mad * 1.4826).replace(0.0, np.nan)

        z_df = centered.div(robust_scale, axis=0).replace([np.inf, -np.inf], np.nan)
        valid_feature = z_df.notna().sum(axis=1) >= 3
        return z_df.loc[valid_feature].fillna(0.0)

    @staticmethod
    def _calc_qc_centroid_residuals(
        log_df: pd.DataFrame,
        qc_cols: pd.Index,
    ) -> pd.Series:
        """Calculate QC-sample residuals against the pooled QC median profile."""
        if len(qc_cols) < 2:
            return pd.Series(dtype=float)

        qc_log = log_df[qc_cols].astype(float)
        qc_centroid = qc_log.median(axis=1, skipna=True)
        residual = qc_log.sub(qc_centroid, axis=0)

        residual_arr = residual.to_numpy(dtype=float, copy=True)
        residual_arr[~np.isfinite(residual_arr)] = np.nan
        sample_residual = np.sqrt(np.nanmedian(np.square(residual_arr), axis=0))

        residual_series = pd.Series(sample_residual, index=qc_cols, dtype=float)
        return residual_series.replace([np.inf, -np.inf], np.nan).dropna()

    @classmethod
    def _calc_qc_precision_values(
        cls,
        log_df: pd.DataFrame,
        qc_cols: pd.Index,
    ) -> dict[str, float]:
        """Calculate pooled-QC centroid compactness metrics."""
        metrics = {
            "qc_centroid_median": float("nan"),
            "qc_centroid_iqr": float("nan"),
        }
        if len(qc_cols) < 2:
            return metrics

        qc_residuals = cls._calc_qc_centroid_residuals(log_df, qc_cols)
        if qc_residuals.empty:
            return metrics

        metrics["qc_centroid_median"] = cls._finite_or_nan(qc_residuals.median())
        metrics["qc_centroid_iqr"] = cls._series_iqr(qc_residuals)
        return metrics

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

    @staticmethod
    def _calc_ma_trend_bias(
        a_vals: np.ndarray,
        m_vals: np.ndarray,
        n_bins: int = 30,
        min_bin_size: int = 20,
    ) -> float:
        """Estimate intensity-dependent MA trend amplitude from binned medians."""
        a_arr = np.asarray(a_vals, dtype=float)
        m_arr = np.asarray(m_vals, dtype=float)
        finite_mask = np.isfinite(a_arr) & np.isfinite(m_arr)
        a_arr = a_arr[finite_mask]
        m_arr = m_arr[finite_mask]
        if a_arr.size < max(3 * min_bin_size, 30):
            return float("nan")

        bin_count = min(n_bins, max(3, int(a_arr.size // min_bin_size)))
        if bin_count < 3:
            return float("nan")

        order = np.argsort(a_arr)
        trend_medians = []
        for bin_idx in np.array_split(order, bin_count):
            if bin_idx.size < 3:
                continue
            trend_medians.append(float(np.nanmedian(m_arr[bin_idx])))

        if len(trend_medians) < 3:
            return float("nan")
        trend_arr = np.asarray(trend_medians, dtype=float)
        trend_center = np.nanmedian(trend_arr)
        return float(np.nanmedian(np.abs(trend_arr - trend_center)))

    @classmethod
    def _calc_ma_metric_values(
        cls,
        log_df: pd.DataFrame,
        max_points: Optional[int] = 50000,
        flat_positions: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Calculate MA spread and binned-trend bias metrics."""
        metrics = {
            "ma_spread": float("nan"),
            "ma_trend_bias": float("nan"),
            "ma_spearman": float("nan"),
        }
        eval_positions = flat_positions
        if eval_positions is None:
            eval_positions = cls._select_ma_flat_positions(
                log_df=log_df,
                max_points=max_points,
            )

        a_vals, m_vals = cls.calc_ma_arrays(log_df, flat_positions=eval_positions)
        if len(m_vals) == 0:
            return metrics

        m_median = np.nanmedian(m_vals)
        metrics["ma_spread"] = float(np.nanmedian(np.abs(m_vals - m_median)))

        metrics["ma_trend_bias"] = cls._calc_ma_trend_bias(a_vals, m_vals)
        rho_val = stats.spearmanr(a_vals, m_vals)[0]
        metrics["ma_spearman"] = cls._finite_or_nan(rho_val)
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
        empty = {
            "geometry": {
                "raw_dist": np.array([], dtype=float),
                "norm_dist": np.array([], dtype=float),
                "rank_loss": float("nan"),
                "median_relative_delta": float("nan"),
                "neighborhood_trustworthiness": float("nan"),
                "n_neighbors": float("nan"),
            },
        }

        log_raw = su._extract_log2_target(raw_obj)
        log_norm = su._extract_log2_target(norm_obj)
        if log_raw is None or log_norm is None:
            return empty

        sample_cols = (
            raw_obj._actual_sample.columns.intersection(log_raw.columns)
            .intersection(log_norm.columns)
            .sort_values()
        )
        if len(sample_cols) < 3:
            return empty

        raw_sample = log_raw[sample_cols].astype(float)
        norm_sample = log_norm[sample_cols].astype(float)
        valid_features = raw_sample.index.intersection(norm_sample.index)
        raw_sample = raw_sample.loc[valid_features]
        norm_sample = norm_sample.loc[valid_features]

        finite_rows = np.isfinite(raw_sample.to_numpy()).any(axis=1) & np.isfinite(
            norm_sample.to_numpy()
        ).any(axis=1)
        raw_sample = raw_sample.loc[finite_rows]
        norm_sample = norm_sample.loc[finite_rows]

        if raw_sample.empty:
            return empty

        if max_features is not None and raw_sample.shape[0] > max_features:
            rng = np.random.default_rng(seed)
            keep_idx = rng.choice(raw_sample.index, size=max_features, replace=False)
            raw_sample = raw_sample.loc[keep_idx]
            norm_sample = norm_sample.loc[keep_idx]

        raw_z = cls._robust_feature_zscore(raw_sample)
        norm_z = cls._robust_feature_zscore(norm_sample)
        valid_z_features = raw_z.index.intersection(norm_z.index)
        raw_z = raw_z.loc[valid_z_features]
        norm_z = norm_z.loc[valid_z_features]

        if raw_z.shape[0] >= 2:
            # Divide by sqrt(n_features) so Euclidean distances remain comparable
            # across feature subsampling levels.
            scale = np.sqrt(float(raw_z.shape[0]))
            raw_geom_dist = pdist(raw_z.T.to_numpy(dtype=float), metric="euclidean")
            norm_geom_dist = pdist(norm_z.T.to_numpy(dtype=float), metric="euclidean")
            raw_geom_dist = raw_geom_dist / scale
            norm_geom_dist = norm_geom_dist / scale

            valid_geom = np.isfinite(raw_geom_dist) & np.isfinite(norm_geom_dist)
            raw_geom_dist = raw_geom_dist[valid_geom]
            norm_geom_dist = norm_geom_dist[valid_geom]
            if raw_geom_dist.size > 0:
                denominator = np.maximum(np.abs(raw_geom_dist), np.finfo(float).eps)
                relative_delta = np.abs(norm_geom_dist - raw_geom_dist) / denominator
                empty["geometry"] = {
                    "raw_dist": raw_geom_dist,
                    "norm_dist": norm_geom_dist,
                    "rank_loss": cls._rank_loss_from_distances(
                        raw_geom_dist, norm_geom_dist
                    ),
                    "median_relative_delta": float(np.median(relative_delta)),
                    "neighborhood_trustworthiness": float("nan"),
                    "n_neighbors": float("nan"),
                }

            n_samples = raw_z.shape[1]
            if n_samples >= 3:
                n_neighbors = max(1, min(5, (n_samples - 1) // 2))
                try:
                    trust_value = trustworthiness(
                        X=raw_z.T.to_numpy(dtype=float),
                        X_embedded=norm_z.T.to_numpy(dtype=float),
                        n_neighbors=n_neighbors,
                        metric="euclidean",
                    )
                except (ValueError, FloatingPointError):
                    trust_value = float("nan")

                empty["geometry"]["neighborhood_trustworthiness"] = cls._finite_or_nan(
                    trust_value
                )
                empty["geometry"]["n_neighbors"] = float(n_neighbors)

        return empty

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

    def _sample_structure_guardrail_metrics(
        self,
        norm_obj: core_classes.MetaboInt,
        max_features: Optional[int] = 5000,
    ) -> dict[str, float]:
        """Quantify local sample-neighborhood preservation without labels."""
        metrics = {
            "robust_distance_rank_loss": float("nan"),
            "robust_distance_relative_delta": float("nan"),
            "sample_neighborhood_trustworthiness": float("nan"),
        }

        structure = self._calc_sample_structure_arrays(
            raw_obj=self,
            norm_obj=norm_obj,
            max_features=max_features,
            seed=int(self.attrs.get("global_seed", 123)),
        )
        geom_metrics = structure["geometry"]

        metrics["robust_distance_rank_loss"] = self._finite_or_nan(
            geom_metrics.get("rank_loss")
        )
        metrics["robust_distance_relative_delta"] = self._finite_or_nan(
            geom_metrics.get("median_relative_delta")
        )
        metrics["sample_neighborhood_trustworthiness"] = self._finite_or_nan(
            geom_metrics.get("neighborhood_trustworthiness")
        )

        return metrics

    def calc_auto_norm_candidate_metrics(
        self,
        norm_obj: core_classes.MetaboInt,
        max_points: Optional[int] = 50000,
        ma_eval_positions: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Calculate component scores for Auto normalization."""
        metrics = {
            "qc_centroid_median_before": float("nan"),
            "qc_centroid_median_after": float("nan"),
            "qc_centroid_iqr_before": float("nan"),
            "qc_centroid_iqr_after": float("nan"),
            "rle_center_offset_before": float("nan"),
            "rle_center_offset_after": float("nan"),
            "rle_spread_before": float("nan"),
            "rle_spread_after": float("nan"),
            "ma_spread_before": float("nan"),
            "ma_spread_after": float("nan"),
            "ma_trend_bias_before": float("nan"),
            "ma_trend_bias_after": float("nan"),
            "robust_distance_rank_loss": float("nan"),
            "robust_distance_relative_delta": float("nan"),
            "sample_neighborhood_trustworthiness": float("nan"),
            "qc_precision_change_score": float("nan"),
            "rle_alignment_change_score": float("nan"),
            "ma_bias_change_score": float("nan"),
            self._AUTO_NEIGHBORHOOD_SCORE_COL: float("nan"),
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
            qc_before = self._calc_qc_precision_values(log_raw, qc_cols)
            qc_after = self._calc_qc_precision_values(log_norm, qc_cols)
            metrics["qc_centroid_median_before"] = qc_before["qc_centroid_median"]
            metrics["qc_centroid_median_after"] = qc_after["qc_centroid_median"]
            metrics["qc_centroid_iqr_before"] = qc_before["qc_centroid_iqr"]
            metrics["qc_centroid_iqr_after"] = qc_after["qc_centroid_iqr"]

            metrics["qc_precision_change_score"] = (
                MetaboIntNormalizer._weighted_mean_score(
                    [
                        (
                            MetaboIntNormalizer._relative_change_lower_better(
                                qc_before["qc_centroid_median"],
                                qc_after["qc_centroid_median"],
                            ),
                            2.0,
                        ),
                        (
                            MetaboIntNormalizer._relative_change_lower_better(
                                qc_before["qc_centroid_iqr"],
                                qc_after["qc_centroid_iqr"],
                            ),
                            1.0,
                        ),
                    ],
                    clip_values=False,
                )
            )

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
                            MetaboIntNormalizer._relative_change_lower_better(
                                rle_before["rle_center_offset"],
                                rle_after["rle_center_offset"],
                            ),
                            3.0,
                        ),
                        (
                            MetaboIntNormalizer._relative_change_lower_better(
                                rle_before["rle_spread"],
                                rle_after["rle_spread"],
                            ),
                            2.0,
                        ),
                    ],
                    clip_values=False,
                )
            )

        ma_before = self._calc_ma_metric_values(
            log_raw,
            max_points=max_points,
            flat_positions=ma_eval_positions,
        )
        ma_after = self._calc_ma_metric_values(
            log_norm,
            max_points=max_points,
            flat_positions=ma_eval_positions,
        )
        metrics["ma_spread_before"] = ma_before["ma_spread"]
        metrics["ma_spread_after"] = ma_after["ma_spread"]
        metrics["ma_trend_bias_before"] = ma_before["ma_trend_bias"]
        metrics["ma_trend_bias_after"] = ma_after["ma_trend_bias"]
        metrics["ma_bias_change_score"] = MetaboIntNormalizer._weighted_mean_score(
            [
                (
                    MetaboIntNormalizer._relative_change_lower_better(
                        ma_before["ma_spread"],
                        ma_after["ma_spread"],
                    ),
                    7.0,
                ),
                (
                    MetaboIntNormalizer._relative_change_lower_better(
                        ma_before["ma_trend_bias"],
                        ma_after["ma_trend_bias"],
                    ),
                    3.0,
                ),
            ],
            clip_values=False,
        )

        metrics.update(self._sample_structure_guardrail_metrics(norm_obj))
        trustworthiness_score = self._finite_or_nan(
            metrics["sample_neighborhood_trustworthiness"]
        )
        if np.isfinite(trustworthiness_score):
            metrics[self._AUTO_NEIGHBORHOOD_SCORE_COL] = float(
                0.5 * np.clip(trustworthiness_score, 0.0, 1.0)
            )
        return metrics

    @classmethod
    def _score_auto_norm_candidates(
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
                if score_col == cls._AUTO_NEIGHBORHOOD_SCORE_COL:
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
            "qc_precision_change_score",
            "rle_alignment_change_score",
            "ma_bias_change_score",
            cls._AUTO_NEIGHBORHOOD_SCORE_COL,
            "selection_margin",
        ]
        summary = score_df.reindex(columns=summary_cols).copy()
        summary = summary.rename(columns={"auto_score": "overall_score"})

        for col in [
            "overall_score",
            "qc_precision_change_score",
            "rle_alignment_change_score",
            "ma_bias_change_score",
            cls._AUTO_NEIGHBORHOOD_SCORE_COL,
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
            "qc_precision_change_score",
            "rle_alignment_change_score",
            "ma_bias_change_score",
            cls._AUTO_NEIGHBORHOOD_SCORE_COL,
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

    def _apply_normalization_strategy(
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

    def _apply_auto_normalization(
        self, df_target: pd.DataFrame
    ) -> "MetaboIntNormalizer":
        """Evaluate fixed strategies and select the best normalization method."""
        logger.info(
            "Auto normalization evaluates fixed strategies on a common log-like "
            "view using before-to-after QC/RLE/MA improvements and local "
            "sample-neighborhood preservation."
        )

        records: list[dict[str, object]] = []
        candidate_outputs: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
        log_raw_eval = self._extract_auto_eval_target(self)
        if log_raw_eval is not None:
            eval_cols = df_target.columns.intersection(log_raw_eval.columns, sort=False)
            log_raw_eval = log_raw_eval.loc[:, eval_cols]
        ma_eval_positions = (
            self._select_ma_flat_positions(log_raw_eval, max_points=50000)
            if log_raw_eval is not None
            else None
        )

        for method in self._AUTO_CANDIDATES:
            apply_external_log = self._uses_external_log_for_method(method)
            logger.info(
                f"Evaluating Auto normalization candidate: {method} "
                f"(log_transform={apply_external_log})."
            )
            try:
                df_norm, meta_stamps = self._apply_normalization_strategy(
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
                    max_points=50000,
                    ma_eval_positions=ma_eval_positions,
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

        scored_candidates = self._score_auto_norm_candidates(records)
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
            return self._apply_auto_normalization(df_target)

        apply_external_log = self._uses_external_log_for_method(method)
        df_norm, meta_stamps = self._apply_normalization_strategy(
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

        # Retrieve the comprehensive QA metrics from the data passport
        quality_metrics = self.attrs.get("normalization_quality_metrics", {})
        if quality_metrics:
            metrics["quality_metrics"] = quality_metrics

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

        # 3. QA Metrics Engine
        logger.info("Calculating normalization-related metrics...")
        quality_metrics = self.calc_norm_quality_metrics(
            raw_obj=self, norm_obj=clean_obj, max_points=50000
        )
        clean_obj.attrs["normalization_quality_metrics"] = quality_metrics

        # 4. Visualization Phase (2-Stage)
        logger.info("Generating diagnostic plots for normalization...")
        vis = MetaboVisualizerNormalizer(raw_obj=self, norm_obj=clean_obj)

        grid_path = None
        fig_grid = vis.plot_normalization_summary_grid(metrics=quality_metrics)
        if fig_grid:
            grid_path = os.path.join(
                output_dir, f"Normalization_Dashboard_{suffix}.svg"
            )
            vis.save_and_show_pw(pw_obj=fig_grid, file_path=grid_path)

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
        self.pal = {"Before Norm": "tab:gray", "After Norm": "tab:red"}

    def plot_auto_selection_stacked_bar(
        self,
        auto_summary: list[dict[str, Any]] | pd.DataFrame | None = None,
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

        score_cols = [
            "qc_precision_change_score",
            "rle_alignment_change_score",
            "ma_bias_change_score",
            "sample_neighborhood_score",
        ]
        label_map = {
            "qc_precision_change_score": "QC precision change",
            "rle_alignment_change_score": "RLE alignment change",
            "ma_bias_change_score": "MA bias change",
            "sample_neighborhood_score": "Sample neighborhood",
        }
        contribution_weights = MetaboIntNormalizer._AUTO_SCORE_COMPONENT_WEIGHTS
        color_map = {
            "qc_precision_change_score": "tab:red",
            "rle_alignment_change_score": "#f2a6a6",
            "ma_bias_change_score": "#595959",
            "sample_neighborhood_score": "#bdbdbd",
        }
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

        ax = pw.Brick(figsize=(4.0, 4.0), label="auto_norm_stacked_bar")
        y_pos = np.arange(len(plot_df))
        left = np.zeros(len(plot_df), dtype=float)

        for col in score_cols:
            contribution_col = f"contribution_{col}"
            values = plot_df[contribution_col].to_numpy(dtype=float)
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
            left += values

        for i, row in plot_df.iterrows():
            total_score = row["overall_score"]
            if row["status"] == "ok" and np.isfinite(total_score):
                label = f"{float(total_score):.3f}"
                ax.text(
                    min(float(left[i]) + 0.015, 1.02),
                    y_pos[i],
                    label,
                    va="center",
                    ha="left",
                    fontsize=9,
                    color="0.15",
                )
            else:
                ax.text(
                    0.015,
                    y_pos[i],
                    "failed",
                    va="center",
                    ha="left",
                    fontsize=9,
                    color="0.45",
                    style="italic",
                )

        selected_method = self.norm.attrs.get("auto_selected_method", "N/A")
        selected_score = self.norm.attrs.get("auto_selected_score", np.nan)
        if isinstance(selected_score, (int, float, np.floating)) and np.isfinite(
            selected_score
        ):
            title = (
                "Auto Normalization Selection\n"
                f"{_method_label(selected_method)} | Overall {selected_score:.3f}"
            )
        else:
            title = f"Auto Normalization Selection\n{_method_label(selected_method)}"

        ax.set_yticks(y_pos)
        ax.set_yticklabels([_method_label(m) for m in plot_df["method"]])

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
            group_title="Score Component",
            legend_cols=1,
        )
        ax.set_xlim(0, 1.12)
        ax.set_ylim(-0.5, len(plot_df) - 0.5)
        ax.invert_yaxis()
        self._apply_standard_format(
            ax=ax,
            title=title,
            xlabel="Weighted contribution to overall score",
            append_stage=False,
        )
        return ax

    @staticmethod
    def _add_metric_note(
        ax: plt.Axes, lines: list[str], loc: str = "lower right"
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
            fontsize=9,
            verticalalignment=verticalalignment,
            horizontalalignment=horizontalalignment,
            clip_on=False,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="none",
                alpha=0.70,
            ),
        )

    def _plot_qc_centroid_residuals(
        self, ax: plt.Axes | None = None, max_points: int = 50000
    ) -> plt.Figure | plt.Axes:
        """Plot QC-sample compactness against the pooled QC centroid."""
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(4, 4))
        else:
            current_ax = ax
            fig = current_ax.figure

        plot_records = []
        stats_lines = []
        for label, obj in self.stages:
            log_d = su._extract_log2_target(obj)
            if log_d is None or log_d.empty:
                continue

            qc_cols = obj._qc.columns.intersection(log_d.columns)
            if len(qc_cols) < 2:
                continue

            qc_residuals = MetaboIntNormalizer._calc_qc_centroid_residuals(
                log_d, qc_cols
            )
            if qc_residuals.empty:
                continue

            qc_iqr = qc_residuals.quantile(0.75) - qc_residuals.quantile(0.25)
            stats_lines.append(
                f"{label.replace(' Norm', '')}: {qc_residuals.median():.3f} / {qc_iqr:.3f}"
            )

            if len(qc_residuals) > max_points:
                qc_residuals = qc_residuals.sample(
                    n=max_points,
                    random_state=int(self.norm.attrs.get("global_seed", 123)),
                )

            plot_records.append(
                pd.DataFrame(
                    {
                        "Stage": label,
                        "QC centroid residual": qc_residuals.to_numpy(),
                    }
                )
            )

        if plot_records:
            plot_df = pd.concat(plot_records, ignore_index=True)
            sns.boxplot(
                data=plot_df,
                x="Stage",
                y="QC centroid residual",
                hue="Stage",
                order=["Before Norm", "After Norm"],
                hue_order=["Before Norm", "After Norm"],
                palette=self.pal,
                showfliers=False,
                ax=current_ax,
                width=0.56,
                linewidth=1.2,
                legend=False,
            )
            sns.stripplot(
                data=plot_df,
                x="Stage",
                y="QC centroid residual",
                hue="Stage",
                order=["Before Norm", "After Norm"],
                hue_order=["Before Norm", "After Norm"],
                palette=self.pal,
                dodge=False,
                jitter=0.12,
                alpha=0.65,
                size=3.0,
                edgecolor="k",
                linewidth=0.3,
                ax=current_ax,
                legend=False,
                zorder=4,
            )
        else:
            current_ax.text(
                0.5,
                0.5,
                "Insufficient QC data",
                transform=current_ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )

        note_lines = ["Median / IQR", *stats_lines] if stats_lines else []
        self._add_metric_note(current_ax, note_lines, loc="upper right")
        self._apply_standard_format(
            ax=current_ax,
            title="Pooled QC Precision",
            xlabel="Pipeline Stage",
            ylabel="QC centroid residual",
            append_stage=False,
        )
        return fig if ax is None else current_ax

    def _plot_qc_rle_boxplot(
        self, ax: plt.Axes | None = None, max_points: int = 50000
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
            x_base = np.arange(len(metric_order), dtype=float)
            bar_width = 0.34
            offsets = {
                "Before Norm": -bar_width / 2,
                "After Norm": bar_width / 2,
            }
            bar_label_records = []

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

                bar_container = current_ax.bar(
                    x_base + offsets[stage],
                    values,
                    yerr=np.vstack([lower_errors, upper_errors]),
                    width=bar_width,
                    color=self.pal[stage],
                    edgecolor="k",
                    linewidth=1.0,
                    alpha=0.78,
                    label=stage,
                    zorder=3,
                    error_kw={
                        "ecolor": "0.20",
                        "elinewidth": 1.0,
                        "capsize": 3.0,
                        "capthick": 1.0,
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
                            "x": patch.get_x() + patch.get_width() / 2.0,
                            "value": value,
                            "y": value + upper_error,
                        }
                    )

            y_max = float(np.nanmax(plot_df[["Value", "Q75"]].to_numpy(dtype=float)))
            y_upper = y_max * 1.35 if y_max > 0 else 1.0
            current_ax.set_ylim(0, y_upper)
            label_offset = y_upper * 0.025
            for label_record in bar_label_records:
                value = MetaboIntNormalizer._finite_or_nan(label_record["value"])
                if not np.isfinite(value):
                    continue
                current_ax.text(
                    label_record["x"],
                    label_record["y"] + label_offset,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="0.15",
                    zorder=4,
                )
            current_ax.set_xlim(-0.5, len(metric_order) - 0.5)
            current_ax.set_xticks(x_base)
            current_ax.set_xticklabels(["RLE center\noffset", "RLE\nspread"])

            bracket_top = y_upper
            for metric_idx, metric in enumerate(metric_order):
                p_val = self._paired_wilcoxon_pvalue(
                    sample_metric_values.get("Before Norm", {}).get(metric),
                    sample_metric_values.get("After Norm", {}).get(metric),
                )
                y_level = y_max * (1.12 + 0.10 * metric_idx)
                y_level = max(y_level, y_upper * (0.70 + 0.08 * metric_idx))
                bracket_top = max(bracket_top, y_level * 1.08)
                self._add_pairwise_significance(
                    ax=current_ax,
                    x_left=x_base[metric_idx] + offsets["Before Norm"],
                    x_right=x_base[metric_idx] + offsets["After Norm"],
                    y=y_level,
                    text=self._pvalue_to_stars(p_val),
                    height=y_upper * 0.025,
                )
            current_ax.set_ylim(0, bracket_top)
        else:
            current_ax.text(
                0.5,
                0.5,
                "Insufficient QC data",
                transform=current_ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )

        self._apply_standard_format(
            ax=current_ax,
            title="QC RLE Alignment",
            xlabel="QC RLE Metric",
            ylabel="Metric Value",
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
            fontsize=10,
            color="0.20",
            clip_on=False,
            zorder=6,
        )

    def _plot_stage_legend_atom(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Render a standalone legend for before/after normalization stages."""
        import matplotlib.patches as mpatches

        if ax is None:
            _, ax = plt.subplots(figsize=(0.9, 4.0))

        ax.axis("off")
        handles = [
            mpatches.Patch(
                facecolor=self.pal["Before Norm"],
                edgecolor="k",
                linewidth=1.0,
                alpha=0.78,
                label="Before Norm",
            ),
            mpatches.Patch(
                facecolor=self.pal["After Norm"],
                edgecolor="k",
                linewidth=1.0,
                alpha=0.78,
                label="After Norm",
            ),
        ]
        ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.5))
        self._format_single_legend(
            ax=ax,
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            group_title="Stage",
            legend_cols=1,
        )
        return ax

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
                        fontsize=9,
                        verticalalignment="bottom",
                        horizontalalignment="right",
                        clip_on=False,
                        bbox=dict(
                            boxstyle="round,pad=0.4",
                            facecolor="white",
                            edgecolor="none",
                            alpha=0.6,
                        ),
                    )

            if current_ax.get_legend_handles_labels()[0]:
                current_ax.legend(loc="best")
                self._format_single_legend(current_ax, group_title="Stage")

        if return_fig:
            plt.tight_layout()
            return fig
        return ax_qc, ax_sample

    def _plot_ma_scatter(
        self,
        metrics: dict[str, Any] | None = None,
        ax_before: plt.Axes | None = None,
        ax_after: plt.Axes | None = None,
        cax: plt.Axes | None = None,
        max_plot_points: Optional[int] = 200000,
    ) -> plt.Figure | tuple[plt.Axes, plt.Axes]:
        """Plot true MA hexbin scatter using a continuous density scale."""
        return_fig = False
        if ax_before is None or ax_after is None:
            fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(8, 4))
            return_fig = True

        color_map = pu.custom_linear_cmap(
            color_list=["white", "tab:red"], n_colors=256, cmin=0.1, cmax=1.0
        )

        stages_data = []
        all_a_vals = []

        for label, obj in self.stages:
            log_d = su._extract_log2_target(obj)
            if log_d is not None and not log_d.empty:
                ma_plot_positions = MetaboIntNormalizer._select_ma_flat_positions(
                    log_df=log_d,
                    max_points=max_plot_points,
                )
                a_vals, m_vals = MetaboIntNormalizer.calc_ma_arrays(
                    log_d,
                    flat_positions=ma_plot_positions,
                )
                stages_data.append((label, a_vals, m_vals))
                all_a_vals.extend(a_vals)
            else:
                stages_data.append((label, [], []))

        if all_a_vals:
            a_min, a_max = np.nanmin(all_a_vals), np.nanmax(all_a_vals)
            margin_x = (a_max - a_min) * 0.08
            extent = (a_min - margin_x, a_max + margin_x, -5, 5)
        else:
            extent = (0, 25, -5, 5)

        hb_list = []
        for i, (label, x_val, y_val) in enumerate(stages_data):
            current_ax = ax_before if i == 0 else ax_after
            stage_title = {
                "Before Norm": "Before Normalization",
                "After Norm": "After Normalization",
            }.get(label, label)
            if len(x_val) > 0:
                hb = current_ax.hexbin(
                    x=x_val,
                    y=y_val,
                    gridsize=80,
                    extent=extent,
                    cmap=color_map,
                    mincnt=1,
                    bins="log",
                    linewidths=0,
                    alpha=0.95,
                )
                hb_list.append(hb)

                if metrics and "MA" in metrics and label in metrics["MA"]:
                    m_dict = metrics["MA"][label]
                    annot_text = (
                        f"Spread (MAD): {m_dict.get('MAD', 0):.3f}\n"
                        f"Trend bias: {m_dict.get('TrendBias', 0):.3f}"
                    )
                    current_ax.text(
                        0.96,
                        0.02,
                        annot_text,
                        transform=current_ax.transAxes,
                        fontsize=9,
                        verticalalignment="bottom",
                        horizontalalignment="right",
                        clip_on=False,
                        bbox=dict(
                            boxstyle="round,pad=0.4",
                            facecolor="white",
                            edgecolor="none",
                            alpha=0.6,
                        ),
                    )

            current_ax.axhline(0, color="k", linestyle="--", linewidth=1.5)

            self._apply_standard_format(
                ax=current_ax,
                title=f"MA Plot ({stage_title})",
                xlabel="Average Log2 Intensity (A)",
                ylabel="Mean-centered Log2 Intensity (M)",
                append_stage=False,
            )
            current_ax.set_xlim(extent[0], extent[1])
            current_ax.set_ylim(extent[2], extent[3])

        if cax is not None and hb_list:
            cb = plt.colorbar(hb_list[-1], cax=cax)
            cb.set_label("Log10(Count)")
            cb.outline.set_linewidth(1.0)

        if return_fig:
            plt.tight_layout()
            return fig
        return ax_before, ax_after

    def _plot_structure_scatter(
        self,
        ax: plt.Axes | None = None,
        raw_dist: np.ndarray | None = None,
        norm_dist: np.ndarray | None = None,
        title: str = "Sample Structure Preservation",
        xlabel: str = "Raw sample distance",
        ylabel: str = "Normalized sample distance",
        note_lines: list[str] | None = None,
        max_pairs: int = 20000,
    ) -> plt.Figure | plt.Axes:
        """Plot raw-vs-normalized pairwise sample distances."""
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(4, 4))
        else:
            current_ax = ax
            fig = current_ax.figure

        raw_dist = np.asarray(raw_dist if raw_dist is not None else [], dtype=float)
        norm_dist = np.asarray(norm_dist if norm_dist is not None else [], dtype=float)
        valid_dist = np.isfinite(raw_dist) & np.isfinite(norm_dist)
        raw_dist = raw_dist[valid_dist]
        norm_dist = norm_dist[valid_dist]

        if raw_dist.size == 0 or norm_dist.size == 0:
            current_ax.text(
                0.5,
                0.5,
                "Insufficient Sample pairs",
                transform=current_ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
            self._apply_standard_format(
                ax=current_ax,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                append_stage=False,
            )
            return fig if ax is None else current_ax

        if raw_dist.size > max_pairs:
            rng = np.random.default_rng(self.norm.attrs.get("global_seed", 123))
            keep = rng.choice(raw_dist.size, size=max_pairs, replace=False)
            x_plot = raw_dist[keep]
            y_plot = norm_dist[keep]
        else:
            x_plot = raw_dist
            y_plot = norm_dist

        current_ax.scatter(
            x_plot,
            y_plot,
            s=12,
            color="tab:red",
            alpha=0.35,
            edgecolors="none",
            rasterized=True,
        )

        lim_max = float(np.nanmax([raw_dist.max(), norm_dist.max(), 1.0]))
        lim_pad = max(lim_max * 0.04, 0.02)
        current_ax.plot(
            [0, lim_max + lim_pad],
            [0, lim_max + lim_pad],
            color="0.25",
            linestyle="--",
            linewidth=1.2,
            zorder=1,
        )
        current_ax.set_xlim(0, lim_max + lim_pad)
        current_ax.set_ylim(0, lim_max + lim_pad)
        current_ax.set_aspect("equal", adjustable="box")

        self._add_metric_note(current_ax, note_lines or [])
        self._apply_standard_format(
            ax=current_ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            append_stage=False,
        )
        return fig if ax is None else current_ax

    def _plot_sample_structure_preservation(
        self,
        ax_geom: plt.Axes | None = None,
        max_features: int = 5000,
        max_pairs: int = 20000,
    ) -> plt.Axes | plt.Figure:
        """Plot robust geometric sample-structure preservation."""
        if ax_geom is None:
            created_fig, ax_geom = plt.subplots(figsize=(4, 4))
        else:
            created_fig = None

        structure = MetaboIntNormalizer._calc_sample_structure_arrays(
            raw_obj=self.raw,
            norm_obj=self.norm,
            max_features=max_features,
            seed=int(self.norm.attrs.get("global_seed", 123)),
        )
        geom_metrics = structure["geometry"]

        geom_rank_loss = MetaboIntNormalizer._finite_or_nan(
            geom_metrics.get("rank_loss")
        )
        geom_rho = 1.0 - geom_rank_loss if np.isfinite(geom_rank_loss) else float("nan")
        geom_delta = MetaboIntNormalizer._finite_or_nan(
            geom_metrics.get("median_relative_delta")
        )
        trust_value = MetaboIntNormalizer._finite_or_nan(
            geom_metrics.get("neighborhood_trustworthiness")
        )
        n_neighbors = MetaboIntNormalizer._finite_or_nan(
            geom_metrics.get("n_neighbors")
        )
        geom_note = []
        if np.isfinite(trust_value):
            if np.isfinite(n_neighbors):
                geom_note.append(
                    f"Trustworthiness (k={int(n_neighbors)}): {trust_value:.3f}"
                )
            else:
                geom_note.append(f"Trustworthiness: {trust_value:.3f}")
        if np.isfinite(geom_rho):
            geom_note.append(f"Distance rank rho: {geom_rho:.3f}")
        if np.isfinite(geom_delta):
            geom_note.append(f"Median rel. |delta d|: {geom_delta:.3f}")

        self._plot_structure_scatter(
            ax=ax_geom,
            raw_dist=geom_metrics["raw_dist"],
            norm_dist=geom_metrics["norm_dist"],
            title="Sample Structure Preservation",
            xlabel="Raw robust distance",
            ylabel="Normalized robust distance",
            note_lines=geom_note,
            max_pairs=max_pairs,
        )

        if created_fig is not None:
            plt.tight_layout()
            return created_fig
        return ax_geom

    def _plot_distance_preservation(
        self,
        ax: plt.Axes | None = None,
        max_features: int = 5000,
        max_pairs: int = 20000,
    ) -> plt.Figure | plt.Axes:
        """Backward-compatible wrapper for sample-neighborhood preservation."""
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(4, 4))
        else:
            current_ax = ax
            fig = current_ax.figure
        self._plot_sample_structure_preservation(
            ax_geom=current_ax,
            max_features=max_features,
            max_pairs=max_pairs,
        )
        return fig if ax is None else current_ax

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
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.6,
                ),
            )

        # 2. Force Legend in Upper Left
        if handles:
            ax.legend(handles=handles)
            self._format_single_legend(
                ax=ax,
                group_title="Stage",
                loc="upper left",
                bbox_to_anchor=None,
            )

        return fig if ax is None else ax

    def plot_normalization_summary_grid(self, metrics: dict[str, Any]) -> object | None:
        """Combine score-aligned normalization diagnostics into a PW dashboard."""
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping summary grid.")
            return None

        pw.clear()

        auto_summary = self.norm.attrs.get("normalization_auto_summary")
        is_auto = bool(auto_summary)

        # Row 1 in Auto mode: Auto score | QC Precision | QC Alignment | Legend.
        # Row 1 in fixed-method mode: QC Precision | QC Alignment | Legend.
        qc_panel_size = (3.7, 4.0) if is_auto else (5.6, 4.0)
        ax_qc_precision = pw.Brick(figsize=qc_panel_size, label="QC_Precision")
        self._plot_qc_centroid_residuals(ax=ax_qc_precision)

        ax_qc_alignment = pw.Brick(figsize=qc_panel_size, label="QC_Alignment")
        self._plot_qc_rle_boxplot(ax=ax_qc_alignment)

        ax_stage_legend = pw.Brick(figsize=(0.9, 4.0), label="Stage_Legend")
        self._plot_stage_legend_atom(ax=ax_stage_legend)

        if is_auto:
            ax_auto = self.plot_auto_selection_stacked_bar(auto_summary=auto_summary)
            if ax_auto is None:
                ax_auto = pw.Brick(figsize=(4.0, 4.0), label="Auto_Score_Spacer")
                ax_auto.axis("off")
            row1 = ax_auto | ax_qc_precision | ax_qc_alignment | ax_stage_legend
        else:
            row1 = ax_qc_precision | ax_qc_alignment | ax_stage_legend

        # Row 2: MA before | MA after | colorbar | sample-neighborhood structure.
        ax_ma_before = pw.Brick(figsize=(3.9, 4.0), label="MA_Before")
        ax_ma_after = pw.Brick(figsize=(3.9, 4.0), label="MA_After")
        ax_cb = pw.Brick(figsize=(0.2, 4.0), label="MA_CB")

        self._plot_ma_scatter(
            metrics=metrics, ax_before=ax_ma_before, ax_after=ax_ma_after, cax=ax_cb
        )

        ax_geom_structure = pw.Brick(figsize=(4.0, 4.0), label="Geometric_Structure")
        self._plot_sample_structure_preservation(ax_geom=ax_geom_structure)

        row2 = ax_ma_before | ax_ma_after | ax_cb | ax_geom_structure

        return row1 / row2
