# src/pimqc/imputation.py
"""
Script purpose: Execute mechanism-aware missing-value imputation.

execute_imputation() reads MAR/MNAR feature labels from MetaboInt attributes,
log-transforms the working matrix, imputes MNAR features with QRILC or
LOD-style constants, and handles MAR features with Median, MinProb, KNN, LLS,
BPCA, or AUTO selection. AUTO mode masks observed values, benchmarks candidates
with stratified NRMSE and distribution-preservation metrics, and applies the
selected algorithm feature-wise.
The method reconstructs the original intensity scale, preserves metadata,
stores candidate and QA metrics, writes the imputed matrix, and renders KDE
and NRMSE diagnostics.
"""

import os
import copy
import math
import numpy as np
import pandas as pd
from functools import cached_property

import matplotlib.pyplot as plt

import scipy.stats as stats
from sklearn.impute import KNNImputer
from loguru import logger
from typing import Any, Callable, Dict, Optional

from . import io_utils as iu
from . import stat_utils as su
from . import plot_utils as pu
from . import core_classes
from . import visualizer_classes


class BayesianPCAImputer:
    """Bayesian PCA missing-value estimator adapted from pcaMethods BPCA.

    The implementation follows the structure of the pcaMethods R port of
    Oba's BPCA algorithm: initialize principal axes with SVD, then iteratively
    update scores, loadings, residual precision (tau), and loading precision
    terms (alpha). Input rows are observations and columns are variables.
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 100,
        threshold: float = 1e-4,
    ) -> None:
        """Initialize BPCA model settings."""
        self.n_components = max(1, int(n_components))
        self.max_iter = max(1, int(max_iter))
        self.threshold = float(threshold)

    @staticmethod
    def _safe_inverse(mat: np.ndarray) -> np.ndarray:
        """Invert a small matrix with pseudo-inverse fallback."""
        try:
            return np.linalg.inv(mat)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(mat)

    def _initialize_model(self, y: np.ndarray) -> dict[str, Any]:
        """Initialize the BPCA working state from an incomplete matrix."""
        rows, cols = y.shape
        nans = np.isnan(y)
        yest = np.where(nans, 0.0, y)

        max_components = max(1, min(self.n_components, rows, cols))
        if rows > 1 and cols > 1:
            cov_y = np.cov(yest, rowvar=False)
            cov_y = np.atleast_2d(cov_y)
        else:
            cov_y = np.eye(cols) * np.nanvar(yest)

        cov_y = np.nan_to_num(cov_y, nan=0.0, posinf=0.0, neginf=0.0)
        u, s, _ = np.linalg.svd(cov_y, full_matrices=False)
        s = np.clip(s[:max_components], a_min=0.0, a_max=None)

        mean = np.nanmean(y, axis=0)
        if np.isnan(mean).any():
            global_mean = np.nanmean(y)
            if np.isnan(global_mean):
                global_mean = 0.0
            mean = np.where(np.isnan(mean), global_mean, mean)

        pa = u[:, :max_components] @ np.diag(np.sqrt(s))
        residual_var = float(np.trace(cov_y) - np.sum(s))
        tau = 1.0 / residual_var if residual_var > 1e-10 else 1e10
        tau = float(np.clip(tau, 1e-10, 1e10))

        galpha0 = 1e-10
        balpha0 = 1.0
        alpha_denom = tau * np.diag(pa.T @ pa) + 2 * galpha0 / balpha0
        alpha = (2 * galpha0 + cols) / np.maximum(alpha_denom, 1e-12)

        return {
            "rows": rows,
            "cols": cols,
            "comps": max_components,
            "yest": yest.copy(),
            "row_miss": np.where(nans.sum(axis=1) != 0)[0],
            "row_nomiss": np.where(nans.sum(axis=1) == 0)[0],
            "nans": nans,
            "mean": mean,
            "pa": pa,
            "tau": tau,
            "scores": np.zeros((rows, max_components), dtype=float),
            "galpha0": galpha0,
            "balpha0": balpha0,
            "alpha": alpha,
            "gmu0": 0.001,
            "btau0": 1.0,
            "gtau0": 1e-10,
            "sigw": np.eye(max_components),
        }

    def _do_step(self, model: dict[str, Any], y: np.ndarray) -> dict[str, Any]:
        """Perform one BPCA EM/Bayesian update step."""
        rows = model["rows"]
        cols = model["cols"]
        comps = model["comps"]
        pa = model["pa"]
        tau = model["tau"]
        sigw = model["sigw"]
        mean = model["mean"]
        nans = model["nans"]

        scores = np.zeros((rows, comps), dtype=float)
        t_mat = np.zeros((cols, comps), dtype=float)
        tr_s = 0.0

        rx = np.eye(comps) + tau * (pa.T @ pa) + sigw
        rx_inv = self._safe_inverse(rx)

        idx_nomiss = model["row_nomiss"]
        if len(idx_nomiss) > 0:
            dy = y[idx_nomiss, :] - mean
            x = tau * rx_inv @ pa.T @ dy.T
            t_mat += dy.T @ x.T
            tr_s += float(np.sum(dy * dy))
            scores[idx_nomiss, :] = x.T

        for i in model["row_miss"]:
            missing = nans[i, :]
            observed = ~missing

            dyo = y[i, observed] - mean[observed]
            wm = pa[missing, :]
            wo = pa[observed, :]

            rx_obs = rx - tau * (wm.T @ wm)
            rx_obs_inv = self._safe_inverse(rx_obs)

            ex = tau * wo.T @ dyo.reshape(-1, 1)
            x = rx_obs_inv @ ex
            dym = (wm @ x).ravel()

            dy_full = np.zeros(cols, dtype=float)
            dy_full[observed] = dyo
            dy_full[missing] = dym

            model["yest"][i, :] = dy_full + mean
            t_mat += dy_full.reshape(-1, 1) @ x.reshape(1, -1)

            if missing.any():
                t_mat[missing, :] += wm @ rx_obs_inv
                tr_s += float(
                    dy_full @ dy_full
                    + np.sum(missing) / tau
                    + np.trace(wm @ rx_obs_inv @ wm.T)
                )
            else:
                tr_s += float(dy_full @ dy_full)

            scores[i, :] = x.ravel()

        t_mat /= rows
        tr_s /= rows

        dw = rx_inv + tau * t_mat.T @ pa @ rx_inv + np.diag(model["alpha"]) / rows
        dw_inv = self._safe_inverse(dw)

        pa_new = t_mat @ dw_inv
        tau_num = cols + 2 * model["gtau0"] / rows
        tau_den = (
            tr_s
            - np.trace(t_mat.T @ pa_new)
            + (
                float(np.dot(mean, mean)) * model["gmu0"]
                + 2 * model["gtau0"] / model["btau0"]
            )
            / rows
        )
        tau_new = float(tau_num / max(float(tau_den), 1e-12))
        tau_new = float(np.clip(tau_new, 1e-10, 1e10))

        sigw_new = dw_inv * (cols / rows)
        alpha_denom = (
            tau_new * np.diag(pa_new.T @ pa_new)
            + np.diag(sigw_new)
            + 2 * model["galpha0"] / model["balpha0"]
        )
        alpha_new = (2 * model["galpha0"] + cols) / np.maximum(alpha_denom, 1e-12)

        model["scores"] = scores
        model["pa"] = pa_new
        model["tau"] = tau_new
        model["sigw"] = sigw_new
        model["alpha"] = alpha_new
        return model

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Estimate missing values in an observation-by-variable matrix."""
        y = np.asarray(y, dtype=float)
        if y.ndim != 2:
            raise ValueError("BPCA input must be a 2D matrix.")

        if not np.isnan(y).any():
            return y.copy()

        if y.shape[0] < 2 or y.shape[1] < 2:
            means = np.nanmean(y, axis=0)
            global_mean = np.nanmean(y)
            if np.isnan(global_mean):
                global_mean = 0.0
            means = np.where(np.isnan(means), global_mean, means)
            return np.where(np.isnan(y), means, y)

        model = self._initialize_model(y)
        tau_old = 1000.0

        for step in range(1, self.max_iter + 1):
            model = self._do_step(model, y)
            if step % 10 == 0:
                dtau = abs(np.log10(model["tau"]) - np.log10(tau_old))
                if dtau < self.threshold:
                    break
                tau_old = model["tau"]

        result = np.where(np.isnan(y), model["yest"], y)
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


class MetaboIntImputer(core_classes.MetaboInt):
    """Missing value imputation engine with hybrid stratified evaluation."""

    _metadata = ["attrs", "stats"]

    def __init__(
        self,
        *args: object,
        pipeline_params: Optional[Dict[str, Any]] = None,
        mar_method: Optional[str] = None,
        mnar_method: Optional[str] = None,
        mnar_fraction: Optional[float] = None,
        knn_neighbors: Optional[int] = None,
        lls_neighbors: Optional[int] = None,
        bpca_components: Optional[int] = None,
        bpca_max_iter: Optional[int] = None,
        bpca_tol: Optional[float] = None,
        sim_mask_ratio: Optional[float] = None,
        **kwargs: object,
    ) -> None:
        """Initialize MetaboIntImputer with parameters and metadata.

        Args:
            *args: Variable arguments passed to pandas DataFrame.
            pipeline_params: Global configuration dictionary.
            mar_method: Method for MAR features.
            mnar_method: Method for MNAR features.
            mnar_fraction: Multiplier for LOD-based MNAR imputation.
            knn_neighbors: Number of neighbors for the KNN algorithm.
            lls_neighbors: Number of neighbors for the LLS algorithm.
            bpca_components: Number of principal components for BPCA.
            bpca_max_iter: Maximum BPCA EM/Bayesian update steps.
            bpca_tol: BPCA precision-change convergence threshold.
            sim_mask_ratio: Ratio for simulated masking during evaluation.
            **kwargs: Keyword arguments for the DataFrame constructor.
        """
        super().__init__(*args, pipeline_params=pipeline_params, **kwargs)

        # 1. Base defaults matching pipeline_parameters.toml
        imp_configs = {
            "mar_method": "Auto",
            "mnar_method": "QRILC",
            "mnar_fraction": 0.5,
            "knn_neighbors": 5,
            "lls_neighbors": 15,
            "bpca_components": 2,
            "bpca_max_iter": 100,
            "bpca_tol": 1e-4,
            "sim_mask_ratio": 0.05,
        }

        # 2. TOML global configuration overrides base defaults
        if pipeline_params and "MetaboIntImputer" in pipeline_params:
            imp_configs.update(pipeline_params["MetaboIntImputer"])

        # 3. Explicit kwargs override TOML (Highest priority)
        local_args = locals()
        explicit_params = [
            "mar_method",
            "mnar_method",
            "mnar_fraction",
            "knn_neighbors",
            "lls_neighbors",
            "bpca_components",
            "bpca_max_iter",
            "bpca_tol",
            "sim_mask_ratio",
        ]
        for param in explicit_params:
            if local_args[param] is not None:
                imp_configs[param] = local_args[param]

        self.attrs.update(imp_configs)

    @property
    def _constructor(self) -> type["MetaboIntImputer"]:
        """Return the class constructor for stable subclassing."""
        return MetaboIntImputer

    def __finalize__(
        self,
        other: object,
        method: Optional[str] = None,
        **kwargs: object,
    ) -> "MetaboIntImputer":
        """Ensure custom metadata (attrs) is preserved safely."""
        try:
            super().__finalize__(other, method=method, **kwargs)
        except ValueError:
            pass  # Bypass array comparison ValueError during concat

        if method == "concat" and hasattr(other, "objs"):
            for obj in other.objs:
                if hasattr(obj, "attrs") and obj.attrs:
                    self.attrs = copy.deepcopy(obj.attrs)
                    break
        elif hasattr(other, "attrs"):
            self.attrs = copy.deepcopy(other.attrs)
        if hasattr(other, "stats"):
            self.stats = copy.deepcopy(other.stats)

        return self

    # ====================================================================
    # Imputation-related Metrics
    # ====================================================================
    def calc_imp_quality_metrics(
        self, raw_obj: core_classes.MetaboInt, imp_obj: core_classes.MetaboInt
    ) -> dict[str, Any]:
        """Calculate post-imputation distribution QA metrics for the passport.

        Computes Jensen-Shannon and Wasserstein distances for final QC and sample
        matrices separately. These values are retained as post-imputation QA
        diagnostics; AUTO selection instead uses masked-value metrics.

        Returns:
            Contains the quantified post-imputation QA metrics.
        """
        metrics = {
            "JSD": {"QC": {}, "Sample": {}},
            "Wasserstein": {"QC": {}, "Sample": {}},
            "JSD_Score": {"QC": {}, "Sample": {}},
            "Wasserstein_Score": {"QC": {}, "Sample": {}},
        }
        raw_log = np.log2(raw_obj.astype(float).replace({0: np.nan}) + 1.0)
        imp_log = np.log2(imp_obj.astype(float) + 1.0)

        qc_cols = imp_obj._qc.columns.intersection(raw_log.columns)
        sam_cols = imp_obj._actual_sample.columns.intersection(raw_log.columns)

        for grp, cols in [("QC", qc_cols), ("Sample", sam_cols)]:
            if cols.empty:
                continue

            r_slice = raw_log[cols].values.flatten()
            i_slice = imp_log[cols].values.flatten()

            # 1. Data Before Imputation (All)
            obs = r_slice[~np.isnan(r_slice)]

            # 2. Data After Imputation (All)
            imp_all = i_slice[~np.isnan(i_slice)]

            # 3. Imputed Data (Patches only)
            mask_missing = np.isnan(r_slice)
            imp_only = i_slice[mask_missing]
            imp_only = imp_only[~np.isnan(imp_only)]

            if len(obs) > 0 and len(imp_all) > 0:
                dist_1 = su.calc_distribution_distance_metrics(obs, imp_all)
                jsd_val = su.finite_or_nan(dist_1.get("jsd"))
                wd_val = su.finite_or_nan(dist_1.get("wasserstein_normalized"))
                metrics["JSD"][grp]["Before vs After (All)"] = jsd_val
                metrics["Wasserstein"][grp]["Before vs After (All)"] = wd_val
                if np.isfinite(jsd_val):
                    metrics["JSD_Score"][grp]["Before vs After (All)"] = float(
                        np.clip(1.0 - jsd_val, 0.0, 1.0)
                    )
                if np.isfinite(wd_val):
                    metrics["Wasserstein_Score"][grp][
                        "Before vs After (All)"
                    ] = float(1.0 / (1.0 + max(wd_val, 0.0)))

            if len(obs) > 0 and len(imp_only) > 0:
                dist_2 = su.calc_distribution_distance_metrics(obs, imp_only)
                jsd_val = su.finite_or_nan(dist_2.get("jsd"))
                wd_val = su.finite_or_nan(dist_2.get("wasserstein_normalized"))
                metrics["JSD"][grp]["Before vs Imputed Only"] = jsd_val
                metrics["Wasserstein"][grp]["Before vs Imputed Only"] = wd_val
                if np.isfinite(jsd_val):
                    metrics["JSD_Score"][grp]["Before vs Imputed Only"] = float(
                        np.clip(1.0 - jsd_val, 0.0, 1.0)
                    )
                if np.isfinite(wd_val):
                    metrics["Wasserstein_Score"][grp][
                        "Before vs Imputed Only"
                    ] = float(1.0 / (1.0 + max(wd_val, 0.0)))

        return metrics

    # ====================================================================
    # Core Algorithms (Log2 Space)
    # ====================================================================
    @staticmethod
    def impute_by_constant(
        df_log: pd.DataFrame, fraction: float = 1.0, imp_mode: str = "row"
    ) -> pd.DataFrame:
        """Imputes missing values using a constant LOD heuristic.

        Args:
            df: The dataset (typically log-transformed).
            fraction: The heuristic multiplier (e.g., 0.5 for half-minimum).
            imp_mode: "row" (feature-wise), "column" (sample-wise), or "global".
            is_log2: If True, executes fractional math in linear space safely.

        Returns:
            Dataframe with constant imputation applied.
        """
        if imp_mode in ("row", "row-wise", "row min"):
            raw_mins = df_log.min(axis=1)
        elif imp_mode in ("column", "column-wise", "column min"):
            raw_mins = df_log.min(axis=0)
        else:  # elif imp_mode in ("global", "global min"):
            raw_mins = df_log.min().min()

        linear_mins = np.exp2(raw_mins) - 1.0
        target_mins = np.log2((linear_mins * fraction) + 1.0)

        # 3. Broadcast the computed minimums to fill NaNs
        if imp_mode in ("row", "row-wise", "row min"):
            return df_log.apply(lambda x: x.fillna(target_mins[x.name]), axis=1)
        else:
            return df_log.fillna(target_mins)

    @staticmethod
    def impute_by_qrilc(
        df_log: pd.DataFrame, tune_sigma: float = 1.0, global_seed: int = 123
    ) -> pd.DataFrame:
        """Impute missing values using QRILC logic for left-censored data.

        This follows the imputeLCMD::impute.QRILC orientation: for each sample
        column, fit observed sample quantiles against theoretical normal
        quantiles, then draw missing values from the left tail of the estimated
        censored distribution. The project matrix stores features in rows and
        samples in columns, matching imputeLCMD's input orientation.

        Ref:
            Missing value imputation approach for mass spectrometry-based
            metabolomics data (Scientific reports, 2018)
        """

        rng = np.random.default_rng(global_seed)
        arr = df_log.to_numpy(dtype=float)
        res_arr = arr.copy()
        n_features = arr.shape[0]
        upper_q = 0.99
        probs = np.arange(0.001, upper_q + 0.001 + 1e-12, 0.01)

        for col_idx in range(arr.shape[1]):
            sample_vec = arr[:, col_idx]
            missing_mask = np.isnan(sample_vec)
            n_missing = int(np.sum(missing_mask))
            if n_missing == 0:
                continue

            observed = sample_vec[~missing_mask]
            if observed.size < 3:
                fallback = float(np.nanmin(observed)) if observed.size else 0.0
                res_arr[missing_mask, col_idx] = fallback
                continue

            p_missing = n_missing / float(n_features)
            q_normal = stats.norm.ppf(
                np.linspace(p_missing + 0.001, upper_q + 0.001, len(probs))
            )
            q_sample = np.quantile(observed, probs, method="linear")
            slope, intercept = np.polyfit(q_normal, q_sample, deg=1)

            center = float(intercept)
            slope_abs = max(abs(float(slope)), 1e-12)
            # imputeLCMD passes the fitted scale-like coefficient to rtmvnorm's
            # covariance argument. In one dimension this corresponds to a
            # standard deviation of sqrt(scale * tune_sigma).
            scale = math.sqrt(max(slope_abs * float(tune_sigma), 1e-12))
            upper = stats.norm.ppf(
                p_missing + 0.001,
                loc=center,
                scale=slope_abs,
            )
            upper_std = (upper - center) / scale
            drawn = stats.truncnorm.rvs(
                a=-np.inf,
                b=upper_std,
                loc=center,
                scale=scale,
                size=n_missing,
                random_state=rng,
            )
            res_arr[missing_mask, col_idx] = np.clip(drawn, a_min=0.0, a_max=None)

        return pd.DataFrame(res_arr, index=df_log.index, columns=df_log.columns)

    @staticmethod
    def impute_by_knn(df_log: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
        """Impute missing values using K-Nearest Neighbors algorithm."""
        # Scale neighbor count for isolated small groups.
        n_samples = df_log.shape[1]
        safe_k = min(n_neighbors, n_samples - 1)

        if safe_k < 1:
            # Fallback to feature median if insufficient neighbors (e.g., n=1)
            return df_log.apply(lambda x: x.fillna(x.median()), axis=1).fillna(0.0)

        imputer = KNNImputer(n_neighbors=safe_k, weights="distance")
        arr_imp = imputer.fit_transform(df_log.T).T

        return pd.DataFrame(arr_imp, index=df_log.index, columns=df_log.columns)

    @staticmethod
    def impute_by_lls(df_log: pd.DataFrame, n_neighbors: int = 15) -> pd.DataFrame:
        """Impute missing values using Local Least Squares (LLS) regression.

        Finds 'k' complete features that are highly correlated with the target
        feature, and constructs a local linear regression model to predict
        the missing values.
        """
        arr_log = df_log.values
        res_arr = arr_log.copy()

        # 1. Identify complete features to serve as the candidate neighbor pool
        complete_mask = ~np.isnan(arr_log).any(axis=1)
        complete_features = arr_log[complete_mask]
        n_complete = complete_features.shape[0]

        # Fallback: If dataset is too sparse and lacks complete features
        if n_complete < 2:
            logger.debug(
                "Insufficient complete features for LLS. Falling back to median."
            )
            return df_log.apply(lambda x: x.fillna(x.median()), axis=1).fillna(0.0)

        safe_k = min(n_neighbors, n_complete)

        for i in range(arr_log.shape[0]):
            row = arr_log[i]
            missing_mask = np.isnan(row)

            # Skip if no missing values
            if not missing_mask.any():
                continue

            obs_mask = ~missing_mask
            w_obs = row[obs_mask]

            # Fallback: Need at least 3 points for stable linear regression
            if obs_mask.sum() < 3:
                res_arr[i, missing_mask] = (
                    np.nanmedian(row) if obs_mask.sum() > 0 else 0.0
                )
                continue

            # 2. Vectorized Pearson correlation to find closest complete features
            A_obs = complete_features[:, obs_mask]

            w_mean = np.mean(w_obs)
            A_mean = np.mean(A_obs, axis=1, keepdims=True)

            w_centered = w_obs - w_mean
            A_centered = A_obs - A_mean

            cov = np.sum(A_centered * w_centered, axis=1)
            var_w = np.sum(w_centered**2)
            var_A = np.sum(A_centered**2, axis=1)

            denom = np.sqrt(var_w * var_A)
            corr = np.zeros_like(cov)
            valid_corr = denom > 1e-9
            # Use absolute correlation since negative correlation is also
            # useful for regression
            corr[valid_corr] = np.abs(cov[valid_corr] / denom[valid_corr])

            # 3. Select top K neighbors
            top_k_idx = np.argsort(corr)[-safe_k:]

            # 4. Construct matrices for Least Squares estimation
            # A_mat: neighbors' observed values (Shape: n_neighbors x n_observed)
            A_mat = A_obs[top_k_idx]
            # B_mat: neighbors' values at target's missing positions
            B_mat = complete_features[top_k_idx][:, missing_mask]

            # 5. Solve linear system: A_mat.T * x = w_obs
            try:
                # Used for numerical stability over matrix inverse
                x, _, _, _ = np.linalg.lstsq(A_mat.T, w_obs, rcond=None)

                # Predict: x.T * B_mat
                w_miss = x.T @ B_mat
                # Prevent negative intensities in log space fallback
                w_miss = np.clip(w_miss, a_min=0.0, a_max=None)
                res_arr[i, missing_mask] = w_miss
            except np.linalg.LinAlgError:
                # Fallback if matrix is singular or highly collinear
                res_arr[i, missing_mask] = np.nanmedian(row)

        return pd.DataFrame(res_arr, index=df_log.index, columns=df_log.columns)

    @staticmethod
    def impute_by_bpca(
        df_log: pd.DataFrame,
        n_components: int = 2,
        max_iter: int = 100,
        threshold: float = 1e-4,
    ) -> pd.DataFrame:
        """Impute missing values using Bayesian PCA in log2 space.

        pcaMethods treats rows as observations and columns as variables. The
        project matrix stores features in rows and samples in columns, so this
        wrapper transposes the matrix before fitting BPCA and restores the
        original orientation afterward.
        """
        if df_log.empty or not df_log.isna().any().any():
            return df_log.copy()

        arr = df_log.to_numpy(dtype=float)
        if arr.shape[0] < 2 or arr.shape[1] < 2:
            return df_log.apply(lambda x: x.fillna(x.median()), axis=1).fillna(0.0)

        safe_components = max(1, min(int(n_components), arr.shape[0], arr.shape[1]))
        imputer = BayesianPCAImputer(
            n_components=safe_components,
            max_iter=max_iter,
            threshold=threshold,
        )

        # BPCA returns an imputed observation-by-variable matrix.
        arr_imp = imputer.fit_transform(arr.T).T
        arr_imp = np.where(np.isnan(arr), arr_imp, arr)
        arr_imp = np.clip(arr_imp, a_min=0.0, a_max=None)

        return pd.DataFrame(arr_imp, index=df_log.index, columns=df_log.columns)

    @staticmethod
    def impute_by_minprob(df_log: pd.DataFrame, global_seed: int = 123) -> pd.DataFrame:
        """Impute using a normal distribution to simulate values below LOD.

        This method adopts a left-shifted Gaussian distribution (Perseus style)
        without hard clipping, preserving the natural variance of the unobserved
        low-abundance tail.
        """
        rng = np.random.default_rng(global_seed)
        res_df = df_log.copy()
        for col in res_df.columns:
            s = res_df[col]
            if s.isna().sum() == 0:
                continue

            valid = s.dropna()
            m, sd = valid.mean(), valid.std()

            # Shift the distribution leftward to simulate the missing tail
            # standard parameters: shift by -1.8 std, width of 0.3 std
            shift_mean = m - 1.8 * sd
            shift_std = max(0.3 * sd, 0.01)

            # Draw random values simulating noise below the detection limit
            drawn = rng.normal(loc=shift_mean, scale=shift_std, size=s.isna().sum())

            # Prevent negative intensities in linear space.
            # Log2 values must be >= 0 so that exp2(x) - 1.0 >= 0
            drawn = np.clip(drawn, a_min=0.0, a_max=None)
            res_df.loc[s.isna(), col] = drawn

        return res_df

    def _apply_isolated(
        self,
        df_slice: pd.DataFrame,
        imp_func: Callable[..., pd.DataFrame],
        **kwargs: object,
    ) -> pd.DataFrame:
        """Executes imputation independently on QC and biological samples.

        Prevents variance leakage between technical replicates and
        actual biological samples during complex ML imputation.
        """
        qc_cols = self._qc.columns.intersection(df_slice.columns)
        sam_cols = self._actual_sample.columns.intersection(df_slice.columns)

        res_dfs = []
        global_fallback = None  # Lazy evaluation for edge cases

        # 1. Impute QC using strictly QC context
        if not qc_cols.empty:
            res_qc = imp_func(df_slice[qc_cols], **kwargs)
            if res_qc.isna().any().any():
                global_fallback = imp_func(df_slice, **kwargs)
                res_qc = res_qc.combine_first(global_fallback[qc_cols])
            res_dfs.append(res_qc)

        # 2. Impute Samples using strictly Sample context
        if not sam_cols.empty:
            res_sam = imp_func(df_slice[sam_cols], **kwargs)
            if res_sam.isna().any().any():
                if global_fallback is None:
                    global_fallback = imp_func(df_slice, **kwargs)
                res_sam = res_sam.combine_first(global_fallback[sam_cols])
            res_dfs.append(res_sam)

        if not res_dfs:
            return df_slice

        # Reconstruct matrix ensuring original column order
        return pd.concat(res_dfs, axis=1)[df_slice.columns]

    # ====================================================================
    # Evaluation Logic (Hybrid Masking & Stratified NRMSE)
    # ====================================================================

    @staticmethod
    def generate_gmm_noise_mask(
        df_log: pd.DataFrame,
        mask_ratio: float,
        noise_factor: float = 1.5,
        global_seed: int = 123,
        batch_array: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Generate a MNAR mask using GMM probability scoring, optimized for
        batch-wise evaluation.
        """
        from sklearn.mixture import GaussianMixture

        rng = np.random.default_rng(global_seed)
        shape = df_log.shape
        mask_arr = np.zeros(shape, dtype=bool)

        # Fallback to Global GMM if no batch metadata is provided
        if batch_array is None or len(np.unique(batch_array)) <= 1:
            valid_mask = ~df_log.isna().values
            valid_data = df_log.values[valid_mask].reshape(-1, 1)
            target_nas = int(valid_mask.sum() * mask_ratio)

            if target_nas == 0 or len(valid_data) < 10:
                return pd.DataFrame(False, index=df_log.index, columns=df_log.columns)

            gmm = GaussianMixture(n_components=2, random_state=global_seed)
            gmm.fit(valid_data)
            lower_cluster_idx = np.argmin(gmm.means_)

            base_prob = gmm.predict_proba(valid_data)[:, lower_cluster_idx]
            final_score = base_prob + rng.uniform(0, noise_factor, size=base_prob.shape)
            cutoff_score = np.sort(final_score)[-target_nas]

            mask_arr[valid_mask] = final_score >= cutoff_score
            return pd.DataFrame(mask_arr, index=df_log.index, columns=df_log.columns)

        # Advanced Logic: Batch-wise independent GMM masking
        unique_batches = np.unique(batch_array)
        for b in unique_batches:
            b_cols_idx = np.where(batch_array == b)[0]
            b_data = df_log.iloc[:, b_cols_idx].values

            valid_mask_b = ~np.isnan(b_data)
            valid_data_b = b_data[valid_mask_b].reshape(-1, 1)
            target_nas_b = int(valid_mask_b.sum() * mask_ratio)

            if target_nas_b == 0:
                continue

            # Defensive mechanism for extremely small batches
            if len(valid_data_b) < 10:
                if len(valid_data_b) > 0:
                    cutoff_val = np.percentile(
                        valid_data_b, (target_nas_b / len(valid_data_b)) * 100
                    )
                    b_mask = np.zeros_like(b_data, dtype=bool)
                    b_mask[valid_mask_b] = valid_data_b.flatten() <= cutoff_val
                    mask_arr[:, b_cols_idx] = b_mask
                continue

            try:
                gmm = GaussianMixture(n_components=2, random_state=global_seed)
                gmm.fit(valid_data_b)
                lower_cluster_idx = np.argmin(gmm.means_)

                base_prob = gmm.predict_proba(valid_data_b)[:, lower_cluster_idx]
                final_score = base_prob + rng.uniform(
                    0, noise_factor, size=base_prob.shape
                )
                cutoff_score = np.sort(final_score)[-target_nas_b]

                b_mask = np.zeros_like(b_data, dtype=bool)
                b_mask[valid_mask_b] = final_score >= cutoff_score
                mask_arr[:, b_cols_idx] = b_mask
            except Exception as e:
                # Soft fallback to percentile truncation if GMM fails to
                # converge on edge-case batches
                logger.debug(
                    f"GMM failed for batch {b}: {e}. "
                    "Falling back to empirical percentile."
                )
                cutoff_val = np.percentile(
                    valid_data_b, (target_nas_b / len(valid_data_b)) * 100
                )
                b_mask = np.zeros_like(b_data, dtype=bool)
                b_mask[valid_mask_b] = valid_data_b.flatten() <= cutoff_val
                mask_arr[:, b_cols_idx] = b_mask

        return pd.DataFrame(mask_arr, index=df_log.index, columns=df_log.columns)

    @staticmethod
    def compute_stratified_nrmse(
        df_true: pd.DataFrame,
        df_imp: pd.DataFrame,
        mask_df: pd.DataFrame,
        lod_q: float = 0.25,
    ) -> dict[str, float]:
        """Calculate NRMSE stratified by low and high abundance regions."""
        feat_meds = df_true.median(axis=1).fillna(0)
        lod_val = feat_meds.quantile(lod_q)

        t_all = df_true.values[mask_df.values]
        p_all = df_imp.values[mask_df.values]
        med_all = np.tile(feat_meds.values[:, None], (1, df_true.shape[1]))[
            mask_df.values
        ]

        low_m, hi_m = (med_all <= lod_val), (med_all > lod_val)

        def _get_nrmse(t: np.ndarray, p: np.ndarray) -> float:
            if len(t) < 2 or (np.max(t) - np.min(t)) < 1e-9:
                return np.nan
            rmse = np.sqrt(np.mean((t - p) ** 2))
            return float(rmse / (np.max(t) - np.min(t)))

        # 1. Compile the metrics dictionary
        metrics = {
            "NRMSE_Total": _get_nrmse(t_all, p_all),
            "NRMSE_Low": _get_nrmse(t_all[low_m], p_all[low_m]),
            "NRMSE_High": _get_nrmse(t_all[hi_m], p_all[hi_m]),
            "Count_Low": int(np.sum(low_m)),
            "Count_High": int(np.sum(hi_m)),
            "Threshold": float(lod_val),
        }

        # 2. Return exactly 3 objects to match the unpacking logic
        return metrics, t_all, p_all

    @staticmethod
    def _low_value_reference_impute(masked_df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing entries with one-half of the global observed minimum."""
        observed = masked_df.to_numpy(dtype=float)
        observed = observed[np.isfinite(observed)]
        if observed.size == 0:
            fill_value = 0.0
        else:
            linear_observed = np.exp2(observed) - 1.0
            positive_values = linear_observed[
                np.isfinite(linear_observed) & (linear_observed > 0)
            ]
            if positive_values.size == 0:
                fill_value = float(np.nanmin(observed))
            else:
                fill_value = float(np.log2(np.nanmin(positive_values) * 0.5 + 1.0))
        return masked_df.fillna(fill_value)

    @staticmethod
    def _reference_improvement_score(reference: object, value: object) -> float:
        """Score improvement over a lower-is-better reference metric."""
        score = su.relative_change_lower_better(reference, value)
        return float(np.clip(score, 0.0, 1.0)) if np.isfinite(score) else float("nan")

    def _evaluate_imputation_candidate(
        self,
        df_log: pd.DataFrame,
        idx_mar: pd.Index,
        target_cols: pd.Index,
        method: str,
        ratio: float = 0.05,
        global_seed: int = 123,
        batch_array: Optional[np.ndarray] = None,
    ) -> tuple:
        """Evaluate one MAR imputation candidate on an artificial mask."""
        mar_data = df_log.loc[idx_mar, target_cols].astype(float)
        mar_data.attrs["is_logged"] = True

        mask = self.generate_gmm_noise_mask(
            mar_data,
            ratio,
            noise_factor=1.5,
            global_seed=global_seed,
            batch_array=batch_array,
        )

        masked_df = mar_data.copy()
        masked_df[mask] = np.nan

        method_key = str(method).replace(" ", "").replace("-", "").upper()
        if method_key in ("HALFGLOBALMIN", "HALFGLOBALMINREFERENCE", "LOWVALUEREF"):
            imp_res = self._low_value_reference_impute(masked_df)
        elif method in ("Prob", "prob", "MinProb", "minprob"):
            imp_res = self._apply_isolated(
                masked_df, self.impute_by_minprob, global_seed=global_seed
            )
        elif method in ("knn", "KNN"):
            k_val = self.attrs.get("knn_neighbors", 5)
            imp_res = self._apply_isolated(
                masked_df, self.impute_by_knn, n_neighbors=k_val
            )
        elif method in ("lls", "LLS"):
            k_val = self.attrs.get("lls_neighbors", 15)
            imp_res = self._apply_isolated(
                masked_df, self.impute_by_lls, n_neighbors=k_val
            )
        elif method in ("bpca", "BPCA"):
            imp_res = self._apply_isolated(
                masked_df,
                self.impute_by_bpca,
                n_components=self.attrs.get("bpca_components", 2),
                max_iter=self.attrs.get("bpca_max_iter", 100),
                threshold=self.attrs.get("bpca_tol", 1e-4),
            )
        elif method_key in ("QRILC", "QRLIC"):
            imp_res = self._apply_isolated(
                masked_df,
                self.impute_by_qrilc,
                global_seed=global_seed,
            )
        else:
            imp_res = self._apply_isolated(
                masked_df, lambda df: df.apply(lambda x: x.fillna(x.median()), axis=1)
            )
        imp_res.attrs["is_logged"] = True

        eval_met, t_vals, p_vals = self.compute_stratified_nrmse(
            mar_data, imp_res, mask
        )
        dist_metrics = su.calc_distribution_distance_metrics(t_vals, p_vals)
        structure_metrics = su.calc_sample_structure_preservation(
            raw_obj=mar_data,
            transformed_obj=imp_res,
            sample_cols=target_cols,
            max_features=5000,
            seed=global_seed,
        )
        eval_met.update(
            {
                "JSD_Total": dist_metrics["jsd"],
                "Wasserstein_Total": dist_metrics["wasserstein"],
                "Wasserstein_Normalized": dist_metrics["wasserstein_normalized"],
                "Sample_Structure_Score": structure_metrics[
                    "sample_structure_composite_preservation"
                ],
                "Trustworthiness": structure_metrics[
                    "sample_structure_trustworthiness"
                ],
                "Distance_Rank_Preservation": structure_metrics[
                    "sample_structure_rank_preservation"
                ],
                "Distance_Scale_Preservation": structure_metrics[
                    "sample_structure_scale_preservation"
                ],
            }
        )
        return eval_met, t_vals, p_vals

    @staticmethod
    def _score_imputation_candidates(
        cache: dict[str, tuple[dict[str, float], np.ndarray, np.ndarray]],
        reference_metrics: dict[str, float],
    ) -> pd.DataFrame:
        """Score MAR candidates against a half-global-min reference."""
        rows = []
        for method, (metrics, _, _) in cache.items():
            rows.append(
                {
                    "method": method,
                    "nrmse_total": metrics.get("NRMSE_Total"),
                    "nrmse_low": metrics.get("NRMSE_Low"),
                    "jsd_total": metrics.get("JSD_Total"),
                    "wasserstein_normalized": metrics.get(
                        "Wasserstein_Normalized"
                    ),
                    "sample_structure_score": metrics.get("Sample_Structure_Score"),
                }
            )
        score_df = pd.DataFrame(rows)
        if score_df.empty:
            return score_df

        ref_nrmse_total = reference_metrics.get("NRMSE_Total")
        ref_nrmse_low = reference_metrics.get("NRMSE_Low")
        ref_jsd = reference_metrics.get("JSD_Total")
        ref_wasserstein = reference_metrics.get("Wasserstein_Normalized")

        score_df["nrmse_total_score"] = score_df["nrmse_total"].apply(
            lambda value: MetaboIntImputer._reference_improvement_score(
                ref_nrmse_total, value
            )
        )
        score_df["nrmse_low_score"] = score_df["nrmse_low"].apply(
            lambda value: MetaboIntImputer._reference_improvement_score(
                ref_nrmse_low, value
            )
        )
        score_df["jsd_score"] = score_df["jsd_total"].apply(
            lambda value: MetaboIntImputer._reference_improvement_score(
                ref_jsd, value
            )
        )
        score_df["wasserstein_score"] = score_df["wasserstein_normalized"].apply(
            lambda value: MetaboIntImputer._reference_improvement_score(
                ref_wasserstein, value
            )
        )

        reconstruction_scores = []
        distribution_scores = []
        auto_scores = []
        for row in score_df.itertuples():
            reconstruction_score = su.weighted_mean_score(
                [
                    (row.nrmse_total_score, 0.70),
                    (row.nrmse_low_score, 0.30),
                ]
            )
            distribution_score = su.weighted_mean_score(
                [
                    (row.jsd_score, 0.50),
                    (row.wasserstein_score, 0.50),
                ]
            )
            auto_score = su.weighted_mean_score(
                [
                    (reconstruction_score, 0.65),
                    (distribution_score, 0.20),
                    (row.sample_structure_score, 0.15),
                ]
            )
            reconstruction_scores.append(reconstruction_score)
            distribution_scores.append(distribution_score)
            auto_scores.append(auto_score)

        score_df["reconstruction_score"] = reconstruction_scores
        score_df["distribution_preservation_score"] = distribution_scores
        score_df["sample_structure_score"] = pd.to_numeric(
            score_df["sample_structure_score"], errors="coerce"
        )
        score_df["auto_score"] = auto_scores
        return score_df

    def _select_best_imputation_method(
        self,
        df_log: pd.DataFrame,
        idx_mar: pd.Index,
        target_cols: pd.Index,
        ratio: float = 0.05,
        global_seed: int = 123,
        batch_array: Optional[np.ndarray] = None,
    ) -> tuple:
        """Select the best MAR imputer from masked-reconstruction benchmarks."""
        candidates = ["KNN", "MinProb", "QRILC", "Median", "LLS", "BPCA"]
        best_method = "KNN"
        cache = {}
        reference_metrics, _, _ = self._evaluate_imputation_candidate(
            df_log=df_log,
            idx_mar=idx_mar,
            target_cols=target_cols,
            method="HalfGlobalMinReference",
            ratio=ratio,
            global_seed=global_seed,
            batch_array=batch_array,
        )

        for cand in candidates:
            logger.info(f'Simulating "{cand}" on MAR subset...')
            emet, tv, pv = self._evaluate_imputation_candidate(
                df_log=df_log,
                idx_mar=idx_mar,
                target_cols=target_cols,
                method=cand,
                ratio=ratio,
                global_seed=global_seed,
                batch_array=batch_array,
            )
            cache[cand] = (emet, tv, pv)

        score_df = self._score_imputation_candidates(cache, reference_metrics)
        if not score_df.empty and score_df["auto_score"].notna().any():
            score_df = score_df.sort_values(
                by=["auto_score", "nrmse_total", "method"],
                ascending=[False, True, True],
            )
            best_method = str(score_df.iloc[0]["method"])
            for row in score_df.itertuples():
                metrics = cache[row.method][0]
                metrics["Reconstruction_Score"] = su.finite_or_nan(
                    row.reconstruction_score
                )
                metrics["Distribution_Preservation_Score"] = su.finite_or_nan(
                    row.distribution_preservation_score
                )
                metrics["JSD_Score"] = su.finite_or_nan(row.jsd_score)
                metrics["Wasserstein_Score"] = su.finite_or_nan(row.wasserstein_score)
                metrics["Sample_Structure_Score"] = su.finite_or_nan(
                    row.sample_structure_score
                )
                metrics["Auto_Score"] = su.finite_or_nan(row.auto_score)
                metrics["Low_Value_Reference_NRMSE_Total"] = su.finite_or_nan(
                    reference_metrics.get("NRMSE_Total")
                )
                metrics["Low_Value_Reference_NRMSE_Low"] = su.finite_or_nan(
                    reference_metrics.get("NRMSE_Low")
                )
                metrics["Low_Value_Reference_JSD_Total"] = su.finite_or_nan(
                    reference_metrics.get("JSD_Total")
                )
                metrics["Low_Value_Reference_Wasserstein_Normalized"] = (
                    su.finite_or_nan(reference_metrics.get("Wasserstein_Normalized"))
                )
        else:
            best_method = min(
                cache,
                key=lambda method: cache[method][0].get("NRMSE_Total", float("inf")),
            )

        best_score = cache[best_method][0].get("Auto_Score", float("nan"))
        logger.info(
            f"Optimal MAR algorithm selected: {best_method} "
            f"(score={best_score:.3f})"
        )
        return best_method, cache

    @cached_property
    def imputation_metrics(self) -> Dict[str, Any]:
        """Extracts key parameters and performance metrics from imputation.

        Returns:
            dict: A structured dictionary of imputation metadata for reporting.
        """
        mar_req = self.attrs.get("mar_requested", "auto")
        mar_sel = self.attrs.get("selected_mar_method", "Unknown")
        mnar_meth = self.attrs.get("mnar_method", "row")
        mnar_frac = self.attrs.get("mnar_fraction", 0.5)

        status = self.attrs.get("imputation_status", "Pending")
        if (
            status == "Skipped"
            or mnar_frac is None
            or str(mnar_meth).upper() in {"QRILC", "NOT REQUIRED"}
        ):
            reported_mnar_frac = None
        else:
            reported_mnar_frac = float(mnar_frac)

        def _safe_round(val: object) -> float:
            if pd.isna(val):
                return float("nan")
            return round(float(val), 4)

        raw_mets = self.attrs.get("candidate_metrics", {})
        perf_dict = {}
        for cand, mets in raw_mets.items():
            perf_dict[cand] = {
                "nrmse_low": _safe_round(mets.get("nrmse_low")),
                "nrmse_high": _safe_round(mets.get("nrmse_high")),
                "nrmse_total": _safe_round(mets.get("nrmse_total")),
            }

        if status == "Skipped":
            idx_mar = pd.Index([])
            idx_mnar = pd.Index([])
        else:
            raw_mar = pd.Index(self.attrs.get("idx_mar", []))
            raw_mnar = pd.Index(self.attrs.get("idx_mnar", []))
            idx_mar = raw_mar.intersection(self.index)
            idx_mnar = raw_mnar.intersection(self.index)

        # Retrieve the unified QA metrics (JSD) from the data passport
        qa_metrics = self.attrs.get("imputation_qa_metrics", {})

        metrics = {
            "imputation_status": status,
            "strategies": {
                "mar_method_requested": mar_req,
                "mar_method_selected": mar_sel,
                "mnar_method": mnar_meth,
                "mnar_fraction": reported_mnar_frac,
            },
            "performance": perf_dict,
            "feature_distribution": {
                "mar_count": len(idx_mar),
                "mnar_count": len(idx_mnar),
            },
            "qa_metrics": qa_metrics,
            "skip_reason": self.attrs.get("imputation_skip_reason"),
        }

        return metrics

    @iu._exe_time
    def execute_imputation(
        self,
        mar_method: str = None,
        mnar_method: str = None,
        mnar_fraction: float = None,
        knn_neighbors: int = None,
        lls_neighbors: int = None,
        sim_ratio: float = None,
        output_dir: str = None,
    ) -> pd.DataFrame:
        """Executes hybrid imputation and exports complete visualizations."""
        # ====================================================================
        # 1. Parameter Extraction & Priority Fallback
        # ====================================================================
        _mnar = mnar_method or self.attrs.get("mnar_method", "QRILC")
        _frac = (
            mnar_fraction
            if mnar_fraction is not None
            else self.attrs.get("mnar_fraction", 0.5)
        )

        _mar = mar_method or self.attrs.get("mar_method", "Auto")
        _knn_k = (
            knn_neighbors
            if knn_neighbors is not None
            else self.attrs.get("knn_neighbors", 5)
        )
        _lls_k = (
            lls_neighbors
            if lls_neighbors is not None
            else self.attrs.get("lls_neighbors", 15)
        )
        _bpca_k = self.attrs.get("bpca_components", 2)
        _bpca_max_iter = self.attrs.get("bpca_max_iter", 100)
        _bpca_tol = self.attrs.get("bpca_tol", 1e-4)
        _ratio = (
            sim_ratio
            if sim_ratio is not None
            else self.attrs.get("sim_mask_ratio", 0.05)
        )

        _seed = self.attrs.get("global_seed", 123)
        target_cols = self.columns.difference(self._blank.columns)
        target_matrix = self.loc[:, target_cols]

        if not target_matrix.isna().any().any():
            logger.info(
                "No missing values detected in target samples. "
                "Bypassing imputation and propagating the matrix unchanged."
            )
            imputed_obj = self.copy().__finalize__(self)
            imputed_obj.attrs["pipeline_stage"] = "Imputation"
            imputed_obj.attrs["imputation_status"] = "Skipped"
            imputed_obj.attrs["imputation_skip_reason"] = (
                "No missing values detected in target samples."
            )
            imputed_obj.attrs["selected_mar_method"] = "Not required"
            imputed_obj.attrs["mar_requested"] = mar_method or self.attrs.get(
                "mar_method", "auto"
            )
            imputed_obj.attrs["mnar_method"] = "Not required"
            imputed_obj.attrs["mnar_fraction"] = None
            imputed_obj.attrs["candidate_metrics"] = {}
            imputed_obj.attrs["imputation_qa_metrics"] = {}

            if output_dir:
                iu._check_dir_exists(output_dir, handle="makedirs")
                imputed_obj.to_csv(
                    os.path.join(output_dir, "Imputed_Data_NotRequired.csv")
                )

            logger.success("Missing value imputation skipped: no missing values found.")
            return imputed_obj

        batch_col = self.attrs.get("batch", "Batch")
        batch_array = target_cols.get_level_values(batch_col).values

        mnar_info = (
            f"{_mnar}"
            if (str(_mnar).upper() == "QRILC")
            else (f"{_mnar} (LOD={_frac}x)")
        )

        _mar_clean = str(_mar).upper()
        if _mar_clean in ("AUTO", "BEST"):
            mar_info = (
                f"Auto (Evaluating KNN={_knn_k}, LLS (K={_lls_k}), "
                f"BPCA (PCs={_bpca_k}), MinProb, Median)"
            )
        elif _mar_clean == "KNN":
            mar_info = f"KNN (K={_knn_k})"
        elif _mar_clean == "LLS":
            mar_info = f"LLS (K={_lls_k})"
        elif _mar_clean == "BPCA":
            mar_info = f"BPCA (PCs={_bpca_k}, MaxIter={_bpca_max_iter})"
        else:
            mar_info = f"{_mar}"

        logger.info(
            f"Hybrid Imputation Engine Initialized. "
            f"MAR: {mar_info} | MNAR: {mnar_info} | Sim_Mask: {_ratio}"
        )

        df_log = np.log2(self.astype(float).replace({0: np.nan}) + 1.0)

        # ====================================================================
        # 2. ROUTE A: MNAR -> Localized LOD Imputation or QRILC
        # ====================================================================
        idx_mnar = pd.Index(self.attrs.get("idx_mnar", [])).intersection(df_log.index)

        if len(idx_mnar) > 0:
            logger.info(f"Applying {_mnar} to {len(idx_mnar)} MNAR features.")

            if str(_mnar).upper() == "QRILC":
                mnar_imp = MetaboIntImputer.impute_by_qrilc(
                    df_log=df_log.loc[idx_mnar, target_cols], global_seed=_seed
                )
            else:
                mnar_imp = MetaboIntImputer.impute_by_constant(
                    df_log=df_log.loc[idx_mnar, target_cols],
                    fraction=_frac,
                    imp_mode=_mnar,
                )

            df_log.loc[idx_mnar, target_cols] = mnar_imp
        else:
            logger.info("MNAR index empty. Bypassing MNAR imputation.")

        # ====================================================================
        # 3. ROUTE B: MAR -> ML Simulation & Impute
        # ====================================================================
        idx_mar = pd.Index(self.attrs.get("idx_mar", [])).intersection(df_log.index)
        cache, eval_met, t_vals, p_vals = {}, {}, [], []
        is_auto = _mar in ("auto", "Auto", "Best", "best")

        if len(idx_mar) > 0:
            if is_auto:
                _mar, cache = self._select_best_imputation_method(
                    df_log, idx_mar, target_cols, _ratio, _seed, batch_array
                )
                eval_met, t_vals, p_vals = cache[_mar]
            else:
                eval_met, t_vals, p_vals = self._evaluate_imputation_candidate(
                    df_log, idx_mar, target_cols, _mar, _ratio, _seed, batch_array
                )

            logger.info(f"Executing isolated '{_mar}' on MAR features.")
            mar_slice = df_log.loc[idx_mar, target_cols]

            if _mar in ("Prob", "prob", "MinProb", "minprob"):
                mar_imp = self._apply_isolated(
                    mar_slice, self.impute_by_minprob, global_seed=_seed
                )
            elif _mar in ("knn", "KNN"):
                mar_imp = self._apply_isolated(
                    mar_slice, self.impute_by_knn, n_neighbors=_knn_k
                )
            elif _mar in ("lls", "LLS"):
                mar_imp = self._apply_isolated(
                    mar_slice, self.impute_by_lls, n_neighbors=_lls_k
                )
            elif _mar in ("bpca", "BPCA"):
                mar_imp = self._apply_isolated(
                    mar_slice,
                    self.impute_by_bpca,
                    n_components=_bpca_k,
                    max_iter=_bpca_max_iter,
                    threshold=_bpca_tol,
                )
            elif str(_mar).upper() in ("QRILC", "QRLIC"):
                mar_imp = self._apply_isolated(
                    mar_slice,
                    self.impute_by_qrilc,
                    global_seed=_seed,
                )
            else:
                mar_imp = self._apply_isolated(
                    mar_slice,
                    lambda df: df.apply(lambda x: x.fillna(x.median()), axis=1),
                )

            df_log.loc[idx_mar, target_cols] = mar_imp

        # ====================================================================
        # 4. FINALIZATION: Matrix Reconstruction & Passport Update
        # ====================================================================
        final_log = pd.concat(
            [df_log[target_cols], df_log[self._blank.columns]], axis=1
        )[self.columns]

        res_val = np.exp2(final_log) - 1.0
        imputed_obj = self._constructor(res_val).__finalize__(self)

        imputed_obj.attrs["pipeline_stage"] = "Imputation"
        imputed_obj.attrs["imputation_status"] = "Completed"
        display_mar_method = MetaboVisualizerImputer._format_imputation_method_label(
            str(_mar)
        )
        imputed_obj.attrs["selected_mar_method"] = display_mar_method
        imputed_obj.attrs["mar_requested"] = mar_method or self.attrs.get(
            "mar_method", "auto"
        )
        imputed_obj.attrs["mnar_method"] = _mnar
        imputed_obj.attrs["mnar_fraction"] = _frac

        eval_source = cache if cache else {display_mar_method: (eval_met, t_vals, p_vals)}
        cand_mets = {}
        for m_name, (m_eval, _, _) in eval_source.items():
            cand_mets[m_name] = {
                "nrmse_low": m_eval.get("NRMSE_Low", float("nan")),
                "nrmse_high": m_eval.get("NRMSE_High", float("nan")),
                "nrmse_total": m_eval.get("NRMSE_Total", float("nan")),
                "jsd_total": m_eval.get("JSD_Total", float("nan")),
                "wasserstein_total": m_eval.get("Wasserstein_Total", float("nan")),
                "wasserstein_normalized": m_eval.get(
                    "Wasserstein_Normalized", float("nan")
                ),
                "reconstruction_score": m_eval.get(
                    "Reconstruction_Score", float("nan")
                ),
                "distribution_preservation_score": m_eval.get(
                    "Distribution_Preservation_Score", float("nan")
                ),
                "jsd_score": m_eval.get("JSD_Score", float("nan")),
                "wasserstein_score": m_eval.get("Wasserstein_Score", float("nan")),
                "sample_structure_score": m_eval.get(
                    "Sample_Structure_Score", float("nan")
                ),
                "trustworthiness": m_eval.get("Trustworthiness", float("nan")),
                "distance_rank_preservation": m_eval.get(
                    "Distance_Rank_Preservation", float("nan")
                ),
                "distance_scale_preservation": m_eval.get(
                    "Distance_Scale_Preservation", float("nan")
                ),
                "auto_score": m_eval.get("Auto_Score", float("nan")),
            }
        imputed_obj.attrs["candidate_metrics"] = cand_mets

        # ====================================================================
        # 5. EXPORT & VISUALIZATIONS
        # ====================================================================
        logger.info("Calculating imputation-related metrics...")
        qa_metrics = self.calc_imp_quality_metrics(
            raw_obj=self, imp_obj=imputed_obj
        )
        imputed_obj.attrs["imputation_qa_metrics"] = qa_metrics

        if output_dir:
            iu._check_dir_exists(output_dir, handle="makedirs")
            imputed_obj.to_csv(os.path.join(output_dir, f"Imputed_Data_{_mar}.csv"))

            logger.info("Generating diagnostic plots for imputation...")
            vis = MetaboVisualizerImputer(raw_obj=self, imp_obj=imputed_obj)

            if len(idx_mar) > 0:
                benchmark_results = (
                    cache if cache else {_mar: (eval_met, t_vals, p_vals)}
                )
                if is_auto:
                    fig_grid = vis.plot_imputation_auto_dashboard(
                        benchmark_results, best_method=_mar
                    )
                else:
                    fig_grid = vis.plot_imputation_method_dashboard(
                        metrics=eval_met,
                        true_vals=t_vals,
                        pred_vals=p_vals,
                        method_name=display_mar_method,
                    )
                if fig_grid is not None:
                    grid_path = os.path.join(
                        output_dir, f"Imputation_Dashboard_{display_mar_method}.svg"
                    )
                    vis.save_and_show_pw(
                        pw_obj=fig_grid,
                        file_path=grid_path,
                        width="60%" if is_auto else "45%",
                    )
                    logger.info(f"Imputation dashboard saved as: {grid_path}")

                fig_candidates = (
                    vis.plot_imputation_nrmse_appendix_grid(benchmark_results)
                    if is_auto and cache
                    else None
                )
                if fig_candidates is not None:
                    candidate_path = os.path.join(
                        output_dir,
                        f"Imputation_Candidate_Dashboard_{display_mar_method}.svg",
                    )
                    vis.save_and_show_pw(
                        pw_obj=fig_candidates,
                        file_path=candidate_path,
                        width="60%",
                    )
                    logger.info(
                        "Imputer candidate NRMSE grid saved as: "
                        f"{candidate_path}"
                    )

                # Manuscript-only article dashboards are retained for manual
                # figure assembly and are not generated by routine execution.
                # if is_auto:
                #     reconstruction_article = (
                #         vis.plot_imputation_reconstruction_article_dashboard(
                #             results_dict=benchmark_results,
                #             best_method=_mar,
                #         )
                #     )
                #     if reconstruction_article is not None:
                #         reconstruction_path = os.path.join(
                #             output_dir,
                #             (
                #                 "Imputation_Reconstruction_Article_Dashboard_"
                #                 f"{display_mar_method}.svg"
                #             ),
                #         )
                #         vis.save_and_show_pw(
                #             pw_obj=reconstruction_article,
                #             file_path=reconstruction_path,
                #             width="45%",
                #         )
                #         logger.info(
                #             "Imputation reconstruction article dashboard saved as: "
                #             f"{reconstruction_path}"
                #         )

                # if is_auto:
                #     preservation_article = (
                #         vis.plot_imputation_preservation_article_dashboard(
                #             results_dict=benchmark_results,
                #             best_method=_mar,
                #         )
                #     )
                #     if preservation_article is not None:
                #         preservation_path = os.path.join(
                #             output_dir,
                #             (
                #                 "Imputation_Preservation_Article_Dashboard_"
                #                 f"{display_mar_method}.svg"
                #             ),
                #         )
                #         vis.save_and_show_pw(
                #             pw_obj=preservation_article,
                #             file_path=preservation_path,
                #             width="45%",
                #         )
                #         logger.info(
                #             "Imputation preservation article dashboard saved as: "
                #             f"{preservation_path}"
                #         )

        logger.success("Missing value imputation completed successfully.")
        return imputed_obj


class MetaboVisualizerImputer(visualizer_classes.BaseMetaboVisualizer):
    """Visualization suite for evaluating imputation accuracy."""

    def __init__(
        self,
        raw_obj: core_classes.MetaboInt,
        imp_obj: core_classes.MetaboInt,
    ) -> None:
        """Initialize the imputation visualizer."""
        super().__init__(metabo_obj=imp_obj)
        self.raw_obj = raw_obj.astype(float).replace({0: np.nan})
        self.imp_obj = imp_obj.astype(float)

    def _plot_masked_distribution_fidelity(
        self,
        true_vals: np.ndarray,
        pred_vals: np.ndarray,
        metrics: dict[str, float] | None = None,
        ax: plt.Axes | None = None,
        compact_title: bool = False,
        article_compact: bool = False,
        show_legend: bool = False,
    ) -> plt.Figure | plt.Axes:
        """Plot score-aligned density fidelity for pooled masked nonblank values."""
        from scipy.stats import gaussian_kde

        return_fig = ax is None
        if ax is None:
            fig, ax = plt.subplots(figsize=(5.0, 4.0))

        truth = np.asarray(true_vals, dtype=float)
        reconstruction = np.asarray(pred_vals, dtype=float)
        truth = truth[np.isfinite(truth)]
        reconstruction = reconstruction[np.isfinite(reconstruction)]

        if truth.size < 2 or reconstruction.size < 2:
            ax.text(
                0.5,
                0.5,
                "Insufficient masked values for density estimation.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                bbox=pu.ai_ready_text_bbox(),
                zorder=10,
            )
            self._apply_standard_format(
                ax=ax,
                title="Distribution Fidelity"
                if compact_title
                else "Masked-Value Distribution Fidelity",
                xlabel="Log2 Intensity",
                ylabel="Relative Density",
                append_stage=False,
            )
            return fig if return_fig else ax

        x_min = min(float(np.min(truth)), float(np.min(reconstruction)))
        x_max = max(float(np.max(truth)), float(np.max(reconstruction)))
        margin = (x_max - x_min) * 0.10 if x_max > x_min else 1.0
        x_grid = np.linspace(x_min - margin, x_max + margin, 500)

        def _evaluate_kde(values: np.ndarray, seed: int) -> np.ndarray:
            kde_values = values.copy()
            if np.nanstd(kde_values) < 1e-6:
                rng = np.random.default_rng(seed)
                kde_values += rng.normal(0.0, 1e-4, size=kde_values.size)
            return gaussian_kde(kde_values)(x_grid)

        truth_density = _evaluate_kde(truth, seed=123)
        reconstruction_density = _evaluate_kde(reconstruction, seed=456)
        y_max = max(
            float(np.nanmax(truth_density)), float(np.nanmax(reconstruction_density))
        )

        pu.mark_preserve_alpha(ax)
        ax.fill_between(
            x_grid,
            truth_density,
            color="tab:gray",
            alpha=0.20,
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(
            x_grid,
            truth_density,
            color="black",
            linestyle="--",
            linewidth=0.75 if article_compact else 1.5,
            zorder=2,
        )
        ax.plot(
            x_grid,
            reconstruction_density,
            color=pu.PRIMARY_ACCENT_COLOR,
            linewidth=1.0 if article_compact else 2.0,
            zorder=3,
        )
        ax.set_ylim(0.0, y_max * 1.35)

        if metrics:
            jsd = su.finite_or_nan(metrics.get("JSD_Total"))
            wasserstein = su.finite_or_nan(
                metrics.get("Wasserstein_Normalized")
            )
            annotation_lines = []
            if np.isfinite(jsd):
                annotation_lines.append(
                    f"JSD: {jsd:.3f}"
                    if article_compact
                    else f"Jensen-Shannon distance: {jsd:.3f}"
                )
            if np.isfinite(wasserstein):
                annotation_lines.append(
                    f"W: {wasserstein:.3f}"
                    if article_compact
                    else f"Normalized Wasserstein distance: {wasserstein:.3f}"
                )
            if annotation_lines:
                ax.text(
                    0.96,
                    0.96,
                    "\n".join(annotation_lines),
                    transform=ax.transAxes,
                    fontsize=4.25 if article_compact else pu.DEFAULT_ANNOTATION_FONTSIZE,
                    horizontalalignment="right",
                    verticalalignment="top",
                    bbox=pu.ai_ready_text_bbox(pad=0.25 if article_compact else 0.4),
                    zorder=10,
                )

        if show_legend:
            import matplotlib.lines as mlines

            ax.legend(
                handles=[
                    mlines.Line2D(
                        [],
                        [],
                        color="black",
                        linestyle="--",
                        linewidth=0.75 if article_compact else 1.5,
                        label="Known masked values",
                    ),
                    mlines.Line2D(
                        [],
                        [],
                        color=pu.PRIMARY_ACCENT_COLOR,
                        linewidth=1.0 if article_compact else 2.0,
                        label="Reconstructed values",
                    ),
                ]
            )
            self._format_single_legend(
                ax=ax,
                group_title="Masked-value density",
                loc="lower right",
                bbox_to_anchor=None,
                max_item_rows=6,
            )

        self._apply_standard_format(
            ax=ax,
            title="Distribution Fidelity"
            if compact_title
            else "Masked-Value Distribution Fidelity",
            xlabel="Log2 Intensity",
            ylabel="Relative Density",
            append_stage=False,
        )
        return fig if return_fig else ax

    def _plot_kde_standalone_legend(
        self,
        ax: plt.Axes,
        legend_cols: int = 3,
        loc: str = "upper left",
        bbox_to_anchor: tuple[float, float] | None = (0.0, 1.0),
    ) -> plt.Axes:
        """Draw a standalone legend for masked-density fidelity overlays."""
        import matplotlib.lines as mlines

        ax.axis("off")
        handles = [
            mlines.Line2D(
                [],
                [],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Known masked values",
            ),
            mlines.Line2D(
                [],
                [],
                color=pu.PRIMARY_ACCENT_COLOR,
                linestyle="-",
                linewidth=2.0,
                label="Reconstructed values",
            ),
        ]
        ax.legend(
            handles=handles,
            title="Masked-value density",
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            ncol=legend_cols,
            frameon=True,
            edgecolor="k",
            borderaxespad=0.0,
        )
        self._format_single_legend(
            ax=ax,
            group_title="Masked-value density",
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            legend_cols=legend_cols,
            borderaxespad=0.0,
        )
        return ax

    def _plot_nrmse_scatter(
        self,
        true_vals: np.ndarray,
        pred_vals: np.ndarray,
        metrics: dict[str, float],
        method_name: str = "",
        axis_lims: tuple[float, float] | None = None,
        compact_title: bool = False,
        show_method_in_title: bool = True,
        show_colorbar: bool = True,
        article_compact: bool = False,
        ax: plt.Axes | None = None,
    ) -> plt.Figure | plt.Axes:
        """Plot hexbin scatter of true vs imputed values from mask test."""
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(5, 4))
        else:
            current_ax = ax
            fig = current_ax.figure

        if axis_lims is not None:
            ax_min, ax_max = axis_lims
            extent = (ax_min, ax_max, ax_min, ax_max)
            lim_min, lim_max = ax_min, ax_max
        else:
            d_min = min(true_vals.min(), pred_vals.min())
            d_max = max(true_vals.max(), pred_vals.max())
            margin = (d_max - d_min) * 0.05

            lim_min, lim_max = d_min - margin, d_max + margin
            extent = (lim_min, lim_max, lim_min, lim_max)

        color_map = pu.custom_linear_cmap(
            color_list=["white", pu.PRIMARY_ACCENT_COLOR],
            n_colors=256,
            cmin=0.1,
            cmax=1.0,
        )

        hb = current_ax.hexbin(
            x=true_vals,
            y=pred_vals,
            gridsize=40,
            extent=extent,
            cmap=color_map,
            mincnt=1,
        )

        current_ax.plot(
            [lim_min, lim_max],
            [lim_min, lim_max],
            color="tab:gray",
            linestyle="--",
            linewidth=0.6 if article_compact else 1.0,
            zorder=3,
        )

        threshold = metrics.get("Threshold")
        if threshold is not None:
            current_ax.axvline(
                x=threshold,
                color="tab:gray",
                linestyle="--",
                linewidth=0.6 if article_compact else 1.0,
            )
            current_ax.axhline(
                y=threshold,
                color="tab:gray",
                linestyle="--",
                linewidth=0.6 if article_compact else 1.0,
            )

        nrmse_total = float(metrics.get("NRMSE_Total", np.nan))
        nrmse_low = float(metrics.get("NRMSE_Low", np.nan))
        annot_text = (
            f"NRMSE Total: {nrmse_total:.4f}\n"
            f"NRMSE Low: {nrmse_low:.4f}"
        )

        current_ax.text(
            0.96,
            0.02,
            annot_text,
            transform=current_ax.transAxes,
            fontsize=4.25 if article_compact else pu.DEFAULT_ANNOTATION_FONTSIZE,
            verticalalignment="bottom",
            horizontalalignment="right",
            clip_on=False,
            bbox=pu.ai_ready_text_bbox(pad=0.25 if article_compact else 0.4),
            zorder=10,
        )

        title_str = "MAR Masked Simulation"
        if method_name:
            is_selected = method_name.strip().startswith("*")
            clean_name = method_name.replace("*", "").strip()
            clean_upper = clean_name.upper()
            if clean_upper in ("KNN", "LLS", "BPCA"):
                display = clean_upper
            elif clean_upper in ("QRILC", "QRLIC"):
                display = "QRILC"
            elif clean_upper in ("MINPROB", "PROB"):
                display = "MinProb"
            elif clean_upper == "MEDIAN":
                display = "Median"
            else:
                display = clean_name.title()

            if is_selected:
                display = f"* {display}"

            if show_method_in_title:
                title_str = display if compact_title else f"{title_str} ({display})"

        self._apply_standard_format(
            ax=current_ax,
            title=title_str,
            xlabel="True Intensity (Log2)",
            ylabel="Imputed Intensity (Log2)",
            append_stage=False,
        )

        if axis_lims is not None:
            current_ax.set_xlim(ax_min, ax_max)
            current_ax.set_ylim(ax_min, ax_max)

        if show_colorbar:
            cb = fig.colorbar(hb, ax=current_ax)
            cb.set_label("Log10(Count)")

        if ax is None:
            return fig
        return current_ax

    @staticmethod
    def _format_imputation_method_label(method_name: str) -> str:
        """Return a compact display label for an imputation method."""
        method_map = {
            "KNN": "KNN",
            "LLS": "LLS",
            "BPCA": "BPCA",
            "QRILC": "QRILC",
            "MINPROB": "MinProb",
            "PROB": "MinProb",
            "MEDIAN": "Median",
        }
        return method_map.get(str(method_name).upper(), str(method_name))

    @staticmethod
    def _method_key(method_name: str) -> str:
        """Normalize imputation method labels for robust matching."""
        return str(method_name).replace(" ", "").replace("-", "").upper()

    def plot_imputation_score_summary(
        self,
        results_dict: dict[str, tuple[dict[str, float], np.ndarray, np.ndarray]],
        best_method: str,
        ax: plt.Axes | None = None,
        show_legend: bool = False,
    ) -> plt.Axes:
        """Plot MAR imputation AUTO score components."""
        try:
            import patchworklib as pw
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError("patchworklib is required for this plot.")

        if ax is None:
            current_ax = pw.Brick(figsize=(4, 4), label="imputation_nrmse_summary")
        else:
            current_ax = ax

        summary_rows = []
        best_key = self._method_key(best_method)
        for method_name, (metrics, _, _) in results_dict.items():
            nrmse_total = float(metrics.get("NRMSE_Total", np.nan))
            summary_rows.append(
                {
                    "method": method_name,
                    "label": self._format_imputation_method_label(method_name),
                    "nrmse_total": nrmse_total,
                    "reconstruction_score": metrics.get("Reconstruction_Score"),
                    "distribution_preservation_score": metrics.get(
                        "Distribution_Preservation_Score"
                    ),
                    "sample_structure_score": metrics.get("Sample_Structure_Score"),
                    "auto_score": metrics.get("Auto_Score"),
                    "selected": self._method_key(method_name) == best_key,
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.replace([np.inf, -np.inf], np.nan)
        has_auto_score = summary_df["auto_score"].notna().any()
        if has_auto_score:
            summary_df = summary_df.dropna(subset=["auto_score"])
            summary_df = summary_df.sort_values(
                by=["auto_score", "nrmse_total", "label"],
                ascending=[False, True, True],
            ).reset_index(drop=True)
        else:
            summary_df = summary_df.dropna(subset=["nrmse_total"])
            summary_df = summary_df.sort_values(
                by=["nrmse_total", "label"], ascending=[False, True]
            ).reset_index(drop=True)

        if summary_df.empty:
            current_ax.axis("off")
            return current_ax

        y_pos = np.arange(len(summary_df))
        if has_auto_score:
            score_cols = [
                "reconstruction_score",
                "distribution_preservation_score",
                "sample_structure_score",
            ]
            weights = {
                "reconstruction_score": 0.65,
                "distribution_preservation_score": 0.20,
                "sample_structure_score": 0.15,
            }
            label_map = {
                "reconstruction_score": "Masked reconstruction",
                "distribution_preservation_score": "Distribution fidelity",
                "sample_structure_score": "Sample structure preservation",
            }
            color_map = {
                "reconstruction_score": pu.get_equivalent_hex(
                    pu.PRIMARY_ACCENT_COLOR, alpha=1.0
                ),
                "distribution_preservation_score": pu.get_equivalent_hex(
                    "tab:gray", alpha=0.75
                ),
                "sample_structure_score": pu.get_equivalent_hex(
                    "tab:gray", alpha=0.45
                ),
            }
            left = np.zeros(len(summary_df), dtype=float)
            for score_col in score_cols:
                left_start = left.copy()
                values = []
                for _, row in summary_df.iterrows():
                    available_weight = sum(
                        weights[col]
                        for col in score_cols
                        if np.isfinite(su.finite_or_nan(row.get(col)))
                    )
                    if available_weight <= 0:
                        values.append(0.0)
                        continue
                    score_value = np.clip(
                        su.finite_or_nan(row.get(score_col)), 0.0, 1.0
                    )
                    values.append(score_value * weights[score_col] / available_weight)
                values_arr = np.asarray(values, dtype=float)
                current_ax.barh(
                    y_pos,
                    values_arr,
                    left=left,
                    color=color_map[score_col],
                    edgecolor="k",
                    linewidth=0.5,
                    height=0.58,
                    label=label_map[score_col],
                )
                for y_idx, row in enumerate(summary_df.itertuples()):
                    score_value = su.finite_or_nan(getattr(row, score_col))
                    if values_arr[y_idx] < 0.10 or not np.isfinite(score_value):
                        continue
                    face_color = color_map[score_col]
                    current_ax.text(
                        left_start[y_idx] + values_arr[y_idx] / 2.0,
                        y_idx,
                        f"{score_value:.2f}",
                        va="center",
                        ha="center",
                        fontsize=9.5,
                        color=pu.get_contrast_color(face_color),
                        clip_on=True,
                    )
                left += values_arr

        else:
            selected_color = pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=1.0)
            background_color = pu.get_equivalent_hex("tab:gray", alpha=0.75)
            bar_colors = [
                selected_color if bool(row.selected) else background_color
                for row in summary_df.itertuples()
            ]
            current_ax.barh(
                y_pos,
                summary_df["nrmse_total"],
                color=bar_colors,
                edgecolor="k",
                linewidth=0.5,
                height=0.58,
            )
            left = summary_df["nrmse_total"].to_numpy(dtype=float)

        y_labels = [
            f"* {row.label}" if bool(row.selected) else str(row.label)
            for row in summary_df.itertuples()
        ]
        current_ax.set_yticks(y_pos)
        current_ax.set_yticklabels(y_labels)
        current_ax.invert_yaxis()

        if has_auto_score:
            x_upper = float(np.nanmax(left)) if left.size else 1.0
            x_upper = min(1.08, max(x_upper + 0.08, x_upper * 1.10, 0.20))
            current_ax.set_xlim(0, x_upper)
            label_values = summary_df["auto_score"].to_numpy(dtype=float)
        else:
            xmax = float(summary_df["nrmse_total"].max())
            current_ax.set_xlim(0, xmax * 1.2 if xmax > 0 else 1)
            label_values = summary_df["nrmse_total"].to_numpy(dtype=float)

        for y_idx, row in enumerate(summary_df.itertuples()):
            value = float(label_values[y_idx])
            label_x = float(left[y_idx])
            current_ax.text(
                min(label_x + 0.015, current_ax.get_xlim()[1] * 0.97),
                y_idx,
                f"{value:.3f}" if has_auto_score else f"{value:.4f}",
                va="center",
                ha="left",
                fontsize=10.5,
            )

        title = (
            "Auto Imputation Method Selection"
            if has_auto_score
            else "MAR Imputer Ranking"
        )
        if show_legend and has_auto_score:
            legend_handles = [
                mpatches.Patch(
                    facecolor=color_map[score_col],
                    edgecolor="k",
                    linewidth=0.5,
                    label=label_map[score_col],
                )
                for score_col in score_cols
            ]
            current_ax.legend(handles=legend_handles)
            self._format_single_legend(
                ax=current_ax,
                group_title="AUTO imputation score components",
                loc="lower right",
                bbox_to_anchor=None,
                max_item_rows=6,
            )
        self._apply_standard_format(
            ax=current_ax,
            title=title,
            xlabel=(
                "Weighted contribution to overall score"
                if has_auto_score
                else "NRMSE Total"
            ),
            append_stage=False,
        )
        current_ax.tick_params(axis="y", length=0)
        return current_ax

    def plot_imputation_score_legend(
        self,
        ax: plt.Axes,
        legend_cols: int | None = None,
        fontsize: float = 9.0,
        title_fontsize: float = 10.0,
        article_compact: bool = False,
    ) -> plt.Axes:
        """Draw a standalone legend for MAR imputation score components."""
        import matplotlib.patches as mpatches

        ax.axis("off")
        legend_linewidth = 0.5 if article_compact else 1.0
        handles = [
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=1.0),
                edgecolor="k",
                linewidth=legend_linewidth,
                label="Masked reconstruction",
            ),
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=0.75),
                edgecolor="k",
                linewidth=legend_linewidth,
                label="Distribution fidelity",
            ),
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=0.45),
                edgecolor="k",
                linewidth=legend_linewidth,
                label="Sample structure preservation",
            ),
        ]
        ax.legend(handles=handles)
        self._format_single_legend(
            ax=ax,
            group_title="AUTO imputation score components",
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            legend_cols=legend_cols,
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

    def plot_imputation_dashboard_legend(
        self,
        ax: plt.Axes,
        fontsize: float = 9.0,
        title_fontsize: float = 10.0,
    ) -> plt.Axes:
        """Draw score-component and masked-density legends in one panel."""
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches

        score_handles = [
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=1.0),
                edgecolor="k",
                linewidth=1.0,
                label="Masked reconstruction",
            ),
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=0.75),
                edgecolor="k",
                linewidth=1.0,
                label="Distribution fidelity",
            ),
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=0.45),
                edgecolor="k",
                linewidth=1.0,
                label="Sample structure preservation",
            ),
        ]
        density_handles = [
            mlines.Line2D(
                [],
                [],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Known masked values",
            ),
            mlines.Line2D(
                [],
                [],
                color=pu.PRIMARY_ACCENT_COLOR,
                linestyle="-",
                linewidth=2.0,
                label="Reconstructed values",
            ),
        ]

        self._plot_grouped_standalone_legends(
            ax=ax,
            legend_groups=[
                ("AUTO imputation score components", score_handles),
                ("Masked-value density reference", density_handles),
            ],
            loc="upper left",
            start_bbox=(0.0, 1.0),
            row_gap=0.04,
            max_item_rows=6,
            borderaxespad=0.0,
            fontsize=fontsize,
            title_fontsize=title_fontsize,
        )
        return ax

    def plot_imputation_article_score_legend(self, ax: plt.Axes) -> plt.Axes:
        """Draw a right-side score legend for the imputation article panel."""
        return self.plot_imputation_score_legend(
            ax=ax,
            fontsize=pu.ARTICLE_LEGEND_FONTSIZE,
            title_fontsize=pu.ARTICLE_LEGEND_TITLE_FONTSIZE,
            article_compact=True,
        )

    def plot_imputation_article_density_legend(self, ax: plt.Axes) -> plt.Axes:
        """Draw a right-side density legend for the imputation article panel."""
        import matplotlib.lines as mlines

        ax.axis("off")
        density_handles = [
            mlines.Line2D(
                [],
                [],
                color="black",
                linestyle="--",
                linewidth=0.75,
                label="Known masked values",
            ),
            mlines.Line2D(
                [],
                [],
                color=pu.PRIMARY_ACCENT_COLOR,
                linestyle="-",
                linewidth=1.0,
                label="Reconstructed values",
            ),
        ]
        ax.legend(handles=density_handles)
        self._format_single_legend(
            ax=ax,
            group_title="Masked-value density reference",
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            max_item_rows=6,
            borderaxespad=0.0,
            fontsize=pu.ARTICLE_LEGEND_FONTSIZE,
            title_fontsize=pu.ARTICLE_LEGEND_TITLE_FONTSIZE,
            handlelength=1.0,
            handletextpad=0.3,
            labelspacing=0.25,
            borderpad=0.3,
        )
        self._apply_article_legend_style(
            ax=ax,
            fontsize=pu.ARTICLE_LEGEND_FONTSIZE,
            title_fontsize=pu.ARTICLE_LEGEND_TITLE_FONTSIZE,
        )
        return ax

    def _resolve_article_benchmark_item(
        self,
        results_dict: dict[str, tuple[dict[str, float], np.ndarray, np.ndarray]],
        best_method: str,
    ) -> tuple[str, tuple[dict[str, float], np.ndarray, np.ndarray]] | None:
        """Return the selected AUTO benchmark tuple without changing candidate order."""
        selected_key = self._method_key(best_method)
        for method_name, item in results_dict.items():
            if self._method_key(method_name) == selected_key:
                return method_name, item
        return next(iter(results_dict.items()), None)

    def plot_imputation_reconstruction_article_dashboard(
        self,
        results_dict: dict[str, tuple[dict[str, float], np.ndarray, np.ndarray]],
        best_method: str,
    ) -> object | None:
        """Create a compact AUTO selection and masked-reconstruction manuscript panel."""
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping imputation article panel.")
            return None

        best_item = self._resolve_article_benchmark_item(results_dict, best_method)
        if best_item is None:
            return None

        method_name, (metrics, true_vals, pred_vals) = best_item
        pw.clear()
        panel_height = 1.75

        summary_ax = pw.Brick(
            figsize=(1.85, panel_height), label="article_imputation_summary"
        )
        self.plot_imputation_score_summary(
            results_dict=results_dict,
            best_method=best_method,
            ax=summary_ax,
            show_legend=False,
        )
        self._apply_article_panel_format(
            summary_ax,
            title="Auto Imputation Method Selection",
        )

        scatter_ax = pw.Brick(
            figsize=(1.85, panel_height), label="article_imputation_masked_nrmse"
        )
        self._plot_nrmse_scatter(
            true_vals=true_vals,
            pred_vals=pred_vals,
            metrics=metrics,
            method_name=self._format_imputation_method_label(method_name),
            compact_title=False,
            show_method_in_title=False,
            show_colorbar=False,
            article_compact=True,
            ax=scatter_ax,
        )
        self._apply_article_panel_format(
            scatter_ax,
            title="MAR Masked Simulation",
        )
        scatter_ax.set_xlabel("True log2 intensity")
        scatter_ax.set_ylabel("Reconstructed log2 intensity")

        legend_ax = pw.Brick(
            figsize=(1.30, panel_height), label="article_imputation_score_legend"
        )
        self.plot_imputation_article_score_legend(ax=legend_ax)
        return summary_ax | scatter_ax | legend_ax

    def plot_imputation_preservation_article_dashboard(
        self,
        results_dict: dict[str, tuple[dict[str, float], np.ndarray, np.ndarray]],
        best_method: str,
    ) -> object | None:
        """Create a compact fidelity and sample-structure manuscript panel."""
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping imputation article panel.")
            return None

        best_item = self._resolve_article_benchmark_item(results_dict, best_method)
        if best_item is None:
            return None

        _, (metrics, true_vals, pred_vals) = best_item
        pw.clear()
        panel_height = 1.75

        density_ax = pw.Brick(
            figsize=(1.85, panel_height), label="article_imputation_density"
        )
        self._plot_masked_distribution_fidelity(
            true_vals=true_vals,
            pred_vals=pred_vals,
            metrics=metrics,
            ax=density_ax,
            compact_title=False,
            article_compact=True,
            show_legend=False,
        )
        self._apply_article_panel_format(
            density_ax,
            title="Distribution Fidelity",
        )

        sample_ax = pw.Brick(
            figsize=(1.85, panel_height), label="article_imputation_sample_structure"
        )
        pu.plot_sample_structure_change_map(
            ax=sample_ax,
            raw_obj=self.raw_obj,
            transformed_obj=self.imp_obj,
            structure_metrics=metrics,
            title="Sample Structure Change Map",
            compact_style=True,
        )
        self._apply_article_panel_format(
            sample_ax,
            title="Sample Structure Change Map",
        )

        legend_ax = pw.Brick(
            figsize=(1.30, panel_height), label="article_imputation_density_legend"
        )
        self.plot_imputation_article_density_legend(ax=legend_ax)
        return density_ax | sample_ax | legend_ax

    def plot_imputation_structure_metrics(
        self,
        results_dict: dict[str, tuple[dict[str, float], np.ndarray, np.ndarray]],
        best_method: str,
        metric_group: str = "structure",
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot preservation metrics for MAR imputation candidates."""
        try:
            import patchworklib as pw
        except ImportError:
            raise ImportError("patchworklib is required for this plot.")

        if ax is None:
            current_ax = pw.Brick(figsize=(4.0, 4.0), label="imputation_structure")
        else:
            current_ax = ax

        rows = []
        best_key = self._method_key(best_method)
        for method_name, (metrics, _, _) in results_dict.items():
            rows.append(
                {
                    "method": method_name,
                    "label": self._format_imputation_method_label(method_name),
                    "Jensen-Shannon preservation": metrics.get("JSD_Score"),
                    "Wasserstein preservation": metrics.get("Wasserstein_Score"),
                    "Trustworthiness": metrics.get("Trustworthiness"),
                    "Distance rank preservation": metrics.get(
                        "Distance_Rank_Preservation"
                    ),
                    "Distance scale preservation": metrics.get(
                        "Distance_Scale_Preservation"
                    ),
                    "auto_score": metrics.get("Auto_Score"),
                    "selected": self._method_key(method_name) == best_key,
                }
            )

        group_key = str(metric_group).lower().strip()
        if group_key in {"distribution", "dist"}:
            metric_cols = [
                "Jensen-Shannon preservation",
                "Wasserstein preservation",
            ]
            metric_labels = [
                "Jensen-Shannon\npreservation",
                "Wasserstein\npreservation",
            ]
            title = "Distribution Preservation"
        elif group_key in {"structure", "sample", "sample_structure"}:
            metric_cols = [
                "Trustworthiness",
                "Distance rank preservation",
                "Distance scale preservation",
            ]
            metric_labels = [
                "Trustworthiness",
                "Distance-rank\npreservation",
                "Distance-scale\npreservation",
            ]
            title = "Sample Structure Preservation"
        else:
            raise ValueError(
                "metric_group must be 'distribution' or 'structure'."
            )

        metric_df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
        for col in ["auto_score", *metric_cols]:
            metric_df[col] = pd.to_numeric(metric_df[col], errors="coerce")
        metric_df = metric_df.dropna(subset=metric_cols, how="all")
        metric_df = metric_df.sort_values(
            by=["auto_score", "label"], ascending=[False, True]
        ).reset_index(drop=True)

        if metric_df.empty:
            current_ax.axis("off")
            return current_ax

        matrix = metric_df[metric_cols].to_numpy(dtype=float)
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
        current_ax.set_xticks(np.arange(len(metric_cols)))
        current_ax.set_xticklabels(metric_labels)
        current_ax.set_yticks(np.arange(len(metric_df)))
        current_ax.set_yticklabels(
            [
                f"* {row.label}" if bool(row.selected) else str(row.label)
                for row in metric_df.itertuples()
            ]
        )
        current_ax.set_xticks(np.arange(-0.5, len(metric_cols), 1), minor=True)
        current_ax.set_yticks(np.arange(-0.5, len(metric_df), 1), minor=True)
        grid_lw = 1.0
        current_ax.grid(which="minor", color="k", linestyle="-", linewidth=grid_lw)
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
            ax=current_ax,
            title=title,
            xlabel="",
            ylabel="",
            append_stage=False,
        )
        pu.rotate_xticks_if_overlapping(current_ax)
        for spine in current_ax.spines.values():
            spine.set_visible(False)
        current_ax.tick_params(axis="both", length=0)
        return current_ax

    def plot_imputation_preservation_scorecard(
        self,
        results_dict: dict[str, tuple[dict[str, float], np.ndarray, np.ndarray]],
        best_method: str,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot distribution and sample-structure preservation scores together."""
        try:
            import patchworklib as pw
        except ImportError:
            raise ImportError("patchworklib is required for this plot.")

        if ax is None:
            current_ax = pw.Brick(figsize=(4.8, 4.0), label="imputation_scorecard")
        else:
            current_ax = ax

        best_key = self._method_key(best_method)
        rows = []
        for method_name, (metrics, _, _) in results_dict.items():
            rows.append(
                {
                    "method": method_name,
                    "label": self._format_imputation_method_label(method_name),
                    "Jensen-Shannon preservation": metrics.get("JSD_Score"),
                    "Wasserstein preservation": metrics.get("Wasserstein_Score"),
                    "Trustworthiness": metrics.get("Trustworthiness"),
                    "Distance rank preservation": metrics.get(
                        "Distance_Rank_Preservation"
                    ),
                    "Distance scale preservation": metrics.get(
                        "Distance_Scale_Preservation"
                    ),
                    "auto_score": metrics.get("Auto_Score"),
                    "selected": self._method_key(method_name) == best_key,
                }
            )

        metric_cols = [
            "Jensen-Shannon preservation",
            "Wasserstein preservation",
            "Trustworthiness",
            "Distance rank preservation",
            "Distance scale preservation",
        ]
        metric_labels = [
            "Jensen-Shannon\npreservation",
            "Wasserstein\npreservation",
            "Trustworthiness",
            "Distance-rank\npreservation",
            "Distance-scale\npreservation",
        ]
        metric_df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
        for col in ["auto_score", *metric_cols]:
            metric_df[col] = pd.to_numeric(metric_df[col], errors="coerce")
        metric_df = metric_df.dropna(subset=metric_cols, how="all")
        metric_df = metric_df.sort_values(
            by=["auto_score", "label"], ascending=[False, True]
        ).reset_index(drop=True)

        if metric_df.empty:
            current_ax.axis("off")
            return current_ax

        matrix = metric_df[metric_cols].to_numpy(dtype=float)
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
        current_ax.set_xticks(np.arange(len(metric_cols)))
        current_ax.set_xticklabels(metric_labels)
        current_ax.set_yticks(np.arange(len(metric_df)))
        current_ax.set_yticklabels(
            [
                f"* {row.label}" if bool(row.selected) else str(row.label)
                for row in metric_df.itertuples()
            ]
        )
        n_rows, n_cols = matrix.shape
        for x_pos in np.arange(-0.5, n_cols, 1.0):
            current_ax.plot(
                [x_pos, x_pos],
                [-0.5, n_rows - 0.5],
                color="k",
                linewidth=1.0,
                zorder=3,
            )
        for y_pos in np.arange(-0.5, n_rows, 1.0):
            current_ax.plot(
                [-0.5, n_cols - 0.5],
                [y_pos, y_pos],
                color="k",
                linewidth=1.0,
                zorder=3,
            )

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

        dist_color = pu.get_equivalent_hex("tab:gray", alpha=0.75)
        struct_color = pu.get_equivalent_hex("tab:gray", alpha=0.45)
        group_specs = [
            (-0.5, 2.0, "Distribution fidelity", dist_color),
            (1.5, 3.0, "Sample structure preservation", struct_color),
        ]
        for x_start, width, label, face_color in group_specs:
            current_ax.add_patch(
                plt.Rectangle(
                    (x_start, -1.05),
                    width,
                    0.38,
                    facecolor=face_color,
                    edgecolor="k",
                    linewidth=1.0,
                    zorder=5,
                    clip_on=False,
                )
            )
            current_ax.text(
                x_start + width / 2.0,
                -0.86,
                label,
                ha="center",
                va="center",
                fontsize=9.5,
                color=pu.get_contrast_color(face_color),
                zorder=6,
                clip_on=False,
            )

        current_ax.set_ylim(len(metric_df) - 0.5, -1.18)
        self._apply_standard_format(
            ax=current_ax,
            title="Candidate Preservation Scorecard",
            xlabel="",
            ylabel="",
            append_stage=False,
            tick_fontsize=12,
        )
        pu.rotate_xticks_if_overlapping(current_ax)
        for spine in current_ax.spines.values():
            spine.set_visible(False)
        current_ax.tick_params(axis="both", length=0)
        return current_ax

    def plot_imputation_auto_dashboard(
        self,
        results_dict: dict[str, tuple[dict[str, float], np.ndarray, np.ndarray]],
        best_method: str,
    ) -> object | None:
        """Create the final MAR imputation Auto-selection dashboard."""
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("Module 'patchworklib' not found. Skipping grid.")
            return None

        if not results_dict:
            return None

        g_min, g_max = float("inf"), float("-inf")
        sorted_items = sorted(
            results_dict.items(),
            key=lambda item: (
                float(item[1][0].get("Auto_Score", np.nan)),
                -float(item[1][0].get("NRMSE_Total", np.nan)),
                self._format_imputation_method_label(item[0]),
            ),
            reverse=True,
        )

        for _, (_, true_vals, pred_vals) in sorted_items:
            g_min = min(g_min, true_vals.min(), pred_vals.min())
            g_max = max(g_max, true_vals.max(), pred_vals.max())

        margin = (g_max - g_min) * 0.05
        shared_lims = (g_min - margin, g_max + margin)
        best_key = self._method_key(best_method)
        best_item = next(
            (
                item
                for item in sorted_items
                if self._method_key(item[0]) == best_key
            ),
            sorted_items[0],
        )

        pw.clear()
        ax_summary = pw.Brick(figsize=(4.3, 4.0), label="imputation_score_summary")
        self.plot_imputation_score_summary(
            results_dict=results_dict,
            best_method=best_method,
            ax=ax_summary,
        )
        ax_scorecard = pw.Brick(figsize=(6.0, 4.0), label="imputation_scorecard")
        self.plot_imputation_preservation_scorecard(
            results_dict=results_dict,
            best_method=best_method,
            ax=ax_scorecard,
        )
        ax_legend = pw.Brick(figsize=(2.2, 4.0), label="imputation_dashboard_legend")
        self.plot_imputation_dashboard_legend(ax=ax_legend)

        method_name, (metrics, true_vals, pred_vals) = best_item
        ax_best_scatter = pw.Brick(figsize=(4.0, 4.0), label="best_nrmse_scatter")
        self._plot_nrmse_scatter(
            true_vals,
            pred_vals,
            metrics,
            method_name=self._format_imputation_method_label(method_name),
            axis_lims=shared_lims,
            compact_title=False,
            show_method_in_title=False,
            ax=ax_best_scatter,
        )

        ax_density = pw.Brick(
            figsize=(4.25, 4.0), label="imputation_masked_density"
        )
        self._plot_masked_distribution_fidelity(
            true_vals=true_vals,
            pred_vals=pred_vals,
            metrics=metrics,
            ax=ax_density,
            compact_title=False,
        )
        ax_sample_structure = pw.Brick(
            figsize=(4.25, 4.0), label="imputation_sample_structure_preservation"
        )
        pu.plot_sample_structure_change_map(
            ax=ax_sample_structure,
            raw_obj=self.raw_obj,
            transformed_obj=self.imp_obj,
            structure_metrics=metrics,
            title="Sample Structure Change Map",
        )

        top_row = ax_summary | ax_scorecard | ax_legend
        diagnostic_row = ax_best_scatter | ax_density | ax_sample_structure

        return top_row / diagnostic_row

    def plot_imputation_method_dashboard(
        self,
        metrics: dict[str, float],
        true_vals: np.ndarray,
        pred_vals: np.ndarray,
        method_name: str,
    ) -> object | None:
        """Create a fixed-method imputation dashboard without Auto score panels."""
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("Module 'patchworklib' not found. Skipping grid.")
            return None

        if true_vals is None or pred_vals is None or len(true_vals) == 0:
            return None

        d_min = min(float(np.nanmin(true_vals)), float(np.nanmin(pred_vals)))
        d_max = max(float(np.nanmax(true_vals)), float(np.nanmax(pred_vals)))
        margin = (d_max - d_min) * 0.05 if d_max > d_min else 1.0
        shared_lims = (d_min - margin, d_max + margin)

        pw.clear()
        ax_scatter = pw.Brick(figsize=(4.0, 4.0), label="method_nrmse_scatter")
        self._plot_nrmse_scatter(
            true_vals,
            pred_vals,
            metrics,
            method_name=self._format_imputation_method_label(method_name),
            axis_lims=shared_lims,
            compact_title=False,
            show_method_in_title=False,
            ax=ax_scatter,
        )

        ax_density = pw.Brick(figsize=(4.0, 4.0), label="method_masked_density")
        self._plot_masked_distribution_fidelity(
            true_vals=true_vals,
            pred_vals=pred_vals,
            metrics=metrics,
            ax=ax_density,
            compact_title=False,
        )

        ax_sample_structure = pw.Brick(
            figsize=(4.0, 4.0), label="method_sample_structure_preservation"
        )
        pu.plot_sample_structure_change_map(
            ax=ax_sample_structure,
            raw_obj=self.raw_obj,
            transformed_obj=self.imp_obj,
            structure_metrics=metrics,
            title="Sample Structure Change Map",
        )

        ax_legend = pw.Brick(figsize=(1.2, 8.0), label="method_kde_legend")
        self._plot_kde_standalone_legend(
            ax=ax_legend,
            legend_cols=1,
            loc="center left",
            bbox_to_anchor=(0.0, 0.5),
        )

        return ax_scatter | ax_density | ax_sample_structure | ax_legend

    def plot_imputation_nrmse_appendix_grid(
        self,
        results_dict: dict[str, tuple[dict[str, float], np.ndarray, np.ndarray]],
    ) -> object | None:
        """Create a 2 x 3 appendix grid of candidate masked-reconstruction plots."""
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("Module 'patchworklib' not found. Skipping grid.")
            return None

        if not results_dict:
            return None

        sorted_items = sorted(
            results_dict.items(),
            key=lambda item: (
                float(item[1][0].get("Auto_Score", np.nan)),
                -float(item[1][0].get("NRMSE_Total", np.nan)),
                self._format_imputation_method_label(item[0]),
            ),
            reverse=True,
        )
        g_min, g_max = float("inf"), float("-inf")
        for _, (_, true_vals, pred_vals) in sorted_items:
            g_min = min(g_min, true_vals.min(), pred_vals.min())
            g_max = max(g_max, true_vals.max(), pred_vals.max())

        margin = (g_max - g_min) * 0.05
        shared_lims = (g_min - margin, g_max + margin)
        best_key = self._method_key(sorted_items[0][0])

        pw.clear()
        scatter_bricks: list[object] = []
        for idx, (method_name, (metrics, true_vals, pred_vals)) in enumerate(
            sorted_items[:6]
        ):
            ax_scatter = pw.Brick(
                figsize=(3.6, 3.6), label=f"nrmse_appendix_scatter_{idx + 1}"
            )
            display_method = self._format_imputation_method_label(method_name)
            if self._method_key(method_name) == best_key:
                display_method = f"* {display_method}"
            self._plot_nrmse_scatter(
                true_vals,
                pred_vals,
                metrics,
                method_name=display_method,
                axis_lims=shared_lims,
                compact_title=False,
                ax=ax_scatter,
            )
            scatter_bricks.append(ax_scatter)

        while len(scatter_bricks) < 6:
            ax_blank = pw.Brick(
                figsize=(3.6, 3.6),
                label=f"nrmse_appendix_scatter_blank_{len(scatter_bricks)}",
            )
            ax_blank.axis("off")
            scatter_bricks.append(ax_blank)

        return (
            scatter_bricks[0] | scatter_bricks[1] | scatter_bricks[2]
        ) / (scatter_bricks[3] | scatter_bricks[4] | scatter_bricks[5])

    def plot_imputation_density_overlay(
        self,
        true_vals: np.ndarray,
        pred_vals: np.ndarray,
        metrics: dict[str, float] | None = None,
    ) -> object | None:
        """Create a standalone score-aligned masked-density fidelity panel."""
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping density overlay.")
            return None

        pw.clear()

        ax_density = pw.Brick(figsize=(5.0, 4.0), label="masked_density_fidelity")
        self._plot_masked_distribution_fidelity(
            true_vals=true_vals,
            pred_vals=pred_vals,
            metrics=metrics,
            ax=ax_density,
            compact_title=False,
        )
        ax_legend = pw.Brick(figsize=(2.2, 4), label="masked_density_legend")
        self._plot_kde_standalone_legend(ax=ax_legend, legend_cols=1)

        return ax_density | ax_legend
