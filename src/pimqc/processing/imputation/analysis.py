"""Mechanism-aware missing-value imputation and candidate selection.

MetaboIntImputer applies MNAR methods such as QRILC or censored-value estimates
and MAR methods including Median, MinProb, KNN, LLS, BPCA, and AUTO selection.
It evaluates masked reconstruction, distribution fidelity, and sample-structure
preservation, then records the selected strategy and imputed stage metrics.
"""

import math
from functools import cached_property
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
from loguru import logger
from sklearn.impute import KNNImputer

from ...config import resolve_stage_config
from ...constants import DEFAULT_RANDOM_SEED
from ...core import model
from ...runtime import log_execution_time
from ...statistics import metrics as su
from ...statistics import sample_structure as structure_stats
from ...statistics import selection as selection_utils
from ..stage import StageResult
from .methods import IMPUTATION_METHODS
from .runner import ImputationStageRunner


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

        dw = (
            rx_inv
            + tau * t_mat.T @ pa @ rx_inv
            + np.diag(model["alpha"]) / rows
        )
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
        alpha_new = (2 * model["galpha0"] + cols) / np.maximum(
            alpha_denom, 1e-12
        )

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


class MetaboIntImputer(model.MetaboInt):
    """Missing value imputation engine with hybrid stratified evaluation."""

    _metadata = ["attrs", "stats"]
    _RUNTIME_CONFIG_KEYS = frozenset(
        {
            "mar_method",
            "mnar_method",
            "mnar_fraction",
            "knn_neighbors",
            "lls_neighbors",
            "bpca_components",
            "bpca_max_iter",
            "bpca_tol",
            "sim_mask_ratio",
            "global_seed",
        }
    )

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

        imp_configs = resolve_stage_config(
            pipeline_params,
            "MetaboIntImputer",
            {
                "mar_method": "Auto",
                "mnar_method": "QRILC",
                "mnar_fraction": 0.5,
                "knn_neighbors": 5,
                "lls_neighbors": 15,
                "bpca_components": 2,
                "bpca_max_iter": 100,
                "bpca_tol": 1e-4,
                "sim_mask_ratio": 0.05,
            },
            {
                "mar_method": mar_method,
                "mnar_method": mnar_method,
                "mnar_fraction": mnar_fraction,
                "knn_neighbors": knn_neighbors,
                "lls_neighbors": lls_neighbors,
                "bpca_components": bpca_components,
                "bpca_max_iter": bpca_max_iter,
                "bpca_tol": bpca_tol,
                "sim_mask_ratio": sim_mask_ratio,
            },
        )

        self.attrs.update(imp_configs)

    @property
    def _constructor(self) -> type["MetaboIntImputer"]:
        """Return the class constructor for stable subclassing."""
        return MetaboIntImputer

    # =========================================================================
    # Imputation-related Metrics
    # =========================================================================
    def calc_imp_quality_metrics(
        self, raw_obj: model.MetaboInt, imp_obj: model.MetaboInt
    ) -> dict[str, Any]:
        """
        Calculate post-imputation distribution QA metrics for the passport.

        Computes Jensen-Shannon and Wasserstein distances for final QC and
        sample
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

            # Data Before Imputation (All)
            obs = r_slice[~np.isnan(r_slice)]

            # Data After Imputation (All)
            imp_all = i_slice[~np.isnan(i_slice)]

            # Imputed Data (Patches only)
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

    # =========================================================================
    # Core Algorithms (Log2 Space)
    # =========================================================================
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

        # Broadcast the computed minimums to fill NaNs
        if imp_mode in ("row", "row-wise", "row min"):
            return df_log.apply(lambda x: x.fillna(target_mins[x.name]), axis=1)
        else:
            return df_log.fillna(target_mins)

    @staticmethod
    def impute_by_qrilc(
        df_log: pd.DataFrame,
        tune_sigma: float = 1.0,
        global_seed: int = DEFAULT_RANDOM_SEED,
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
            res_arr[missing_mask, col_idx] = np.clip(
                drawn, a_min=0.0, a_max=None
            )

        return pd.DataFrame(res_arr, index=df_log.index, columns=df_log.columns)

    @staticmethod
    def impute_by_knn(
        df_log: pd.DataFrame, n_neighbors: int = 5
    ) -> pd.DataFrame:
        """Impute missing values using K-Nearest Neighbors algorithm."""
        # Scale neighbor count for isolated small groups.
        n_samples = df_log.shape[1]
        safe_k = min(n_neighbors, n_samples - 1)

        if safe_k < 1:
            # Fallback to feature median if insufficient neighbors (e.g., n=1)
            return df_log.apply(lambda x: x.fillna(x.median()), axis=1).fillna(
                0.0
            )

        imputer = KNNImputer(n_neighbors=safe_k, weights="distance")
        arr_imp = imputer.fit_transform(df_log.T).T

        return pd.DataFrame(arr_imp, index=df_log.index, columns=df_log.columns)

    @staticmethod
    def impute_by_lls(
        df_log: pd.DataFrame, n_neighbors: int = 15
    ) -> pd.DataFrame:
        """Impute missing values using Local Least Squares (LLS) regression.

        Finds 'k' complete features that are highly correlated with the target
        feature, and constructs a local linear regression model to predict
        the missing values.
        """
        arr_log = df_log.values
        res_arr = arr_log.copy()

        # Identify complete features to serve as the candidate neighbor pool
        complete_mask = ~np.isnan(arr_log).any(axis=1)
        complete_features = arr_log[complete_mask]
        n_complete = complete_features.shape[0]

        # Fallback: If dataset is too sparse and lacks complete features
        if n_complete < 2:
            logger.debug(
                "Insufficient complete features for LLS. "
                "Falling back to median."
            )
            return df_log.apply(lambda x: x.fillna(x.median()), axis=1).fillna(
                0.0
            )

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

            # Vectorized Pearson correlation to find closest complete
            # features
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

            # Select top K neighbors
            top_k_idx = np.argsort(corr)[-safe_k:]

            # Construct matrices for Least Squares estimation
            # A_mat: neighbors' observed values (Shape: n_neighbors x
            # n_observed)
            A_mat = A_obs[top_k_idx]
            # B_mat: neighbors' values at target's missing positions
            B_mat = complete_features[top_k_idx][:, missing_mask]

            # Solve linear system: A_mat.T * x = w_obs
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
            return df_log.apply(lambda x: x.fillna(x.median()), axis=1).fillna(
                0.0
            )

        safe_components = max(
            1, min(int(n_components), arr.shape[0], arr.shape[1])
        )
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
    def impute_by_minprob(
        df_log: pd.DataFrame,
        global_seed: int = DEFAULT_RANDOM_SEED,
    ) -> pd.DataFrame:
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
            drawn = rng.normal(
                loc=shift_mean, scale=shift_std, size=s.isna().sum()
            )

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

        # Impute QC using strictly QC context
        if not qc_cols.empty:
            res_qc = imp_func(df_slice[qc_cols], **kwargs)
            if res_qc.isna().any().any():
                global_fallback = imp_func(df_slice, **kwargs)
                res_qc = res_qc.combine_first(global_fallback[qc_cols])
            res_dfs.append(res_qc)

        # Impute Samples using strictly Sample context
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

    # =========================================================================
    # Evaluation Logic (Hybrid Masking & Stratified NRMSE)
    # =========================================================================

    @staticmethod
    def generate_gmm_noise_mask(
        df_log: pd.DataFrame,
        mask_ratio: float,
        noise_factor: float = 1.5,
        global_seed: int = DEFAULT_RANDOM_SEED,
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
                return pd.DataFrame(
                    False, index=df_log.index, columns=df_log.columns
                )

            gmm = GaussianMixture(n_components=2, random_state=global_seed)
            gmm.fit(valid_data)
            lower_cluster_idx = np.argmin(gmm.means_)

            base_prob = gmm.predict_proba(valid_data)[:, lower_cluster_idx]
            final_score = base_prob + rng.uniform(
                0, noise_factor, size=base_prob.shape
            )
            cutoff_score = np.sort(final_score)[-target_nas]

            mask_arr[valid_mask] = final_score >= cutoff_score
            return pd.DataFrame(
                mask_arr, index=df_log.index, columns=df_log.columns
            )

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

                base_prob = gmm.predict_proba(valid_data_b)[
                    :, lower_cluster_idx
                ]
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

        return pd.DataFrame(
            mask_arr, index=df_log.index, columns=df_log.columns
        )

    @staticmethod
    def compute_stratified_nrmse(
        df_true: pd.DataFrame,
        df_imp: pd.DataFrame,
        mask_df: pd.DataFrame,
        lod_q: float = 0.25,
    ) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
        """Calculate NRMSE by feature-median abundance strata.

        Masked observations inherit the median of their source feature. The
        ``lod_q`` quantile of those feature medians separates the low and high
        strata; it is an evaluation cutoff, not an analytical detection limit.
        """
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

        # Compile the metrics dictionary
        metrics = {
            "NRMSE_Total": _get_nrmse(t_all, p_all),
            "NRMSE_Low": _get_nrmse(t_all[low_m], p_all[low_m]),
            "NRMSE_High": _get_nrmse(t_all[hi_m], p_all[hi_m]),
            "Count_Low": int(np.sum(low_m)),
            "Count_High": int(np.sum(hi_m)),
            "Threshold": float(lod_val),
            "Threshold_Quantile": float(lod_q),
        }

        # Return exactly 3 objects to match the unpacking logic
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
                fill_value = float(
                    np.log2(np.nanmin(positive_values) * 0.5 + 1.0)
                )
        return masked_df.fillna(fill_value)

    @staticmethod
    def _reference_improvement_score(reference: object, value: object) -> float:
        """Score improvement over a lower-is-better reference metric."""
        score = su.relative_change_lower_better(reference, value)
        return (
            float(np.clip(score, 0.0, 1.0))
            if np.isfinite(score)
            else float("nan")
        )

    def _evaluate_imputation_candidate(
        self,
        df_log: pd.DataFrame,
        idx_mar: pd.Index,
        target_cols: pd.Index,
        method: str,
        ratio: float = 0.05,
        global_seed: int = DEFAULT_RANDOM_SEED,
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
        if method_key in (
            "HALFGLOBALMIN",
            "HALFGLOBALMINREFERENCE",
            "LOWVALUEREF",
        ):
            imp_res = self._low_value_reference_impute(masked_df)
        else:
            method_spec = IMPUTATION_METHODS.resolve(method)
            method_key = method_spec.key

        if method_key == "MINPROB":
            imp_res = self._apply_isolated(
                masked_df, self.impute_by_minprob, global_seed=global_seed
            )
        elif method_key == "KNN":
            k_val = self.attrs.get("knn_neighbors", 5)
            imp_res = self._apply_isolated(
                masked_df, self.impute_by_knn, n_neighbors=k_val
            )
        elif method_key == "LLS":
            k_val = self.attrs.get("lls_neighbors", 15)
            imp_res = self._apply_isolated(
                masked_df, self.impute_by_lls, n_neighbors=k_val
            )
        elif method_key == "BPCA":
            imp_res = self._apply_isolated(
                masked_df,
                self.impute_by_bpca,
                n_components=self.attrs.get("bpca_components", 2),
                max_iter=self.attrs.get("bpca_max_iter", 100),
                threshold=self.attrs.get("bpca_tol", 1e-4),
            )
        elif method_key == "QRILC":
            imp_res = self._apply_isolated(
                masked_df,
                self.impute_by_qrilc,
                global_seed=global_seed,
            )
        elif method_key == "MEDIAN":
            imp_res = self._apply_isolated(
                masked_df,
                lambda df: df.apply(lambda x: x.fillna(x.median()), axis=1),
            )
        imp_res.attrs["is_logged"] = True

        eval_met, t_vals, p_vals = self.compute_stratified_nrmse(
            mar_data, imp_res, mask
        )
        dist_metrics = su.calc_distribution_distance_metrics(t_vals, p_vals)
        structure_metrics = structure_stats.calc_sample_structure_preservation(
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
                "Wasserstein_Normalized": dist_metrics[
                    "wasserstein_normalized"
                ],
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
                    "sample_structure_score": metrics.get(
                        "Sample_Structure_Score"
                    ),
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
        score_df["wasserstein_score"] = score_df[
            "wasserstein_normalized"
        ].apply(
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
        global_seed: int = DEFAULT_RANDOM_SEED,
        batch_array: Optional[np.ndarray] = None,
    ) -> tuple:
        """Select the best MAR imputer from masked-reconstruction benchmarks."""
        candidates = ["KNN", "MinProb", "QRILC", "Median", "LLS", "BPCA"]
        selected_method = "KNN"
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
            score_df = selection_utils.rank_candidates(
                score_df,
                score_column="auto_score",
                tie_breakers=(("nrmse_total", True), ("method", True)),
            )
            selected_method = str(score_df.iloc[0]["method"])
            for row in score_df.itertuples():
                metrics = cache[row.method][0]
                metrics["Reconstruction_Score"] = su.finite_or_nan(
                    row.reconstruction_score
                )
                metrics["Distribution_Preservation_Score"] = su.finite_or_nan(
                    row.distribution_preservation_score
                )
                metrics["JSD_Score"] = su.finite_or_nan(row.jsd_score)
                metrics["Wasserstein_Score"] = su.finite_or_nan(
                    row.wasserstein_score
                )
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
                    su.finite_or_nan(
                        reference_metrics.get("Wasserstein_Normalized")
                    )
                )
        else:
            selected_method = min(
                cache,
                key=lambda method: cache[method][0].get(
                    "NRMSE_Total", float("inf")
                ),
            )

        best_score = cache[selected_method][0].get("Auto_Score", float("nan"))
        logger.info(
            f"Optimal MAR algorithm selected: {selected_method} "
            f"(score={best_score:.3f})"
        )
        return selected_method, cache

    @cached_property
    def imputation_metrics(self) -> Dict[str, Any]:
        """Extracts key parameters and performance metrics from imputation.

        Returns:
            dict: A structured dictionary of imputation metadata for reporting.
        """
        requested_method = self.attrs.get("requested_method", "auto")
        selected_method = self.attrs.get("selected_method", "Unknown")
        selected_label = self.attrs.get("selected_label", selected_method)
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

        raw_results = self.attrs.get("candidate_results", {})
        candidate_results = []
        candidate_metric_keys = (
            "nrmse_low",
            "nrmse_high",
            "nrmse_total",
            "jsd_total",
            "wasserstein_total",
            "wasserstein_normalized",
            "reconstruction_score",
            "distribution_preservation_score",
            "jsd_score",
            "wasserstein_score",
            "sample_structure_score",
            "trustworthiness",
            "distance_rank_preservation",
            "distance_scale_preservation",
            "auto_score",
        )
        for method, metrics in raw_results.items():
            metrics = metrics if isinstance(metrics, dict) else {}
            record = {
                "method": method,
                "selected": method == selected_method,
                "status": "ok",
            }
            record.update(
                {
                    metric: _safe_round(metrics.get(metric))
                    for metric in candidate_metric_keys
                }
            )
            candidate_results.append(record)

        is_auto = bool(self.attrs.get("is_auto", False))
        selected_score = next(
            (
                record["auto_score"]
                for record in candidate_results
                if record["selected"]
            ),
            None,
        )
        valid_scores = sorted(
            (
                score
                for record in candidate_results
                if (score := record.get("auto_score")) is not None
                and np.isfinite(score)
            ),
            reverse=True,
        )
        selection_margin = (
            valid_scores[0] - valid_scores[1]
            if is_auto and len(valid_scores) > 1
            else None
        )

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
                "mnar_method": mnar_meth,
                "mnar_fraction": reported_mnar_frac,
            },
            "selection": {
                "requested_method": requested_method,
                "selected_method": selected_method,
                "selected_label": selected_label,
                "is_auto": is_auto,
                "selected_score": selected_score,
                "selection_margin": selection_margin,
                "candidate_results": candidate_results,
            },
            "feature_distribution": {
                "mar_count": len(idx_mar),
                "mnar_count": len(idx_mnar),
            },
            "qa_metrics": qa_metrics,
            "skip_reason": self.attrs.get("imputation_skip_reason"),
        }

        return metrics

    def transform_imputation(self) -> StageResult["MetaboIntImputer"]:
        """Perform imputation without writing files or rendering figures."""
        # =====================================================================
        # Parameter Extraction
        # =====================================================================
        # StageRunner resolves defaults, TOML settings, and call-time notebook
        # overrides before this calculation-only method is reached.
        _mnar = self.attrs.get("mnar_method", "QRILC")
        _frac = self.attrs.get("mnar_fraction", 0.5)
        requested_mar = self.attrs.get("mar_method", "Auto")
        _mar = requested_mar
        _knn_k = self.attrs.get("knn_neighbors", 5)
        _lls_k = self.attrs.get("lls_neighbors", 15)
        _bpca_k = self.attrs.get("bpca_components", 2)
        _bpca_max_iter = self.attrs.get("bpca_max_iter", 100)
        _bpca_tol = self.attrs.get("bpca_tol", 1e-4)
        _ratio = self.attrs.get("sim_mask_ratio", 0.05)

        _seed = self.attrs.get("global_seed", DEFAULT_RANDOM_SEED)
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
            imputed_obj.attrs["selected_method"] = "Not required"
            imputed_obj.attrs["selected_label"] = "Not required"
            imputed_obj.attrs["requested_method"] = requested_mar
            imputed_obj.attrs["is_auto"] = False
            imputed_obj.attrs["mnar_method"] = "Not required"
            imputed_obj.attrs["mnar_fraction"] = None
            imputed_obj.attrs["candidate_results"] = {}
            imputed_obj.attrs["imputation_qa_metrics"] = {}

            logger.success(
                "Missing value imputation skipped: no missing values found."
            )
            return StageResult(
                data=imputed_obj,
                metrics=imputed_obj.imputation_metrics,
                candidates={},
                metadata={
                    "requested_method": requested_mar,
                    "selected_method": "Not required",
                    "selected_label": "Not required",
                    "is_auto": False,
                    "idx_mar_count": 0,
                    "skipped": True,
                },
            )

        batch_col = self.attrs.get("batch", "Batch")
        batch_array = target_cols.get_level_values(batch_col).values

        mnar_info = (
            f"{_mnar}"
            if (str(_mnar).upper() == "QRILC")
            else (f"{_mnar} (LOD={_frac}x)")
        )

        mar_spec = IMPUTATION_METHODS.resolve(_mar)
        if mar_spec.key == "AUTO":
            mar_info = (
                f"Auto (Evaluating KNN={_knn_k}, LLS (K={_lls_k}), "
                f"BPCA (PCs={_bpca_k}), MinProb, Median)"
            )
        elif mar_spec.key == "KNN":
            mar_info = f"KNN (K={_knn_k})"
        elif mar_spec.key == "LLS":
            mar_info = f"LLS (K={_lls_k})"
        elif mar_spec.key == "BPCA":
            mar_info = f"BPCA (PCs={_bpca_k}, MaxIter={_bpca_max_iter})"
        else:
            mar_info = f"{_mar}"

        logger.info(
            f"Hybrid Imputation Engine Initialized. "
            f"MAR: {mar_info} | MNAR: {mnar_info} | Sim_Mask: {_ratio}"
        )

        df_log = np.log2(self.astype(float).replace({0: np.nan}) + 1.0)

        # =====================================================================
        # MNAR Route: Localized LOD Imputation or QRILC
        # =====================================================================
        idx_mnar = pd.Index(self.attrs.get("idx_mnar", [])).intersection(
            df_log.index
        )

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

        # =====================================================================
        # MAR Route: Candidate Evaluation and Imputation
        # =====================================================================
        idx_mar = pd.Index(self.attrs.get("idx_mar", [])).intersection(
            df_log.index
        )
        cache, eval_met, t_vals, p_vals = {}, {}, [], []
        is_auto = mar_spec.key == "AUTO"

        if len(idx_mar) > 0:
            if is_auto:
                _mar, cache = self._select_best_imputation_method(
                    df_log, idx_mar, target_cols, _ratio, _seed, batch_array
                )
                eval_met, t_vals, p_vals = cache[_mar]
                mar_spec = IMPUTATION_METHODS.resolve(_mar)
            else:
                eval_met, t_vals, p_vals = self._evaluate_imputation_candidate(
                    df_log,
                    idx_mar,
                    target_cols,
                    _mar,
                    _ratio,
                    _seed,
                    batch_array,
                )

            logger.info(f"Executing isolated '{_mar}' on MAR features.")
            mar_slice = df_log.loc[idx_mar, target_cols]

            if mar_spec.key == "MINPROB":
                mar_imp = self._apply_isolated(
                    mar_slice, self.impute_by_minprob, global_seed=_seed
                )
            elif mar_spec.key == "KNN":
                mar_imp = self._apply_isolated(
                    mar_slice, self.impute_by_knn, n_neighbors=_knn_k
                )
            elif mar_spec.key == "LLS":
                mar_imp = self._apply_isolated(
                    mar_slice, self.impute_by_lls, n_neighbors=_lls_k
                )
            elif mar_spec.key == "BPCA":
                mar_imp = self._apply_isolated(
                    mar_slice,
                    self.impute_by_bpca,
                    n_components=_bpca_k,
                    max_iter=_bpca_max_iter,
                    threshold=_bpca_tol,
                )
            elif mar_spec.key == "QRILC":
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

        # =====================================================================
        # Matrix Reconstruction and Passport Update
        # =====================================================================
        final_log = pd.concat(
            [df_log[target_cols], df_log[self._blank.columns]], axis=1
        )[self.columns]

        res_val = np.exp2(final_log) - 1.0
        imputed_obj = self._constructor(res_val).__finalize__(self)

        imputed_obj.attrs["pipeline_stage"] = "Imputation"
        imputed_obj.attrs["imputation_status"] = "Completed"
        display_mar_method = IMPUTATION_METHODS.display_name(_mar)
        imputed_obj.attrs["selected_method"] = _mar
        imputed_obj.attrs["selected_label"] = display_mar_method
        # Keep the requested mode separate from the selected candidate.  AUTO
        # reuses ``_mar`` for the winning method during candidate selection.
        imputed_obj.attrs["requested_method"] = requested_mar
        imputed_obj.attrs["is_auto"] = is_auto
        imputed_obj.attrs["mnar_method"] = _mnar
        imputed_obj.attrs["mnar_fraction"] = _frac

        eval_source = (
            cache if cache else {display_mar_method: (eval_met, t_vals, p_vals)}
        )
        cand_mets = {}
        for m_name, (m_eval, _, _) in eval_source.items():
            cand_mets[m_name] = {
                "nrmse_low": m_eval.get("NRMSE_Low", float("nan")),
                "nrmse_high": m_eval.get("NRMSE_High", float("nan")),
                "nrmse_total": m_eval.get("NRMSE_Total", float("nan")),
                "jsd_total": m_eval.get("JSD_Total", float("nan")),
                "wasserstein_total": m_eval.get(
                    "Wasserstein_Total", float("nan")
                ),
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
                "wasserstein_score": m_eval.get(
                    "Wasserstein_Score", float("nan")
                ),
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
        imputed_obj.attrs["candidate_results"] = cand_mets

        # =====================================================================
        # Quality Metrics and Result Assembly
        # =====================================================================
        logger.info("Calculating imputation-related metrics...")
        qa_metrics = self.calc_imp_quality_metrics(
            raw_obj=self, imp_obj=imputed_obj
        )
        imputed_obj.attrs["imputation_qa_metrics"] = qa_metrics

        benchmark_results = (
            (cache if cache else {_mar: (eval_met, t_vals, p_vals)})
            if len(idx_mar) > 0
            else {}
        )
        return StageResult(
            data=imputed_obj,
            metrics=imputed_obj.imputation_metrics,
            candidates=benchmark_results,
            metadata={
                "selected_method": _mar,
                "selected_label": display_mar_method,
                "requested_method": requested_mar,
                "is_auto": is_auto,
                "idx_mar_count": len(idx_mar),
                "has_candidate_cache": bool(cache),
                "skipped": False,
            },
        )

    @log_execution_time
    def run_imputation(
        self,
        output_dir: str | None = None,
        **runtime_overrides: object,
    ) -> StageResult["MetaboIntImputer"]:
        """Return the structured missing-value imputation stage result.

        Keyword settings use the same names as the imputation configuration,
        such as ``mar_method``, ``mnar_method``, ``knn_neighbors``,
        ``bpca_components``, ``sim_mask_ratio``, and ``global_seed``. They
        take precedence over TOML settings and built-in defaults for this
        processor instance.
        """
        runner = ImputationStageRunner(
            processor=self,
            output_dir=output_dir,
            runtime_overrides=runtime_overrides,
            allowed_override_keys=self._RUNTIME_CONFIG_KEYS,
        )
        result = runner.run()
        logger.success("Missing value imputation completed successfully.")
        return result
