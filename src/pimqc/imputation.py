# src/pimqc/imputation.py
"""
Missing value imputation module with biological-aware evaluation.
"""

import os
import copy
import numpy as np
import pandas as pd
from functools import cached_property

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
from sklearn.impute import KNNImputer
from loguru import logger
from typing import Dict, Any, Optional

from . import io_utils as iu
from . import stat_utils as su
from . import plot_utils as pu
from . import core_classes
from . import visualizer_classes


class MetaboIntImputer(core_classes.MetaboInt):
    """Missing value imputation engine with hybrid stratified evaluation."""

    _metadata = ["attrs","stats"]

    def __init__(
        self,
        *args,
        pipeline_params: Optional[Dict[str, Any]] = None,
        mar_method: Optional[str] = None,
        mnar_method: Optional[str] = None,
        mnar_fraction: Optional[float] = None,
        knn_neighbors: Optional[int] = None,
        lls_neighbors: Optional[int] = None,
        sim_mask_ratio: Optional[float] = None,
        **kwargs
    ):
        """Initialize MetaboIntImputer with parameters and metadata.

        Args:
            *args: Variable arguments passed to pandas DataFrame.
            pipeline_params: Global configuration dictionary.
            mar_method: Method for MAR features.
            mnar_method: Method for MNAR features.
            mnar_fraction: Multiplier for LOD-based MNAR imputation.
            knn_neighbors: Number of neighbors for the KNN algorithm.
            lls_neighbors: Number of neighbors for the LLS algorithm.
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
            "sim_mask_ratio": 0.05,
        }

        # 2. TOML global configuration overrides base defaults
        if pipeline_params and "MetaboIntImputer" in pipeline_params:
            imp_configs.update(pipeline_params["MetaboIntImputer"])

        # 3. Explicit kwargs override TOML (Highest priority)
        local_args = locals()
        explicit_params = [
            "mar_method", "mnar_method", "mnar_fraction", 
            "knn_neighbors", "lls_neighbors", "sim_mask_ratio"
        ]
        for param in explicit_params:
            if local_args[param] is not None:
                imp_configs[param] = local_args[param]

        self.attrs.update(imp_configs)

    @property
    def _constructor(self):
        """Return the class constructor for stable subclassing."""
        return MetaboIntImputer

    def __finalize__(self, other, method=None, **kwargs):
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
    def calc_imp_quality_metrics(self, raw_obj, imp_obj):
        """Calculate QA metrics (JSD) and prepare KDE plotting data.
        
        Computes Jensen-Shannon Divergence for dual combinations and
        constructs a long-form DataFrame for KDE plotting.
        
        Returns:
            metrics (dict): Contains the quantified JSD evaluation scores.
            df_kde (pd.DataFrame): Long-form DataFrame strictly for plotting.
        """
        metrics = {"JSD": {"QC": {}, "Sample": {}}}
        dfs = []
        
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
                jsd_1 = su.calc_jsd_similarity(obs, imp_all)
                val_1 = (
                    jsd_1.get("JSD", jsd_1.get("jsd", np.nan))
                    if isinstance(jsd_1, dict) else jsd_1
                )
                metrics["JSD"][grp]["Before vs After (All)"] = float(val_1)
                
            if len(obs) > 0 and len(imp_only) > 0:
                jsd_2 = su.calc_jsd_similarity(obs, imp_only)
                val_2 = (
                    jsd_2.get("JSD", jsd_2.get("jsd", np.nan))
                    if isinstance(jsd_2, dict) else jsd_2
                )
                metrics["JSD"][grp]["Before vs Imputed Only"] = float(val_2)
                
            # Compile plotting data
            if len(obs) > 0:
                dfs.append(pd.DataFrame({
                    "Log2_Intensity": obs, "Group": grp, 
                    "Type": "Before Imputation"
                }))
            if len(imp_all) > 0:
                dfs.append(pd.DataFrame({
                    "Log2_Intensity": imp_all, "Group": grp, 
                    "Type": "After Imputation"
                }))
            if len(imp_only) > 0:
                dfs.append(pd.DataFrame({
                    "Log2_Intensity": imp_only, "Group": grp, 
                    "Type": "Imputed Data"
                }))
                
        df_kde = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        return metrics, df_kde

    # ====================================================================
    # Core Algorithms (Log2 Space)
    # ====================================================================
    @staticmethod
    def impute_by_constant(df_log, fraction = 1.0, imp_mode = "row"):
        """Imputes missing values using a constant LOD heuristic.

        Args:
            df: The dataset (typically log-transformed).
            fraction: The heuristic multiplier (e.g., 0.5 for half-minimum).
            imp_mode: "row" (feature-wise), "column" (sample-wise), or "global".
            is_log2: If True, executes fractional math in linear space safely.

        Returns:
            Dataframe with constant imputation applied.
        """
        if imp_mode in ("row","row-wise","row min"):
            raw_mins = df_log.min(axis=1)
        elif imp_mode in ("column","column-wise","column min"):
            raw_mins = df_log.min(axis=0)
        else: # elif imp_mode in ("global", "global min"):
            raw_mins = df_log.min().min()

        linear_mins = np.exp2(raw_mins) - 1.0
        target_mins = np.log2((linear_mins * fraction) + 1.0)

        # 3. Broadcast the computed minimums to fill NaNs
        if imp_mode in ("row", "row-wise", "row min"):
            return df_log.apply(lambda x: x.fillna(target_mins[x.name]), axis=1)
        else:
            return df_log.fillna(target_mins)

    @staticmethod
    def impute_by_qrilc(df_log, tune_sigma=1.0, global_seed=123):
        """Impute missing values using QRILC logic for left-censored data.
        
        Approximates Quantile Regression Imputation of Left-Censored data by 
        estimating the feature-wise underlying normal distribution using robust 
        estimators (Median/MAD), then drawing randomly from the truncated left 
        tail (below the observed minimum) to preserve low-abundance variance.
        
        Ref:
            Missing value imputation approach for mass spectrometry-based 
            metabolomics data (Scientific reports, 2018)
        """
        
        
        # Initialize numpy random generator for deterministic results
        rng = np.random.default_rng(global_seed)
        res_df = df_log.copy()

        def _qrilc_row(row):
            n_missing = row.isna().sum()
            if n_missing == 0:
                return row
            
            valid = row.dropna()
            # Fallback for features with extremely few observed values
            if len(valid) < 3:
                fallback_val = valid.min() if len(valid) > 0 else 0.0
                return row.fillna(fallback_val)

            # 1. Estimate LOD (Truncation point) for this specific feature
            lod = valid.min()

            # 2. Estimate robust parameters for the underlying distribution.
            # Using Median and MAD prevents the bias that would occur if we 
            # used mean/std on left-censored data.
            mu = np.median(valid)
            mad = np.median(np.abs(valid - mu))
            sigma = (mad * 1.4826) * tune_sigma

            # Prevent zero variance which crashes truncnorm
            if sigma < 1e-6:
                sigma = 0.01
                
            # 3. Define standard normal quantiles for the truncation limits
            a, b = -np.inf, (lod - mu) / sigma

            # 4. Draw random samples strictly from the truncated left tail
            drawn = stats.truncnorm.rvs(
                a=a, b=b, loc=mu, scale=sigma, size=n_missing, random_state=rng
            )
            
            # Prevent negative intensities in linear space (exp2(x) - 1 >= 0)
            drawn = np.clip(drawn, a_min=0.0, a_max=None)
            
            # Fill the missing values
            row_copy = row.copy()
            row_copy[row.isna()] = drawn
            return row_copy

        # Apply the QRILC logic feature-wise (row-wise)
        return res_df.apply(_qrilc_row, axis=1)
    
    @staticmethod
    def impute_by_knn(df_log, n_neighbors:int=5):
        """Impute missing values using K-Nearest Neighbors algorithm."""
        # [CRITICAL FIX]: Dynamic neighbor scaling for isolated small groups
        n_samples = df_log.shape[1]
        safe_k = min(n_neighbors, n_samples - 1)
        
        if safe_k < 1:
            # Fallback to feature median if insufficient neighbors (e.g., n=1)
            return df_log.apply(
                lambda x: x.fillna(x.median()), axis=1
            ).fillna(0.0)
            
        imputer = KNNImputer(n_neighbors=safe_k, weights="distance")
        arr_imp = imputer.fit_transform(df_log.T).T
        
        return pd.DataFrame(
            arr_imp, index=df_log.index, columns=df_log.columns
        )

    @staticmethod
    def impute_by_lls(df_log, n_neighbors: int = 15):
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
                "Insufficient complete features for LLS."
                "Falling back to median.")
            return df_log.apply(
                lambda x: x.fillna(x.median()), axis=1).fillna(0.0)
            
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
                    np.nanmedian(row) if obs_mask.sum() > 0 else 0.0)
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
                
        return pd.DataFrame(
            res_arr, index=df_log.index, columns=df_log.columns)
    
    @staticmethod
    def impute_by_minprob(df_log, global_seed = 123):
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
                loc=shift_mean, scale=shift_std, size=s.isna().sum())
            
            # [CRITICAL FIX]: Prevent negative intensities in linear space
            # Log2 values must be >= 0 so that exp2(x) - 1.0 >= 0
            drawn = np.clip(drawn, a_min=0.0, a_max=None)
            res_df.loc[s.isna(), col] = drawn
            
        return res_df

    def _apply_isolated(self, df_slice, imp_func, **kwargs):
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

    # @staticmethod
    # def generate_abundance_mask(df_log, mask_ratio, noise_factor=1.0):
    #     """Generate an abundance-dependent mask for MNAR simulation."""
    #     np.random.seed(42)
    #     shape = df_log.shape
    #     valid_mask = ~df_log.isna()
    #     target_nas = int(valid_mask.values.sum() * mask_ratio)
    
    #     if target_nas == 0:
    #         return pd.DataFrame(
    #             False, index=df_log.index, columns=df_log.columns
    #         )
    
    #     feat_meds = df_log.median(axis=1).fillna(0)
    #     log_meds = np.log2(feat_meds + 1.0)
    #     max_v = log_meds.max() if log_meds.max() > 0 else 1.0
    #     rel_abd = log_meds / max_v
    
    #     # Base probability weight is inversely proportional to abundance
    #     weight_mat = np.tile((1.0 - rel_abd).values[:, None], (1, shape[1]))
    #     final_score = weight_mat + np.random.uniform(0, noise_factor, shape)
    #     final_score[~valid_mask.values] = -1.0
    
    #     cutoff = np.sort(final_score.flatten())[-target_nas]
    #     mask_arr = (final_score >= cutoff) & valid_mask.values
    
    #     return pd.DataFrame(
    #         mask_arr, index=df_log.index, columns=df_log.columns
    #     )

    @staticmethod
    def generate_gmm_noise_mask(
        df_log: pd.DataFrame, 
        mask_ratio: float, 
        noise_factor: float = 1.5,
        global_seed: int = 123,
        batch_array: Optional[np.ndarray] = None
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
                    False, index=df_log.index, columns=df_log.columns)
                
            gmm = GaussianMixture(n_components=2, random_state=global_seed)
            gmm.fit(valid_data)
            lower_cluster_idx = np.argmin(gmm.means_)
            
            base_prob = gmm.predict_proba(valid_data)[:, lower_cluster_idx]
            final_score = base_prob + rng.uniform(
                0, noise_factor, size=base_prob.shape)
            cutoff_score = np.sort(final_score)[-target_nas]
            
            mask_arr[valid_mask] = final_score >= cutoff_score
            return pd.DataFrame(
                mask_arr, index=df_log.index, columns=df_log.columns)

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
                        valid_data_b, (target_nas_b / len(valid_data_b)) * 100)
                    b_mask = np.zeros_like(b_data, dtype=bool)
                    b_mask[valid_mask_b] = valid_data_b.flatten() <= cutoff_val
                    mask_arr[:, b_cols_idx] = b_mask
                continue

            try:
                gmm = GaussianMixture(n_components=2, random_state=global_seed)
                gmm.fit(valid_data_b)
                lower_cluster_idx = np.argmin(gmm.means_)
                
                base_prob = gmm.predict_proba(valid_data_b)[
                    :, lower_cluster_idx]
                final_score = base_prob + rng.uniform(
                    0, noise_factor, size=base_prob.shape)
                cutoff_score = np.sort(final_score)[-target_nas_b]
                
                b_mask = np.zeros_like(b_data, dtype=bool)
                b_mask[valid_mask_b] = final_score >= cutoff_score
                mask_arr[:, b_cols_idx] = b_mask
            except Exception as e:
                # Soft fallback to percentile truncation if GMM fails to 
                # converge on edge-case batches
                logger.debug(
                    f"GMM failed for batch {b}: {e}. "
                    "Falling back to empirical percentile.")
                cutoff_val = np.percentile(
                    valid_data_b, (target_nas_b / len(valid_data_b)) * 100)
                b_mask = np.zeros_like(b_data, dtype=bool)
                b_mask[valid_mask_b] = valid_data_b.flatten() <= cutoff_val
                mask_arr[:, b_cols_idx] = b_mask

        return pd.DataFrame(
            mask_arr, index=df_log.index, columns=df_log.columns)

    @staticmethod
    def compute_stratified_nrmse(df_true, df_imp, mask_df, lod_q=0.25):
        """Calculate NRMSE stratified by low and high abundance regions."""
        feat_meds = df_true.median(axis=1).fillna(0)
        lod_val = feat_meds.quantile(lod_q)
        
        t_all = df_true.values[mask_df.values]
        p_all = df_imp.values[mask_df.values]
        med_all = np.tile(
            feat_meds.values[:, None], (1, df_true.shape[1])
        )[mask_df.values]
        
        low_m, hi_m = (med_all <= lod_val), (med_all > lod_val)
        
        def _get_nrmse(t, p):
            if len(t) < 2 or (np.max(t) - np.min(t)) < 1e-9:
                return np.nan
            rmse = np.sqrt(np.mean((t - p)**2))
            return float(rmse / (np.max(t) - np.min(t)))
            
        # 1. Compile the metrics dictionary
        metrics = {
            "NRMSE_Total": _get_nrmse(t_all, p_all),
            "NRMSE_Low": _get_nrmse(t_all[low_m], p_all[low_m]),
            "NRMSE_High": _get_nrmse(t_all[hi_m], p_all[hi_m]),
            "Count_Low": int(np.sum(low_m)),
            "Count_High": int(np.sum(hi_m)),
            "Threshold": float(lod_val)
        }
        
        # 2. Return exactly 3 objects to match the unpacking logic
        return metrics, t_all, p_all

    def run_benchmark_simulation(
        self,
        df_log: pd.DataFrame,
        idx_mar: pd.Index,
        target_cols: pd.Index,
        method: str,
        ratio: float = 0.05,
        global_seed: int = 123,
        batch_array: Optional[np.ndarray] = None
    ) -> tuple:
        """Runs MNAR/MAR mask simulation strictly on MAR features."""
        # 1. Isolate the MAR subset for benchmarking
        mar_data = df_log.loc[idx_mar, target_cols].astype(float)
        
        # 2. Generate the boolean mask with batch-awareness
        mask = self.generate_gmm_noise_mask(
            mar_data, ratio, noise_factor=1.5, global_seed=global_seed,
            batch_array=batch_array
        )
        
        # 3. Apply the mask to create the simulated missing dataset
        masked_df = mar_data.copy()
        masked_df[mask] = np.nan
        
        # 4. Execute imputation on the masked subset with absolute isolation
        if method in ("Prob","prob", "MinProb", "minprob"):
            imp_res = self._apply_isolated(
                masked_df, self.impute_by_minprob, global_seed=global_seed)
        elif method in ("knn", "KNN"):
            k_val = self.attrs.get("knn_neighbors", 5)
            imp_res = self._apply_isolated(
                masked_df, self.impute_by_knn, n_neighbors=k_val)
        elif method in ("lls", "LLS"):
            k_val = self.attrs.get("lls_neighbors", 15) 
            imp_res = self._apply_isolated(
                masked_df, self.impute_by_lls, n_neighbors=k_val)
        else:
            imp_res = self._apply_isolated(
                masked_df, 
                lambda df: df.apply(lambda x: x.fillna(x.median()), axis=1))
            
        # 5. Compute fidelity strictly on the artificially masked locations
        eval_met, t_vals, p_vals = self.compute_stratified_nrmse(
            mar_data, imp_res, mask)
        return eval_met, t_vals, p_vals
    
    # ====================================================================
    # Execution & Auto-Selection
    # ====================================================================
    def select_best_algorithm(
        self,
        df_log: pd.DataFrame,
        idx_mar: pd.Index,
        target_cols: pd.Index,
        ratio: float = 0.05,
        global_seed: int = 123,
        batch_array: Optional[np.ndarray] = None
    ) -> tuple:
        """Autonomously selects the best algorithm using MAR-only subset."""
        candidates = ["KNN", "MinProb", "Median", "LLS"]
        best_method = "KNN"
        best_nrmse = float("inf")
        cache = {}

        for cand in candidates:
            logger.info(f'Simulating "{cand}" on MAR subset...')
            emet, tv, pv = self.run_benchmark_simulation(
                df_log=df_log, idx_mar=idx_mar, target_cols=target_cols,
                method=cand, ratio=ratio, global_seed=global_seed,
                batch_array=batch_array
            )
            cache[cand] = (emet, tv, pv)
            
            nrmse_total = emet.get("NRMSE_Total", float("inf")) 
            if nrmse_total < best_nrmse:
                best_nrmse = nrmse_total
                best_method = cand

        logger.info(f"Optimal MAR algorithm selected: {best_method}")
        return best_method, cache

    @cached_property
    def imputation_metrics(self):
        """Extracts key parameters and performance metrics from imputation.

        Returns:
            dict: A structured dictionary of imputation metadata for reporting.
        """
        mar_req = self.attrs.get("mar_requested", "auto")
        mar_sel = self.attrs.get("selected_mar_method", "Unknown")
        mnar_meth = self.attrs.get("mnar_method", "row")
        mnar_frac = self.attrs.get("mnar_fraction", 0.5)
        
        reported_mnar_frac = None if str(mnar_meth).upper() == "QRILC" else (
            float(mnar_frac))
        status = self.attrs.get("imputation_status", "Pending")

        def _safe_round(val):
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
        output_dir: str = None
    ) -> pd.DataFrame:
        """Executes hybrid imputation and exports complete visualizations."""
        # ====================================================================
        # 1. Parameter Extraction & Priority Fallback
        # ====================================================================
        _mnar = mnar_method or self.attrs.get("mnar_method", "QRILC")
        _frac = (
            mnar_fraction if mnar_fraction is not None 
            else self.attrs.get("mnar_fraction", 0.5))
        
        _mar = mar_method or self.attrs.get("mar_method", "Auto")
        _knn_k = (
            knn_neighbors if knn_neighbors is not None 
            else self.attrs.get("knn_neighbors", 5))
        _lls_k = (
            lls_neighbors if lls_neighbors is not None 
            else self.attrs.get("lls_neighbors", 15))
        _ratio = (
            sim_ratio if sim_ratio is not None 
            else self.attrs.get("sim_mask_ratio", 0.05))

        _seed = self.attrs.get("global_seed", 123)
        target_cols = self.columns.difference(self._blank.columns)
        
        batch_col = self.attrs.get("batch", "Batch")
        batch_array = target_cols.get_level_values(batch_col).values
        
        mnar_info = f"{_mnar}" if (str(_mnar).upper() == "QRILC") else (
            f"{_mnar} (LOD={_frac}x)")
        
        _mar_clean = str(_mar).upper()
        if _mar_clean in ("AUTO", "BEST"):
            mar_info = f"Auto (Evaluating KNN={_knn_k}, LLS (K={_lls_k}), MinProb, Median)"
        elif _mar_clean == "KNN":
            mar_info = f"KNN (K={_knn_k})"
        elif _mar_clean == "LLS":
            mar_info = f"LLS (K={_lls_k})"
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
        idx_mnar = pd.Index(self.attrs.get("idx_mnar", [])).intersection(
            df_log.index)
            
        if len(idx_mnar) > 0:
            logger.info(f"Applying {_mnar} to {len(idx_mnar)} MNAR features.")
            
            if str(_mnar).upper() == "QRILC":
                mnar_imp = MetaboIntImputer.impute_by_qrilc(
                    df_log=df_log.loc[idx_mnar, target_cols],
                    global_seed=_seed
                )
            else:
                mnar_imp = MetaboIntImputer.impute_by_constant(
                    df_log=df_log.loc[idx_mnar, target_cols],
                    fraction=_frac, imp_mode=_mnar,
                )
                
            df_log.loc[idx_mnar, target_cols] = mnar_imp
        else:
            logger.info("MNAR index empty. Bypassing MNAR imputation.")

        # ====================================================================
        # 3. ROUTE B: MAR -> ML Simulation & Impute
        # ====================================================================
        idx_mar = pd.Index(self.attrs.get("idx_mar", [])).intersection(
            df_log.index)
        cache, eval_met, t_vals, p_vals = {}, {}, [], []
        is_auto = (_mar in ("auto", "Auto", "Best", "best"))
        
        if len(idx_mar) > 0:
            if is_auto:
                _mar, cache = self.select_best_algorithm(
                    df_log, idx_mar, target_cols, _ratio, _seed, batch_array
                )
                eval_met, t_vals, p_vals = cache[_mar]
            else:
                eval_met, t_vals, p_vals = self.run_benchmark_simulation(
                    df_log, idx_mar, target_cols, _mar, _ratio, _seed,
                    batch_array
                )
                
            logger.info(f"Executing isolated '{_mar}' on MAR features.")
            mar_slice = df_log.loc[idx_mar, target_cols]
            
            if _mar in ("Prob","prob", "MinProb", "minprob"):
                mar_imp = self._apply_isolated(
                    mar_slice, self.impute_by_minprob, global_seed=_seed)
            elif _mar in ("knn", "KNN"):
                mar_imp = self._apply_isolated(
                    mar_slice, self.impute_by_knn, n_neighbors=_knn_k)
            elif _mar in ("lls", "LLS"):
                mar_imp = self._apply_isolated(
                    mar_slice, self.impute_by_lls, n_neighbors=_lls_k)
            else:
                mar_imp = self._apply_isolated(
                    mar_slice, 
                    lambda df: df.apply(lambda x: x.fillna(x.median()), axis=1))
                
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
        imputed_obj.attrs["selected_mar_method"] = _mar
        imputed_obj.attrs["mar_requested"] = mar_method or self.attrs.get(
            "mar_method", "auto"
        )
        imputed_obj.attrs["mnar_method"] = _mnar
        imputed_obj.attrs["mnar_fraction"] = _frac
        
        cand_mets = {}
        for m_name, (m_eval, _, _) in cache.items():
            cand_mets[m_name] = {
                "nrmse_low": m_eval.get("NRMSE_Low", float("nan")),
                "nrmse_high": m_eval.get("NRMSE_High", float("nan")),
                "nrmse_total": m_eval.get("NRMSE_Total", float("nan")),
            }
        imputed_obj.attrs["candidate_metrics"] = cand_mets

        # ====================================================================
        # 5. EXPORT & VISUALIZATIONS
        # ====================================================================
        logger.info("Calculating imputation-related metrics...")
        qa_metrics, df_kde = self.calc_imp_quality_metrics(
            raw_obj=self, imp_obj=imputed_obj
        )
        imputed_obj.attrs["imputation_qa_metrics"] = qa_metrics

        if output_dir:
            iu._check_dir_exists(output_dir, handle="makedirs")
            imputed_obj.to_csv(
                os.path.join(output_dir, f"Imputed_Data_{_mar}.csv")
            )
            
            logger.info("Generating diagnostic plots for imputation...")
            vis = MetaboVisualizerImputer(
                raw_obj=self, imp_obj=imputed_obj, df_kde=df_kde
            )
                        
            if len(idx_mar) > 0 and cache:
                if is_auto:
                    fig_cands = vis.plot_multi_nrmse_scatters(cache)
                    vis.save_and_show_pw(
                        pw_obj=fig_cands, 
                        file_path=os.path.join(output_dir, "Imputer_Candidates"),
                        width="80%"
                    )
                
                fig_grid = vis.plot_imputation_summary_grid(
                    t_vals, p_vals, eval_met, _mar, metrics=qa_metrics
                )
                if fig_grid:
                    grid_path = os.path.join(
                        output_dir, f"Imputation_Dashboard_{_mar}.svg")
                    vis.save_and_show_pw(pw_obj=fig_grid, file_path=grid_path)
                    logger.info(
                        f"Imputer summary dashboard saved as: {grid_path}")

        logger.success("Missing value imputation completed successfully.")
        return imputed_obj

class MetaboVisualizerImputer(visualizer_classes.BaseMetaboVisualizer):
    """Visualization suite for evaluating imputation accuracy."""

    def __init__(self, raw_obj, imp_obj, df_kde=None):
        """Initialize the visualizer with pre-calculated KDE data arrays."""
        super().__init__(metabo_obj=imp_obj)
        self.raw_obj = raw_obj.astype(float).replace({0: np.nan})
        self.imp_obj = imp_obj.astype(float)
        self.df_kde = df_kde if df_kde is not None else pd.DataFrame()

    def _plot_imputed_kde_overlay(
        self, metrics=None, ax_qc=None, ax_sample=None
    ):
        """Plot KDE overlay with independent styling and precise Z-orders."""
        from scipy.stats import gaussian_kde
        import matplotlib.colors as mcolors  # 引入颜色解析模块
        
        return_fig = False
        
        if self.df_kde.empty:
            if ax_qc is None or ax_sample is None:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "No valid data to plot.", ha="center")
                return fig
            return ax_qc, ax_sample

        if ax_qc is None or ax_sample is None:
            fig, (ax_qc, ax_sample) = plt.subplots(
                1, 2, figsize=(11, 4), gridspec_kw={'width_ratios': [1, 1]}
            )
            return_fig = True

        layer_styles = {
            "Before Imputation": {
                "color": "black", "fill": True, "alpha": 0.2, 
                "ls": "--", "lw": 1.5, "z": 1
            },
            "After Imputation": {
                "color": "tab:gray", "fill": False, "alpha": 1.0, 
                "ls": "-", "lw": 2.0, "z": 2
            },
            "Imputed Data": {
                "color": "tab:red", "fill": False, "alpha": 1.0, 
                "ls": "-", "lw": 2.0, "z": 3
            }
        }
        
        plot_order = [
            "Before Imputation", 
            "After Imputation", 
            "Imputed Data"
        ]

        for grp, ax in [("QC", ax_qc), ("Sample", ax_sample)]:
            subset_grp = self.df_kde[self.df_kde["Group"] == grp]
            
            if subset_grp.empty:
                ax.text(0.5, 0.5, f"No {grp} data available.", ha="center")
                self._apply_standard_format(
                    ax=ax, title=f"Density Overlay ({grp})"
                )
                continue
                
            x_min = subset_grp["Log2_Intensity"].min()
            x_max = subset_grp["Log2_Intensity"].max()
            x_margin = (x_max - x_min) * 0.1 if x_max > x_min else 1.0
            x_grid = np.linspace(x_min - x_margin, x_max + x_margin, 500)
            
            total_baseline = len(subset_grp[
                subset_grp["Type"] == "After Imputation"
            ])
            
            for t in plot_order:
                subset = subset_grp[subset_grp["Type"] == t]
                if len(subset) > 1 and total_baseline > 0:
                    vals = subset["Log2_Intensity"].values
                    
                    if np.std(vals) < 1e-6:
                        rng = np.random.default_rng(123)
                        vals = vals + rng.normal(0, 1e-4, size=len(vals))
                        
                    kde = gaussian_kde(vals)
                    y_scaled = kde(x_grid) * (len(subset) / total_baseline)
                    
                    cfg = layer_styles[t]
                    if cfg["fill"]:
                        ax.fill_between(
                            x_grid, y_scaled, color=cfg["color"], 
                            alpha=cfg["alpha"], zorder=cfg["z"], linewidth=0
                        )
                        ax.plot(
                            x_grid, y_scaled, color=cfg["color"], 
                            linestyle=cfg["ls"], linewidth=cfg["lw"], 
                            zorder=cfg["z"]
                        )
                        # [核心修复区] 彻底解耦图例背景与边框的透明度
                        fc = mcolors.to_rgba(cfg["color"], alpha=cfg["alpha"])
                        ec = mcolors.to_rgba(cfg["color"], alpha=1.0)
                        
                        ax.fill_between(
                            [], [], facecolor=fc, edgecolor=ec, 
                            linestyle=cfg["ls"], linewidth=cfg["lw"], label=t
                        )
                    else:
                        ax.plot(
                            x_grid, y_scaled, color=cfg["color"], 
                            linestyle=cfg["ls"], linewidth=cfg["lw"], 
                            alpha=cfg["alpha"], zorder=cfg["z"], label=t
                        )

            if metrics and "JSD" in metrics and grp in metrics["JSD"]:
                m_grp = metrics["JSD"][grp]
                jsd_1 = m_grp.get("Before vs After (All)", np.nan)
                jsd_2 = m_grp.get("Before vs Imputed Only", np.nan)
                
                lines = ["Jensen-Shannon Divergence:"]
                if not pd.isna(jsd_1):
                    lines.append(f"Before vs After: {float(jsd_1):.3f}")
                if not pd.isna(jsd_2):
                    lines.append(f"Before vs Imputed: {float(jsd_2):.3f}")
                
                if len(lines) > 1:
                    annot_text = "\n".join(lines)
                    ax.text(
                        0.96, 0.96, annot_text, transform=ax.transAxes,
                        fontsize=9, verticalalignment="top", 
                        horizontalalignment="right", clip_on=False,
                        bbox=dict(
                            boxstyle="round,pad=0.4", facecolor="white", 
                            edgecolor="none", alpha=0.6))
                
            self._apply_standard_format(
                ax=ax, title=f"Density Overlay ({grp})",
                xlabel="Log2 Intensity", ylabel="Relative Density", 
                append_stage=False
            )
            
            if ax.get_legend():
                ax.get_legend().remove()
                
            if grp == "Sample":
                self._format_single_legend(
                    ax=ax, title="Data Type",
                    bbox_to_anchor=(1.05, 1), loc="upper left"
                )
                
        if return_fig:
            plt.tight_layout()
            return fig
            
        return ax_qc, ax_sample
    
    def _plot_nrmse_scatter(
        self, true_vals, pred_vals, metrics, method_name="", 
        axis_lims=None, ax=None
    ):
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
            color_list=["white", "tab:red"], n_colors=256, cmin=0.1, cmax=1.0
        )

        hb = current_ax.hexbin(
            x=true_vals, y=pred_vals, gridsize=40, extent=extent,
            cmap=color_map, mincnt=1,
        )
        
        current_ax.plot(
            [lim_min, lim_max], [lim_min, lim_max], 
            color="tab:gray", linestyle="--", linewidth=1.0, zorder=3
        )

        threshold = metrics.get("Threshold")
        if threshold is not None:
            current_ax.axvline(
                x=threshold, color="tab:gray", linestyle="--", 
                linewidth=1.0, alpha=0.8
            )
            current_ax.axhline(
                y=threshold, color="tab:gray", linestyle="--", 
                linewidth=1.0, alpha=0.8
            )

        annot_text = (
            f"NRMSE_Total: {metrics['NRMSE_Total']:.4f}\n"
            f"NRMSE_Low:   {metrics['NRMSE_Low']:.4f}\n"
            f"NRMSE_High:  {metrics['NRMSE_High']:.4f}"
        )

        current_ax.text(
            0.96, 0.02, annot_text, transform=current_ax.transAxes,
            fontsize=9, verticalalignment="bottom", 
            horizontalalignment="right", clip_on=False,
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor="white", 
                edgecolor="none", alpha=0.6
            )
        )
        
        title_str = "MAR Masked Simulation"
        if method_name:
            clean_name = method_name.replace("*", "")
            if clean_name.upper() in ("KNN", "LLS"):
                display = method_name.upper()
            elif clean_name in ("MinProb", "minprob", "Prob", "prob"):
                display = method_name
            else:
                display = method_name.title()
                
            title_str += f"\n({display})"
            
        self._apply_standard_format(
            ax=current_ax, title=title_str,
            xlabel="True Intensity (Log2)", ylabel="Imputed Intensity (Log2)",
            append_stage=False
        )
        
        if axis_lims is not None:
            current_ax.set_xlim(ax_min, ax_max)
            current_ax.set_ylim(ax_min, ax_max)
        
        cb = fig.colorbar(hb, ax=current_ax)
        cb.set_label("Log10(Count)")
        
        if ax is None:
            return fig
        return current_ax

    def plot_multi_nrmse_scatters(self, results_dict: dict):
        """
        Plot a dynamically sized grid of NRMSE scatter plots for all candidates.
        """
        try:
            import math
            import operator
            from functools import reduce
            import patchworklib as pw
        except ImportError:
            logger.warning("Module 'patchworklib' not found. Skipping grid.")
            return None

        n_plots = len(results_dict)
        if n_plots == 0:
            return None

        # 1. Dynamically calculate optimal column count
        layout_map = {
            1: 1, 2: 2, 3: 3, 4: 4, 
            5: 3, 6: 3, 7: 4, 8: 4, 9: 5
        }
        # Fallback to square root logic for >9 plots, capped at 4 columns
        n_cols = layout_map.get(
            n_plots, min(4, math.ceil(math.sqrt(n_plots)))
        )

        # 2. Determine global axis limits and identify the best method
        g_min, g_max = float("inf"), float("-inf")
        best_method = None
        best_nrmse = float("inf")

        for method_name, (metrics, true_vals, pred_vals) in results_dict.items():
            g_min = min(g_min, true_vals.min(), pred_vals.min())
            g_max = max(g_max, true_vals.max(), pred_vals.max())
            
            if metrics["NRMSE_Total"] < best_nrmse:
                best_nrmse = metrics["NRMSE_Total"]
                best_method = method_name

        margin = (g_max - g_min) * 0.05
        shared_lims = (g_min - margin, g_max + margin)

        # 3. Generate individual scatter plots (Bricks)
        pw.clear()
        bricks = []
        
        for method_name, (metrics, true_vals, pred_vals) in results_dict.items():
            ax = pw.Brick(figsize=(4, 4), label=f"nrmse_{method_name}")
            
            # Highlight the best performing method with an asterisk
            display_name = (
                f"*{method_name}" if method_name == best_method else method_name
            )

            self._plot_nrmse_scatter(
                true_vals, pred_vals, metrics, 
                method_name=display_name, axis_lims=shared_lims, ax=ax
            )
            bricks.append(ax)

        # 4. Pad the last row with empty placeholders to maintain grid structure
        while len(bricks) % n_cols != 0:
            empty_ax = pw.Brick(figsize=(4, 4), label=f"empty_{len(bricks)}")
            empty_ax.axis("off")
            bricks.append(empty_ax)

        # 5. Dynamically stitch the grid using reduce
        rows = []
        for i in range(0, len(bricks), n_cols):
            row_bricks = bricks[i : i + n_cols]
            # Equivalent to: row_bricks[0] | row_bricks[1] | ... | row_bricks[n]
            row_grid = reduce(operator.or_, row_bricks)
            rows.append(row_grid)

        # Equivalent to: rows[0] / rows[1] / ... / rows[n]
        final_grid = reduce(operator.truediv, rows)

        return final_grid

    def plot_imputation_summary_grid(self, t, p, met, method, metrics=None):
        """Combine NRMSE scatter and split KDE density subplots."""
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping summary grid.")
            return None
            
        pw.clear()
        
        ax1 = pw.Brick(figsize=(4, 4), label="NRMSE")
        self._plot_nrmse_scatter(t, p, met, method, ax=ax1)
        
        ax_qc = pw.Brick(figsize=(4, 4), label="KDE_QC")
        ax_sample = pw.Brick(figsize=(4, 4), label="KDE_Sample")
        
        self._plot_imputed_kde_overlay(
            metrics=metrics, ax_qc=ax_qc, ax_sample=ax_sample
        )
        
        return ax1 | ax_qc | ax_sample