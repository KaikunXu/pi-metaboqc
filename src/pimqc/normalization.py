# src/pimqc/normalization.py
"""
Purpose of script: Data normalization module for MetaboInt.
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
def _numba_vsn_nll(params, fit_data):
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

    _metadata = ["attrs","stats"]

    def __init__(
        self,
        *args: Any,
        pipeline_params: Optional[Dict[str, Any]] = None,
        norm_method: Optional[str] = None,
        robust_log: Optional[bool] = None,
        **kwargs: Any
    ) -> None:
        """Initialize MetaboIntNormalizer with parameters and metadata.

        Args:
            *args: Variable length arguments passed to DataFrame.
            pipeline_params: Global configuration dictionary from TOML.
            norm_method: Normalization method (e.g., 'VSN', 'Quantile').
            robust_log: Whether to apply robust log transformation.
            **kwargs: Keyword arguments passed to DataFrame constructor.
        """
        super().__init__(
            *args, pipeline_params=pipeline_params, **kwargs
        )

        # 1. Base defaults matching pipeline_parameters.toml
        norm_configs = {
            "norm_method": "VSN",
            "robust_log": False
        }

        # 2. TOML global configuration overrides base defaults
        if pipeline_params and "MetaboIntNormalizer" in pipeline_params:
            norm_configs.update(pipeline_params["MetaboIntNormalizer"])

        # 3. Explicit kwargs override TOML (Highest priority)
        if norm_method is not None:
            norm_configs["norm_method"] = norm_method
        if robust_log is not None:
            norm_configs["robust_log"] = robust_log

        # 4. Finalize state strictly into lifecycle attributes
        self.attrs.update(norm_configs)

    @property
    def _constructor(self):
        """Override pandas constructor to return current subclass type."""
        return MetaboIntNormalizer

    def __finalize__(self, other, method=None, **kwargs):
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
    def calc_ma_arrays(df_log):
        """Calculate flattened A and M values for MA-plot visualization.
        
        Args:
            df_log: DataFrame of log2 intensities (Features x Samples).
            
        Returns:
            a_flat, m_flat: Flattened 1D numpy arrays with NaNs removed.
        """
        # A: The average log2 intensity for each feature (X-axis)
        a_vals = df_log.mean(axis="columns")
        
        # M: The deviation of each sample from that feature's mean (Y-axis)
        m_df = df_log.sub(a_vals, axis="index")
        
        # Optimized vectorized implementation using numpy repeat
        # This is much faster than broadcasting or tiling DataFrames
        a_flat = np.repeat(a_vals.values, m_df.shape[1])
        m_flat = m_df.values.flatten()
        
        # Filter out NaNs to ensure hexbin/scatter stability
        valid = ~np.isnan(a_flat) & ~np.isnan(m_flat)
        return a_flat[valid], m_flat[valid]

    def calc_norm_quality_metrics(
        self, raw_obj, norm_obj, max_points: Optional[int] = 50000
    ) -> dict:
        """Calculate normalization-related metrics for technical reporting.

        Computes Jensen-Shannon Divergence (JSD), MA plot statistics (MAD,
        Spearman), and eCDF alignment (Wasserstein, KS). 
        Supports statistical subsampling to prevent O(N log N) bottlenecks.
        """
        metrics = {
            "JSD": {"QC": {}, "Sample": {}},
            "MA": {"Before Norm": {}, "After Norm": {}},
            "eCDF": {"Before Norm": {}, "After Norm": {}}
        }

        log_raw = su._extract_log2_target(raw_obj)
        log_norm = su._extract_log2_target(norm_obj)

        if log_raw is None or log_norm is None:
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
                if isinstance(qc_jsd, dict) else float(qc_jsd)
            )

        if not sam_cols.empty:
            raw_vals = _subsample(log_raw[sam_cols].values.flatten())
            norm_vals = _subsample(log_norm[sam_cols].values.flatten())
            sam_jsd = su.calc_jsd_similarity(raw_vals, norm_vals)
            
            metrics["JSD"]["Sample"]["Before vs After"] = (
                float(sam_jsd.get("JSD", sam_jsd.get("jsd", np.nan)))
                if isinstance(sam_jsd, dict) else float(sam_jsd)
            )

        # 2. MA Plot & 3. eCDF Distribution Metrics
        stages = [("Before Norm", log_raw), ("After Norm", log_norm)]
        for stage, df in stages:

            # ----------------------------------------------------
            # MA Metrics (MAD, Spearman rho)
            # ----------------------------------------------------
            a_vals, m_vals = self.calc_ma_arrays(df)
            if len(m_vals) > 0:
                m_median = np.median(m_vals)
                mad_val = float(np.median(np.abs(m_vals - m_median)))

                # Subsample for Spearman only if max_points is specified
                if max_points is not None and len(a_vals) > max_points:
                    np.random.seed(g_seed)
                    idx = np.random.choice(
                        len(a_vals), max_points, replace=False
                    )
                    a_sub, m_sub = a_vals[idx], m_vals[idx]
                else:
                    a_sub, m_sub = a_vals, m_vals

                rho_val = float(stats.spearmanr(a_sub, m_sub)[0])
                metrics["MA"][stage] = {"MAD": mad_val, "Spearman": rho_val}

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
                            vals = np.random.choice(
                                vals, 10000, replace=False
                            )

                        w_dists.append(
                            stats.wasserstein_distance(vals, pooled_ref)
                        )
                        ks_dists.append(stats.ks_2samp(vals, pooled_ref)[0])

                metrics["eCDF"][stage] = {
                    "Wasserstein": float(np.mean(w_dists)),
                    "KS": float(np.mean(ks_dists))
                }
        return metrics
    
    # ====================================================================
    # Mathematical Operators (Sample-wise & Global)
    # ====================================================================
    @staticmethod
    def calc_tic_normalization(df):
        """Apply Total Ion Current (TIC) normalization sample-wise."""
        col_sums = df.sum(axis="index").replace(0, 1)
        return df.div(col_sums, axis="columns") * col_sums.median()

    @staticmethod
    def calc_median_normalization(df):
        """Apply Median normalization sample-wise."""
        col_medians = df.median(axis="index").replace(0, 1)
        return df.div(col_medians, axis="columns") * col_medians.median()
    
    @staticmethod
    def calc_pqn_normalization(df, qc_cols=None):
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
    def calc_vsn_normalization(df):
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
            _numba_vsn_nll, x0=x0, args=(fit_data,),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-5}
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
        
        vsn_meta = {
            "vsn_scale": float(b),
            "vsn_shift": float(pure_shift)
        }
        return res_df, vsn_meta
    
    @staticmethod
    def calc_quantile_normalization(df):
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
        n_jobs: int = -1
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
                norm_factor = 2 ** shift
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
    def apply_normalization(self):
        """Execute normalization workflow to generate a Clean_Dataset.
        
        Implements different execution orders:
        - TIC/Median/PQN: Normalization -> Robust Log
        - Quantile: Robust Log -> Normalization
        - VSN: Intrinsic glog (No manual log needed)
        """
        # 1. Define target by concatenating QC and Samples
        # Use intersection to preserve the original column sequence (Inject Order)
        df_target = pd.concat([self._qc, self._actual_sample], axis=1)
        ordered_cols = self.columns.intersection(df_target.columns)
        df_target = df_target[ordered_cols].copy()
        
        if df_target.empty:
            raise ValueError("No target samples (QC/Actual) available.")

        method = self.attrs.get("norm_method", "None").upper()
        is_log = self.attrs.get("robust_log", False)
        
        # Passport stamps for the generated Clean_Dataset
        meta_stamps = {"norm_method": method, "is_logged": False}

        # -------------------------------------------------------------
        # Category A: Linear Scale Methods (TIC, Median, PQN) or None
        # Logic: Normalize first to correct drift, then Log for variance.
        # -------------------------------------------------------------
        if method in ["TIC", "MEDIAN", "PQN", "MDFC", "NONE"]:
            if method == "TIC":
                df_target = self.calc_tic_normalization(df_target)
            elif method == "MEDIAN":
                df_target = self.calc_median_normalization(df_target)
            elif method == "PQN":
                qc_cols = self._qc.columns
                df_target = self.calc_pqn_normalization(df_target, qc_cols)
            elif method == "MDFC":
                # MDFC routing using QC samples as the reference
                qc_cols = self._qc.columns
                n_cores = self.attrs.get("n_jobs", -1)
                df_target = self.calc_mdfc_normalization(
                    df_target, qc_cols=qc_cols, n_jobs=n_cores)
            
            # Apply log transform after sample-wise normalization
            if is_log:
                df_target = su.robust_log2_transform(df_target)
                meta_stamps["is_logged"] = True

        # -------------------------------------------------------------
        # Category B: Distribution Alignment (Quantile)
        # Logic: Log transform first, then align distributions.
        # -------------------------------------------------------------
        elif method == "QUANTILE":
            if is_log:
                df_target = su.robust_log2_transform(df_target)
                meta_stamps["is_logged"] = True
            df_target = self.calc_quantile_normalization(df_target)

        # -------------------------------------------------------------
        # Category C: Variance Stabilizing Normalization (VSN)
        # Logic: Intrinsic glog. robust_log is forced to False via config.
        # -------------------------------------------------------------
        elif method == "VSN":
            df_target, vsn_meta = self.calc_vsn_normalization(df_target)
            meta_stamps.update(vsn_meta)
            meta_stamps["is_logged"] = True  # Output is already in glog scale

        # -------------------------------------------------------------
        # Finalization: Create the Clean_Dataset object
        # -------------------------------------------------------------
        clean_obj = self._constructor(df_target).__finalize__(self)
        clean_obj.attrs.update(meta_stamps)
        
        # Clear legacy scaling attributes to ensure decoupling
        clean_obj.attrs.pop("is_scaled", None)
        clean_obj.attrs.pop("scale_method", None)
        
        return clean_obj
    
    @cached_property
    def normalization_metrics(self):
        """Extracts configuration and QA metrics from the workflow."""
        curr_stage = self.attrs.get("pipeline_stage", "Unknown")
        
        metrics = {
            "current_stage": curr_stage,
            "strategies": {
                "normalization_method": self.attrs.get("norm_method", "None"),
                "log_transform_active": self.attrs.get("is_logged", False)
            }
        }

        if self.attrs.get("norm_method", "None").upper() == "VSN":
            metrics["vsn_parameters"] = {
                "vsn_scale": self.attrs.get("vsn_scale", float("nan")),
                "vsn_shift": self.attrs.get("vsn_shift", float("nan"))
            }

        # Retrieve the comprehensive QA metrics from the data passport
        quality_metrics = self.attrs.get("normalization_quality_metrics", {})
        if quality_metrics:
            metrics["quality_metrics"] = quality_metrics

        return metrics

    @iu._exe_time
    def execute_normalization(self, output_dir):
        """Execute workflow, save outputs, and generate plots."""
        iu._check_dir_exists(dir_path=output_dir, handle="makedirs")

        method = self.attrs.get("norm_method", "None")
        is_log = self.attrs.get("robust_log", False)

        blank_count = len(self._blank.columns)
        if blank_count > 0:
            logger.info(f"Permanently dropping {blank_count} Blank samples.")

        logger.info(
            f"Applying Normalization | Method: {method} | Log: {is_log}"
        )

        # 1. Execute Core Calculation
        clean_obj = self.apply_normalization()

        # 2. Dynamic Suffix and Export
        suffix_parts = [method]
        if is_log and method.upper() != "VSN":
            suffix_parts.append("Log2")
        suffix = "_".join(suffix_parts)
        
        filename = f"Normalized_Data_{suffix}.csv"
        file_path = os.path.join(output_dir, filename)
        
        clean_obj.attrs["pipeline_stage"] = "Normalization"
        clean_obj.to_csv(
            path_or_buf=file_path, na_rep="NA", encoding="utf-8-sig"
        )

        # 3. QA Metrics Engine
        logger.info("Calculating normalization-related metrics...")
        quality_metrics = self.calc_norm_quality_metrics(
            raw_obj=self, norm_obj=clean_obj, max_points=50000
        )
        clean_obj.attrs["normalization_quality_metrics"] = quality_metrics

        # 4. Visualization Phase (2-Stage)
        logger.info("Generating diagnostic plots for normalization...")
        vis = MetaboVisualizerNormalizer(raw_obj=self, norm_obj=clean_obj)

        fig_grid = vis.plot_normalization_summary_grid(metrics=quality_metrics)
        if fig_grid:
            grid_path = os.path.join(
                output_dir, f"Normalization_Dashboard_{suffix}.svg"
            )
            vis.save_and_show_pw(pw_obj=fig_grid, file_path=grid_path)
            
        logger.info(f"Normalization summary dashboard saved as: {grid_path}")
        logger.success("Data normalization completed successfully.")
        return clean_obj

class MetaboVisualizerNormalizer(visualizer_classes.BaseMetaboVisualizer):
    """2-Stage Visualization Suite (Before vs After Normalization).

    Generates high-contrast diagnostic plots evaluating the efficacy of the 
    global variance stabilization and normalization preprocessing.
    """

    def __init__(self, raw_obj, norm_obj):
        """Initialize with pre- and post-normalization datasets."""
        super().__init__(metabo_obj=norm_obj)
        self.raw = raw_obj
        self.norm = norm_obj
        self.stages = [("Before Norm", self.raw), ("After Norm", self.norm)]
        self.pal = {"Before Norm": "tab:gray", "After Norm": "tab:red"}

    def _plot_rle_boxplot(self, ax=None):
        """Plot Relative Log Expression (RLE) grouped by Sample Type."""
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(4, 4))
        else:
            current_ax = ax
            fig = current_ax.figure

        st_col = self.norm.attrs.get("sample_type", "Sample Type")
        s_dict = self.norm.attrs.get("sample_dict", {})
        qc_label = s_dict.get("QC sample", "QC")
        act_label = s_dict.get("Actual sample", "Sample")

        df_list = []
        for label, obj in self.stages:
            log_d = su._extract_log2_target(obj)
            if log_d is None or log_d.empty:
                continue
                
            feature_medians = log_d.median(axis=1)
            rle_df = log_d.sub(feature_medians, axis=0)
            
            df_flat = rle_df.T.reset_index()
            melted = df_flat.melt(
                id_vars=list(rle_df.columns.names),
                var_name="Feature", value_name="RLE"
            )
            melted["Stage"] = label
            df_list.append(melted)

        if df_list:
            plot_df = pd.concat(df_list, ignore_index=True)
            plot_df = plot_df.dropna(subset=["RLE"])
            plot_df = plot_df[plot_df[st_col].isin([qc_label, act_label])]

            box_palette = {qc_label: "tab:red", act_label: "tab:gray"}

            sns.boxplot(
                data=plot_df, x="Stage", y="RLE", hue=st_col,
                hue_order=[qc_label, act_label], palette=box_palette,
                showfliers=False, ax=current_ax, width=0.6, 
                dodge=True, linewidth=1.2
            )

        current_ax.axhline(0, color="k", linestyle="--", linewidth=1.0)
        
        self._apply_standard_format(
            ax=current_ax, title="Relative Log Expression",
            xlabel="Pipeline Stage", ylabel="RLE Deviation", 
            append_stage=False
        )
        
        if current_ax.get_legend():
            current_ax.legend(loc="best")
            self._format_single_legend(current_ax, title="Sample Type")

        return fig if ax is None else current_ax

    def _plot_density_kde(self, metrics=None, ax_qc=None, ax_sample=None):
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
                    cols = obj._actual_sample.columns.intersection(
                        log_d.columns
                    )
                else:
                    cols = []

                if len(cols) > 0:
                    vals = log_d[cols].values.flatten()
                    vals = vals[~np.isnan(vals)]
                    if len(vals) > 0:
                        sns.kdeplot(
                            vals, ax=current_ax, label=label, 
                            color=self.pal[label], linewidth=2, alpha=0.8
                        )

            self._apply_standard_format(
                ax=current_ax, title=f"Density Overlay ({grp})",
                xlabel="Log2 Intensity", ylabel="Density", append_stage=False
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
                        0.96, 0.02, annot_text, transform=current_ax.transAxes,
                        fontsize=9, verticalalignment="bottom",
                        horizontalalignment="right", clip_on=False,
                        bbox=dict(
                            boxstyle="round,pad=0.4", facecolor="white",
                            edgecolor="none", alpha=0.6
                        )
                    )
                
            if current_ax.get_legend_handles_labels()[0]:
                current_ax.legend(loc="best")
                self._format_single_legend(current_ax, title="Stage")

        if return_fig:
            plt.tight_layout()
            return fig
        return ax_qc, ax_sample

    def _plot_ma_scatter(
        self, metrics=None, ax_before=None, ax_after=None, cax=None
    ):
        """Plot true MA hexbin scatter using a continuous density scale."""
        return_fig = False
        if ax_before is None or ax_after is None:
            fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(8, 4))
            return_fig = True

        color_map = pu.custom_linear_cmap(
            color_list=["white", "tab:red"], n_colors=256, 
            cmin=0.1, cmax=1.0
        )
        
        stages_data = []
        all_a_vals = []
        
        for label, obj in self.stages:
            log_d = su._extract_log2_target(obj)
            if log_d is not None and not log_d.empty:
                a_vals, m_vals = MetaboIntNormalizer.calc_ma_arrays(log_d)
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
            if len(x_val) > 0:
                hb = current_ax.hexbin(
                    x=x_val, y=y_val, gridsize=80, extent=extent,
                    cmap=color_map, mincnt=1, bins="log", 
                    linewidths=0, alpha=0.95
                )
                hb_list.append(hb)

                if metrics and "MA" in metrics and label in metrics["MA"]:
                    m_dict = metrics["MA"][label]
                    annot_text = (
                        f"Spread (MAD): {m_dict.get('MAD', 0):.3f}\n"
                        f"Bias (Spearman \u03c1): "
                        f"{m_dict.get('Spearman', 0):.3f}"
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

            current_ax.axhline(0, color="k", linestyle="--", linewidth=1.5)
            
            self._apply_standard_format(
                ax=current_ax, title=f"MA Plot ({label})",
                xlabel="Average Log2 Intensity (A)", 
                ylabel="Log2 Fold Change (M)", append_stage=False
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

    def _plot_ecdf_overlay(self, metrics=None, ax=None):
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
                if len(vals) == 0: continue

                vals_sorted = np.sort(vals)
                p = np.linspace(0, 1, len(vals_sorted))
                z = 2 if label == "After Norm" else 1
                ax.plot(vals_sorted, p, color=self.pal[label],
                        alpha=0.2, linewidth=1.0, zorder=z)
                
            ax.plot([], [], color=self.pal[label], label=label, linewidth=2)
            # Create handles for the legend
            handles.append(mlines.Line2D(
                [], [], color=self.pal[label], label=label, linewidth=2))

        self._apply_standard_format(
            ax=ax, title="eCDF Distribution Alignment",
            xlabel="Log2 Intensity", ylabel="Cumulative Probability",
            append_stage=False
        )

        # 1. Inject Metrics Text Box (Lower Right)
        if metrics and "eCDF" in metrics:
            lines = ["Dist. Alignment (W / KS)"]
            for label in ["Before Norm", "After Norm"]:
                m_dict = metrics["eCDF"].get(label, {})
                if m_dict:
                    lines.append(
                        f"{label}: {m_dict.get('Wasserstein', 0):.2f} / "
                        f"{m_dict.get('KS', 0):.3f}")
            
            ax.text(0.96, 0.02, "\n".join(lines), transform=ax.transAxes, 
                    fontsize=9, verticalalignment="bottom", 
                    horizontalalignment="right", clip_on=False,
                    bbox=dict(
                        boxstyle="round,pad=0.4", facecolor="white", 
                        edgecolor="none", alpha=0.6))

        # 2. Force Legend in Upper Left
        if handles:
            ax.legend(handles=handles)
            self._format_single_legend(
                ax=ax, title="Stage", loc="upper left", bbox_to_anchor=None)

        return fig if ax is None else ax

    def plot_normalization_summary_grid(self, metrics):
        """Combine RLE, KDE, Hexbin MA, and eCDF plots into a 2x3 grid."""
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping summary grid.")
            return None

        pw.clear()

        # Row 1: RLE | KDE QC | KDE Sample (Total Width = 4+4+4 = 12.0)
        ax_rle = pw.Brick(figsize=(4, 4), label="RLE")
        self._plot_rle_boxplot(ax=ax_rle)
        
        ax_qc = pw.Brick(figsize=(4, 4), label="KDE_QC")
        ax_sam = pw.Brick(figsize=(4, 4), label="KDE_SAM")
        self._plot_density_kde(metrics=metrics, ax_qc=ax_qc, ax_sample=ax_sam)
        
        row1 = ax_rle | ax_qc | ax_sam

        # Row 2: eCDF Overlay | MA Before | MA After | Colorbar
        # Adjusted width for exact alignment: 4.0 + 3.9 + 3.9 + 0.2 = 12.0
        ax_ecdf = pw.Brick(figsize=(4, 4), label="eCDF")
        self._plot_ecdf_overlay(metrics=metrics, ax=ax_ecdf)

        ax_ma_before = pw.Brick(figsize=(3.9, 4), label="MA_Before")
        ax_ma_after = pw.Brick(figsize=(3.9, 4), label="MA_After")
        ax_cb = pw.Brick(figsize=(0.2, 4), label="MA_CB")
        
        self._plot_ma_scatter(
            metrics=metrics, ax_before=ax_ma_before, 
            ax_after=ax_ma_after, cax=ax_cb
        )
        
        row2 = ax_ecdf | ax_ma_before | ax_ma_after | ax_cb

        return row1 / row2