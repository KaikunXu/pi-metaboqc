"""
Purpose of script: Execute Quality control-based signal drift correction.
"""

import os
import re
import math
import copy
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed
from loguru import logger

from numba import njit, prange

from . import io_utils as iu
from . import plot_utils as pu
from . import core_classes
from . import visualizer_classes

warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=RuntimeWarning)

# ==============================================================================
# Numba JIT Engines for Fast Robust QC-RLSC (LOESS with iterative reweighting)
# ==============================================================================

@njit(fastmath=True)
def _tricube_kernel(x):
    """Tricube weight function based on spatial distance."""
    abs_x = abs(x)
    if abs_x >= 1.0:
        return 0.0
    return (1.0 - abs_x**3)**3

@njit(fastmath=True)
def _bisquare_weight(res, s):
    """Bisquare weight function based on residuals for robust iterations."""
    # If perfect fit (median residual is 0), keep only points with tiny error
    if s <= 1e-9: 
        return 1.0 if abs(res) <= 1e-9 else 0.0
    v = res / (6.0 * s)
    if abs(v) >= 1.0:
        return 0.0
    return (1.0 - v*v)**2

@njit(fastmath=True)
def _numba_loess_1d_core(x, y, x_pred, frac, delta):
    """Core Weighted Least Squares pass applying spatial and robust weights."""
    n = len(x)
    m = len(x_pred)
    y_pred = np.zeros(m)
    
    k = int(math.ceil(n * frac))
    if k < 2: k = 2
    if k > n: k = n

    for i in range(m):
        x0 = x_pred[i]
        diffs = np.abs(x - x0)
        sorted_diffs = np.sort(diffs)
        h = sorted_diffs[k-1]
        
        if h <= 0.0:
            h = 1e-9
            
        sum_w = 0.0
        sum_wx = 0.0
        sum_wy = 0.0
        
        # Pass 1: Compute weighted mean (combining Tricube and Delta weights)
        for j in range(n):
            w = _tricube_kernel((x[j] - x0) / h) * delta[j]
            sum_w += w
            sum_wx += w * x[j]
            sum_wy += w * y[j]
            
        if sum_w <= 0:
            y_pred[i] = np.mean(y)
            continue
            
        x_bar = sum_wx / sum_w
        y_bar = sum_wy / sum_w
        
        numerator = 0.0
        denominator = 0.0
        
        # Pass 2: Calculate the slope for local linear regression
        for j in range(n):
            w = _tricube_kernel((x[j] - x0) / h) * delta[j]
            dev_x = x[j] - x_bar
            numerator += w * dev_x * (y[j] - y_bar)
            denominator += w * dev_x * dev_x
            
        if denominator <= 0:
            beta1 = 0.0
        else:
            beta1 = numerator / denominator
            
        beta0 = y_bar - beta1 * x_bar
        y_pred[i] = beta0 + beta1 * x0
        
    return y_pred

@njit(fastmath=True)
def _numba_loess_robust(x, y, x_pred, frac, max_iter):
    """Executes LOESS with robust iterative reweighting."""
    n = len(x)
    # Initial robust weights are all 1.0 (degrades to standard LOESS initially)
    delta = np.ones(n)
    
    # Execute robust iterations (replicates statsmodels max_iter=3 logic)
    for iteration in range(max_iter):
        # Predict only on training points (QCs) to calculate residuals
        y_fit = _numba_loess_1d_core(x, y, x, frac, delta)
        residuals = np.abs(y - y_fit)
        
        # Calculate the median of residuals
        s = np.median(residuals)
        
        # Update robust weights for each QC point
        for j in range(n):
            delta[j] = _bisquare_weight(residuals[j], s)
            
    # Final prediction on all target samples (x_pred) using converged delta
    return _numba_loess_1d_core(x, y, x_pred, frac, delta)

@njit(parallel=True, fastmath=True)
def _numba_batch_qc_rlsc(data, qc_mask, injection_orders, frac, max_iter):
    """Multi-core parallelized batch prediction engine."""
    rows, cols = data.shape
    predicted_matrix = np.zeros((rows, cols))
    x_qc_all = injection_orders[qc_mask]
    
    for i in prange(rows):
        row_data = data[i, :]
        y_qc_all = row_data[qc_mask]
        
        valid_mask = ~np.isnan(y_qc_all)
        valid_count = np.sum(valid_mask)
        
        if valid_count < 3:
            mean_val = 0.0
            if valid_count > 0:
                mean_val = np.sum(y_qc_all[valid_mask]) / valid_count
            else:
                mean_val = np.nan
            predicted_matrix[i, :] = mean_val
            continue
            
        clean_x = x_qc_all[valid_mask]
        clean_y = y_qc_all[valid_mask]
        
        # Call robust LOESS with iteration mechanism
        predicted_matrix[i, :] = _numba_loess_robust(
            clean_x, clean_y, injection_orders, frac, max_iter
        )
        
    return predicted_matrix


class MetaboIntCorrector(core_classes.MetaboInt):
    """Quality control-based signal drift correction."""

    _metadata = ["attrs"]

    def __init__(
        self,
        *args,
        pipeline_params=None,
        batch="Batch",
        inject_order="Inject Order",
        base_est="QC-RLSC",
        frac=0.3,
        n_tree=500,
        svr_kernel="rbf",
        svr_c=1.0,
        svr_gamma="scale",
        n_jobs=iu.__max_threading__,
        **kwargs
    ):
        """
        Initialize the signal correction class with explicit hyperparameters.

        Args:
            *args: Variable length arguments for pandas DataFrame.
            pipeline_params: Configuration dictionary for the pipeline.
            batch: Column name representing batch information.
            inject_order: Column name representing injection order.
            base_est: Estimator type ("QC-RLSC", "QC-RFSC", or "QC-SVR").
            frac: Smoothing parameter for LOESS.
            n_tree: Number of estimators for Random Forest.
            svr_kernel: Kernel type for SVR.
            svr_c: Regularization parameter for SVR.
            svr_gamma: Kernel coefficient for SVR.
            n_jobs: Number of parallel jobs.
            **kwargs: Extra arguments for pandas DataFrame.
        """
        super().__init__(*args, pipeline_params=pipeline_params, **kwargs)

        sc_configs = {
            "batch": batch, "inject_order": inject_order,
            "base_est": base_est, "frac": frac, "n_tree": n_tree,
            "svr_kernel": svr_kernel, "svr_c": svr_c, "svr_gamma": svr_gamma,
            "n_jobs": n_jobs
        }

        if pipeline_params and "MetaboIntCorrector" in pipeline_params:
            sc_configs.update(pipeline_params["MetaboIntCorrector"])

        self.attrs.update(sc_configs)

    @property
    def _constructor(self):
        """Override constructor to return MetaboIntCorrector."""
        return MetaboIntCorrector

    def __finalize__(self, other, method=None, **kwargs):
        """Explicitly preserve custom attributes during pandas operations."""
        super().__finalize__(other, method=method, **kwargs)
        if hasattr(other, "attrs"):
            self.attrs = copy.deepcopy(other.attrs)
        return self

    # =========================================================================
    # Statistical Utility Methods
    # =========================================================================

    @staticmethod
    def extract_qc_rsd_series(df_obj):
        """Extracts the RSD series for QC samples across all features.

        Args:
            df_obj: MetaboInt object or standard pandas DataFrame.

        Returns:
            pd.Series: Feature-wise Relative Standard Deviation (RSD).
        """
        if hasattr(df_obj, "_qc") and not df_obj._qc.empty:
            qc_data = df_obj._qc.astype(float)
        else:
            # Fallback for raw DataFrames without internal properties
            sample_type_col = df_obj.attrs.get("sample_type", "Sample Type")
            qc_label = df_obj.attrs.get("sample_dict", {}).get(
                "QC sample", "QC")
            mask = df_obj.columns.get_level_values(sample_type_col) == qc_label
            qc_data = df_obj.loc[:, mask].astype(float)
            
        return (qc_data.std(axis=1, ddof=1) / qc_data.mean(axis=1)).dropna()

    @staticmethod
    def calculate_median_qc_rsd(df_obj):
        """Calculates the scalar median RSD of QC samples."""
        rsd_series = MetaboIntCorrector.extract_qc_rsd_series(df_obj)
        if rsd_series.empty:
            return float("nan")
        return float(rsd_series.median())

    # =========================================================================
    # Algorithm Factory & Fitting Logic
    # =========================================================================
    def build_correction_pipeline(
        self, method, frac, n_tree, global_seed, svr_kernel, svr_c, svr_gamma):
        """Construct the ML pipeline based on explicit hyperparameters."""
        name = method.upper()
        
        # Note: LOESS/QC-RLSC is handled upstream by Numba JIT engines.
        # This pipeline builder is strictly for Scikit-Learn models.
        
        if name in ("RF", "RANDOM FOREST", "QC-RFSC"):
            # Revert: Remove StandardScaler for RF to match alpha version
            return RandomForestRegressor(
                n_estimators=n_tree, random_state=global_seed)
        elif name in ("SVR", "QC-SVR"):
            # Maintain Scaling for SVR to ensure numerical stability
            return make_pipeline(
                StandardScaler(), 
                SVR(kernel=svr_kernel, C=svr_c, gamma=svr_gamma))
        else:
            raise ValueError(
                f"Unsupported correction method for Joblib pipeline: '{method}'. "
                "Supported ML methods are: 'QC-RFSC', 'QC-SVR'. "
                "(Note: 'QC-RLSC' is executed via Numba, not Joblib).")

    def _fit_predict_feature(
        self, feat_idx, raw_vals, qc_mask, inject_order_array, method,
        kwargs_dict
    ):
        """Fit model on QCs and predict drift using SKLearn regressors."""
        qc_x = inject_order_array[qc_mask].reshape(-1, 1)
        qc_y = raw_vals[qc_mask]
        valid = ~np.isnan(qc_y)
        
        if valid.sum() < 1:
            return feat_idx, np.full(len(inject_order_array), np.nan)

        model = self.build_correction_pipeline(method=method, **kwargs_dict)
        try:
            if "SVR" in method.upper():
                y_scaler = StandardScaler()
                y_scaled = y_scaler.fit_transform(qc_y[valid].reshape(-1, 1))
                model.fit(qc_x[valid], y_scaled.ravel())
                pred_scaled = model.predict(inject_order_array.reshape(-1, 1))
                pred_y = y_scaler.inverse_transform(
                    pred_scaled.reshape(-1, 1)
                ).ravel()
                
            elif "RF" in method.upper():
                model.fit(qc_x[valid], qc_y[valid])
                pred_y = model.predict(inject_order_array.reshape(-1, 1))
                
            # The LOWESS branch has been completely removed
            return feat_idx, pred_y
            
        except Exception as e:
            logger.debug(f"Feature {feat_idx} fit failed via {method}: {e}")
            return feat_idx, np.full(len(inject_order_array), np.nan)

    # =========================================================================
    # Mathematical Core Phases
    # =========================================================================

    def _calculate_qc_baseline_means(
        self, batch_col, sample_type_col, qc_label):
        """Calculate and broadcast basic mean intensity of QC by batch."""
        # Isolate QC samples from the main matrix
        qc_df = self.loc[
            :, self.columns.get_level_values(sample_type_col) == qc_label]
        batch_levels = qc_df.columns.get_level_values(batch_col)
        
        # Calculate batch-wise mean for all features
        int_base = qc_df.transpose().groupby(batch_levels).mean().transpose()
        
        # Broadcast batch means to all samples in the original dataframe
        base_int_broadcasted = pd.DataFrame(
            index=self.index, columns=self.columns)
        for batch in self.columns.get_level_values(batch_col).unique():
            mask = self.columns.get_level_values(batch_col) == batch
            bc_block = pd.concat([int_base[batch]] * mask.sum(), axis=1)
            base_int_broadcasted.loc[:, mask] = bc_block.values
        return base_int_broadcasted
    
    def _calculate_predicted_matrix(
        self, batch_col, sample_type_col, inject_order_col, qc_label, method,
        n_jobs, kwargs_dict
    ):
        """Calculate continuous drift baseline with Numba acceleration."""
        pred_df = self.copy()
        
        for batch in self.columns.get_level_values(batch_col).unique():
            mask = self.columns.get_level_values(batch_col) == batch
            batch_data = self.loc[:, mask]
            qc_mask = batch_data.columns.get_level_values(
                sample_type_col) == qc_label
            inject_order= batch_data.columns.get_level_values(
                inject_order_col).values
            
            # ==============================================================
            # Fast-track Numba JIT pipeline
            # ==============================================================
            if method.upper() in ("LOESS", "LOWESS", "QC-RLSC"):
                logger.info(f"Executing Numba JIT QC-RLSC for batch: {batch}")
                
                # Explicitly extract parameters, prioritizing frac
                frac = kwargs_dict.get("frac", kwargs_dict.get("frac", 0.75))
                max_iter = kwargs_dict.get("max_iter", 3)
                
                # Pass directly to Numba engine for instant matrix prediction
                pred_matrix = _numba_batch_qc_rlsc(
                    data=batch_data.values, 
                    qc_mask=qc_mask, 
                    injection_orders=inject_order, 
                    frac=frac,
                    max_iter=max_iter
                )
                pred_df.loc[:, mask] = pred_matrix
                
            # ==============================================================
            # Joblib Channel for QC-SVR and QC-RFSC
            # ==============================================================
            else:
                log_msg = f"Using Joblib Parallel {method} for batch: {batch}"
                logger.info(log_msg)
                
                feature_indices = batch_data.index
                matrix_values = batch_data.values
                
                tasks = [delayed(self._fit_predict_feature)(
                    feature_indices[i], matrix_values[i, :], qc_mask,
                    inject_order, method, kwargs_dict
                ) for i in range(len(feature_indices))]
                
                results = Parallel(n_jobs=n_jobs)(iu.get_custom_progress(
                    tasks, len(tasks), desc=f"SC [{batch}]"
                ))
                
                for feat_idx, pred_vals in results:
                    pred_df.loc[feat_idx, mask] = pred_vals
        
        pred_df[pred_df <= 0] = np.nan
        return pred_df

    # =========================================================================
    # Execution Stream
    # =========================================================================

    @iu._exe_time
    def execute_signal_correction(self, output_dir):
        """
        Execute complete signal correction workflow matching alpha results.
        """
        # Explicit variable extraction to increase modularity
        sample_type_col = self.attrs.get("sample_type", "Sample Type")
        batch_col = self.attrs.get("batch", "Batch")
        inject_order_col = self.attrs.get("inject_order", "Inject Order")
        
        sample_dict = self.attrs.get("sample_dict", {})
        qc_label = sample_dict.get("QC sample", "QC")
        actual_label = sample_dict.get("Actual sample", "Sample")
        
        method = self.attrs.get("base_est", "QC-RLSC")
        n_jobs = self.attrs.get("n_jobs", -1)
        
        pipe_params = self.attrs.get("pipeline_parameters", {})
        bound_type = pipe_params.get("MetaboInt", {}).get("boundary", "IQR")
        
        kwargs_dict = {
            "frac": self.attrs.get("frac"), "n_tree": self.attrs.get("n_tree"),
            "global_seed": self.attrs.get("global_seed", 123),
            "svr_kernel": self.attrs.get("svr_kernel"),
            "svr_c": self.attrs.get("svr_c"), 
            "svr_gamma": self.attrs.get("svr_gamma")
        }

        iu._check_dir_exists(output_dir, handle="makedirs")

        # Phase 1: Signal Drift Prediction
        base_int_broadcasted = self._calculate_qc_baseline_means(
            batch_col, sample_type_col, qc_label)
        pred_df = self._calculate_predicted_matrix(
            batch_col, sample_type_col, inject_order_col, qc_label, method,
            n_jobs, kwargs_dict
        )
        
        pred_df.to_csv(os.path.join(
            output_dir, f"QC_Fit_Baseline_{method}.csv"
        ))
        
        # Phase 2: Intra-batch Correction
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            intra_df = self._constructor(
                base_int_broadcasted * (self / pred_df)
            ).__finalize__(self)

        intra_path = os.path.join(
            output_dir, f"Intra_Batch_Corrected_{method}.csv")
        intra_df.to_csv(intra_path)
        logger.info(
            f"Intra-correction completed, saved as : {intra_path}")

        # Phase 3: Inter-batch Alignment
        inter_df = intra_df.copy()
        if len(self.columns.get_level_values(batch_col).unique()) > 1:
            intra_qc = intra_df.loc[:, 
                intra_df.columns.get_level_values(sample_type_col) == qc_label
            ]
            bt_qc_mean = intra_qc.transpose().groupby(batch_col).mean(
                ).transpose()
            global_mean = intra_qc.mean(axis=1)
            
            for batch in self.columns.get_level_values(batch_col).unique():
                mask = inter_df.columns.get_level_values(batch_col) == batch
                inter_df.loc[:, mask] = inter_df.loc[:, mask].multiply(
                    global_mean / bt_qc_mean[batch], axis=0
                )

        inter_path = os.path.join(
            output_dir, f"Inter_Batch_Corrected_{method}.csv")
        inter_df.to_csv(inter_path)
        logger.info(
            f"Inter-correction completed, saved as : {inter_path}")
        
        # Phase 4: Visualization Suite
        vis = MetaboVisualizerCorrector(self)
        # 4.1: QC RSD Progression Boxplot
        fig_rsd = vis.plot_corr_rsd(
            self, intra_df, inter_df, sample_type_col, qc_label)
        vis.save_and_close_fig(
            fig_rsd, os.path.join(
                output_dir, f"QC_RSD_Boxplot_{method}")
        )
        
        if len(self.valid_is) > 0:
            # 4.2: 3-Stage Scatter Panels per Internal Standard
            fig_dict = vis.plot_is_int_order_scatter(
                self, intra_df, inter_df, pred_df, self.valid_is,
                sample_type_col, batch_col, inject_order_col, qc_label,
                actual_label, bound_type
            )
            for feat, fig in fig_dict.items():
                safe_feat = re.sub(r"[^a-zA-Z0-9]", "_", feat)
                vis.save_and_close_fig(
                    fig, os.path.join(
                        output_dir, "Internal_Standard_Scatters",
                        f"IS_Scatter_{safe_feat}")
                )
            
            # 4.3: Predicted Baseline Overlay Grid
            fig_pred = vis.plot_pred_baseline_is(
                self, pred_df, self.valid_is, sample_type_col, batch_col, 
                inject_order_col, qc_label, actual_label
            )
            vis.save_and_close_fig(
                fig_pred,
                os.path.join(output_dir, f"Pred_Base_IS_{method}")
            )
        
        # ==========================================================
        # Calculate standardized RSD metrics utilizing static method
        raw_rsd = MetaboIntCorrector.calculate_median_qc_rsd(self)
        
        # 1. Update Intra-batch passport
        intra_df.attrs["pipeline_stage"] = "Intra-batch correction"
        intra_df.attrs["qc_rsd_baseline"] = raw_rsd
        intra_df.attrs["qc_rsd_current"] = (
            MetaboIntCorrector.calculate_median_qc_rsd(intra_df)
        )
        
        # 2. Update Inter-batch passport
        inter_df.attrs["pipeline_stage"] = "Inter-batch correction"
        inter_df.attrs["qc_rsd_baseline"] = raw_rsd
        inter_df.attrs["qc_rsd_current"] = (
            MetaboIntCorrector.calculate_median_qc_rsd(inter_df)
        )
        # ==========================================================

        logger.success(
            "Data signal drift and batch-effect correction completed."
        )
        return intra_df, inter_df

    @property
    def correction_metrics(self):
        """
        Extracts drift and batch correction performance metrics.
        """

        stage = self.attrs.get("pipeline_stage", "Unknown")
        
        rsd_base = self.attrs.get("qc_rsd_baseline")
        rsd_curr = self.attrs.get("qc_rsd_current")

        metrics = {
            "correction_status": stage,
            "methodology": {
                "base_est": self.attrs.get("base_est", "QC-SVR"),
                "frac": self.attrs.get("frac", 0.3),
                "svr_kernel": self.attrs.get("svr_kernel", "rbf"),
                "n_tree": self.attrs.get("n_tree", 500),
                "svr_c": self.attrs.get("svr_c", 100.0),
                "svr_gamma": self.attrs.get("svr_gamma", 1.0),
            },
            "performance": {
                "median_qc_rsd_baseline": rsd_base,
                "median_qc_rsd_current": rsd_curr,
                "absolute_rsd_reduction": None,
                "relative_noise_reduction": None
            }
        }

        # Calculate improvement ratio if values exist
        if rsd_base and rsd_curr and rsd_base > 0:
            abs_red = rsd_base - rsd_curr
            rel_red = abs_red / rsd_base
            metrics["performance"]["absolute_rsd_reduction"] = abs_red
            metrics["performance"]["relative_noise_reduction"] = rel_red
        return metrics


class MetaboVisualizerCorrector(visualizer_classes.BaseMetaboVisualizer):
    """Visualization suite matching original alpha output styles."""

    def __init__(self, corr_obj):
        """Initialize with a computed MetaboIntCorrector object."""
        super().__init__(metabo_obj=corr_obj)
        self.corr = corr_obj

    # =========================================================================
    # Evaluation & Diagnostic Plotters
    # =========================================================================

    def plot_corr_rsd(
        self, raw_df, intra_df, inter_df, sample_type_col, qc_label, ax=None
    ):
        """Plots RSD boxplots across different signal correction stages.

        Args:
            raw_df: Uncorrected MetaboInt object.
            intra_df: Intra-batch corrected MetaboInt object.
            inter_df: Inter-batch corrected MetaboInt object.
            sample_type_col: Column name indicating sample types.
            qc_label: Label used for QC samples.
            ax: Optional matplotlib Axes object for plotting.

        Returns:
            A matplotlib Figure or Axes object containing the boxplots.
        """
        # 1. Calculate raw RSD series using centralized static method
        rsd_raw = MetaboIntCorrector.extract_qc_rsd_series(raw_df)
        rsd_intra = MetaboIntCorrector.extract_qc_rsd_series(intra_df)
        rsd_inter = MetaboIntCorrector.extract_qc_rsd_series(inter_df)

        # 2. Aggregate RSD statistics into a long-form DataFrame
        plot_df = pd.DataFrame({
            "Original": rsd_raw,
            "Intra-batch\ncorrected": rsd_intra,
            "Inter-batch\ncorrected": rsd_inter
        }).melt(var_name="Stage", value_name="RSD")

        if ax is None:
            fig, current_ax = plt.subplots(figsize=(4, 4))
        else:
            current_ax = ax
            fig = current_ax.figure
            
        sns.boxplot(
            data=plot_df, x="Stage", y="RSD", hue="Stage", width=0.6, 
            showfliers=False, palette=pu.extract_linear_cmap(
                pu.custom_linear_cmap(["white", "tab:red"], 3), 0, 1
            ), ax=current_ax
        )
        
        # 3. Extract and format median RSD values directly from series.
        median_raw = rsd_raw.median() * 100
        median_intra = rsd_intra.median() * 100
        median_inter = rsd_inter.median() * 100

        annot_text = (
            f"Median RSD:\n"
            f"Original: {median_raw:.2f}%\n"
            f"Intra-batch: {median_intra:.2f}%\n"
            f"Inter-batch: {median_inter:.2f}%"
        )

        current_ax.text(
            0.96, 0.98, annot_text, transform=current_ax.transAxes,
            fontsize=10, verticalalignment="top",
            horizontalalignment="right", clip_on=False,
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor="white",
                edgecolor="none", alpha=0.6))

        # 4. Apply standard axis formatting.
        self._apply_standard_format(
            current_ax, ylabel="RSD (%)", append_stage=False
        )
        pu.change_axis_format(current_ax, "percentage", "y")
        
        if ax is None:
            return fig
            
        return current_ax

    def plot_single_is_scatter(
        self, df, feat, sample_type, batch, inject_order, qc_label,
        actual_label, ylabel, boundary, ax=None
    ):
        """Plot a single scatter panel with calculated boundaries."""
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(7.5, 3))
        else:
            current_ax = ax
            fig = current_ax.figure
            
        # Extract and format data for plotting
        plot_data = df.int_order_info(feat_type="IS").reset_index()
        
        # [BUG FIX]: Explicitly sort to render QC samples on the top layer
        plot_data[sample_type] = pd.Categorical(
            plot_data[sample_type], categories=[actual_label, qc_label],
            ordered=True
        )
        plot_data = plot_data.sort_values(sample_type)
        
        sns.scatterplot(
            data=plot_data, x=inject_order, y=feat, hue=sample_type,
            style=batch, s=40, edgecolor="k", palette=self.pal,
            hue_order=[qc_label, actual_label], markers=self.style_map, 
            style_order=self.all_batches, ax=current_ax
        )
        
        # Calculate and draw Shewhart boundary lines
        solid_line, lower_limit, upper_limit = core_classes.MetaboInt(
            ).calculate_boundaries(plot_data[feat], boundary)
        for y, linestyle in zip(
            [solid_line, lower_limit, upper_limit], ["-", "--", "--"]):
            current_ax.axhline(y, color="k", linestyle=linestyle)
            
        self._apply_standard_format(
            current_ax, xlabel=inject_order, ylabel=ylabel, append_stage=False)
        pu.change_axis_format(current_ax, "scientific notation", "y")
        return fig

    def plot_is_int_order_scatter(
        self, raw, intra, inter, pred, valid, sample_type, batch,
        inject_order, qc_label, actual_label, boundary
    ):
        """Reconstruct original 3-row scatter layout with baseline overlay."""
        fig_dict = {}
        for feat in valid:
            fig = plt.figure(figsize=(7.5, 9), layout="constrained")
            stages = [
                ("Raw Intensity", raw), 
                ("After Intra-batch \nCorrected", intra), 
                ("After Inter-batch \nCorrected", inter)
            ]
            
            for n, (ylabel, df) in enumerate(stages):
                ax = plt.subplot(3, 1, n + 1)
                self.plot_single_is_scatter(
                    df, feat, sample_type, batch, inject_order, qc_label,
                    actual_label, ylabel, boundary, ax)
                
                # Overlay red dashed baseline specifically on the raw plot
                if ylabel == "Raw Intensity":
                    pred_info = pred.int_order_info(
                        feat_type="IS").reset_index()
                    for batch_id in pred_info[batch].unique():
                        sns.lineplot(
                            data=pred_info[pred_info[batch] == batch_id],
                            x=inject_order, y=feat, 
                            color="k", linestyle="-", ax=ax, zorder=3
                        )
                
                # Manage legends: Only the bottom plot shows the unified legend
                if n == 2:
                    self._format_multi_legends(
                        ax=ax, group_titles=[sample_type, batch])
                elif ax.get_legend():
                    ax.legend().remove()
            
            plt.close(fig)
            fig_dict[feat] = fig
        return fig_dict

    def plot_pred_baseline_is(
        self, raw, pred, valid, sample_type, batch, inject_order, qc_label,
        actual_label):
        """Reconstruct original multi-panel baseline overlay grid."""
        num_cols = 2
        num_rows = int(np.ceil(len(valid) / num_cols))
        fig = plt.figure(
            figsize=(7.5 * num_cols, 3 * num_rows), layout="constrained")
        
        for n, feat in enumerate(valid):
            ax = plt.subplot(num_rows, num_cols, n + 1)
            plot_data = raw.int_order_info(feat_type="IS").reset_index()
            
            # [BUG FIX]: Explicitly sort to render QC samples on the top layer
            plot_data[sample_type] = pd.Categorical(
                plot_data[sample_type], categories=[actual_label, qc_label],
                ordered=True
            )
            plot_data = plot_data.sort_values(sample_type)
            
            # Render raw scatter points
            sns.scatterplot(
                data=plot_data, x=inject_order, y=feat, hue=sample_type,
                style=batch, s=40,  edgecolor="k", palette=self.pal,
                hue_order=[qc_label, actual_label], 
                markers=self.style_map, style_order=self.all_batches, ax=ax
            )
            
            # Overlay continuous black baseline prediction
            pred_info = pred.int_order_info(feat_type="IS").reset_index()
            for batch_id in pred_info[batch].unique():
                sns.lineplot(
                    data=pred_info[pred_info[batch] == batch_id],
                    x=inject_order, y=feat, color="k", ax=ax
                )
                
            self._apply_standard_format(
                ax, xlabel=inject_order, ylabel=feat, append_stage=False)
            
            # Management of legends in grid
            if n == len(valid) - 1:
                self._format_multi_legends(
                    ax=ax, group_titles=[sample_type, batch])
            elif ax.get_legend():
                ax.legend().remove()
        return fig