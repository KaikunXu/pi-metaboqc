"""
Purpose of script: Execute Quality control-based signal drift correction.
"""

import os
import re
import copy
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from loguru import logger

from . import io_utils as iu
from . import plot_utils as pu
from . import core_classes
from . import visualizer_classes

warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=RuntimeWarning)


class _LoessSmoother(BaseEstimator, RegressorMixin):
    """Loess corrector implementation following scikit-learn API."""

    def __init__(self, span=0.3):
        """
        Initialize the LOESS smoother.

        Args:
            span: Fraction of data used for estimating each y-value.
        """
        self.span = span
        self.interpolator_ = None

    def fit(self, x, y):
        """Fit the loess algorithm matching alpha version logic."""
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        try:
            # Revert: Match alpha version lowess parameters
            y_fit = lowess(
                endog=y_arr, exog=x_arr, frac=self.span, it=3, 
                missing="drop", is_sorted=True, return_sorted=False
            )
            # Revert: No extrapolation to match alpha version boundary behavior
            self.interpolator_ = interp1d(x=x_arr, y=y_fit, bounds_error=False)
        except Exception:
            pass
        return self

    def predict(self, x):
        """Predict intensity based on injection order."""
        try:
            if self.interpolator_ is None:
                raise NotFittedError("Loess smoother is not fitted yet.")
            return self.interpolator_(x)
        except NotFittedError:
            return np.full(shape=x.shape, fill_value=np.nan)


class MetaboIntCorrector(core_classes.MetaboInt):
    """Quality control-based signal drift correction."""

    _metadata = ["attrs"]

    def __init__(
        self,
        *args,
        pipeline_params=None,
        mode="POS",
        batch="Batch",
        inject_order="Inject Order",
        base_est="QC-RLSC",
        span=0.3,
        n_tree=500,
        random_state=12345,
        svr_kernel="rbf",
        svr_c=1.0,
        svr_gamma="scale",
        n_jobs=-1,
        **kwargs
    ):
        """
        Initialize the signal correction class with explicit hyperparameters.

        Args:
            *args: Variable length arguments for pandas DataFrame.
            pipeline_params: Configuration dictionary for the pipeline.
            mode: MS Polarity ("POS" or "NEG").
            batch: Column name representing batch information.
            inject_order: Column name representing injection order.
            base_est: Estimator type ("QC-RLSC", "QC-RFSC", or "QC-SVR").
            span: Smoothing parameter for LOESS.
            n_tree: Number of estimators for Random Forest.
            random_state: Seed for random number generators.
            svr_kernel: Kernel type for SVR.
            svr_c: Regularization parameter for SVR.
            svr_gamma: Kernel coefficient for SVR.
            n_jobs: Number of parallel jobs.
            **kwargs: Extra arguments for pandas DataFrame.
        """
        super().__init__(*args, pipeline_params=pipeline_params, **kwargs)

        sc_configs = {
            "mode": mode, "batch": batch, "inject_order": inject_order,
            "base_est": base_est, "span": span, "n_tree": n_tree,
            "random_state": random_state, "svr_kernel": svr_kernel,
            "svr_c": svr_c, "svr_gamma": svr_gamma, "n_jobs": n_jobs
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
    # Algorithm Factory & Fitting Logic
    # =========================================================================

    def build_correction_pipeline(
        self, method, span, n_tree, random_state, svr_kernel, svr_c, svr_gamma
    ):
        """Construct the ML pipeline based on explicit hyperparameters."""
        name = method.upper()
        if name in ("LOESS", "LOWESS", "QC-RLSC"):
            return _LoessSmoother(span=span)
        elif name in ("RF", "RANDOM FOREST", "QC-RFSC"):
            # Revert: Remove StandardScaler for RF to match alpha version
            return RandomForestRegressor(
                n_estimators=n_tree, random_state=random_state
            )
        elif name in ("SVR", "QC-SVR"):
            # Maintain Scaling for SVR to ensure numerical stability
            return make_pipeline(
                StandardScaler(), 
                SVR(kernel=svr_kernel, C=svr_c, gamma=svr_gamma)
            )
        return _LoessSmoother(span=span)

    def _fit_predict_feature(
        self, feat_idx, raw_vals, qc_mask, io_arr, method, kwargs_dict
    ):
        """Fit model on QCs and predict drift matching alpha version logic."""
        qc_x = io_arr[qc_mask].reshape(-1, 1)
        qc_y = raw_vals[qc_mask]
        valid = ~np.isnan(qc_y)
        
        # Revert: Use relaxed threshold (min 1 valid QC) matching alpha version
        if valid.sum() < 1:
            return feat_idx, np.full(len(io_arr), np.nan)

        model = self.build_correction_pipeline(method=method, **kwargs_dict)
        try:
            if "SVR" in method.upper():
                y_scaler = StandardScaler()
                y_scaled = y_scaler.fit_transform(qc_y[valid].reshape(-1, 1))
                model.fit(qc_x[valid], y_scaled.ravel())
                pred_scaled = model.predict(io_arr.reshape(-1, 1))
                pred_y = y_scaler.inverse_transform(
                    pred_scaled.reshape(-1, 1)
                ).ravel()
            else:
                # LOESS/RF fitting logic
                x_fit = qc_x[valid] if "RF" in method.upper() else io_arr[qc_mask]
                model.fit(x_fit, qc_y[valid])
                pred_y = model.predict(io_arr)
            return feat_idx, pred_y
        except Exception:
            return feat_idx, np.full(len(io_arr), np.nan)

    # =========================================================================
    # Mathematical Core Phases
    # =========================================================================

    def _calculate_qc_baseline_means(self, bt_col, st_col, qc_lbl):
        """Calculate and broadcast basic mean intensity of QC by batch."""
        # Isolate QC samples from the main matrix
        qc_df = self.loc[:, self.columns.get_level_values(st_col) == qc_lbl]
        batch_levels = qc_df.columns.get_level_values(bt_col)
        
        # Calculate batch-wise mean for all features
        int_base = qc_df.transpose().groupby(batch_levels).mean().transpose()
        
        # Broadcast batch means to all samples in the original dataframe
        int_base_bc = pd.DataFrame(index=self.index, columns=self.columns)
        for batch in self.columns.get_level_values(bt_col).unique():
            mask = self.columns.get_level_values(bt_col) == batch
            bc_block = pd.concat([int_base[batch]] * mask.sum(), axis=1)
            int_base_bc.loc[:, mask] = bc_block.values
        return int_base_bc

    def _calculate_predicted_matrix(
        self, bt_col, st_col, io_col, qc_lbl, method, n_jobs, kwargs_dict
    ):
        """Calculate continuous drift baseline with parallel progress bar."""
        pred_df = self.copy()
        for batch in self.columns.get_level_values(bt_col).unique():
            # Filter samples belonging to current batch
            mask = self.columns.get_level_values(bt_col) == batch
            batch_data = self.loc[:, mask]
            qc_m = batch_data.columns.get_level_values(st_col) == qc_lbl
            io_a = batch_data.columns.get_level_values(io_col).values
            
            # Prepare parallel fitting tasks
            tasks = [delayed(self._fit_predict_feature)(
                idx, row.values, qc_m, io_a, method, kwargs_dict
            ) for idx, row in batch_data.iterrows()]
            
            # Execute with custom progress bar from io_utils
            results = Parallel(n_jobs=n_jobs)(iu.get_custom_progress(
                tasks, len(tasks), desc=f"SC [{batch}]"
            ))
            for feat_idx, pred_vals in results:
                pred_df.loc[feat_idx, mask] = pred_vals
        
        # Revert: Global truncation to NaN for non-positive predicted baselines
        pred_df[pred_df <= 0] = np.nan
        return pred_df

    # =========================================================================
    # Execution Stream
    # =========================================================================

    @iu._exe_time
    def execute_signal_correction(self, output_dir):
        """Execute complete signal correction workflow matching alpha results."""
        # Explicit variable extraction to increase modularity
        st_col = self.attrs.get("sample_type", "Sample Type")
        bt_col = self.attrs.get("batch", "Batch")
        io_col = self.attrs.get("inject_order", "Inject Order")
        
        sample_dict = self.attrs.get("sample_dict", {})
        qc_lbl = sample_dict.get("QC sample", "QC")
        act_lbl = sample_dict.get("Actual sample", "Sample")
        
        mode = self.attrs.get("mode", "POS")
        method = self.attrs.get("base_est", "QC-RLSC")
        n_jobs = self.attrs.get("n_jobs", -1)
        
        pipe_params = self.attrs.get("pipeline_parameters", {})
        bound_type = pipe_params.get("MetaboInt", {}).get("boundary", "IQR")
        
        kwargs_dict = {
            "span": self.attrs.get("span"), "n_tree": self.attrs.get("n_tree"),
            "random_state": self.attrs.get("random_state"),
            "svr_kernel": self.attrs.get("svr_kernel"),
            "svr_c": self.attrs.get("svr_c"), 
            "svr_gamma": self.attrs.get("svr_gamma")
        }

        iu._check_dir_exists(output_dir, handle="makedirs")

        # Phase 1: Signal Drift Prediction
        int_base_bc = self._calculate_qc_baseline_means(bt_col, st_col, qc_lbl)
        pred_df = self._calculate_predicted_matrix(
            bt_col, st_col, io_col, qc_lbl, method, n_jobs, kwargs_dict
        )
        
        pred_df.to_csv(os.path.join(
            output_dir, f"QC_Fit_Baseline_{method}_{mode}.csv"
        ))
        logger.info("Baseline fitting completed")
        
        # Phase 2: Intra-batch Correction
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            intra_df = self._constructor(
                int_base_bc * (self / pred_df)
            ).__finalize__(self)

        intra_path = os.path.join(
            output_dir, f"Intra_Batch_Corrected_{method}_{mode}.csv")
        intra_df.to_csv(intra_path)
        logger.info(
            f"Intra-correction completed, saved as : {intra_path}")
        logger.info(
            f"MetaboInt shape after intra-correction: {intra_df.shape}")

        # Phase 3: Inter-batch Alignment
        inter_df = intra_df.copy()
        if len(self.columns.get_level_values(bt_col).unique()) > 1:
            intra_qc = intra_df.loc[:, 
                intra_df.columns.get_level_values(st_col) == qc_lbl
            ]
            bt_qc_mean = intra_qc.transpose().groupby(bt_col).mean().transpose()
            global_mean = intra_qc.mean(axis=1)
            
            for batch in self.columns.get_level_values(bt_col).unique():
                mask = inter_df.columns.get_level_values(bt_col) == batch
                inter_df.loc[:, mask] = inter_df.loc[:, mask].multiply(
                    global_mean / bt_qc_mean[batch], axis=0
                )

        inter_path = os.path.join(
            output_dir, f"Inter_Batch_Corrected_{method}_{mode}.csv")
        inter_df.to_csv(intra_path)
        logger.info(
            f"Inter-correction completed, saved as : {inter_path}")
        logger.info(
            f"MetaboInt shape after inter-correction: {inter_df.shape}")
        
        # Phase 4: Visualization Suite
        vis = MetaboVisualizerCorrector(self)
        # 4.1: QC RSD Progression Boxplot
        fig_rsd = vis.plot_corr_rsd(self, intra_df, inter_df, st_col, qc_lbl)
        vis.save_and_close_fig(
            fig_rsd, os.path.join(
                output_dir, f"QC_RSD_Boxplot_{method}_{mode}.pdf")
        )
        
        if len(self.valid_is) > 0:
            # 4.2: 3-Stage Scatter Panels per Internal Standard
            fig_dict = vis.plot_is_int_order_scatter(
                self, intra_df, inter_df, pred_df, self.valid_is, st_col, 
                bt_col, io_col, qc_lbl, act_lbl, bound_type
            )
            for feat, fig in fig_dict.items():
                safe_feat = re.sub(r'[^a-zA-Z0-9]', '_', feat)
                vis.save_and_close_fig(
                    fig,
                    os.path.join(output_dir, f"Scatter_{safe_feat}_{mode}.pdf")
                )
            
            # 4.3: Predicted Baseline Overlay Grid
            fig_pred = vis.plot_pred_baseline_is(
                self, pred_df, self.valid_is, st_col, bt_col, io_col, 
                qc_lbl, act_lbl
            )
            vis.save_and_close_fig(
                fig_pred,
                os.path.join(output_dir, f"Pred_Base_IS_{method}_{mode}.pdf")
            )
            
        logger.success(
            "Data signal drift and batch-effect correction "
            "completed.")
        # ==========================================================
        intra_df.attrs["pipeline_stage"] = "Intra-batch correction"
        inter_df.attrs["pipeline_stage"] = "Inter-batch correction"
        # ==========================================================

        return intra_df, inter_df


class MetaboVisualizerCorrector(visualizer_classes.BaseMetaboVisualizer):
    """Visualization suite matching original alpha output styles."""

    def __init__(self, corr_obj):
        """Initialize with a computed MetaboIntCorrector object."""
        super().__init__(metabo_obj=corr_obj)
        self.corr = corr_obj

    # =========================================================================
    # Evaluation & Diagnostic Plotters
    # =========================================================================

    def plot_corr_rsd(self, raw_df, intra_df, inter_df, st_col, qc_lbl, ax=None):
        """Plot RSD boxplots across different correction stages."""
        def get_rsd(df):
            qc = df.loc[:, df.columns.get_level_values(st_col) == qc_lbl]
            return (qc.std(axis=1, ddof=1) / qc.mean(axis=1)).dropna()

        # Aggregate RSD statistics
        plot_df = pd.DataFrame({
            "Original": get_rsd(raw_df), 
            "Intra-batch\ncorrected": get_rsd(intra_df),
            "Inter-batch\ncorrected": get_rsd(inter_df)
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
        
        self._apply_standard_format(
            current_ax, ylabel="RSD (%)",append_stage=False)
        pu.change_axis_format(current_ax, "percentage", "y")
        return fig

    def plot_single_is_scatter(
        self, df, feat, st, bt, io, qcl, actl, yl, bnd, ax=None
    ):
        """Plot a single scatter panel with calculated boundaries."""
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(7.5, 3))
        else:
            current_ax = ax
            fig = current_ax.figure
            
        # Extract and format data for plotting
        p_data = df.int_order_info(feat_type="IS").reset_index()
        
        # [BUG FIX]: Explicitly sort to render QC samples on the top layer
        p_data[st] = pd.Categorical(
            p_data[st], categories=[actl, qcl], ordered=True
        )
        p_data = p_data.sort_values(st)
        
        sns.scatterplot(
            data=p_data, x=io, y=feat, hue=st, style=bt, s=40, edgecolor="k", 
            palette=self.pal, hue_order=[qcl, actl], markers=self.style_map, 
            style_order=self.all_batches, ax=current_ax
        )
        
        # Calculate and draw Shewhart boundary lines
        s, l, u = core_classes.MetaboInt().calculate_boundaries(
            p_data[feat], bnd
        )
        for y, ls in zip([s, l, u], ["-", "--", "--"]):
            current_ax.axhline(y, color="k", linestyle=ls)
            
        self._apply_standard_format(
            current_ax, xlabel=io, ylabel=yl, append_stage=False)
        pu.change_axis_format(current_ax, "scientific notation", "y")
        return fig

    def plot_is_int_order_scatter(
        self, raw, intra, inter, pred, valid, st, bt, io, qcl, actl, bnd
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
            
            for n, (yl, df) in enumerate(stages):
                ax = plt.subplot(3, 1, n + 1)
                self.plot_single_is_scatter(
                    df, feat, st, bt, io, qcl, actl, yl, bnd, ax)
                
                # Overlay red dashed baseline specifically on the raw plot
                if yl == "Raw Intensity":
                    p_info = pred.int_order_info(feat_type="IS").reset_index()
                    for b in p_info[bt].unique():
                        sns.lineplot(
                            data=p_info[p_info[bt] == b], x=io, y=feat, 
                            color="tab:red", linestyle="--", ax=ax, zorder=3
                        )
                
                # Manage legends: Only the bottom plot shows the unified legend
                if n == 2:
                    self._format_multi_legends(ax=ax, group_titles=[st, bt])
                elif ax.get_legend():
                    ax.legend().remove()
            
            plt.close(fig)
            fig_dict[feat] = fig
        return fig_dict

    def plot_pred_baseline_is(self, raw, pred, valid, st, bt, io, qcl, actl):
        """Reconstruct original multi-panel baseline overlay grid."""
        nc = 2
        nr = int(np.ceil(len(valid) / nc))
        fig = plt.figure(figsize=(7.5 * nc, 3 * nr), layout="constrained")
        
        for n, feat in enumerate(valid):
            ax = plt.subplot(nr, nc, n + 1)
            p_data = raw.int_order_info(feat_type="IS").reset_index()
            
            # [BUG FIX]: Explicitly sort to render QC samples on the top layer
            p_data[st] = pd.Categorical(
                p_data[st], categories=[actl, qcl], ordered=True
            )
            p_data = p_data.sort_values(st)
            
            # Render raw scatter points
            sns.scatterplot(
                data=p_data, x=io, y=feat, hue=st, style=bt, s=40, 
                edgecolor="k", palette=self.pal, hue_order=[qcl, actl], 
                markers=self.style_map, style_order=self.all_batches, ax=ax
            )
            
            # Overlay continuous black baseline prediction
            p_info = pred.int_order_info(feat_type="IS").reset_index()
            for b in p_info[bt].unique():
                sns.lineplot(
                    data=p_info[p_info[bt] == b], x=io, y=feat, color="k", ax=ax
                )
                
            self._apply_standard_format(
                ax, xlabel=io, ylabel=feat, append_stage=False)
            
            # Management of legends in grid
            if n == len(valid) - 1:
                self._format_multi_legends(ax=ax, group_titles=[st, bt])
            elif ax.get_legend():
                ax.legend().remove()
        return fig