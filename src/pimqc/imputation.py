# src/pimqc/imputation.py
"""
Missing value imputation module with biological-aware evaluation.
"""

import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import KNNImputer
from loguru import logger

from . import core_classes
from . import io_utils as iu
from . import visualizer_classes


class MetaboIntImputer(core_classes.MetaboInt):
    """Missing value imputation engine with hybrid stratified evaluation."""

    _metadata = ["attrs"]

    def __init__(
        self,
        *args,
        pipeline_params=None,
        method="probabilistic",
        knn_neighbors=5,
        **kwargs
    ):
        """Initialize MetaboIntImputer with parameters and metadata.

        Args:
            *args: Variable length arguments passed to pandas DataFrame.
            pipeline_params: Global configuration dictionary.
            method: Default imputation algorithm to be used.
            knn_neighbors: Number of neighbors for the KNN algorithm.
            **kwargs: Keyword arguments for the DataFrame constructor.
        """
        super().__init__(*args, pipeline_params=pipeline_params, **kwargs)

        configs = {
            "method": method,
            "knn_neighbors": knn_neighbors,
        }

        if pipeline_params and "MetaboIntImputer" in pipeline_params:
            configs.update(pipeline_params["MetaboIntImputer"])

        self.attrs.update(configs)

    @property
    def _constructor(self):
        """Return the class constructor for stable subclassing."""
        return MetaboIntImputer

    def __finalize__(self, other, method=None, **kwargs):
        """Ensure custom metadata (attrs) is preserved during operations."""
        super().__finalize__(other, method=method, **kwargs)
        if hasattr(other, "attrs"):
            self.attrs = copy.deepcopy(other.attrs)
        return self

    # ====================================================================
    # Core Algorithms (Log2 Space)
    # ====================================================================

    @staticmethod
    def impute_by_constant(df_log, fraction=1.0, mode="global"):
        """Impute missing values using a defined minimum value.

        Args:
            df_log: Input log-transformed DataFrame.
            fraction: Multiplier for the minimum value. Default is 1.0.
            mode: 'global' for the overall minimum, 'column' for column min.

        Returns:
            pd.DataFrame: Imputed data.
        """
        if mode == "column":
            fill_vals = df_log.min(axis=0) * fraction
            return df_log.fillna(fill_vals)
        else:
            fill_val = df_log.min().min() * fraction
            return df_log.fillna(fill_val)

    @staticmethod
    def impute_by_knn(df_log, n_neighbors):
        """Impute missing values using K-Nearest Neighbors algorithm."""
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
        # Transpose since KNNImputer works on (samples x features)
        arr_imp = imputer.fit_transform(df_log.T).T
        return pd.DataFrame(
            arr_imp, index=df_log.index, columns=df_log.columns
        )

    @staticmethod
    def impute_by_prob(df_log):
        """Impute using a normal distribution to simulate values below LOD.
        
        This method adopts a left-shifted Gaussian distribution (Perseus style)
        without hard clipping, preserving the natural variance of the unobserved
        low-abundance tail.
        """
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
            drawn = np.random.normal(
                loc=shift_mean, 
                scale=shift_std, 
                size=s.isna().sum()
            )
            
            # [CRITICAL FIX]: Removed np.clip(a_max=lod). 
            # Forcing a clip causes generated values to pile up exactly at 
            # the minimum, deteriorating the method into a constant imputation.
            res_df.loc[s.isna(), col] = drawn
            
        return res_df

    # ====================================================================
    # Evaluation Logic (Hybrid Masking & Stratified NRMSE)
    # ====================================================================

    @staticmethod
    def generate_abundance_mask(df_log, mask_ratio, noise_factor=1.0):
        """Generate an abundance-dependent mask for MNAR simulation."""
        np.random.seed(42)
        shape = df_log.shape
        valid_mask = ~df_log.isna()
        target_nas = int(valid_mask.values.sum() * mask_ratio)
        
        if target_nas == 0:
            return pd.DataFrame(
                False, index=df_log.index, columns=df_log.columns
            )
            
        feat_meds = df_log.median(axis=1).fillna(0)
        log_meds = np.log2(feat_meds + 1.0)
        max_v = log_meds.max() if log_meds.max() > 0 else 1.0
        rel_abd = log_meds / max_v
        
        # Base probability weight is inversely proportional to abundance
        weight_mat = np.tile((1.0 - rel_abd).values[:, None], (1, shape[1]))
        final_score = weight_mat + np.random.uniform(0, noise_factor, shape)
        final_score[~valid_mask.values] = -1.0
        
        cutoff = np.sort(final_score.flatten())[-target_nas]
        mask_arr = (final_score >= cutoff) & valid_mask.values
        
        return pd.DataFrame(
            mask_arr, index=df_log.index, columns=df_log.columns
        )

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
        
        low_m, hi_m = med_all <= lod_val, med_all > lod_val
        
        def _get_nrmse(t, p):
            if len(t) < 2 or (np.max(t) - np.min(t)) < 1e-9:
                return np.nan
            rmse = np.sqrt(np.mean((t - p)**2))
            return float(rmse / (np.max(t) - np.min(t)))
            
        return {
            "NRMSE_Total": _get_nrmse(t_all, p_all),
            "NRMSE_Low": _get_nrmse(t_all[low_m], p_all[low_m]),
            "NRMSE_High": _get_nrmse(t_all[hi_m], p_all[hi_m]),
            "Count_Low": int(np.sum(low_m)),
            "Count_High": int(np.sum(hi_m))
        }

    def run_benchmark_simulation(self, target_cols, method, mask_ratio):
        """Execute a single simulation run for algorithm benchmarking."""
        k_val = self.attrs.get("knn_neighbors", 5)
        # Cast to float to prevent numpy ufunc crashes with object types
        df_log = np.log2(
            self[target_cols].astype(float).replace({0: np.nan}) + 1.0
        )
        valid_mask = ~df_log.isna()
        
        art_na = self.generate_abundance_mask(df_log, mask_ratio)
        df_mask = df_log.mask(art_na)
        
        if method == "knn":
            df_imp = self.impute_by_knn(df_mask, k_val)
        elif method == "probabilistic":
            df_imp = self.impute_by_prob(df_mask)
        elif method == "global_min":
            df_imp = self.impute_by_constant(
                df_mask, fraction=1.0, mode="global"
            )
        elif method == "column_min":
            df_imp = self.impute_by_constant(
                df_mask, fraction=1.0, mode="column"
            )
        else:
            raise ValueError(f"Unknown imputation method: {method}")
            
        metrics = self.compute_stratified_nrmse(df_log, df_imp, art_na)
        
        return metrics, df_log.values[art_na.values], df_imp.values[art_na.values]

    # ====================================================================
    # Execution & Auto-Selection
    # ====================================================================

    def select_best_algorithm(self, target_cols, mask_ratio):
        """Evaluate candidates and select the best based on NRMSE_Low."""
        cands = ["probabilistic", "knn", "global_min", "column_min"]
        best_m, best_s, cache = "probabilistic", float("inf"), {}
        
        logger.info("Running auto-selection: Benchmarking algorithms.")
        for m in cands:
            try:
                met, t, p = self.run_benchmark_simulation(
                    target_cols, m, mask_ratio
                )
                cache[m] = (met, t, p)
                
                # Output NRMSE Low, High, and Total in loguru
                txt = (
                    f" - [{m.ljust(13)}]: Low={met['NRMSE_Low']:.4f} | "
                    f"High={met['NRMSE_High']:.4f} | "
                    f"Total={met['NRMSE_Total']:.4f}"
                )
                logger.info(txt)
                
                if met["NRMSE_Low"] < best_s:
                    best_s, best_m = met["NRMSE_Low"], m
            except Exception as e:
                logger.warning(f"Algorithm {m} failed: {e}")
                
        logger.success(f"Optimal method selected: '{best_m}'")
        return best_m, cache

    @iu._exe_time
    def execute_imputation(self, method=None, output_dir=None):
        """Execute full imputation process and export visualizations."""
        method = method or self.attrs.get("method", "auto")
        mode = self.attrs.get("mode", "POS")
        k_val = self.attrs.get("knn_neighbors", 5)
        target_cols = self.columns.difference(self._blank.columns)
        
        is_auto = method.lower() == "auto"
        if is_auto:
            method, cache = self.select_best_algorithm(target_cols, 0.05)
            self.attrs["method"] = method
            eval_met, t_vals, p_vals = cache[method]
        else:
            eval_met, t_vals, p_vals = self.run_benchmark_simulation(
                target_cols, method, 0.05
            )
            cache = {method: (eval_met, t_vals, p_vals)}
            
        # Cast to float to avoid object type iteration errors
        df_log = np.log2(self.astype(float).replace({0: np.nan}) + 1.0)
        
        if method == "probabilistic":
            imp_log = self.impute_by_prob(df_log[target_cols])
        elif method == "knn":
            imp_log = self.impute_by_knn(df_log[target_cols], k_val)
        elif method == "global_min":
            imp_log = self.impute_by_constant(
                df_log[target_cols], fraction=1.0, mode="global"
            )
        elif method == "column_min":
            imp_log = self.impute_by_constant(
                df_log[target_cols], fraction=1.0, mode="column"
            )
            
        # Merge imputed actual samples with untouched blank columns
        final_log = pd.concat([
            pd.DataFrame(imp_log), pd.DataFrame(df_log[self._blank.columns])],
        axis=1)[self.columns]
        final_data = self._constructor(final_log).__finalize__(self)
        
        res_val = np.exp2(final_log) - 1.0
        imputed_obj = self._constructor(res_val).__finalize__(self)
        imputed_obj.attrs["pipeline_stage"] = "Imputation"

        # Export numerical results and visualizations
        if output_dir:
            iu._check_dir_exists(output_dir, handle="makedirs")
            imputed_obj.to_csv(
                os.path.join(output_dir, f"Imputed_Data_{method}_{mode}.csv")
            )
            vis = MetaboVisualizerImputer(self, imputed_obj)
            
            vis.save_and_close_fig(
                vis.plot_imputed_kde_overlay(),
                os.path.join(output_dir, f"Impute_KDE_{mode}.pdf")
            )
            
            if is_auto:
                vis.save_and_show_pw(
                    vis.plot_multi_nrmse_scatters(cache),
                    os.path.join(output_dir, f"Imputer_Candidates_{mode}.pdf")
                )
            
            vis.save_and_show_pw(
                vis.plot_imputation_summary_grid(
                    t_vals, p_vals, eval_met, method
                ),
                os.path.join(output_dir, f"Impute_Summary_{method}_{mode}.pdf")
            )

        return imputed_obj


class MetaboVisualizerImputer(visualizer_classes.BaseMetaboVisualizer):
    """Visualization suite for evaluating imputation accuracy."""

    def __init__(self, raw_obj, imp_obj):
        """Initialize the visualizer and cast datasets to float safely."""
        super().__init__(metabo_obj=imp_obj)
        # Casting to float directly prevents NumPy log2 ufunc type errors
        self.raw_obj = raw_obj.astype(float).replace({0: np.nan})
        self.imp_obj = imp_obj.astype(float)

    def _prepare_kde_data(self):
        """Prepare long-form DataFrame for KDE plotting."""
        dfs = []
        raw_log = np.log2(self.raw_obj + 1.0)
        imp_log = np.log2(self.imp_obj + 1.0)
        
        qc_cols = self.imp_obj._qc.columns
        exclude = qc_cols.union(self.imp_obj._blank.columns)
        sample_cols = self.imp_obj.columns.difference(exclude)
        
        for grp, cols in [("QC", qc_cols), ("Sample", sample_cols)]:
            if cols.empty:
                continue
                
            obs = raw_log[cols].values.flatten()
            obs = obs[~np.isnan(obs)]
            if len(obs) > 0:
                dfs.append(pd.DataFrame({
                    "Log2_Intensity": obs, "Group": grp, "Type": "Observed"
                }))
                
            imp = imp_log[cols].values.flatten()
            imp = imp[~np.isnan(imp)]
            if len(imp) > 0:
                dfs.append(pd.DataFrame({
                    "Log2_Intensity": imp, "Group": grp, "Type": "Imputed"
                }))
                
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    def plot_nrmse_scatter(
        self, true_vals, pred_vals, metrics, method_name="", axis_lims=None, ax=None
    ):
        """Plot hexbin scatter of true vs imputed values from mask test."""
        from . import plot_utils as pu
        
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(5, 4))
        else:
            current_ax = ax
            fig = current_ax.figure
            
        # Handle shared limits and binning extents for grid consistency
        if axis_lims is not None:
            ax_min, ax_max = axis_lims
            extent = (ax_min, ax_max, ax_min, ax_max)
            lim_min, lim_max = ax_min, ax_max
        else:
            extent = None
            lim_min = min(true_vals.min(), pred_vals.min())
            lim_max = max(true_vals.max(), pred_vals.max())

        color_map = pu.custom_linear_cmap(
            color_list=["white", "tab:red"], n_colors=256, cmin=0.1, cmax=1.0)

        hb = current_ax.hexbin(
            x=true_vals, y=pred_vals, gridsize=40, extent=extent,
            cmap=color_map,
            mincnt=1, # bins="log"
        )
        
        current_ax.plot(
            [lim_min, lim_max], [lim_min, lim_max], 
            color="tab:gray", linestyle="--", linewidth=1.0, zorder=3
        )
        
        # Format the stratified metrics dictionary into the text block
        textstr = (
            f"NRMSE_Total: {metrics['NRMSE_Total']:.4f}\n"
            f"NRMSE_Low:   {metrics['NRMSE_Low']:.4f}\n"
            f"NRMSE_High:  {metrics['NRMSE_High']:.4f}"
        )
        props = dict(
            boxstyle="round,pad=0.4", facecolor="white", 
            edgecolor="black", alpha=0.9
        )
        current_ax.text(
            0.05, 0.95, textstr, transform=current_ax.transAxes, 
            fontsize=11, verticalalignment="top", bbox=props,
            fontfamily="monospace"
        )
        
        title_str = "Masked Simulation"
        if method_name:
            title_str += f" ({method_name.title()})"
            
        self._apply_standard_format(
            ax=current_ax, title=title_str,
            xlabel="True Intensity (Log2)", ylabel="Imputed Intensity (Log2)"
        )
        
        # Enforce exact axes clipping if global limits are provided
        if axis_lims is not None:
            current_ax.set_xlim(ax_min, ax_max) # pyright: ignore[reportPossiblyUnboundVariable]
            current_ax.set_ylim(ax_min, ax_max) # pyright: ignore[reportPossiblyUnboundVariable]
        
        cb = fig.colorbar(hb, ax=current_ax)
        cb.set_label("Log10(Count)")
        
        if ax is None:
            return fig
        return current_ax

    def plot_multi_nrmse_scatters(self, results_dict):
        """Plot a 2x2 grid of NRMSE scatter plots for all candidate methods."""
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping multi grid.")
            return None
            
        # Calculate global bounds across all candidate methods with 5% padding
        g_min, g_max = float("inf"), float("-inf")
        best_method = None
        best_nrmse = float("inf")
        
        for m, (met, t, p) in results_dict.items():
            g_min = min(g_min, t.min(), p.min())
            g_max = max(g_max, t.max(), p.max())
            # Identify the optimal method based on NRMSE_Low
            if met["NRMSE_Low"] < best_nrmse:
                best_nrmse = met["NRMSE_Low"]
                best_method = m
                
        margin = (g_max - g_min) * 0.05
        shared_lims = (g_min - margin, g_max + margin)

        pw.clear()
        bricks = []
        for m, (met, t, p) in results_dict.items():
            ax = pw.Brick(figsize=(4, 4), label=f"nrmse_{m}")
            # Add an asterisk to the title of the optimal method
            display_name = f"* {m}" if m == best_method else m
            
            self.plot_nrmse_scatter(
                t, p, met, method_name=display_name, axis_lims=shared_lims, ax=ax
            )
            bricks.append(ax)
            
        # Ensure perfect 2x2 symmetry by padding empty bricks if missing
        while len(bricks) < 4:
            e = pw.Brick(figsize=(4, 4), label=f"e_{len(bricks)}")
            e.axis("off")
            bricks.append(e)
            
        return (bricks[0] | bricks[1]) / (bricks[2] | bricks[3])

    def plot_imputed_kde_overlay(self, ax_qc=None, ax_sample=None):
        """Plot KDE overlay, split into distinct QC and Sample subplots.
        
        Args:
            ax_qc: Optional matplotlib axis for the QC plot.
            ax_sample: Optional matplotlib axis for the Sample plot.
            
        Returns:
            fig if no axes are provided, else (ax_qc, ax_sample).
        """
        df_plot = self._prepare_kde_data()
        return_fig = False
        
        if df_plot.empty:
            if ax_qc is None or ax_sample is None:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "No valid data to plot.", ha="center")
                return fig
            return ax_qc, ax_sample

        # Create a 1x2 figure if no external axes are passed
        if ax_qc is None or ax_sample is None:
            fig, (ax_qc, ax_sample) = plt.subplots(1, 2, figsize=(10, 4))
            return_fig = True

        colors = {"Observed": "tab:gray", "Imputed": "tab:red"}
        
        for grp, ax in [("QC", ax_qc), ("Sample", ax_sample)]:
            subset_grp = df_plot[df_plot["Group"] == grp]
            
            if subset_grp.empty:
                ax.text(0.5, 0.5, f"No {grp} data available.", ha="center")
                self._apply_standard_format(
                    ax=ax, title=f"Density Overlay ({grp})"
                )
                continue
                
            for t in ["Observed", "Imputed"]:
                subset = subset_grp[subset_grp["Type"] == t]
                if not subset.empty:
                    sns.kdeplot(
                        data=subset, x="Log2_Intensity", 
                        color=colors.get(t, "black"), ax=ax, 
                        linewidth=2, alpha=0.8, label=t
                    )

            self._apply_standard_format(
                ax=ax, title=f"Density Overlay ({grp})",
                xlabel="Log2 Intensity", ylabel="Density"
            )
            
            # Place legend inside the best location and format it
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="best")
                self._format_single_legend(ax=ax, title="Data Type")
                
        if return_fig:
            plt.tight_layout()
            return fig # pyright: ignore[reportPossiblyUnboundVariable]
            
        return ax_qc, ax_sample

    def plot_imputation_summary_grid(self, t, p, met, method):
        """Combine NRMSE scatter and split KDE density subplots into a grid."""
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping summary grid.")
            return None
            
        pw.clear()
        
        ax1 = pw.Brick(figsize=(4, 4), label="NRMSE")
        self.plot_nrmse_scatter(t, p, met, method, ax=ax1)
        
        ax_qc = pw.Brick(figsize=(4, 4), label="KDE_QC")
        ax_sample = pw.Brick(figsize=(4, 4), label="KDE_Sample")
        
        # Populate the split KDE plots using the provided patchwork bricks
        self.plot_imputed_kde_overlay(ax_qc=ax_qc, ax_sample=ax_sample)
        
        # Assemble 1x3 horizontal layout
        return ax1 | ax_qc | ax_sample