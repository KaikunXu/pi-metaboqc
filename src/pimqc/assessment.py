"""
Purpose of script: Data quality assessment module for MetaboInt.
"""

import os
import re
import copy
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import f, chi2
from loguru import logger

from . import io_utils as iu
from . import plot_utils as pu
from . import core_classes
from . import visualizer_classes
from . import pca_utils

warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=RuntimeWarning)


class MetaboIntAssessor(core_classes.MetaboInt):
    """Data quality assessment computational class for metabolomics."""

    _metadata = ["attrs"]

    def __init__(
        self,
        *args,
        pipeline_params=None,
        mode="POS",
        corr_method="spearman",
        mask=True,
        stat_outlier="All",
        **kwargs
    ):
        """
        Initialize the data quality assessment class.

        Args:
            *args: Variable length arguments passed to pandas DataFrame.
            pipeline_params: Configuration dictionary for the pipeline.
            mode: MS Polarity ("POS" or "NEG").
            corr_method: Correlation method for QC samples.
            mask: Whether to apply an upper triangle mask to corr matrix.
            stat_outlier: Specific statistical outlier detection scope.
            **kwargs: Extra arguments passed to pandas DataFrame.
        """
        super().__init__(*args, pipeline_params=pipeline_params, **kwargs)

        qa_configs = {
            "mode": mode,
            "corr_method": corr_method,
            "mask": mask,
            "stat_outlier": stat_outlier
        }

        if pipeline_params and "MetaboIntAssessor" in pipeline_params:
            qa_configs.update(pipeline_params["MetaboIntAssessor"])

        self.attrs.update(qa_configs)

    @property
    def _constructor(self):
        """Override constructor to return MetaboIntAssessor."""
        return MetaboIntAssessor

    def __finalize__(self, other, method=None, **kwargs):
        """Explicitly preserve custom attributes during pandas operations."""
        super().__finalize__(other, method=method, **kwargs)
        if hasattr(other, "attrs"):
            self.attrs = copy.deepcopy(other.attrs)
        return self

    # =========================================================================
    # Core Statistical Calculations
    # =========================================================================

    def calc_qc_corr_matrix(self, qc_data, method="spearman"):
        """Calculate correlation matrix based on intensity of QC samples."""
        # Compute generic correlation mapping based on user preference
        return qc_data.corr(method=method)

    def calc_batch_qc_corr_matrix(self, corr_mat, qc_data, bt_col):
        """Aggregate QC correlation matrix into batch-level median matrix."""
        batches = qc_data.columns.get_level_values(bt_col)
        
        # Compress rows and columns sequentially to extract medians
        batch_corr = corr_mat.groupby(batches).median()
        batch_corr = batch_corr.transpose().groupby(batches).median()
        
        return batch_corr.transpose()

    def calc_pca_and_outliers(self, features, labels, n_components=2):
        """Execute PCA and construct formatted outlier data structures."""
        alpha = self.attrs.get("alpha", 0.05)
        od_method = self.attrs.get("od_method", "box")
        
        engine = pca_utils.PCAEngine(
            n_components=n_components, alpha=alpha, od_method=od_method
        )
        
        # Execute the unified computation engine
        res = engine.run_pca_workflow(features)
        
        multi_idx = pd.MultiIndex.from_frame(labels)
        pca_scatter = pd.DataFrame(
            res["scores"], index=multi_idx, columns=["PC1", "PC2"]
        )
        pca_var = pd.Series(
            res["variance"], index=["PC1", "PC2"], name="Variance"
        )

        metrics_df = res["metrics"]
        metrics_df.index = multi_idx
        
        # Construct the MultiIndex formatted outliers dataframe directly
        outliers = pd.DataFrame({
            ("SPE-DModX", "SPE-DModX"): metrics_df["OD"],
            ("SPE-DModX", "Outliers (SPE-DModX)"): metrics_df["is_od_outlier"],
            ("HT2", "Hotelling T2 Score"): metrics_df["SD"],
            ("HT2", "Outliers (HT2)"): metrics_df["is_sd_outlier"]
        }, index=multi_idx)

        return {
            "pca_scatter": pca_scatter,
            "pca_variance": pca_var,
            "outliers": outliers,
            "metrics_df": metrics_df,
            "sd_limit": res["sd_limit"],
            "od_limit": res["od_limit"]
        }

    # =========================================================================
    # Pipeline Execution Method
    # =========================================================================

    @iu._exe_time
    def execute_assessment(self, output_dir):
        """Execute the entire QA workflow, save tables, and render plots."""
        # 1. Configuration metadata extraction
        mode = self.attrs.get("mode", "POS")
        st_col = self.attrs.get("sample_type", "Sample Type")
        bt_col = self.attrs.get("batch", "Batch")
        io_col = self.attrs.get("inject_order", "Inject Order")
        sn_col = self.attrs.get("sample_name", "Sample Name")
        
        sample_dict = self.attrs.get("sample_dict", {})
        qc_lbl = sample_dict.get("QC sample", "QC")
        act_lbl = sample_dict.get("Actual sample", "Sample")
        
        corr_method = self.attrs.get("corr_method", "spearman")
        stat_outlier = self.attrs.get("stat_outlier", "All")
        
        pipe_params = self.attrs.get("pipeline_parameters", {})
        bound_type = pipe_params.get("MetaboInt", {}).get("boundary", "IQR")

        iu._check_dir_exists(dir_path=output_dir, handle="makedirs")

        # 2. Compute correlation matrices and data splits
        qc_data = self._qc
        act_mask = self.columns.get_level_values(st_col) == act_lbl
        act_data = self.loc[:, act_mask]
        
        corr_mat = self.calc_qc_corr_matrix(qc_data, method=corr_method)
        batch_corr = self.calc_batch_qc_corr_matrix(corr_mat, qc_data, bt_col)

        # 3. Multivariate PCA metrics via PCAEngine
        # Use explicit engine delegation for feature extraction
        features, labels = pca_utils.PCAEngine.extract_features(
            df=self, 
            st_col=st_col, 
            sn_col=sn_col, 
            act_lbl=act_lbl, 
            qc_lbl=qc_lbl
        )
        pca_res = self.calc_pca_and_outliers(features, labels)

        # 4. Serialize metrics to CSV
        scat_path = os.path.join(output_dir, f"PCA_Scatter_{mode}.csv")
        pca_res["pca_scatter"].to_csv(
            scat_path, encoding="utf-8-sig", na_rep="NA"
        )
        
        out_path = os.path.join(output_dir, f"PCA_Outliers_{mode}.csv")
        pca_res["outliers"].to_csv(
            out_path, encoding="utf-8-sig", na_rep="NA"
        )

        # 5. Initialize Visualizer and generate plots
        vis = MetaboVisualizerAssessor(self)
        batches = qc_data.columns.get_level_values(bt_col).unique()
        
        mask_flag = self.attrs.get("mask", True)
        qc_mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1) \
            if mask_flag else None
            
        # Save individual assessment plots
        vis.save_and_close_fig(
            fig=vis.plot_qc_corr_heatmap(
                corr_matrix=corr_mat, corr_mask=qc_mask, batches=batches, 
                method=corr_method),
            file_path=os.path.join(output_dir, f"QC_Corr_HM_{mode}.pdf"))

        vis.save_and_close_fig(
            fig=vis.plot_batch_corr_heatmap(
                batch_corr_matrix=batch_corr, method=corr_method),
            file_path=os.path.join(output_dir, f"Batch_Corr_HM_{mode}.pdf"))

        vis.save_and_close_fig(
            fig=vis.plot_qc_corr_trend(
                qc_data=qc_data, method=corr_method, bt_col=bt_col, 
                io_col=io_col),
            file_path=os.path.join(output_dir, f"QC_Corr_Trend_{mode}.pdf"))

        vis.save_and_close_fig(
            fig=vis.plot_pca_scatter(
                pca_df=pca_res["pca_scatter"], pca_var=pca_res["pca_variance"],
                st_col=st_col, bt_col=bt_col, qc_lbl=qc_lbl, act_lbl=act_lbl),
            file_path=os.path.join(output_dir, f"QC_AS_PCA_Scatter_{mode}.pdf"))

        vis.save_and_close_fig(
            fig=vis.plot_sd_od_scatter(
                metrics_df=pca_res["metrics_df"], 
                sd_limit=pca_res["sd_limit"], od_limit=pca_res["od_limit"]),
            file_path=os.path.join(output_dir, f"Outlier_Scatter_{mode}.pdf"))

        # Optional statistical outlier barplot
        fig_stat = vis.plot_stat_outliers_bar(
            outliers_df=pca_res["outliers"], st_col=st_col, bt_col=bt_col,
            sn_col=sn_col, act_lbl=act_lbl, target_param=stat_outlier)
        if fig_stat:
            vis.save_and_close_fig(
                fig=fig_stat, 
                file_path=os.path.join(
                    output_dir, f"Statistical_Outliers_Barplot_{mode}.pdf"))

        # QC & Actual Sample RSD distribution
        vis.save_and_close_fig(
            fig=vis.plot_rsd_bar(
                qc_data=qc_data, act_data=act_data, qc_lbl=qc_lbl, 
                act_lbl=act_lbl),
            file_path=os.path.join(output_dir, f"RSD_Barplot_{mode}.pdf"))

        # Internal Standards control chart
        if len(self.valid_is) > 0:
            is_data = self.int_order_info(feat_type="IS")
            vis.save_and_close_fig(
                fig=vis.plot_is_shewhart_chart(
                    is_data=is_data, valid_is=self.valid_is, st_col=st_col, 
                    bt_col=bt_col, io_col=io_col, qc_lbl=qc_lbl, act_lbl=act_lbl,
                    bound_type=bound_type),
                file_path=os.path.join(output_dir, f"Shewhart_IS_{mode}.pdf"))

        # 6. Generate Master Summary Grid via patchworklib
        fig_summary = vis.plot_assessor_summary_grid(
            pca_res=pca_res, qc_data=qc_data, act_data=act_data, 
            batch_corr=batch_corr, corr_mat=corr_mat, qc_mask=qc_mask, 
            batches=batches, method=corr_method, st_col=st_col, bt_col=bt_col,
            qc_lbl=qc_lbl, act_lbl=act_lbl)
        
        grid_path = os.path.join(output_dir, f"Assessor_Grid_{mode}.pdf")
        vis.save_and_show_pw(pw_obj=fig_summary, file_path=grid_path)   
        
        logger.info(f"Assessor summary grid saved as: {grid_path}")
        logger.success("Data quality assessment completed.")

class MetaboVisualizerAssessor(visualizer_classes.BaseMetaboVisualizer):
    """Visualization suite for metabolomics data quality assessment."""

    def __init__(self, qa_obj):
        """Initialize with a computed MetaboIntAssessor object."""
        super().__init__(metabo_obj=qa_obj)

    # =========================================================================
    # Matrix Heatmaps and Systemic Trends Plots
    # =========================================================================

    def _format_heatmap_ticks(self, hm, tick_color_dict):
        """Format labels and assign specific colors for heatmap ticks."""
        def rename_tick(label_text):
            parts = re.split("-", label_text)
            if len(parts) > 4:
                return "-".join([parts[0]] + parts[4:])
            return label_text

        # Update tick label text to a shortened format for readability
        hm.set_xticklabels([rename_tick(e._text) for e in hm.get_xticklabels()])
        hm.set_yticklabels([rename_tick(e._text) for e in hm.get_yticklabels()])

        # Apply specific batch colors to the tick labels for group separation
        for ax_labels in (hm.get_xticklabels(), hm.get_yticklabels()):
            for label in ax_labels:
                for batch, color in tick_color_dict.items():
                    if re.match(pattern=f"^{batch}", string=label._text):
                        label.set_color(color)
                        break

    def plot_qc_corr_heatmap(
        self, corr_matrix, corr_mask, batches, method="spearman", 
        vmin=0.85, vmax=1.0, ax=None
    ):
        """Plot sample-level correlation matrix heatmap of Pooled QCs."""
        n_samples = corr_matrix.shape[0]
        custom_cmap = pu.custom_linear_cmap(["white", "tab:red"], 100)
        tick_colors = pu.extract_linear_cmap(
            cmap=custom_cmap, cmin=0.5, cmax=1.0, n_colors=len(batches)
        )
        tick_color_dict = dict(zip(batches, tick_colors))
        color_map = pu.extract_linear_cmap(custom_cmap, cmin=0.2, cmax=1.0)
        
        if ax is None:
            fig, current_ax = plt.subplots(
                figsize=(n_samples * 0.2 + 1.2, n_samples * 0.2 + 1.2)
            )
        else:
            current_ax = ax
            fig = current_ax.figure
            
        # Create an inset axes for colorbar as a child of current_ax
        # This is the key for patchworklib compatibility
        cbar_ax = current_ax.inset_axes([1.05, 0.1, 0.05, 0.8])
        
        with sns.axes_style("white"):
            hm = sns.heatmap(
                corr_matrix, mask=corr_mask, xticklabels=1, yticklabels=1, 
                vmin=vmin if vmin else corr_matrix.min().min(),
                vmax=vmax, cmap=color_map, annot=False,
                linewidths=0.25, linecolor="white", square=True,
                ax=current_ax, cbar_ax=cbar_ax,
                cbar_kws={
                    "label": f"{method.title()} Correlation",
                    "format": "%.2f"})
            
        self._apply_standard_format(
            ax=current_ax, title="Pooled QCs Correlation", xlabel="Pooled QCs", 
            ylabel="Pooled QCs", title_fontsize=14, label_fontsize=10, 
            tick_fontsize=4
        )
        self._format_heatmap_ticks(hm=hm, tick_color_dict=tick_color_dict)
        return fig

    def plot_batch_corr_heatmap(
        self, batch_corr_matrix, method, vmin=0.85, vmax=1.0, ax=None
    ):
        """Plot inter-batch QC correlation heatmap using median aggregation."""
        n_batches = batch_corr_matrix.shape[0]
        
        if ax is None:
            fig, current_ax = plt.subplots(
                figsize=(n_batches * 0.8 + 2.5, n_batches * 0.8 + 1.5)
            )
        else:
            current_ax = ax
            fig = current_ax.figure
            
        mask = np.triu(np.ones_like(batch_corr_matrix, dtype=bool), k=1)
        custom_cmap = pu.custom_linear_cmap(["white", "tab:red"], 100)
        color_map = pu.extract_linear_cmap(custom_cmap, cmin=0.2, cmax=1.0)
        
        # Create an inset axes for colorbar as a child of current_ax
        cbar_ax = current_ax.inset_axes([1.05, 0.1, 0.05, 0.8])
        
        with sns.axes_style("white"):
            sns.heatmap(
                batch_corr_matrix, mask=mask, annot=True, fmt=".3f", 
                vmin=vmin if vmin else batch_corr_matrix.min().min(), 
                vmax=vmax,
                cmap=color_map, linewidths=0.25, linecolor="white", 
                square=True, ax=current_ax, cbar_ax=cbar_ax,
                cbar_kws={
                    "label": f"{method.title()} Correlation",
                    "format": "%.2f"})

        self._apply_standard_format(
            ax=current_ax, title="Inter-Batch Pooled QC Correlation",
            xlabel="Batch ID", ylabel="Batch ID",
            title_fontsize=14, label_fontsize=12, tick_fontsize=10
        )
        return fig

    def plot_qc_corr_trend(self, qc_data, method, bt_col, io_col, ax=None):
        """Plot correlation of each QC to the global median QC profile."""
        global_median = qc_data.median(axis=1)
        corr_series = qc_data.corrwith(other=global_median, method=method)
        corr_series.name = "Correlation"
        
        plot_df = corr_series.reset_index()
        plot_df[bt_col] = plot_df[bt_col].astype("category")
        
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(7, 3))
        else:
            current_ax = ax
            fig = current_ax.figure
        
        unique_batches = plot_df[bt_col].unique()
        batch_colors = pu.extract_qual_cmap(pu.get_cmap("Set1"))
        batch_pal = dict(zip(unique_batches, batch_colors))
        
        sns.scatterplot(
            data=plot_df, x=io_col, y="Correlation", hue=bt_col, 
            style=bt_col, markers=self.style_map, palette=batch_pal, 
            s=60, edgecolor="k", linewidth=0.5, ax=current_ax, zorder=3
        )
        
        current_ax.axhline(
            0.90, color="k", linestyle="--", linewidth=1.5, zorder=1)
        
        self._apply_standard_format(
            ax=current_ax, title=f"QC vs Median Profile ({method.title()})",
            xlabel=io_col, ylabel="Correlation Coefficient"
        )
        
        if ax is None:
            self._format_single_legend(ax=current_ax, title="Batch")
            # plt.tight_layout()
        else:
            current_ax.legend(
                loc="lower right", **self.LEGEND_KWARGS
            )
            
        return fig

    # =========================================================================
    # Dimensionality Reduction and Outlier Plots
    # =========================================================================

    def plot_pca_scatter(
        self, pca_df, pca_var, st_col, bt_col, qc_lbl, act_lbl,
        x_pc="PC1", y_pc="PC2", draw_ce=True, ax=None
    ):
        """Plot PCA scatter plot with confidence ellipses and QA metrics."""
        # 1. Prepare data following original visual style
        plot_df = pca_df.reset_index().copy()
        plot_df[st_col] = plot_df[st_col].astype("category")
        plot_df = plot_df.sort_values(by=st_col, ascending=False)
        pal_dict = {qc_lbl: "tab:red", act_lbl: "tab:gray"}
        
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(4, 4))
        else:
            current_ax = ax
            fig = current_ax.figure
            
        sns.despine(ax=current_ax)

        # 2. Extract numpy arrays for metric calculations (Safe extraction)
        coords = pca_df[[x_pc, y_pc]].values
        
        def _get_array(col_name):
            if col_name in pca_df.columns:
                return pca_df[col_name].values
            return pca_df.index.get_level_values(col_name).values

        types = _get_array(st_col)
        batches = _get_array(bt_col)

        # 3. Calculate QA metrics using PCAEngine static methods
        rd_score = pca_utils.PCAEngine.calc_relative_dispersion(
            coords=coords, types=types, qc_lbl=qc_lbl, act_lbl=act_lbl
        )
        sil_score = pca_utils.PCAEngine.calc_qc_batch_silhouette(
            coords=coords, types=types, batches=batches, qc_lbl=qc_lbl
        )
        shift_res = pca_utils.PCAEngine.calc_qc_centrality_shift(
            coords=coords, types=types, qc_lbl=qc_lbl, act_lbl=act_lbl
        )
        rel_shift = shift_res["rel_shift"]

        # 4. Format annotation text
        rd_str = f"{rd_score:.4f}" if not pd.isna(rd_score) else "N/A"
        sil_str = f"{sil_score:.4f}" if not pd.isna(sil_score) else "N/A"
        shift_str = f"{rel_shift:.4f}" if not pd.isna(rel_shift) else "N/A"
        
        annot_text = (
            f"Relative Dispersion: {rd_str}\n"
            f"Batch Silhouette: {sil_str}\n"
            f"Centrality Shift: {shift_str}"
        )

        # 5. Render scatter plot following original aesthetic
        sns.scatterplot(
            data=plot_df, x=x_pc, y=y_pc, hue=st_col, style=bt_col,
            s=50, edgecolor="k", palette=pal_dict, linewidth=0.5,
            ax=current_ax, hue_order=[qc_lbl, act_lbl], 
            style_order=self.all_batches, markers=self.style_map
        )
        
        # Draw confidence ellipses if requested
        if draw_ce:
            for group in (qc_lbl, act_lbl):
                sub_df = plot_df[plot_df[st_col] == group]
                if not sub_df.empty:
                    pu.confidence_ellipse(
                        x=sub_df[x_pc], y=sub_df[y_pc], ax=current_ax, 
                        n_std=3.0, alpha=0.1, facecolor=pal_dict[group], 
                        edgecolor=pal_dict[group]
                    )

        # Add metric annotation box at the bottom right
        current_ax.text(
            0.96, 0.02, annot_text, transform=current_ax.transAxes,
            fontsize=9, verticalalignment="bottom", 
            horizontalalignment="right",
            clip_on=False,
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor="white", alpha=0.6
            )
        )
                
        # 6. Apply standard formatting and axis labels
        var_x = pca_var.loc[x_pc] * 100
        var_y = pca_var.loc[y_pc] * 100
        self._apply_standard_format(
            ax=current_ax, xlabel=f"{x_pc} ({var_x:.1f}%)", 
            ylabel=f"{y_pc} ({var_y:.1f}%)",
            title="Pooled QC & Sample PCA Scatter"
        )
        
        self._format_multi_legends(
            ax=current_ax, group_titles=[st_col, bt_col]
        )
        
        current_ax.autoscale()
        return fig

    def plot_sd_od_scatter(self, metrics_df, sd_limit, od_limit, ax=None):
        """Plot SD-OD diagnostic scatter with customized outlier styling."""
        import matplotlib.colors as mcolors
        red_solid = "tab:red"
        red_alpha = mcolors.to_rgba("tab:red", alpha=0.5)
        
        custom_pal = {
            "Normal": "tab:gray", "Strong Outlier": red_alpha,
            "Orthogonal Outlier": red_alpha, "Extreme Outlier": red_solid
        }
        custom_markers = {
            "Normal": "o", "Strong Outlier": "^", 
            "Orthogonal Outlier": "s", "Extreme Outlier": "X"
        }
        
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(4, 4))
        else:
            current_ax = ax
            fig = current_ax.figure
            
        sns.scatterplot(
            data=metrics_df, x="SD", y="OD", hue="Category", 
            style="Category", palette=custom_pal, markers=custom_markers,
            s=50, edgecolor="k", linewidth=0.5, ax=current_ax
        )
        
        current_ax.axvline(
            x=sd_limit, color="k", linestyle="--", alpha=0.6,
            zorder=0, label=f"SD Limit ({sd_limit:.2f})"
        )
        current_ax.axhline(
            y=od_limit, color="k", linestyle="--", alpha=0.6,
            zorder=0, label=f"OD Limit ({od_limit:.2f})"
        )
        
        self._apply_standard_format(
            ax=current_ax, xlabel="Score Distance (Hotelling's $T^2$)", 
            ylabel="Orthogonal Distance (SPE / DModX)",
            title="PCA Outlier Diagnosis"
        )
        
        if current_ax.get_legend():
            current_ax.get_legend().remove()
            
        handles, labels = current_ax.get_legend_handles_labels()
        sc_h, sc_l, ln_h, ln_l = [], [], [], []
        
        for h, l in zip(handles, labels):
            if "Limit" in l:
                ln_h.append(h)
                ln_l.append(l)
            elif l != "Category":
                sc_h.append(h)
                sc_l.append(l)
                
        leg_m = current_ax.legend(
            sc_h, sc_l, title="Category", loc="lower left", 
            bbox_to_anchor=(1.05, 0.5), **self.LEGEND_KWARGS
        )
        current_ax.add_artist(leg_m)
        current_ax.legend(
            ln_h, ln_l, title="Thresholds", loc="upper left", 
            bbox_to_anchor=(1.05, 0.45), **self.LEGEND_KWARGS
        )
        
        current_ax.autoscale()
        return fig

    def plot_stat_outliers_bar(
        self, outliers_df, st_col, bt_col, sn_col, act_lbl, target_param
    ):
        """Plot outlier detection results via bar charts."""
        mask = outliers_df.index.get_level_values(st_col) == act_lbl
        out_df = outliers_df[mask].copy()
        
        idx_df = out_df.index.to_frame()
        new_idx = idx_df[bt_col] + "-" + idx_df[sn_col]
        out_df.index = new_idx
        out_df = out_df.rename_axis(index=["Sample ID"])
        
        # =====================================================================
        # Filter to retain only samples flagged as 'Filtered' (True) 
        # in at least one of the two metrics. This prevents the barplot 
        # from becoming unreadable due to excessive normal samples.
        # =====================================================================
        spe_flags = out_df[("SPE-DModX", "Outliers (SPE-DModX)")]
        ht2_flags = out_df[("HT2", "Outliers (HT2)")]
        out_df = out_df[spe_flags | ht2_flags].copy()
        
        # Graceful exit if no statistical outliers are detected at all
        if out_df.empty:
            from loguru import logger
            logger.info("No outliers detected. Skipping barplot generation.")
            return None
        # =====================================================================
        
        target_col = ("Statistics Outliers", f"Outliers ({target_param})")
        if target_col in out_df.columns:
            out_samps = out_df[out_df[target_col]].index.tolist()
        else:
            out_samps = []

        # Adjusted width multiplier (0.3) for better spacing of fewer bars
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, ncols=1, figsize=(out_df.shape[0] * 0.3 + 2, 7), 
            sharex=True
        )
        
        str_label_map = {"True": "Filtered", "False": "Retained"}
        pal = {"Filtered": "tab:red", "Retained": "tab:gray"}
        
        spe_df = out_df["SPE-DModX"].reset_index()
        spe_df["Status"] = spe_df["Outliers (SPE-DModX)"].astype(str).map(
            str_label_map
        )
        sns.barplot(
            ax=ax1, data=spe_df, x="Sample ID", y="SPE-DModX", 
            edgecolor="k", linewidth=0.25,
            hue="Status", palette=pal, hue_order=["Retained", "Filtered"]
        )
        
        ht2_df = out_df["HT2"].reset_index()
        ht2_df["Status"] = ht2_df["Outliers (HT2)"].astype(str).map(
            str_label_map
        )
        sns.barplot(
            ax=ax2, data=ht2_df, x="Sample ID", y="Hotelling T2 Score", 
            edgecolor="k", linewidth=0.25, 
            hue="Status", palette=pal, hue_order=["Retained", "Filtered"]
        )
        
        pu.change_axis_rotation(ax=ax2, rotation=90, axis="x")
        
        for ax in (ax1, ax2):
            self._apply_standard_format(
                # Slightly increased tick_fontsize since there are fewer bars
                ax=ax, title_fontsize=12, label_fontsize=12, tick_fontsize=8
            )
            self._format_single_legend(
                ax=ax, loc="upper right", bbox_to_anchor=None, ncol=2
            )
            
        for xlabel in ax2.get_xticklabels():
            if xlabel.get_text() in out_samps:
                xlabel.set_color("tab:red")
                
        # plt.tight_layout()
        return fig

    # =========================================================================
    # Intra-Run Stability Validation Plots
    # =========================================================================

    def plot_rsd_bar(self, qc_data, act_data, qc_lbl, act_lbl, ax=None):
        """Plot RSD distribution for QC and Actual samples with dual styles."""
        def _get_rsd_counts(data, label):
            """Calculate feature counts across defined RSD bins."""
            rsd = (data.std(axis=1, ddof=1) / data.mean(axis=1)).rename("RSD")
            bins = [-np.inf, 0.1, 0.2, 0.3, np.inf]
            labels = ["0-10%", "10-20%", "20-30%", ">30%"]
            rsd_range = pd.cut(x=rsd, bins=bins, labels=labels, right=False)
            
            df_counts = rsd_range.value_counts().rename("Counts").reset_index()
            df_counts["Type"] = label
            return df_counts

        qc_df = _get_rsd_counts(qc_data, qc_lbl)
        act_df = _get_rsd_counts(act_data, act_lbl)
        plot_df = pd.concat([qc_df, act_df], ignore_index=True)

        if ax is None:
            fig, current_ax = plt.subplots(figsize=(5.5, 4))
        else:
            current_ax = ax
            fig = current_ax.figure

        # Initialize barplot layout
        sns.barplot(
            data=plot_df, x="RSD", y="Counts", hue="Type", 
            ax=current_ax, hue_order=[qc_lbl, act_lbl]
        )

        import matplotlib.colors as mcolors

        # Apply aesthetics: Use RGBA to ensure stability across PDF backends
        for i, container in enumerate(current_ax.containers):
            # QC (i=0): Solid-like | Actual (i=1): Ghost-like, dashed
            l_style = "-" if i == 0 else "--"
            l_width = 1.0
            alpha_val = 1.0 if i == 0 else 0.4

            for j, bar in enumerate(container):
                base_color = "tab:red" if j == 3 else "tab:gray"
                rgba_color = mcolors.to_rgba(base_color, alpha=alpha_val)
                
                bar.set_facecolor(rgba_color)
                bar.set_edgecolor("black")
                bar.set_linestyle(l_style)
                bar.set_linewidth(l_width)

        # [CRITICAL]: Physically remove zero-height bars to kill ghost labels
        for p in list(current_ax.patches):
            if p.get_height() <= 0:
                p.remove()

        # Update axis limit and annotate AFTER removing empty patches
        max_c = plot_df["Counts"].max()
        current_ax.set_ylim(0, max_c * 1.3)
        pu.show_values_on_bars(
            current_ax, show_percentage=True, position="outside",
            value_format="{:.0f}",pct_type="group",fontsize=8
        )

        # Manually construct legend to ensure correct style mapping
        from matplotlib.patches import Patch
        h_type = [
            Patch(
                facecolor=mcolors.to_rgba("tab:gray", alpha=0.9), 
                edgecolor="black", linestyle="-", linewidth=1.0, 
                label=qc_lbl),
            Patch(
                facecolor=mcolors.to_rgba("tab:gray", alpha=0.4), 
                edgecolor="black", linestyle="--", linewidth=1.0, 
                label=act_lbl)
        ]
        
        # [FIX]: Directly apply custom legend and bypass auto-formatters
        # Use existing global LEGEND_KWARGS for visual consistency
        leg_kwargs = self.LEGEND_KWARGS.copy()
        leg_kwargs.update({"title": "Sample Type", "loc": "best"})
        current_ax.legend(handles=h_type, **leg_kwargs)
        
        self._apply_standard_format(
            ax=current_ax, title="Feature RSD Distribution",
            xlabel="Relative Standard Deviation (RSD) Bin", 
            ylabel="Feature Count"
        )
        
        return fig

    def plot_is_shewhart_chart(
        self, is_data, valid_is, st_col, bt_col, io_col, qc_lbl, act_lbl,
        bound_type
    ):
        """Plot Shewhart control charts for internal standards."""
        plot_df = is_data.reset_index().copy()
        
        plot_df[st_col] = plot_df[st_col].astype("category")
        plot_df[bt_col] = plot_df[bt_col].astype("category")
        plot_df = plot_df.sort_values(by=st_col, ascending=False)
        
        ncols = 2
        nrows = int(np.ceil(len(valid_is) / ncols))
        fig = plt.figure(
            figsize=(7.5 * ncols, 3 * nrows), layout="constrained"
        )
        
        from .core_classes import MetaboInt
        dummy_calc = MetaboInt()
        
        for n, feat in enumerate(valid_is):
            ax = plt.subplot(nrows, ncols, n + 1)
            sns.scatterplot(
                ax=ax, data=plot_df, x=io_col, y=feat, s=40, edgecolor="k", 
                linewidth=0.5, style=bt_col, palette=self.pal,
                hue=st_col, hue_order=[qc_lbl, act_lbl],
                markers=self.style_map
            )

            int_data = is_data[feat].values
            solid, lower, upper = dummy_calc.calculate_boundaries(
                x=int_data, boundary_type=bound_type
            )
            
            ax.axhline(y=solid, color="k", linestyle="-", linewidth=1.5)
            ax.axhline(y=lower, color="k", linestyle="--", linewidth=1.5)
            ax.axhline(y=upper, color="k", linestyle="--", linewidth=1.5)
            
            self._apply_standard_format(
                ax=ax, xlabel=io_col, ylabel=feat, tick_fontsize=11,
                title_fontsize=13
            )
            pu.change_axis_format(
                ax=ax, axis_format="scientific notation", axis="y"
            )

            if n == len(valid_is) - 1:
                self._format_multi_legends(ax=ax, group_titles=[st_col, bt_col])
            elif ax.get_legend():
                ax.legend().remove()
                        
        plt.suptitle(
            t="Shewhart Control Chart of IS", fontsize=14, weight="bold"
        )
        return fig

    # =========================================================================
    # Patchworklib Multi-panel Summary Plots
    # =========================================================================

    def plot_assessor_summary_grid(
        self, pca_res, qc_data, act_data, batch_corr, corr_mat, qc_mask, batches,
        method, st_col, bt_col, qc_lbl, act_lbl
    ):
        """Plot a 2x2 grid summary of key QA metrics using patchworklib."""
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping summary grid.")
            return None
        
        # [CRITICAL]: Clear global patchworklib state to prevent data residue
        pw.clear()
        
        # [0, 0]: QC & Actual RSD Barplot
        ax1 = pw.Brick(figsize=(4, 4))
        self.plot_rsd_bar(
            qc_data=qc_data, act_data=act_data, qc_lbl=qc_lbl, act_lbl=act_lbl, 
            ax=ax1
        )
        
        # [0, 1]: PCA Scatter
        ax2 = pw.Brick(figsize=(4, 4))
        self.plot_pca_scatter(
            pca_df=pca_res["pca_scatter"], pca_var=pca_res["pca_variance"],
            st_col=st_col, bt_col=bt_col, qc_lbl=qc_lbl, act_lbl=act_lbl,
            ax=ax2
        )

        # [1, 0]: Correlation Heatmap (Dynamic Fallback Logic)
        n_batches = batch_corr.shape[0]
        if n_batches == 1:
            n_samples = corr_mat.shape[0]
            ax3 = pw.Brick(
                figsize=(n_samples * 0.18 + 1.0, n_samples * 0.18 + 1.0)
            )
            self.plot_qc_corr_heatmap(
                corr_matrix=corr_mat, corr_mask=qc_mask, batches=batches,
                method=method, ax=ax3
            )
        else:
            ax3 = pw.Brick(
                figsize=(n_batches * 0.8 + 2.5, n_batches * 0.8 + 1.5)
            )
            self.plot_batch_corr_heatmap(
                batch_corr_matrix=batch_corr, method=method, ax=ax3
            )
        
        # [1, 1]: Outlier Scatter
        ax4 = pw.Brick(figsize=(4, 4))
        self.plot_sd_od_scatter(
            metrics_df=pca_res["metrics_df"], 
            sd_limit=pca_res["sd_limit"], od_limit=pca_res["od_limit"],
            ax=ax4
        )

        # Compose the 2x2 layout
        summary_grid = (ax1 | ax2) / (ax3 | ax4)
        return summary_grid