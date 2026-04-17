# src/pimqc/filtering.py
"""
Purpose of script: Invalid feature/sample filtering and visualization.
"""

import os
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from loguru import logger

from . import io_utils as iu
from . import plot_utils as pu
from . import core_classes
from . import visualizer_classes


class MetaboIntFilter(core_classes.MetaboInt):
    """Filtering engine subclassing MetaboInt for metabolomics datasets."""

    _metadata = ["stats", "params"]
    
    _INVALID_STRS = {
        "unknown", "na", "n/a", "nan", "none", "null", "",
        "unassigned", "blank", "blk", "is", "solvent", "wash",
        "sst", "pool", "invalid", "unvalid"
    }

    def __init__(self, data=None, pipeline_params=None, *args, **kwargs):
        """Initialize the filtering engine.

        Args:
            data: Input intensity data.
            pipeline_params: Global configuration dictionary.
            *args: Arguments passed to pandas DataFrame.
            **kwargs: Extra arguments passed to pandas DataFrame.
        """
        super().__init__(data=data, *args, **kwargs)

        if pipeline_params is not None:
            self.attrs["pipeline_parameters"] = pipeline_params

        self.params = self.attrs.get("pipeline_parameters", {})

        if "MetaboIntFilter" not in self.params:
            self.params["MetaboIntFilter"] = {}
        if "feature_counts" not in self.params["MetaboIntFilter"]:
            self.params["MetaboIntFilter"]["feature_counts"] = {}

        self.stats = {
            "mv_group_df": pd.DataFrame(),
            "mv_qc_series": pd.Series(dtype=float),
            "mv_global_series": pd.Series(dtype=float),
            "blank_mean": pd.Series(dtype=float),
            "qc_mean": pd.Series(dtype=float),
            "qc_rsd_all": pd.Series(dtype=float),
            "idx_mar": pd.Index([]),
            "idx_mnar": pd.Index([]),
            "idx_mnar_group": pd.Index([]),
            "idx_mnar_qc": pd.Index([])
        }

    @property
    def _constructor(self):
        """Override constructor to return MetaboIntFilter."""
        return MetaboIntFilter

    def __finalize__(self, other, method=None, **kwargs):
        """Deepcopy custom attributes during pandas operations."""
        self = super().__finalize__(other, method=method, **kwargs)
        for name in self._metadata:
            if hasattr(other, name):
                setattr(self, name, copy.deepcopy(getattr(other, name)))
        return self

    # =========================================================================
    # Internal Utility Methods
    # =========================================================================

    def _get_valid_bio_groups(self):
        """Extract valid biological group names from the column index."""
        meta_params = self.params.get("MetaboInt", {})
        grp_col = meta_params.get("bio_group", "Bio Group")
        sample_dict = meta_params.get("sample_dict", {})
        qc_lbl = sample_dict.get("QC sample", "QC")

        valid_bio_groups = []
        if grp_col in self.columns.names:
            raw_groups = self.columns.get_level_values(grp_col).unique()
            for g in raw_groups:
                if pd.isna(g):
                    continue
                g_str = str(g).strip().lower()
                if g_str in self._INVALID_STRS or g_str == str(qc_lbl).lower():
                    continue
                valid_bio_groups.append(g)

        return valid_bio_groups

    # =========================================================================
    # Filtering Execution Stream
    # =========================================================================

    @iu._exe_time
    def execute_mv_filtering(self, output_dir=None):
        """Execute Stage-1 missing value filter with multi-level plotting."""
        fc = self.params["MetaboIntFilter"]["feature_counts"]
        if "raw" not in fc:
            fc["raw"] = self.shape[0]

        meta_params = self.params.get("MetaboInt", {})
        
        grp_col = meta_params.get("bio_group", "Bio Group")
        group_order = meta_params.get("group_order", None)
        qc_col = meta_params.get("sample_type", "Sample Type")
        
        sample_dict = meta_params.get("sample_dict", {})
        qc_lbl = sample_dict.get("QC sample", "QC")
        fltr_params = self.params.get("MetaboIntFilter", {})

        valid_bio_groups = self._get_valid_bio_groups()

        if qc_col in self.columns.names:
            raw_types = self.columns.get_level_values(qc_col)
            valid_mask = np.array([
                str(t).strip().lower() not in self._INVALID_STRS 
                for t in raw_types
            ])
        else:
            valid_mask = np.ones(self.shape[1], dtype=bool)

        qc_mask = (
            self.columns.get_level_values(qc_col) == qc_lbl 
            if qc_col in self.columns.names 
            else np.zeros(self.shape[1], dtype=bool)
        )

        if qc_mask.any():
            df_qc = self.loc[:, qc_mask]
            self.stats["mv_qc_series"] = df_qc.isna().mean(axis=1)
            
        if valid_mask.any():
            df_valid = self.loc[:, valid_mask]
            self.stats["mv_global_series"] = df_valid.isna().mean(axis=1)
        else:
            self.stats["mv_global_series"] = pd.Series(1.0, index=self.index)

        filter_level = "Group"

        if valid_bio_groups:
            logger.info(
                "High missing value features filter (Stage1): "
                "biological group level")
            mv_tol = fltr_params.get("mv_group_tol", 0.5)
            na_rate = self.isna().T.groupby(level=grp_col).mean().T
            self.stats["mv_group_df"] = na_rate[valid_bio_groups]
            pass_mask = (na_rate[valid_bio_groups] <= mv_tol).any(axis=1)
            
        elif qc_mask.any():
            logger.info(
                "High missing value features filter (Stage1): "
                "pooled QC level")
            filter_level = "QC"
            mv_tol = fltr_params.get("mv_qc_tol", 0.8)
            pass_mask = self.stats["mv_qc_series"] <= mv_tol
            
        else:
            logger.info(
                "High missing value features filter (Stage1): "
                "global sample level")
            filter_level = "Global"
            mv_tol = fltr_params.get("mv_global_tol", 0.7)
            pass_mask = self.stats["mv_global_series"] <= mv_tol

        self.classify_missing_types()
        fc["post_stage1"] = pass_mask.sum()
        df_final = self.loc[self.index[pass_mask]].copy()

        if output_dir:
            iu._check_dir_exists(dir_path=output_dir, handle="makedirs")
            mode = meta_params.get("mode", "POS")
            # ==========================================================
            df_final.attrs["pipeline_stage"] = "High-MV features filter"
            # ==========================================================
            csv_path = os.path.join(
                output_dir, f"High-MV_Filter_{filter_level}_{mode}.csv")
            df_final.to_csv(csv_path, encoding="utf-8-sig", na_rep="NA")
            logger.info(f"Features after High-MV check: {df_final.shape[0]}")
            logger.info(
                "Data after high-missing features filtering saved as: "
                f"{csv_path}")
            
            vis = MetaboVisualizerFilter(engine=df_final)
            
            if not self.stats["idx_mar"].empty or not self.stats[
                "idx_mnar"].empty:
                fig_mnar = vis.plot_missing_classification()
                if fig_mnar:
                    pdf_mnar_path = os.path.join(
                        output_dir,
                        f"High-MV_Filter_MNAR_Classification_{mode}.pdf")
                    vis.save_and_close_fig(fig_mnar, pdf_mnar_path)

            try:
                combined_report = vis.plot_mv_filtering_summary_grid(
                    filter_level=filter_level,
                    group_order=group_order
                )
                if combined_report:
                    report_path = os.path.join(
                        output_dir, f"MV_FLTR_Grid_{mode}.pdf"
                    )
                    vis.save_and_show_pw(
                        pw_obj=combined_report, file_path=report_path)
                    logger.info(
                        f"High-MV features filter summary grid saved as: {report_path}")
            except Exception as e:
                logger.error(
                    f"High-MV features filter summary grids generation failed: {e}")


            plot_tasks = []
            if valid_bio_groups:
                g_tol = fltr_params.get("mv_group_tol", 0.5)
                fig_g = vis.plot_mv_group(
                    mv_df=self.stats["mv_group_df"], tol=g_tol,
                    group_order=group_order)
                plot_tasks.append((fig_g, "Group"))
                
            if qc_mask.any():
                q_tol = fltr_params.get("mv_qc_tol", 0.8)
                fig_q = vis.plot_mv_downgrade(
                    mv_series=self.stats["mv_qc_series"], 
                    level="QC", tol=q_tol)
                plot_tasks.append((fig_q, "QC"))
                
            gl_tol = fltr_params.get("mv_global_tol", 0.7)
            fig_gl = vis.plot_mv_downgrade(
                mv_series=self.stats["mv_global_series"], 
                level="Global", tol=gl_tol)
            plot_tasks.append((fig_gl, "Global"))
            
            for fig, level_name in plot_tasks:
                if fig:
                    pdf_path = os.path.join(
                        output_dir, f"Filter_Stage1_{level_name}_{mode}.pdf")
                    vis.save_and_close_fig(fig, pdf_path)
        
        logger.success("High-MV features filtering completed.")

        return df_final

    @iu._exe_time
    def classify_missing_types(self):
        """Classify features into MAR and MNAR using a hierarchical strategy."""
        meta_params = self.params.get("MetaboInt", {})
        grp_col = meta_params.get("bio_group", "Bio Group")
        qc_col = meta_params.get("sample_type", "Sample Type")
        
        sample_dict = meta_params.get("sample_dict", {})
        qc_lbl = sample_dict.get("QC sample", "QC")
        
        fltr_params = self.params.get("MetaboIntFilter", {})
        grp_mv_tol = fltr_params.get("mnar_group_mv_tol", 0.8)
        qc_mv_tol = fltr_params.get("mnar_qc_mv_tol", 0.0)
        qc_int_pct = fltr_params.get("mnar_intensity_percentile", 0.1)

        idx_mnar_group = pd.Index([])
        idx_mnar_qc = pd.Index([])
        current_idx = self.index

        valid_bio_groups = self._get_valid_bio_groups()

        if valid_bio_groups:
            na_rate_grp = self.isna().T.groupby(level=grp_col).mean().T
            na_rate_valid = na_rate_grp[valid_bio_groups]
            
            cond_grp_mnar = (na_rate_valid >= grp_mv_tol).any(axis=1)
            idx_mnar_group = current_idx[cond_grp_mnar]
            current_idx = current_idx.difference(idx_mnar_group)

        qc_mask = (
            self.columns.get_level_values(qc_col) == qc_lbl 
            if qc_col in self.columns.names 
            else np.zeros(self.shape[1], dtype=bool)
        )
        
        if qc_mask.any() and not current_idx.empty:
            df_qc = self.loc[current_idx, qc_mask]
            
            qc_na_rate = df_qc.isna().mean(axis=1)
            qc_median = df_qc.median(axis=1)
            int_threshold = qc_median.quantile(qc_int_pct)
            
            cond_qc_mv = qc_na_rate > qc_mv_tol
            cond_qc_int = qc_median <= int_threshold
            
            idx_mnar_qc = current_idx[cond_qc_mv | cond_qc_int]
            current_idx = current_idx.difference(idx_mnar_qc)

        idx_mar = current_idx
        idx_mnar_all = idx_mnar_group.union(idx_mnar_qc)
        
        self.stats["idx_mar"] = idx_mar
        self.stats["idx_mnar"] = idx_mnar_all
        self.stats["idx_mnar_group"] = idx_mnar_group
        self.stats["idx_mnar_qc"] = idx_mnar_qc
        
        self.attrs["idx_mar"] = idx_mar
        self.attrs["idx_mnar"] = idx_mnar_all
        self.attrs["idx_mnar_group"] = idx_mnar_group
        self.attrs["idx_mnar_qc"] = idx_mnar_qc
        
        logger.info(
            f"Missing Types: {len(idx_mar)} MAR, {len(idx_mnar_all)} MNAR "
            f"({len(idx_mnar_group)} Group, {len(idx_mnar_qc)} QC)."
        )
        
        if not idx_mnar_all.empty:
            preview_num = min(5, len(idx_mnar_all))
            preview_list = idx_mnar_all[:preview_num].tolist()
            logger.info(f"MNAR Preview (Top {preview_num}): {preview_list}")
        
        return idx_mar, idx_mnar_all

    @iu._exe_time
    def execute_quality_filtering(self, idx_mar=None, idx_mnar=None, output_dir=None):
        """Execute Stage-2 quality filter (Blank Ratio & QC RSD)."""
        if idx_mar is None:
            idx_mar = self.attrs.get("idx_mar")
        if idx_mnar is None:
            idx_mnar = self.attrs.get("idx_mnar")

        if idx_mar is None or idx_mnar is None:
            logger.warning(
                "MAR/MNAR indices not found in pipeline. Recomputing natively. "
                "Warning: If imputation was already applied, this will fail!"
            )
            idx_mar, idx_mnar = self.classify_missing_types()
            
        self.stats["idx_mnar_group"] = self.attrs.get(
            "idx_mnar_group", pd.Index([])
        )
        self.stats["idx_mnar_qc"] = self.attrs.get(
            "idx_mnar_qc", pd.Index([])
        )
        self.stats["idx_mar"] = idx_mar
        self.stats["idx_mnar"] = idx_mnar

        fc = self.params["MetaboIntFilter"]["feature_counts"]
        meta_params = self.params.get("MetaboInt", {})
        
        qc_col = meta_params.get("sample_type", "Sample Type")
        sample_dict = meta_params.get("sample_dict", {})
        qc_lbl = sample_dict.get("QC sample", "QC")
        blk_lbl = sample_dict.get("Blank sample", "Blank")
        
        fltr_params = self.params.get("MetaboIntFilter", {})
        rsd_tol = fltr_params.get("rsd_qc_tol", 0.3)
        blk_tol = fltr_params.get("blank_qc_ratio", 0.2)

        qc_mask = self.columns.get_level_values(qc_col) == qc_lbl
        blk_mask = self.columns.get_level_values(qc_col) == blk_lbl
        current_idx = self.index
        logger.info(f"Features before filtering: {len(current_idx)}")

        if blk_mask.any() and qc_mask.any():
            logger.info(
                "Low quality features filter (Stage2): Blank/QC check.")

            qc_m = self.loc[:, qc_mask].mean(axis=1)
            blk_m = self.loc[:, blk_mask].mean(axis=1)
            
            self.stats.update({"qc_mean": qc_m, "blank_mean": blk_m})
            max_ratio = blk_tol
            
            qc_safe = qc_m.replace(0, np.finfo(float).eps)
            pass_blk = blk_m[blk_m / qc_safe <= max_ratio].index
            current_idx = current_idx.intersection(pass_blk)
            logger.info(f"Features after Blank/QC check: {len(current_idx)}")

        
        fc["post_stage2_blank"] = len(current_idx)

        if qc_mask.any():
            logger.info(
                "Low quality features filter (Stage2): QC RSD check.")
            df_qc = self.loc[current_idx, qc_mask]
            std_qc = df_qc.std(axis=1, ddof=1)
            mean_qc = df_qc.mean(axis=1)
            
            self.stats["qc_rsd_all"] = std_qc / mean_qc
            
            pass_mar = self.stats["qc_rsd_all"].loc[
                idx_mar.intersection(current_idx)]
            final_idx = pass_mar[pass_mar <= rsd_tol].index.union(
                idx_mnar.intersection(current_idx))
            logger.info(f"Features after QC RSD check: {len(final_idx)}")

        else:
            final_idx = current_idx

        fc["post_stage2_rsd"] = len(final_idx)
        df_final = self.loc[final_idx].copy()

        if output_dir:
            iu._check_dir_exists(dir_path=output_dir, handle="makedirs")
            mode = meta_params.get("mode", "POS")
            # ==============================================================
            df_final.attrs["pipeline_stage"] = "Low-quality features filter"
            # ==============================================================
            csv_path = os.path.join(
                output_dir, f"Filtered_Data_Stage2_{mode}.csv"
            )
            df_final.to_csv(csv_path, encoding="utf-8-sig", na_rep="NA")
            logger.info(
                "Data after low-quality features filtering saved as: "
                f"{csv_path}")
            
            vis = MetaboVisualizerFilter(engine=df_final)

            try:
                fig_quality_fltr_summary = vis.plot_quality_filtering_summary_grid()
                if fig_quality_fltr_summary:
                    summary_path = os.path.join(
                        output_dir, f"Quality_FLTR_Grid_{mode}.pdf"
                    )
                    vis.save_and_show_pw(
                        pw_obj=fig_quality_fltr_summary,file_path=summary_path)
                    logger.info(
                        f"Low-quality features filter summary grid saved as: {summary_path}")
            except Exception as e:
                logger.error(f"Stage 2 summary grid generation failed: {e}")

            plot_tasks = [
                (vis.plot_qc_blank_scatter(), "Blank_Scatter"),
                (vis.plot_rsd_dist(idx_mnar=idx_mnar), "RSD_Dist"),
                (vis.plot_filtering_summary(), "Summary")
            ]
            
            for fig, name in plot_tasks:
                if fig:
                    pdf_path = os.path.join(
                        output_dir, f"Filter_Stage2_{name}_{mode}.pdf"
                    )
                    vis.save_and_close_fig(fig, pdf_path)

        logger.success("Low-quality features filtering completed.")

        return df_final


class MetaboVisualizerFilter(visualizer_classes.BaseMetaboVisualizer):
    """Visualization suite for metabolomics filtering results."""

    def __init__(self, engine):
        """Initialize with the filtering engine."""
        super().__init__(metabo_obj=engine)
        self.engine = engine
        self.fltr_params = engine.params.get("MetaboIntFilter", {})

    # =========================================================================
    # Diagnostic Plots
    # =========================================================================

    def plot_filtering_summary(self, ax=None):
        """Plot a bar chart showing feature attrition cascade."""
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(4.5,4))
        else:
            current_ax = ax
            fig = current_ax.figure
        
        fc = self.fltr_params.get("feature_counts", {})
        
        steps = [
            ("Raw\nData", "raw"), 
            ("High-MV\nCheck", "post_stage1"), 
            ("QC/Blank\nCheck", "post_stage2_blank"), 
            ("QC RSD\nCheck", "post_stage2_rsd")
        ]
        
        valid_steps = [(l, k) for l, k in steps if k in fc]
        if not valid_steps:
            return fig if ax is None else current_ax
            
        labels = [item[0] for item in valid_steps]
        counts = [fc[item[1]] for item in valid_steps]

        bar_colors = pu.extract_linear_cmap(
            cmap=pu.custom_linear_cmap(["white", "tab:red"], 100), 
            cmin=0.3, cmax=1.0, n_colors=len(counts)
        )
        
        sns.barplot(
            x=list(labels), y=list(counts), ax=current_ax, hue=list(labels), 
            palette=bar_colors, width=0.6, edgecolor="k", legend=False
        )
        
        pu.show_values_on_bars(
            axs=current_ax, show_percentage=False, fontsize=9, 
            position="outside", value_format="{:.0f}"
        )
        
        self._apply_standard_format(
            ax=current_ax, title="Low-quality Features Removal",
            xlabel="Filtering Steps", ylabel="Retained Features"
        )
        
        if ax is None:
            return fig
            
        return current_ax

    def plot_missing_classification(self, ax=None):
        """Plot the classification boundary for MAR and MNAR features."""
        idx_mnar_grp = self.engine.stats.get("idx_mnar_group")
        idx_mnar_qc = self.engine.stats.get("idx_mnar_qc")
        idx_mar = self.engine.stats.get("idx_mar")
        
        if idx_mar is None:
            return None if ax is None else ax
            
        meta_params = self.engine.params.get("MetaboInt", {})
        qc_col = meta_params.get("sample_type", "Sample Type")
        qc_lbl = meta_params.get("sample_dict", {}).get("QC sample", "QC")
        
        qc_mask = self.engine.columns.get_level_values(qc_col) == qc_lbl
        if not qc_mask.any():
            return None if ax is None else ax
            
        df_qc = self.engine.loc[:, qc_mask]
        
        qc_na_pct = df_qc.isna().mean(axis=1) * 100
        qc_median = np.log2(df_qc.median(axis=1).astype(float) + 1)
        
        types = pd.Series("MAR", index=self.engine.index)
        
        if idx_mnar_grp is not None and not idx_mnar_grp.empty:
            valid_grp = idx_mnar_grp.intersection(self.engine.index)
            if not valid_grp.empty:
                types.loc[valid_grp] = "MNAR (Group)"
                
        if idx_mnar_qc is not None and not idx_mnar_qc.empty:
            valid_qc = idx_mnar_qc.intersection(self.engine.index)
            if not valid_qc.empty:
                types.loc[valid_qc] = "MNAR (QC)"
            
        df_p = pd.DataFrame({
            "Missing_Rate": qc_na_pct,
            "Log2_Intensity": qc_median,
            "Type": types
        })
        
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(5, 4))
        else:
            current_ax = ax
            fig = current_ax.figure
            
        pal = {
            "MAR": "tab:gray", 
            "MNAR (QC)": "tab:red", 
            "MNAR (Group)": "tab:blue"
        }
        all_types = ["MAR", "MNAR (Group)", "MNAR (QC)"]
        hue_order = [t for t in all_types if t in df_p["Type"].unique()]
        
        sns.scatterplot(
            data=df_p, x="Log2_Intensity", y="Missing_Rate", hue="Type",
            palette=pal, hue_order=hue_order,
            s=35, edgecolor="k", linewidth=0.5, alpha=0.75, ax=current_ax
        )
        
        self._apply_standard_format(
            ax=current_ax, title="Missing Value Mechanisms",
            xlabel="Log2(Median QC Intensity + 1)", 
            ylabel="QC Missing Rate (%)"
        )
        
        self._format_single_legend(
            ax=current_ax, title="Mechanism", loc="upper left",
            bbox_to_anchor=(1.05, 1.0)
        )
        
        if ax is None:
            return fig
            
        return current_ax

    def plot_mv_group(self, mv_df, tol, group_order=None, ax=None):
        """Plot histograms of MV ratios across biological groups.

        Args:
            mv_df: DataFrame containing missing value ratios per group.
            tol: Tolerance threshold for missing values.
            group_order: Optional list specifying the rendering order of groups.
            ax: Optional matplotlib Axes object for aggregated plotting.

        Returns:
            A matplotlib Figure or Axes object containing the plots.
        """
        if mv_df.empty:
            return None if ax is None else ax

        # Reorder columns based on the provided categorical group_order
        if group_order is not None:
            valid_order = [g for g in group_order if g in mv_df.columns]
            missing = [g for g in mv_df.columns if g not in valid_order]
            final_order = valid_order + missing
            mv_df = mv_df[final_order]
        else:
            sorted_cols = np.sort(mv_df.columns)
            mv_df = mv_df[sorted_cols]

        mv_df = mv_df * 100
        tol_pct = tol * 100
        n_groups = mv_df.shape[1]

        ncols = min(n_groups, 3)
        nrows = int(np.ceil(n_groups / ncols))

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows), 
            sharey=True, layout="constrained"
        )
        axes_flat = [axes] if n_groups == 1 else axes.flatten()

        for idx, group_name in enumerate(mv_df.columns):
            current_ax = axes_flat[idx]
            
            df_p = pd.DataFrame({
                "MV": mv_df[group_name], 
                "Status": np.where(
                    mv_df[group_name] <= tol_pct, "Retained", "Filtered"
                )
            })
            
            sns.histplot(
                data=df_p, x="MV", hue="Status", 
                hue_order=["Retained", "Filtered"],
                palette={"Retained": "tab:gray", "Filtered": "tab:red"}, 
                multiple="stack", edgecolor="k", bins=np.arange(0, 105, 5), 
                ax=current_ax, legend=False
            )
            
            current_ax.axvline(
                x=tol_pct, color="k", linestyle="--", linewidth=1.5
            )
            
            ylabel = "Feature Count" if idx % ncols == 0 else ""
            self._apply_standard_format(
                ax=current_ax, title=f"Group: {group_name}", 
                xlabel="MV Ratio (%)", ylabel=ylabel
            )
            
            if idx == n_groups - 1:
                # [BUG FIX]: Bypassing the overwrite behavior of the base class
                h_ret = mpatches.Patch(
                    facecolor="tab:gray", edgecolor="k", label="Retained",
                    linestyle="-", linewidth=1.0)
                h_flt = mpatches.Patch(
                    facecolor="tab:red", edgecolor="k", label="Filtered",
                    linestyle="-", linewidth=1.0)
                h_thr = mlines.Line2D(
                    [], [], color="k", linestyle="--", 
                    label=f"Threshold ({tol_pct:.1f}%)"
                )
                current_ax.legend(
                    handles=[h_ret, h_flt, h_thr], title="Status", 
                    loc="upper right", **getattr(self, "LEGEND_KWARGS", {})
                )

        for i in range(n_groups, len(axes_flat)):
            axes_flat[i].set_visible(False)
            
        plt.suptitle(
            t="Stage 1: MV Distribution by Group", fontsize=14, weight="bold"
        )
        return fig

    def plot_qc_blank_scatter(self, ax=None):
        """Plot Log2 scatter of QC vs Blank intensities with double legends."""
        blk_m = self.engine.stats.get("blank_mean")
        qc_m = self.engine.stats.get("qc_mean")
        
        if blk_m is None or qc_m is None or blk_m.empty:
            return None

        df_p = pd.DataFrame({
            "QC": np.log2(qc_m.astype(float) + 1), 
            "Blank": np.log2(blk_m.astype(float) + 1)
        })
        
        tol_blk = self.fltr_params.get("blank_qc_ratio", 0.2)
        max_r = tol_blk
        
        qc_safe = qc_m.replace(0, np.finfo(float).eps)
        df_p["Status"] = np.where(
            blk_m / qc_safe <= max_r, "Retained", "Filtered"
        )

        if ax is None:
            fig, current_ax = plt.subplots(figsize=(4, 4))
        else:
            current_ax = ax
            fig = current_ax.figure

        sns.scatterplot(
            data=df_p, x="QC", y="Blank", ax=current_ax, hue="Status", 
            palette={"Retained": "tab:gray", "Filtered": "tab:red"},
            style="Status", markers={"Retained": "o", "Filtered": "X"},
            s=50, edgecolor="k", linewidth=0.5, alpha=0.8
        )
        
        lims = [
            np.min([current_ax.get_xlim(), current_ax.get_ylim()]),
            np.max([current_ax.get_xlim(), current_ax.get_ylim()])
        ]
        x_l = np.linspace(max(0, lims[0]), lims[1], 200)
        
        current_ax.plot(
            x_l, np.log2(((2**x_l - 1) * max_r) + 1), color="k", 
            linestyle="--", linewidth=1.5, label=f"Blank/QC={tol_blk}"
        )

        self._apply_standard_format(
            ax=current_ax, title="Stage 2.1: Blank Contamination", 
            xlabel="Log2(Mean QC + 1)", ylabel="Log2(Mean Blank + 1)"
        )
        
        handles, labels = current_ax.get_legend_handles_labels()
        if current_ax.get_legend():
            current_ax.get_legend().remove()
        
        sc_h = [
            h for h, l in zip(handles, labels) 
            if l in ["Retained", "Filtered"]
        ]
        sc_l = [l for l in labels if l in ["Retained", "Filtered"]]
        ln_h = [h for h, l in zip(handles, labels) if "Blank/QC" in l]
        ln_l = [l for l in labels if "Blank/QC" in l]
        
        leg1 = current_ax.legend(
            sc_h, sc_l, title="Status", loc="upper left", 
            bbox_to_anchor=(0.02, 0.98), **getattr(self, "LEGEND_KWARGS", {})
        )
        current_ax.add_artist(leg1)
        
        current_ax.legend(
            ln_h, ln_l, loc="lower right", **getattr(self, "LEGEND_KWARGS", {})
        )
        
        if ax is None:
            return fig
            
        return current_ax

    def plot_rsd_dist(self, idx_mnar, ax=None):
        """Plot RSD distribution with consistent bins for MAR and MNAR."""
        rsd_all = self.engine.stats.get("qc_rsd_all")
        if rsd_all is None or rsd_all.empty:
            return None if ax is None else ax

        if ax is None:
            fig, current_ax = plt.subplots(figsize=(4, 4))
        else:
            current_ax = ax
            fig = current_ax.figure
        
        tol = self.fltr_params.get("rsd_qc_tol", 0.3)
        
        idx_mnar_valid = rsd_all.index.intersection(idx_mnar)
        
        types = pd.Series("MAR", index=rsd_all.index)
        if not idx_mnar_valid.empty:
            types.loc[idx_mnar_valid] = "MNAR"
            
        df_p = pd.DataFrame({
            "RSD": rsd_all,
            "Type": types
        })
        
        max_rsd = float(rsd_all.max())
        bin_edges = np.linspace(0, max_rsd, 50)
        
        sns.histplot(
            data=df_p, x="RSD", hue="Type", 
            palette={"MAR": "tab:gray", "MNAR": "tab:red"},
            hue_order=[t for t in ["MAR", "MNAR"] if t in df_p["Type"].unique()],
            bins=bin_edges, kde=True, ax=current_ax, legend=False,
            edgecolor="k", alpha=0.6
        )
            
        current_ax.axvline(
            x=tol, color="k", linestyle="--", linewidth=1.5
        )
        
        # [BUG FIX]: Safe manual handle injection
        handles = []
        if "MAR" in df_p["Type"].values:
            handles.append(mpatches.Patch(
                facecolor="tab:gray", edgecolor="k", linewidth=1.0, label="MAR"))
        if "MNAR" in df_p["Type"].values:
            handles.append(mpatches.Patch(
                facecolor="tab:red", edgecolor="k", linewidth=1.0, label="MNAR"))
            
        handles.append(
            mlines.Line2D(
                [], [], color="k", linestyle="--", 
                label=f"MAR Threshold ({tol})", linewidth=1.0
            )
        )
        
        current_ax.legend(
            handles=handles, title="Feature Type", loc="upper right",
            **getattr(self, "LEGEND_KWARGS", {})
        )

        self._apply_standard_format(
            ax=current_ax, title="Stage 2.2: QC RSD Check", 
            xlabel="RSD", ylabel="Feature Count"
        )
        
        if ax is None:
            return fig
            
        return current_ax

    def plot_mv_downgrade(self, mv_series, level, tol, ax=None):
        """Fallback histogram for QC or Global MV distribution."""
        if mv_series is None or mv_series.empty:
            return None if ax is None else ax
            
        tol_pct = tol * 100
        mv_pct = mv_series * 100
        
        return self.plot_single_mv_hist(
            series=mv_pct, tol=tol_pct, 
            title=f"Stage 1: MV Dist ({level})", 
            xl=f"MV Ratio in {level} (%)", yl="Feature Count", ax=ax
        )

    def plot_single_mv_hist(self, series, tol, title, xl, yl, ax=None):
        """Internal helper for single histogram plotting."""
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(4, 4))
        else:
            current_ax = ax
            fig = current_ax.figure
            
        df = pd.DataFrame({
            "MV": series, 
            "Status": np.where(series <= tol, "Retained", "Filtered")
        })
        
        sns.histplot(
            data=df, x="MV", hue="Status", multiple="stack", edgecolor="k", 
            palette={"Retained": "tab:gray", "Filtered": "tab:red"}, 
            hue_order=["Retained", "Filtered"],
            bins=np.arange(0, 105, 5), ax=current_ax, legend=False
        )
        
        current_ax.axvline(
            x=tol, color="k", linestyle="--", linewidth=1.5
        )
        
        # [BUG FIX]: Bypassing the overwrite behavior of the base class
        h_ret = mpatches.Patch(
            facecolor="tab:gray", edgecolor="k", label="Retained",
            linestyle="-", linewidth=1.0)
        h_flt = mpatches.Patch(
            facecolor="tab:red", edgecolor="k", label="Filtered",
            linestyle="-", linewidth=1.0)
        h_thr = mlines.Line2D(
            [], [], color="k", linestyle="--", label=f"Threshold ({tol:.1f}%)"
        )
        
        current_ax.legend(
            handles=[h_ret, h_flt, h_thr], title="Status", 
            loc="upper right", **getattr(self, "LEGEND_KWARGS", {})
        )
        
        self._apply_standard_format(
            ax=current_ax, title=title, xlabel=xl, ylabel=yl
        )
        
        if ax is None:
            return fig
            
        return current_ax

    # =========================================================================
    # Patchworklib Multi-panel Summary Plots
    # =========================================================================

    def plot_mv_filtering_summary_grid(self, filter_level, group_order=None):
        """Combine Stage 1 MV distribution plots based on filter level.
        
        The layout dynamically adjusts according to the active filter level:
        - "Group": Top rows contain individual group histograms (max 3/row).
        Elements are strictly left-aligned with empty bricks.
        - "QC": A one-row grid (QC and Global side-by-side).
        - "Global": Returns None, as a single plot does not require a grid.
        """
        if filter_level == "Global":
            return None

        try:
            import patchworklib as pw
        except ImportError:
            from loguru import logger
            logger.warning("patchworklib not found. Skipping Stage 1 grid.")
            return None

        pw.clear()

        stats = self.engine.stats
        fltr_params = self.fltr_params

        g_tol = fltr_params.get("mv_group_tol", 0.5)
        q_tol = fltr_params.get("mv_qc_tol", 0.8)
        gl_tol = fltr_params.get("mv_global_tol", 0.7)
        
        has_qc = not stats.get("mv_qc_series", pd.Series(dtype=float)).empty

        if filter_level == "Group":
            mv_df = stats.get("mv_group_df")
            
            # [BUG FIX]: Apply categorical or ASCII sorting before grid building
            if group_order is not None:
                valid_order = [g for g in group_order if g in mv_df.columns]
                missing = [g for g in mv_df.columns if g not in valid_order]
                missing_order = sorted(missing)
                group_cols = valid_order + missing_order
            else:
                group_cols = sorted(mv_df.columns.tolist())
                
            n_groups = len(group_cols)
            
            # Determine global column count for perfect left-alignment
            bottom_len = 2 if has_qc else 1
            max_cols = max(min(n_groups, 3), bottom_len)
            
            # Dynamically build individual bricks for each biological group
            g_bricks = []
            for i, grp in enumerate(group_cols):
                ax_g = pw.Brick(figsize=(4, 4), label=f"G{i}")
                self.plot_single_mv_hist(
                    series=mv_df[grp] * 100, 
                    tol=g_tol * 100, 
                    title=f"Stage 1 Group Dist: {grp}", 
                    xl="MV Ratio (%)", 
                    yl="Feature Count" if i % 3 == 0 else "", 
                    ax=ax_g
                )
                g_bricks.append(ax_g)
                
            # Chunk bricks into rows and pad with empty bricks for alignment
            g_rows = []
            for i in range(0, len(g_bricks), 3):
                chunk = g_bricks[i:i+3]
                current_row = chunk[0]
                for b in chunk[1:]:
                    current_row = current_row | b
                
                # Pad empty bricks to force left-alignment
                if len(chunk) < max_cols:
                    for j in range(len(chunk), max_cols):
                        emp = pw.Brick(figsize=(4, 4), label=f"E_g_{i}_{j}")
                        emp.axis("off")
                        current_row = current_row | emp
                        
                g_rows.append(current_row)
                
            # Build Global and QC bricks
            ax_gl = pw.Brick(figsize=(4, 4), label="GL")
            self.plot_mv_downgrade(
                mv_series=stats.get("mv_global_series"), level="Global", 
                tol=gl_tol, ax=ax_gl
            )

            bottom_bricks = []
            if has_qc:
                ax_qc = pw.Brick(figsize=(4, 4), label="QC")
                self.plot_mv_downgrade(
                    mv_series=stats.get("mv_qc_series"), level="QC", 
                    tol=q_tol, ax=ax_qc
                )
                bottom_bricks = [ax_qc, ax_gl]
            else:
                bottom_bricks = [ax_gl]
                
            row_bottom = bottom_bricks[0]
            for b in bottom_bricks[1:]:
                row_bottom = row_bottom | b
                
            # Pad bottom row to force left-alignment with the group rows
            if len(bottom_bricks) < max_cols:
                for j in range(len(bottom_bricks), max_cols):
                    emp = pw.Brick(figsize=(4, 4), label=f"E_b_{j}")
                    emp.axis("off")
                    row_bottom = row_bottom | emp

            # Combine everything: Group block on top, QC/Global on bottom
            combined_brick = g_rows[0]
            for r in g_rows[1:]:
                combined_brick = combined_brick / r
                
            combined_brick = combined_brick / row_bottom
            return combined_brick

        elif filter_level == "QC":
            ax1 = pw.Brick(figsize=(4, 4), label="A")
            ax2 = pw.Brick(figsize=(4, 4), label="B")

            self.plot_mv_downgrade(
                mv_series=stats.get("mv_qc_series"), level="QC", 
                tol=q_tol, ax=ax1
            )
            self.plot_mv_downgrade(
                mv_series=stats.get("mv_global_series"), level="Global", 
                tol=gl_tol, ax=ax2
            )
            combined_brick = ax1 | ax2
            
            return combined_brick

        return None

    def plot_quality_filtering_summary_grid(
        self
    ):
        """Combine Stage 2 plots into a single figure using patchworklib.
        
        This method aggregates Blank Scatter, RSD Distribution, and 
        Filtering Summary into a 1x3 or custom grid layout.
        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping summary grid.")
            return None

        # [CRITICAL]: Clear global patchworklib state to prevent data residue
        pw.clear()
        # Create bricks for each sub-plot using the patchworklib wrapper
        # We specify different names to avoid brick ID conflicts
        ax1 = pw.Brick(figsize=(4, 4))
        ax2 = pw.Brick(figsize=(4, 4))
        ax3 = pw.Brick(figsize=(4, 4))

        # Explicitly call each plotting method with the allocated axes
        self.plot_qc_blank_scatter(ax=ax1)
        
        # Note: idx_mnar needs to be retrieved from engine stats
        idx_mnar = self.engine.stats.get("idx_mnar", pd.Index([]))
        self.plot_rsd_dist(idx_mnar=idx_mnar, ax=ax2)
        
        self.plot_filtering_summary(ax=ax3)

        # Use patchworklib operator '|' for horizontal concatenation
        # or '/' for vertical stacking. Here we use horizontal.
        combined_brick = (ax1 | ax2 | ax3)
        
        return combined_brick