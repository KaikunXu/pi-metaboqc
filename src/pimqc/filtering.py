"""Module for invalid feature/sample filtering and its visualization."""

import os
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, Any, Optional, List

from . import io_utils as iu
from . import plot_utils as pu
from . import core_classes
from . import visualizer_classes

from loguru import logger


class MetaboIntFilter(core_classes.MetaboInt):
    """Filtering engine subclassing MetaboInt for metabolomics datasets.

    Attributes:
        params: Pipeline configuration parameters.
        stats: Dictionary storing intermediate statistics for visualization.
    """

    _metadata = ["stats", "params"]

    def __init__(
        self, data: Any = None, 
        pipeline_params: Optional[Dict[str, Any]] = None,
        *args: Any, **kwargs: Any
    ) -> None:
        """Initialize the filtering engine as a dataframe subclass."""
        super().__init__(data=data, *args, **kwargs)

        if pipeline_params is not None:
            self.attrs["pipeline_parameters"] = pipeline_params

        self.params: Dict[str, Any] = self.attrs.get(
            "pipeline_parameters", {}
        )

        # Strictly align with the updated JSON block: "MetaboIntFilter"
        if "MetaboIntFilter" not in self.params:
            self.params["MetaboIntFilter"] = {}
        if "feature_counts" not in self.params["MetaboIntFilter"]:
            self.params["MetaboIntFilter"]["feature_counts"] = {}

        self.stats: Dict[str, Any] = {
            "mv_group_df": pd.DataFrame(),
            "mv_qc_series": pd.Series(dtype=float),
            "mv_global_series": pd.Series(dtype=float),
            "blank_mean": pd.Series(dtype=float),
            "qc_mean": pd.Series(dtype=float),
            "qc_rsd_all": pd.Series(dtype=float)
        }

    @property
    def _constructor(self) -> type:
        """Override constructor to return the subclass type."""
        return MetaboIntFilter

    def __finalize__(
        self, other: Any, method: Optional[str] = None, **kwargs: Any
    ) -> "MetaboIntFilter":
        """Explicitly deepcopy custom attributes during pandas operations."""
        self = super().__finalize__(other, method=method, **kwargs)
        for name in self._metadata:
            if hasattr(other, name):
                setattr(self, name, copy.deepcopy(getattr(other, name)))
        return self

    @iu._exe_time
    def execute_mv_fltr(
        self, output_dir: Optional[str] = None
    ) -> "MetaboIntFilter":
        """Execute Stage-1 missing value filter with group validation.

        Args:
            output_dir: Optional directory path to save output files.

        Returns:
            MetaboIntFilter: The missing value filtered dataset.
        """
        fc = self.params["MetaboIntFilter"]["feature_counts"]
        if "raw" not in fc:
            fc["raw"] = self.shape[0]

        # Align with global metadata definitions
        grp_col: str = self.params.get("MetaboInt", {}).get(
            "bio_group", "Bio Group"
        )
        qc_col: str = self.params.get("MetaboInt", {}).get(
            "sample_type", "Sample Type"
        )
        sample_dict = self.params.get("MetaboInt", {}).get("sample_dict", {})
        qc_lbl: str = sample_dict.get("QC sample", "QC")
        
        fltr_params = self.params.get("MetaboIntFilter", {})

        valid_bio_groups: List[str] = []
        invalid_strs = {
            "unknown", "na", "n/a", "nan", "none", "null", "",
            "unassigned", "blank", "blk", "is", "solvent", "wash",
            "sst", "pool", "invalid", "unvalid"
        }

        if grp_col in self.columns.names:
            raw_groups = self.columns.get_level_values(grp_col).unique()
            for g in raw_groups:
                if pd.isna(g):
                    continue
                g_str = str(g).strip().lower()
                if g_str in invalid_strs or g_str == str(qc_lbl).lower():
                    continue
                valid_bio_groups.append(g)

        if qc_col in self.columns.names:
            raw_types = self.columns.get_level_values(qc_col)
            type_strs = [str(t).strip().lower() for t in raw_types]
            valid_mask = np.array(
                [t not in invalid_strs for t in type_strs]
            )
        else:
            valid_mask = np.ones(self.shape[1], dtype=bool)

        filter_level: str = "Group"

        if valid_bio_groups:
            logger.info("Stage 1 MV Filter: Using Biological Group level.")
            mv_tol = fltr_params.get("mv_group_tol", 0.5)
            na_rate = self.isna().groupby(level=grp_col, axis=1).mean()
            self.stats["mv_group_df"] = na_rate[valid_bio_groups]
            pass_mask = (na_rate[valid_bio_groups] <= mv_tol).any(axis=1)
            
        else:
            qc_mask = np.array([False] * self.shape[1])
            if qc_col in self.columns.names:
                qc_mask = self.columns.get_level_values(qc_col) == qc_lbl

            if qc_mask.any():
                logger.info("Stage 1 MV Filter: Using Pooled QC level.")
                filter_level = "QC"
                mv_tol = fltr_params.get("mv_qc_tol", 0.8)
                df_qc = self.loc[:, qc_mask]
                na_rate_qc = df_qc.isna().mean(axis=1)
                self.stats["mv_qc_series"] = na_rate_qc
                pass_mask = na_rate_qc <= mv_tol
                
            else:
                logger.info("Stage 1 MV Filter: Using Global level.")
                filter_level = "Global"
                mv_tol = fltr_params.get("mv_global_tol", 0.7)
                df_global = self.loc[:, valid_mask]
                if df_global.shape[1] == 0:
                    logger.warning("No valid samples for global MV calc.")
                    na_rate_global = pd.Series(
                        [1.0] * self.shape[0], index=self.index
                    )
                    pass_mask = pd.Series(
                        [False] * self.shape[0], index=self.index
                    )
                else:
                    na_rate_global = df_global.isna().mean(axis=1)
                    pass_mask = na_rate_global <= mv_tol
                self.stats["mv_global_series"] = na_rate_global

        valid_idx = self.index[pass_mask]
        df_final = self.loc[valid_idx].copy()
        
        fc["post_stage1"] = df_final.shape[0]

        if output_dir:
            iu._check_dir_exists(dir_path=output_dir, handle="makedirs")
            mode = self.params.get("MetaboInt", {}).get("mode", "POS")

            csv_path = os.path.join(
                output_dir, f"Filtered_Data_Stage1_{mode}.csv"
            )
            df_final.to_csv(csv_path, encoding="utf-8-sig", na_rep="NA")
            
            vis = MetaboVisualizerFilter(engine=df_final)
            
            if filter_level == "Group":
                fig_mv = vis.plot_mv_group()
            else:
                fig_mv = vis.plot_mv_downgrade(level=filter_level)
                
            if fig_mv:
                pdf_path = os.path.join(
                    output_dir, f"Filter_Stage1_MV_{filter_level}_{mode}.pdf"
                )
                vis.save_and_close_fig(fig=fig_mv, file_path=pdf_path)
                
            logger.info(
                f"Stage 1 filter completed. Outputs saved to {output_dir}"
            )

        return df_final

    @iu._exe_time
    def execute_quality_fltr(
        self,
        idx_mar: pd.Index,
        idx_mnar: pd.Index,
        output_dir: Optional[str] = None
    ) -> "MetaboIntFilter":
        """Execute Stage-2 quality filter (Blank Ratio & QC RSD).

        Args:
            idx_mar: Features classified as Missing At Random.
            idx_mnar: Features classified as Missing Not At Random.
            output_dir: Optional directory path to save output files.

        Returns:
            MetaboIntFilter: Quality filtered dataset.
        """
        fc = self.params["MetaboIntFilter"]["feature_counts"]
        
        qc_col: str = self.params.get("MetaboInt", {}).get(
            "sample_type", "Sample Type"
        )
        sample_dict = self.params.get("MetaboInt", {}).get("sample_dict", {})
        qc_lbl: str = sample_dict.get("QC sample", "QC")
        blk_lbl: str = sample_dict.get("Blank sample", "Blank")

        fltr_params = self.params.get("MetaboIntFilter", {})
        rsd_tol: float = fltr_params.get("rsd_qc_tol", 0.3)
        blk_tol: float = fltr_params.get("qc_blank_ratio", 5.0)

        qc_mask = self.columns.get_level_values(qc_col) == qc_lbl
        blk_mask = self.columns.get_level_values(qc_col) == blk_lbl

        df_qc = self.loc[:, qc_mask]
        df_blk = self.loc[:, blk_mask]
        current_idx = self.index

        if blk_mask.any() and qc_mask.any():
            logger.info("Stage 2 Filter: Executing Blank Ratio Check.")
            qc_mean = df_qc.mean(axis=1)
            blk_mean = df_blk.mean(axis=1)
            self.stats["qc_mean"] = qc_mean
            self.stats["blank_mean"] = blk_mean

            qc_mean_safe = qc_mean.replace(0, np.finfo(float).eps)
            blank_ratio = blk_mean / qc_mean_safe

            # The parameter 'qc_blank_ratio' = 5.0 means QC >= 5.0 * Blank.
            # Therefore, the allowed maximum for (Blank / QC) is 1.0 / 5.0.
            max_blank_ratio = 1.0 / blk_tol if blk_tol > 0 else 0.0
            pass_blk = blank_ratio[blank_ratio <= max_blank_ratio].index
            current_idx = current_idx.intersection(pass_blk)
        else:
            logger.warning("Stage 2 Filter: Blank Ratio Check bypassed.")

        fc["post_stage2_blank"] = len(current_idx)

        if qc_mask.any():
            logger.info("Stage 2 Filter: Executing QC RSD Check.")
            self.stats["qc_rsd_all"] = (
                df_qc.std(axis=1, ddof=1) / df_qc.mean(axis=1)
            )
            valid_mar_idx = idx_mar.intersection(current_idx)

            if not valid_mar_idx.empty:
                rsd_mar = self.stats["qc_rsd_all"].loc[valid_mar_idx]
                pass_rsd_mar = rsd_mar[rsd_mar <= rsd_tol].index
            else:
                pass_rsd_mar = pd.Index([])

            valid_mnar_idx = idx_mnar.intersection(current_idx)
            final_idx = pass_rsd_mar.union(valid_mnar_idx)
        else:
            logger.warning("Stage 2 Filter: QC RSD Check bypassed.")
            final_idx = current_idx

        fc["post_stage2_rsd"] = len(final_idx)

        df_final = self.loc[final_idx].copy()

        if output_dir:
            iu._check_dir_exists(dir_path=output_dir, handle="makedirs")
            mode = self.params.get("MetaboInt", {}).get("mode", "POS")

            csv_path = os.path.join(
                output_dir, f"Filtered_Data_Stage2_{mode}.csv"
            )
            df_final.to_csv(csv_path, encoding="utf-8-sig", na_rep="NA")
            
            vis = MetaboVisualizerFilter(engine=df_final)

            fig_blk = vis.plot_qc_blank_scatter()
            if fig_blk:
                pdf_path = os.path.join(
                    output_dir, f"Filter_Stage2_Blank_Scatter_{mode}.pdf"
                )
                vis.save_and_close_fig(fig=fig_blk, file_path=pdf_path)

            fig_rsd = vis.plot_rsd_distribution_with_exemptions(
                idx_mnar=idx_mnar
            )
            if fig_rsd:
                pdf_path = os.path.join(
                    output_dir, f"Filter_Stage2_RSD_Dist_{mode}.pdf"
                )
                vis.save_and_close_fig(fig=fig_rsd, file_path=pdf_path)

            fig_sum = vis.plot_filtering_summary()
            if fig_sum:
                pdf_path = os.path.join(
                    output_dir, f"Filter_Stage2_Summary_{mode}.pdf"
                )
                vis.save_and_close_fig(fig=fig_sum, file_path=pdf_path)
                
            logger.info(
                f"Stage 2 filter completed. Outputs saved to {output_dir}"
            )

        return df_final


class MetaboVisualizerFilter(visualizer_classes.BaseMetaboVisualizer):
    """Visualization suite inheriting from BaseMetaboVisualizer.

    Attributes:
        engine: The configured MetaboIntFilter subclass instance.
    """

    def __init__(self, engine: MetaboIntFilter) -> None:
        """Initialize with the filtering engine to access its stats."""
        super().__init__(metabo_obj=engine)
        self.engine = engine
        
        fltr_params = engine.params.get("MetaboIntFilter", {})
        self.tol_rsd: float = fltr_params.get("rsd_qc_tol", 0.3)
        self.tol_blk: float = fltr_params.get("qc_blank_ratio", 5.0)

    def plot_filtering_summary(self) -> Figure:
        """Plot a bar chart showing feature attrition cascade."""
        fig, ax = plt.subplots(figsize=(4, 4))
        
        fc: Dict[str, int] = self.engine.params.get(
            "MetaboIntFilter", {}).get("feature_counts", {})
        
        labels: List[str] = []
        counts: List[int] = []
        
        if "raw" in fc:
            labels.append("Raw Data")
            counts.append(fc["raw"])
        if "post_stage1" in fc:
            labels.append("High-MV Check")
            counts.append(fc["post_stage1"])
        if "post_stage2_blank" in fc:
            labels.append("Pooled QC/Blank Check")
            counts.append(fc["post_stage2_blank"])
        if "post_stage2_rsd" in fc:
            labels.append("Pooled QC RSD Check")
            counts.append(fc["post_stage2_rsd"])

        if not counts:
            return fig

        n_bars: int = len(counts)
        bar_colors = pu.extract_linear_cmap(
            cmap=pu.custom_linear_cmap(
                color_list=["white", "tab:red"], n_colors=100
            ),
            cmin=0.3,
            cmax=1.0,
            n_colors=n_bars
        )

        sns.barplot(
            x=labels, y=counts, ax=ax, 
            hue=labels, palette=bar_colors, 
            width=0.6, edgecolor="k", legend=False
        )
        
        pu.show_values_on_bars(
            axs=ax, show_percentage=False, fontsize=9, position="outside", 
            value_format="{:.0f}"
        )
        pu.change_axis_rotation(ax=ax, axis="x", rotation=45)
        
        self._apply_standard_format(
            ax=ax, xlabel="Filtering Steps",
            ylabel="Number of Retained Features",
            title="Low-quality Features Removal"
        )
        
        return fig

    def plot_mv_group(self) -> Optional[Figure]:
        """Plot histograms of MV ratios across valid biological groups."""
        mv_df: pd.DataFrame = self.engine.stats.get(
            "mv_group_df", pd.DataFrame()
        )
        if mv_df.empty:
            return None

        grp_col: str = self.engine.params.get("MetaboInt", {}).get(
            "bio_group", "Bio Group"
        )
        if grp_col in self.engine.columns.names:
            group_level = self.engine.columns.get_level_values(grp_col)
            if hasattr(group_level, "categories"):
                cat_order = list(group_level.categories)
            elif hasattr(group_level.dtype, "categories"):
                cat_order = list(group_level.dtype.categories)
            else:
                cat_order = sorted(mv_df.columns.tolist())
                
            valid_cols = [c for c in cat_order if c in mv_df.columns]
            valid_cols += [c for c in mv_df.columns if c not in valid_cols]
            mv_df = mv_df[valid_cols]

        mv_df = mv_df * 100
        
        fltr_params = self.engine.params.get("MetaboIntFilter", {})
        tol_pct: float = fltr_params.get("mv_group_tol", 0.5) * 100
        
        n_groups: int = mv_df.shape[1]
        max_cols: int = 3
        ncols: int = min(n_groups, max_cols)
        nrows: int = int(np.ceil(n_groups / max_cols))

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, 
            figsize=(4 * ncols, 4 * nrows), 
            sharey=True, layout="constrained"
        )
        
        if isinstance(axes, plt.Axes):
            axes_flat = [axes]
        else:
            axes_flat = axes.flatten()

        for idx, group_name in enumerate(mv_df.columns):
            ax = axes_flat[idx]
            
            df_plot = pd.DataFrame({
                "MV": mv_df[group_name],
                "Status": np.where(
                    mv_df[group_name] <= tol_pct, "Retained", "Filtered"
                )
            })

            sns.histplot(
                data=df_plot, x="MV", hue="Status", 
                hue_order=["Retained", "Filtered"],
                palette={"Retained": "tab:gray", "Filtered": "tab:red"}, 
                multiple="stack", edgecolor="k",
                bins=np.arange(0, 105, 5), ax=ax, 
                legend=(idx == n_groups - 1)
            )
            ax.axvline(x=tol_pct, color="k", linestyle="--", linewidth=1.5)

            ylabel: str = "Feature Count" if (idx % max_cols) == 0 else ""
            
            self._apply_standard_format(
                ax=ax, xlabel=f"MV Ratio in '{group_name}' (%)",
                ylabel=ylabel, title=f"Group: {group_name}",
                title_fontsize=12, label_fontsize=12, tick_fontsize=10
            )

            if idx == n_groups - 1:
                self._format_single_legend(
                    fig=fig, ax=ax, loc="upper left", bbox_to_anchor=(1.05, 1.0)
                )

        for idx in range(n_groups, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.suptitle(
            t="Stage 1: Missing Value Distribution by Group", 
            fontsize=14, weight="bold"
        )

        return fig

    def plot_mv_downgrade(self, level: str) -> Optional[Figure]:
        """Plot histogram of MV ratios for downgraded QC or Global level.
        
        Args:
            level: The degradation level used ('QC' or 'Global').
            
        Returns:
            Optional[Figure]: The generated matplotlib figure or None.
        """
        mv_series: Optional[pd.Series] = None
        title_lvl: str = ""
        fltr_params = self.engine.params.get("MetaboIntFilter", {})
        
        if level == "QC":
            mv_series = self.engine.stats.get("mv_qc_series")
            title_lvl = "Pooled QC"
            tol_pct = fltr_params.get("mv_qc_tol", 0.8) * 100
        else:
            mv_series = self.engine.stats.get("mv_global_series")
            title_lvl = "Global Samples"
            tol_pct = fltr_params.get("mv_global_tol", 0.7) * 100
            
        if mv_series is None or mv_series.empty:
            return None

        mv_series = mv_series * 100

        fig, ax = plt.subplots(figsize=(4, 4))
        
        df_plot = pd.DataFrame({
            "MV": mv_series,
            "Status": np.where(mv_series <= tol_pct, "Retained", "Filtered")
        })

        sns.histplot(
            data=df_plot, x="MV", hue="Status", 
            hue_order=["Retained", "Filtered"],
            palette={"Retained": "tab:red", "Filtered": "tab:gray"}, 
            multiple="stack", edgecolor="k",
            bins=np.arange(0, 105, 5), ax=ax, legend=True
        )
        
        ax.axvline(x=tol_pct, color="k", linestyle="--", linewidth=1.5)

        self._apply_standard_format(
            ax=ax, xlabel=f"MV Ratio in {title_lvl} (%)",
            ylabel="Feature Count",
            title=f"Stage 1: MV Distribution ({title_lvl})"
        )
        
        self._format_single_legend(fig=fig, ax=ax)
        
        return fig

    def plot_qc_blank_scatter(self) -> Optional[Figure]:
        """Plot Log2 scatter of QC vs Blank intensities."""
        blk_mean: Optional[pd.Series] = self.engine.stats.get("blank_mean")
        qc_mean: Optional[pd.Series] = self.engine.stats.get("qc_mean")

        if blk_mean is None or qc_mean is None or blk_mean.empty:
            return None

        df_plot = pd.DataFrame({
            "Log2_QC": np.log2(qc_mean + 1),
            "Log2_Blank": np.log2(blk_mean + 1)
        })

        qc_safe = qc_mean.replace(0, np.finfo(float).eps)
        ratio = blk_mean / qc_safe
        
        # Calculate mathematical tolerance for Blank/QC based on QC/Blank target
        max_blank_ratio = 1.0 / self.tol_blk if self.tol_blk > 0 else 0.0
        
        df_plot["Metabolite"] = np.where(
            ratio <= max_blank_ratio, "Retained", "Filtered"
        )

        fig, ax = plt.subplots(figsize=(4, 4))
        
        sns.scatterplot(
            data=df_plot, x="Log2_QC", y="Log2_Blank", ax=ax,
            hue="Metabolite", palette={
                "Retained": "tab:gray", "Filtered": "tab:red"
            },
            style="Metabolite", markers={
                "Retained": "o", "Filtered": "X"
            },
            s=50, edgecolor="k", linewidth=0.5, alpha=0.8
        )
        
        lims_log = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
        ]
        lims_log[0] = max(0, lims_log[0])
        
        x_vals_log = np.linspace(lims_log[0], lims_log[1], 200)
        qc_orig = (2 ** x_vals_log) - 1
        
        # Calculate theoretical maximum allowed blank intensity
        blank_allowed = qc_orig * max_blank_ratio
        y_thresh_log = np.log2(blank_allowed + 1)
        
        ax.plot(
            x_vals_log, y_thresh_log, color="k", linestyle="--", 
            linewidth=1.5, label=f"QC/Blank={self.tol_blk}"
        )

        self._apply_standard_format(
            ax=ax, xlabel="Log2(Mean QC Int + 1)",
            ylabel="Log2(Mean Blank Int + 1)",
            title="Stage 2.1: Blank Contamination Check"
        )
        
        if ax.get_legend():
            ax.get_legend().remove()
            
        handles, labels = ax.get_legend_handles_labels()
        
        sc_handles, sc_labels = [], []
        ln_handles, ln_labels = [], []
        
        for h, l in zip(handles, labels):
            if l in ["Retained", "Filtered"]:
                sc_handles.append(h)
                sc_labels.append(l)
            elif "QC/Blank" in l:
                ln_handles.append(h)
                ln_labels.append(l)

        leg_marker = ax.legend(
            sc_handles, sc_labels, title="Metabolite",
            loc="upper left", bbox_to_anchor=(0.02, 0.98),
            **self.LEGEND_KWARGS
        )
        ax.add_artist(leg_marker)

        ax.legend(
            ln_handles, ln_labels,
            loc="upper left", bbox_to_anchor=(0.9, 0.9),
            **self.LEGEND_KWARGS
        )

        return fig

    def plot_rsd_distribution_with_exemptions(
        self, idx_mnar: pd.Index
    ) -> Optional[Figure]:
        """Plot RSD distribution highlighting exempted MNAR features."""
        rsd_all: Optional[pd.Series] = self.engine.stats.get("qc_rsd_all")
        if rsd_all is None or rsd_all.empty:
            return None

        fig, ax = plt.subplots(figsize=(7, 5))
        rsd_mar = rsd_all.drop(idx_mnar, errors="ignore")
        valid_mnar = rsd_all.index.intersection(idx_mnar)
        rsd_mnar = rsd_all.loc[valid_mnar]

        sns.histplot(
            rsd_mar, color="tab:red", label="MAR (Subject to RSD Filter)",
            ax=ax, kde=True
        )
        sns.histplot(
            rsd_mnar, color="tab:gray", label="MNAR (RSD Exempted)",
            ax=ax, kde=True
        )

        ax.axvline(
            self.tol_rsd, color="k", linestyle="--",
            label=f"Threshold ({self.tol_rsd})"
        )

        self._apply_standard_format(
            ax=ax, xlabel="Relative Standard Deviation (RSD)",
            ylabel="Feature Count",
            title="Stage 2.2: QC RSD & Biological Exemptions"
        )
        
        self._format_single_legend(fig=fig, ax=ax)
        return fig