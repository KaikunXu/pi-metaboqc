"""
Purpose of script: Filter low-quality features and flag abnormal samples.
"""

import os
import warnings
from functools import cached_property
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from . import io_utils as iu
from . import plot_utils as pu
from . import core_classes
from . import visualizer_classes

warnings.filterwarnings(action="ignore", category=RuntimeWarning)


class MetaboIntFLTR(core_classes.MetaboInt):
    """Module for filtering low-quality features in two distinct stages.

    Stage 1 (MV Filtering): Executes Rule 1-3 (Missing Values) and flags 
    abnormal samples. Executed before Signal Correction.
    Stage 2 (Quality Filtering): Executes Rule 4-5 (RSD and Blank Ratio). 
    Executed after Signal Correction but before Normalization.
    
    Note: Dynamic detection allows bypassing Rule 2 if grouping is missing.
    """

    _metadata: List[str] = ["attrs"]

    def __init__(
        self,
        *args: Any,
        pipeline_params: Optional[Dict[str, Any]] = None,
        mv_global_tol: float = 0.7,
        mv_group_tol: float = 0.5,
        mv_qc_tol: float = 0.8,
        rsd_qc_tol: float = 0.7,
        qc_blank_ratio: float = 5.0,
        sample_mv_tol: float = 0.5,
        **kwargs: Any
    ) -> None:
        """Initialize the filtering class with configurable thresholds."""
        super().__init__(*args, pipeline_params=pipeline_params, **kwargs)

        fltr_configs: Dict[str, Any] = {
            "mv_global_tol": mv_global_tol,
            "mv_group_tol": mv_group_tol,
            "mv_qc_tol": mv_qc_tol,
            "rsd_qc_tol": rsd_qc_tol,
            "qc_blank_ratio": qc_blank_ratio,
            "sample_mv_tol": sample_mv_tol
        }

        if pipeline_params and "MetaboIntFLTR" in pipeline_params:
            fltr_configs.update(pipeline_params["MetaboIntFLTR"])

        self.attrs.update(fltr_configs)

    @property
    def _constructor(self) -> type:
        """Override constructor to return MetaboIntFLTR."""
        return MetaboIntFLTR

    def __finalize__(
        self, other: Any, method: Optional[str] = None, **kwargs: Any
    ) -> "MetaboIntFLTR":
        """Explicitly preserve custom attributes during pandas operations."""
        super().__finalize__(other=other, method=method, **kwargs)
        if hasattr(other, "attrs"):
            self.attrs = __import__("copy").deepcopy(x=other.attrs)
        return self

    # ====================================================================
    # Stage 1: Missing Value Masks
    # ====================================================================

    @cached_property
    def mask_mv_global(self) -> pd.Series:
        """Rule 1: Filter features with high missing ratio in all samples."""
        tol = self.attrs["mv_global_tol"]
        mv_ratio = self.isna().mean(axis=1)
        return mv_ratio < tol

    @cached_property
    def mask_mv_group(self) -> pd.Series:
        """Rule 2: Keep features valid in AT LEAST ONE biological group.
        
        Dynamically bypasses this evaluation if bio_group metadata is absent.
        """
        st_col = self.attrs.get("sample_type", "Sample Type")
        bg_col = self.attrs.get("bio_group", "Bio Group")
        
        # Check if the specific metadata level exists for confidential datasets
        if bg_col not in self.columns.names:
            logger.info(
                f"Metadata '{bg_col}' not found. Rule 2 (Group MV) bypassed."
            )
            return pd.Series(data=True, index=self.index)

        act_lbl = self.attrs.get("sample_dict", {}).get("Actual sample", "Sample")
        tol = self.attrs["mv_group_tol"]

        mask_actual = self.columns.get_level_values(level=st_col) == act_lbl
        
        if not mask_actual.any():
            return pd.Series(data=True, index=self.index)

        actual_df = self.loc[:, mask_actual]
        group_labels = actual_df.columns.get_level_values(level=bg_col)
        
        mv_per_group = actual_df.isna().groupby(by=group_labels, axis=1).mean()
        return (mv_per_group < tol).any(axis=1)

    @cached_property
    def mask_mv_qc(self) -> pd.Series:
        """Rule 3: Filter features with high missing ratio in QC samples."""
        tol = self.attrs["mv_qc_tol"]
        if self._qc.empty:
            return pd.Series(data=True, index=self.index)
        
        qc_mv_ratio = self._qc.isna().mean(axis=1)
        return qc_mv_ratio < tol

    # ====================================================================
    # Stage 2: Quality Assessment Masks (RSD & Blank)
    # ====================================================================

    @cached_property
    def mask_rsd_qc(self) -> pd.Series:
        """Rule 4: Filter high-variance features in QC samples."""
        tol = self.attrs["rsd_qc_tol"]
        if self._qc.empty:
            return pd.Series(data=True, index=self.index)
            
        qc_mean = self._qc.mean(axis=1)
        qc_std = self._qc.std(axis=1, ddof=1)
        qc_rsd = qc_std / qc_mean
        
        mask = (qc_rsd <= tol) & (qc_mean > 0)
        return mask.fillna(value=False)

    @cached_property
    def qc_blank_ratio_series(self) -> pd.Series:
        """Calculate the ratio of QC mean vs Blank mean for each feature."""
        st_col = self.attrs.get("sample_type", "Sample Type")
        blk_lbl = self.attrs.get("sample_dict", {}).get("Blank sample", "Blank")
        
        mask_blank = self.columns.get_level_values(level=st_col) == blk_lbl
        
        if not mask_blank.any() or self._qc.empty:
            return pd.Series(data=np.inf, index=self.index)
            
        blank_df = self.loc[:, mask_blank]
        blank_mean = blank_df.mean(axis=1).fillna(value=0.0)
        qc_mean = self._qc.mean(axis=1).fillna(value=0.0)
        
        ratio = qc_mean / blank_mean
        ratio = ratio.replace(to_replace=[np.inf, -np.inf], value=np.nan)
        return ratio

    @cached_property
    def mask_blank_ratio(self) -> pd.Series:
        """Rule 5: Filter artifact features based on QC/Blank ratio."""
        tol = self.attrs["qc_blank_ratio"]
        ratio_series = self.qc_blank_ratio_series
        
        if ratio_series.isin(values=[np.inf]).all():
            logger.info("No Blank or QC samples found; skipping Rule 5.")
            return pd.Series(data=True, index=self.index)
            
        st_col = self.attrs.get("sample_type", "Sample Type")
        blk_lbl = self.attrs.get("sample_dict", {}).get("Blank sample", "Blank")
        mask_blank = self.columns.get_level_values(level=st_col) == blk_lbl
        blank_mean = self.loc[:, mask_blank].mean(axis=1).fillna(value=0.0)
        
        safe_mask = blank_mean == 0
        ratio_mask = ratio_series >= tol
        
        return safe_mask | ratio_mask.fillna(value=False)

    # ====================================================================
    # Execution Engines
    # ====================================================================

    def generate_mv_report(self) -> pd.DataFrame:
        """Combine Missing Value masks into a boolean report."""
        report = pd.DataFrame(index=self.index)
        report["Pass_MV_Global"] = self.mask_mv_global
        
        bg_col = self.attrs.get("bio_group", "Bio Group")
        if bg_col in self.columns.names:
            report["Pass_MV_Group"] = self.mask_mv_group
            
        report["Pass_MV_QC"] = self.mask_mv_qc
        report["Final_Keep_MV"] = report.all(axis=1)
        return report

    def generate_quality_report(self) -> pd.DataFrame:
        """Combine RSD and Blank ratio masks into a boolean report."""
        report = pd.DataFrame(index=self.index)
        report["Pass_RSD_QC"] = self.mask_rsd_qc
        report["Pass_Blank_Ratio"] = self.mask_blank_ratio
        report["Final_Keep_Quality"] = report.all(axis=1)
        return report

    def generate_sample_report(self) -> pd.DataFrame:
        """Evaluate and flag abnormal samples by Missing Value ratio."""
        report = pd.DataFrame(index=self.columns)
        report["Missing_Ratio"] = self.isna().mean(axis=0)
        tol = self.attrs["sample_mv_tol"]
        report["Flag_High_MV"] = report["Missing_Ratio"] >= tol
        return report

    @iu._exe_time
    def execute_mv_fltr(self, output_dir: str) -> "MetaboIntFLTR":
        """Stage 1: Execute MV filtering workflow and flag samples."""
        iu._check_dir_exists(dir_path=output_dir, handle="makedirs")
        mode = self.attrs.get("mode", "POS")

        mv_report = self.generate_mv_report()
        mv_report.to_csv(
            path_or_buf=os.path.join(
                output_dir, f"Feature_MV_Report_{mode}.csv"
            ),
            encoding="utf-8-sig", na_rep="NA"
        )
        n_total = mv_report.shape[0]
        n_keep = mv_report["Final_Keep_MV"].sum()
        logger.info(f"MV Filtering [{mode}]: Kept {n_keep}/{n_total} features.")

        samp_report = self.generate_sample_report()
        if isinstance(samp_report.index, pd.MultiIndex):
            samp_report = samp_report.reset_index()
        samp_report.to_csv(
            path_or_buf=os.path.join(
                output_dir, f"Sample_MV_Warning_{mode}.csv"
            ),
            index=False, encoding="utf-8-sig", na_rep="NA"
        )

        vis = MetaboVisualizerFLTR(fltr_obj=self)
        for name, fig in [
            ("Rule1_MV_Global", vis.plot_mv_global()),
            ("Rule2_MV_Group", vis.plot_mv_group()),
            ("Rule3_MV_QC", vis.plot_mv_qc())
        ]:
            if fig is not None:
                fig.savefig(
                    fname=os.path.join(output_dir, f"{name}_{mode}.pdf"),
                    bbox_inches="tight", dpi=300
                )

        clean_df = self.loc[mv_report["Final_Keep_MV"], :].copy()
        res_obj = MetaboIntFLTR(data=clean_df)
        res_obj.attrs = self.attrs
        return res_obj

    @iu._exe_time
    def execute_quality_fltr(self, output_dir: str) -> "MetaboIntFLTR":
        """Stage 2: Execute RSD and Blank ratio filtering workflow."""
        iu._check_dir_exists(dir_path=output_dir, handle="makedirs")
        mode = self.attrs.get("mode", "POS")

        qual_report = self.generate_quality_report()
        qual_report.to_csv(
            path_or_buf=os.path.join(
                output_dir, f"Feature_Quality_Report_{mode}.csv"
            ),
            encoding="utf-8-sig", na_rep="NA"
        )
        n_total = qual_report.shape[0]
        n_keep = qual_report["Final_Keep_Quality"].sum()
        logger.info(
            f"Quality Filtering [{mode}]: Kept {n_keep}/{n_total} features."
        )

        vis = MetaboVisualizerFLTR(fltr_obj=self)
        for name, fig in [
            ("Rule4_RSD_QC", vis.plot_rsd_qc()),
            ("Rule5_QC_Blank_Ratio", vis.plot_qc_blank_ratio())
        ]:
            if fig is not None:
                fig.savefig(
                    fname=os.path.join(output_dir, f"{name}_{mode}.pdf"),
                    bbox_inches="tight", dpi=300
                )

        clean_df = self.loc[qual_report["Final_Keep_Quality"], :].copy()
        res_obj = MetaboIntFLTR(data=clean_df)
        res_obj.attrs = self.attrs
        return res_obj


class MetaboVisualizerFLTR(visualizer_classes.BaseMetaboVisualizer):
    """Visualization suite for feature filtering metrics."""

    def __init__(self, fltr_obj: "MetaboIntFLTR") -> None:
        """Initialize with a computed MetaboIntFLTR object."""
        super().__init__(metabo_obj=fltr_obj)
        self.fltr = fltr_obj

    def plot_mv_global(self) -> Figure:
        """Rule 1: Plot histogram of global MV ratios."""
        tol = self.attrs.get("mv_global_tol", 0.7) * 100
        df = pd.DataFrame(data={
            "MV": self.fltr.isna().mean(axis=1) * 100, 
            "Pass": self.fltr.mask_mv_global
        })

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
        sns.histplot(
            data=df, x="MV", hue="Pass", hue_order=[True, False],
            palette=self.pal, multiple="stack", edgecolor="k", 
            bins=np.arange(start=0, stop=105, step=5), ax=ax
        )
        
        ax.axvline(
            x=tol, color="k", linestyle="--", linewidth=1.5, 
            label=f"Threshold ({tol:.0f}%)"
        )
        self._apply_standard_format(
            ax=ax, xlabel="Missing Value Ratio in Global Samples (%)", 
            ylabel="Feature Count", title="Rule 1: Global MV Distribution"
        )
        ax.legend(frameon=True, shadow=True)
        plt.close(fig=fig)
        return fig

    def plot_mv_group(self) -> Optional[Figure]:
        """Rule 2: Plot histograms of MV ratios across biological groups."""
        if self.bg_col not in self.fltr.columns.names:
            return None
            
        tol = self.attrs.get("mv_group_tol", 0.5) * 100
        mask_act = self.fltr.columns.get_level_values(
            level=self.st_col) == self.act_lbl
        
        if not mask_act.any():
            return None

        act_df = self.fltr.loc[:, mask_act]
        bg_labels = act_df.columns.get_level_values(level=self.bg_col)
        mv_per_group = act_df.isna().groupby(by=bg_labels, axis=1).mean() * 100
        
        n_groups = mv_per_group.shape[1]
        fig, axes = plt.subplots(
            nrows=1, ncols=n_groups, figsize=(4 * n_groups, 4), sharey=True
        )
        if n_groups == 1:
            axes = [axes]

        for idx, group_name in enumerate(iterable=mv_per_group.columns):
            ax = axes[idx]
            df = pd.DataFrame(data={
                "MV": mv_per_group[group_name], 
                "Pass": mv_per_group[group_name] < tol
            })
            
            sns.histplot(
                data=df, x="MV", hue="Pass", hue_order=[True, False],
                palette=self.pal, multiple="stack", edgecolor="k", 
                bins=np.arange(start=0, stop=105, step=5), ax=ax, legend=False
            )
            ax.axvline(x=tol, color="k", linestyle="--", linewidth=1.5)
            
            ylabel = "Feature Count" if idx == 0 else ""
            self._apply_standard_format(
                ax=ax, xlabel=f"MV Ratio in '{group_name}' (%)", 
                ylabel=ylabel, title_fontsize=12
            )

        plt.suptitle(
            t="Rule 2: Bio Group MV Distribution", fontsize=14, weight="bold"
        )
        plt.tight_layout()
        plt.close(fig=fig)
        return fig

    def plot_mv_qc(self) -> Optional[Figure]:
        """Rule 3: Plot histogram of QC MV ratios."""
        if self.fltr._qc.empty:
            return None

        tol = self.attrs.get("mv_qc_tol", 0.8) * 100
        df = pd.DataFrame(data={
            "MV": self.fltr._qc.isna().mean(axis=1) * 100, 
            "Pass": self.fltr.mask_mv_qc
        })

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
        sns.histplot(
            data=df, x="MV", hue="Pass", hue_order=[True, False],
            palette=self.pal, multiple="stack", edgecolor="k", 
            bins=np.arange(start=0, stop=105, step=5), ax=ax
        )
        
        ax.axvline(
            x=tol, color="k", linestyle="--", linewidth=1.5, 
            label=f"Threshold ({tol:.0f}%)"
        )
        self._apply_standard_format(
            ax=ax, xlabel="Missing Value Ratio in Pooled QC (%)", 
            ylabel="Feature Count", title="Rule 3: Pooled QC MV Distribution"
        )
        ax.legend(frameon=True, shadow=True)
        plt.close(fig=fig)
        return fig

    def plot_rsd_qc(self) -> Optional[Figure]:
        """Rule 4: Plot histogram of QC RSD."""
        if self.fltr._qc.empty:
            return None

        tol = self.attrs.get("rsd_qc_tol", 0.7) * 100
        qc_mean = self.fltr._qc.mean(axis=1)
        qc_std = self.fltr._qc.std(axis=1, ddof=1)
        qc_rsd = (qc_std / qc_mean * 100).fillna(value=np.nan)
        
        df = pd.DataFrame(data={
            "RSD": qc_rsd, "Pass": self.fltr.mask_rsd_qc
        }).dropna()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
        sns.histplot(
            data=df, x="RSD", hue="Pass", hue_order=[True, False],
            palette=self.pal, multiple="stack", edgecolor="k", ax=ax
        )
        
        ax.axvline(
            x=tol, color="k", linestyle="--", linewidth=1.5, 
            label=f"Threshold ({tol:.0f}%)"
        )
        self._apply_standard_format(
            ax=ax, xlabel="RSD in Pooled QC (%)", ylabel="Feature Count", 
            title="Rule 4: QC RSD Distribution"
        )
        ax.legend(frameon=True, shadow=True)
        plt.close(fig=fig)
        return fig

    def plot_qc_blank_ratio(self) -> Optional[Figure]:
        """Rule 5: Plot histogram of log2 QC/Blank ratios with safe features."""
        mask_blk = self.fltr.columns.get_level_values(level=self.st_col) == self.blk_lbl
        
        if not mask_blk.any() or self.fltr._qc.empty:
            return None

        blank_mean = self.fltr.loc[:, mask_blk].mean(axis=1).fillna(value=0.0)
        qc_mean = self.fltr._qc.mean(axis=1).fillna(value=0.0)

        mask_inf = (blank_mean == 0) & (qc_mean > 0)
        mask_valid = (blank_mean > 0) & (qc_mean > 0)
        
        tol = self.attrs.get("qc_blank_ratio", 5.0)
        log2_tol = np.log2(tol)

        df = pd.DataFrame(index=self.fltr.index)
        df["Log2_Ratio"], df["Pass"] = np.nan, False

        if mask_valid.any():
            ratio_valid = qc_mean[mask_valid] / blank_mean[mask_valid]
            df.loc[mask_valid, "Log2_Ratio"] = np.log2(ratio_valid)
            df.loc[mask_valid, "Pass"] = df.loc[mask_valid, "Log2_Ratio"] >= log2_tol
            max_log2 = np.ceil(df["Log2_Ratio"].max())
        else:
            max_log2 = 0.0

        if mask_inf.any():
            inf_val = max_log2 + 2.0
            df.loc[mask_inf, "Log2_Ratio"] = inf_val
            df.loc[mask_inf, "Pass"] = True
        
        df_plot = df.dropna(subset=["Log2_Ratio"])
        if df_plot.empty:
            return None

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
        sns.histplot(
            data=df_plot, x="Log2_Ratio", hue="Pass", hue_order=[True, False],
            palette=self.pal, multiple="stack", edgecolor="k", ax=ax
        )
        
        ax.axvline(
            x=log2_tol, color="k", linestyle="--", linewidth=1.5, 
            label=f"Threshold ({tol:.1f})"
        )
        
        if mask_inf.any():
            ticks = ax.get_xticks()
            new_ticks = [t for t in ticks if t < max_log2 + 1.0] + [inf_val]
            new_labels = [f"{t:g}" for t in new_ticks[:-1]] + ["Blank=0"]
            ax.set_xticks(ticks=new_ticks)
            ax.set_xticklabels(labels=new_labels)
        
        self._apply_standard_format(
            ax=ax, xlabel="Log2 (QC / Blank Mean Ratio)", 
            ylabel="Feature Count", title="Rule 5: QC vs Blank Intensity Ratio"
        )
        ax.legend(frameon=True, shadow=True)
        plt.close(fig=fig)
        return fig