"""Module for invalid feature/sample filtering and its visualization."""

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


class MetaboIntFLTR(core_classes.MetaboInt):
    """Filtering engine subclassing MetaboInt for metabolomics datasets.

    Attributes:
        params: Pipeline configuration parameters.
        feature_counts: Dictionary tracking the number of features.
        stats: Dictionary storing intermediate statistics for visualization.
    """

    # Declare metadata to ensure attributes survive pandas operations
    _metadata = ["feature_counts", "stats", "params"]

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
        self.feature_counts: Dict[str, int] = {"raw": self.shape[0]}

        self.stats: Dict[str, Any] = {
            "mv_group_df": pd.DataFrame(),
            "blank_mean": pd.Series(dtype=float),
            "qc_mean": pd.Series(dtype=float),
            "qc_rsd_all": pd.Series(dtype=float)
        }

    @property
    def _constructor(self) -> type:
        """Override constructor to return the subclass type."""
        return MetaboIntFLTR

    def __finalize__(
        self, other: Any, method: Optional[str] = None, **kwargs: Any
    ) -> "MetaboIntFLTR":
        """Explicitly deepcopy custom attributes during pandas operations."""
        self = super().__finalize__(other, method=method, **kwargs)
        for name in self._metadata:
            if hasattr(other, name):
                setattr(self, name, copy.deepcopy(getattr(other, name)))
        return self

    def execute_mv_fltr(self) -> "MetaboIntFLTR":
        """Execute Stage-1 missing value filter with group validation.

        Returns:
            MetaboIntFLTR: The missing value filtered dataset.
        """
        grp_col: str = self.params.get("MetaboInt", {}).get(
            "bio_group", "Group"
        )
        qc_col: str = self.params.get("MetaboInt", {}).get(
            "sample_type", "Type"
        )
        qc_lbl: str = self.params.get("MetaboInt", {}).get("qc_label", "QC")
        mv_tol: float = self.params.get("Filter", {}).get("mv_tol", 0.8)

        valid_bio_groups: List[str] = []

        if grp_col in self.columns.names:
            raw_groups = self.columns.get_level_values(grp_col).unique()
            invalid_strs = {
                "unknown", "na", "n/a", "nan", "none", "null", "",
                "unassigned", "blank", "blk", "is", "solvent", "wash",
                "sst", "pool"
            }

            for g in raw_groups:
                if pd.isna(g):
                    continue
                g_str = str(g).strip().lower()
                if g_str in invalid_strs or g_str == str(qc_lbl).lower():
                    continue
                valid_bio_groups.append(g)

        if valid_bio_groups:
            na_rate = self.isna().groupby(level=grp_col, axis=1).mean()
            self.stats["mv_group_df"] = na_rate[valid_bio_groups]
            pass_mask = (na_rate[valid_bio_groups] <= mv_tol).any(axis=1)
        else:
            if qc_col in self.columns.names:
                qc_mask = self.columns.get_level_values(qc_col) == qc_lbl
            else:
                qc_mask = np.array([False] * self.shape[1])

            if qc_mask.any():
                df_qc = self.loc[:, qc_mask]
                pass_mask = df_qc.isna().mean(axis=1) <= mv_tol
            else:
                pass_mask = self.isna().mean(axis=1) <= mv_tol

        valid_idx = self.index[pass_mask]
        df_final = self.loc[valid_idx].copy()
        df_final.feature_counts["post_stage1"] = df_final.shape[0]

        return df_final

    def execute_quality_fltr(
        self, idx_mar: pd.Index, idx_mnar: pd.Index
    ) -> "MetaboIntFLTR":
        """Execute Stage-2 quality filter (Blank Ratio & QC RSD).

        Args:
            idx_mar: Features classified as Missing At Random.
            idx_mnar: Features classified as Missing Not At Random.

        Returns:
            MetaboIntFLTR: Quality filtered dataset.
        """
        qc_col: str = self.params.get("MetaboInt", {}).get(
            "sample_type", "Type"
        )
        qc_lbl: str = self.params.get("MetaboInt", {}).get("qc_label", "QC")
        blk_lbl: str = self.params.get("MetaboInt", {}).get(
            "blank_label", "Blank"
        )

        rsd_tol: float = self.params.get("Filter", {}).get("rsd_tol", 0.3)
        blk_tol: float = self.params.get("Filter", {}).get(
            "blank_ratio_tol", 0.3
        )

        qc_mask = self.columns.get_level_values(qc_col) == qc_lbl
        blk_mask = self.columns.get_level_values(qc_col) == blk_lbl

        df_qc = self.loc[:, qc_mask]
        df_blk = self.loc[:, blk_mask]
        current_idx = self.index

        if blk_mask.any() and qc_mask.any():
            qc_mean = df_qc.mean(axis=1)
            blk_mean = df_blk.mean(axis=1)
            self.stats["qc_mean"] = qc_mean
            self.stats["blank_mean"] = blk_mean

            qc_mean_safe = qc_mean.replace(0, np.finfo(float).eps)
            blank_ratio = blk_mean / qc_mean_safe

            pass_blk = blank_ratio[blank_ratio <= blk_tol].index
            current_idx = current_idx.intersection(pass_blk)

        if qc_mask.any():
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
            final_idx = current_idx

        df_final = self.loc[final_idx].copy()
        df_final.feature_counts["post_stage2"] = df_final.shape[0]

        return df_final


class MetaboVisualizerFLTR(visualizer_classes.BaseMetaboVisualizer):
    """Visualization suite inheriting from BaseMetaboVisualizer.

    Attributes:
        engine: The configured MetaboIntFLTR subclass instance.
    """

    def __init__(self, engine: MetaboIntFLTR) -> None:
        """Initialize with the filtering engine to access its stats."""
        # Pass the extracted parameters to the base class
        super().__init__(pipeline_params=engine.params)
        self.engine = engine
        self.tol_mv: float = engine.params.get("Filter", {}).get(
            "mv_tol", 0.8
        )
        self.tol_rsd: float = engine.params.get("Filter", {}).get(
            "rsd_tol", 0.3
        )
        self.tol_blk: float = engine.params.get("Filter", {}).get(
            "blank_ratio_tol", 0.3
        )

    def plot_filtering_summary(self) -> Figure:
        """Plot a bar chart showing feature attrition across stages."""
        fig, ax = plt.subplots(figsize=(8, 5))
        counts: List[int] = [
            self.engine.feature_counts.get("raw", 0),
            self.engine.feature_counts.get("post_stage1", 0),
            self.engine.feature_counts.get("post_stage2", 0)
        ]
        labels: List[str] = [
            "Raw Data", "Post-MV (Stage 1)", "Post-Quality (Stage 2)"
        ]

        sns.barplot(x=labels, y=counts, ax=ax, palette="viridis")
        self._apply_standard_format(
            ax=ax, xlabel="Filtering Stages", ylabel="Number of Features",
            title="Feature Attrition Summary"
        )

        for i, v in enumerate(counts):
            ax.text(i, v + (max(counts) * 0.02), str(v), ha="center")

        plt.tight_layout()
        plt.close(fig)
        return fig

    def plot_mv_group(self) -> Optional[Figure]:
        """Plot histograms of MV ratios across valid biological groups."""
        mv_df: pd.DataFrame = self.engine.stats.get(
            "mv_group_df", pd.DataFrame()
        )
        if mv_df.empty:
            return None

        mv_df = mv_df * 100
        tol_pct: float = self.tol_mv * 100
        n_groups: int = mv_df.shape[1]

        fig, axes = plt.subplots(
            nrows=1, ncols=n_groups, figsize=(4 * n_groups, 4), sharey=True
        )
        if n_groups == 1:
            axes = [axes]

        for idx, group_name in enumerate(mv_df.columns):
            ax = axes[idx]
            df_plot = pd.DataFrame({
                "MV": mv_df[group_name],
                "Pass": mv_df[group_name] <= tol_pct
            })

            sns.histplot(
                data=df_plot, x="MV", hue="Pass", hue_order=[True, False],
                palette="Set2", multiple="stack", edgecolor="k",
                bins=np.arange(0, 105, 5), ax=ax, legend=(idx == 0)
            )
            ax.axvline(x=tol_pct, color="red", linestyle="--", linewidth=1.5)

            ylabel: str = "Feature Count" if idx == 0 else ""
            self._apply_standard_format(
                ax=ax, xlabel=f"MV Ratio in '{group_name}' (%)",
                ylabel=ylabel, title=f"Group: {group_name}"
            )

        plt.suptitle(
            "Stage 1: Missing Value Distribution by Group", weight="bold"
        )
        plt.tight_layout()
        plt.close(fig)
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
        df_plot["Pass"] = ratio <= self.tol_blk

        fig, ax = plt.subplots(figsize=(6, 6))
        sns.scatterplot(
            data=df_plot, x="Log2_QC", y="Log2_Blank", hue="Pass",
            palette={True: "tab:blue", False: "tab:red"},
            alpha=0.6, s=15, edgecolor=None, ax=ax
        )

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
        ]
        ax.plot(lims, lims, "k--", alpha=0.5, label="Ratio = 1.0")

        self._apply_standard_format(
            ax=ax, xlabel="Log2(Mean QC Intensity + 1)",
            ylabel="Log2(Mean Blank Intensity + 1)",
            title="Stage 2.1: Blank Contamination Check"
        )
        ax.legend(frameon=True)
        plt.tight_layout()
        plt.close(fig)
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
            rsd_mar, color="skyblue", label="MAR (Subject to RSD Filter)",
            ax=ax, kde=True
        )
        sns.histplot(
            rsd_mnar, color="salmon", label="MNAR (RSD Exempted)",
            ax=ax, kde=True
        )

        ax.axvline(
            self.tol_rsd, color="red", linestyle="--",
            label=f"Threshold ({self.tol_rsd})"
        )

        self._apply_standard_format(
            ax=ax, xlabel="Relative Standard Deviation (RSD)",
            ylabel="Feature Count",
            title="Stage 2.2: QC RSD & Biological Exemptions"
        )
        ax.legend()
        plt.tight_layout()
        plt.close(fig)
        return fig