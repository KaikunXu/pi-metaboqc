"""
Purpose of script: Data quality assessment for MetaboInt.
"""

import os
import re
import copy
import warnings
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pca import pca
from loguru import logger

from . import io_utils as iu
from . import plot_utils as pu
from . import core_classes
from . import visualizer_classes

warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=RuntimeWarning)


class MetaboIntQA(core_classes.MetaboInt):
    """Data quality assessment computational class for metabolomics.

    This class handles statistical calculations, PCA modeling, and outlier
    detection. Visualization logic is separated to MetaboVisualizerQA.
    """

    _metadata: List[str] = ["attrs"]

    def __init__(
        self,
        *args: Any,
        pipeline_params: Optional[Dict[str, Any]] = None,
        mode: str = "POS",
        corr_method: str = "spearman",
        boundary: str = "IQR",
        mask: bool = True,
        stat_outlier: str = "All",
        **kwargs: Any
    ) -> None:
        """Initialize the data quality assessment class.

        Args:
            *args: Variable length arguments passed to pandas DataFrame.
            pipeline_params: Configuration dictionary for the pipeline.
            mode: MS Polarity ("POS" or "NEG").
            corr_method: Correlation method for QC samples.
            boundary: Strategy to calculate control boundaries (e.g., IQR).
            mask: Whether to apply an upper triangle mask to corr matrix.
            stat_outlier: Specific statistical outlier detection scope.
            **kwargs: Extra arguments passed to pandas DataFrame.
        """
        super().__init__(*args, pipeline_params=pipeline_params, **kwargs)

        qa_configs: Dict[str, Any] = {
            "mode": mode,
            "corr_method": corr_method,
            "boundary": boundary,
            "mask": mask,
            "stat_outlier": stat_outlier
        }

        if pipeline_params and "MetaboIntQA" in pipeline_params:
            qa_configs.update(pipeline_params["MetaboIntQA"])

        self.attrs.update(qa_configs)

    @property
    def _constructor(self) -> type:
        """Override constructor to return MetaboIntQA."""
        return MetaboIntQA

    def __finalize__(
        self, other: Any, method: Optional[str] = None, **kwargs: Any
    ) -> "MetaboIntQA":
        """Explicitly preserve custom attributes during pandas operations."""
        super().__finalize__(other, method=method, **kwargs)
        if hasattr(other, "attrs"):
            self.attrs = copy.deepcopy(other.attrs)
        return self

    @cached_property
    def qc_corr_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix based on intensity of QC samples."""
        method = self.attrs.get("corr_method", "spearman")
        return self._qc.corr(method=method)

    @cached_property
    def qc_corr_mask(self) -> np.ndarray:
        """Generate a boolean mask for the correlation matrix."""
        mat_shape = self.qc_corr_matrix.shape
        if self.attrs.get("mask", True):
            return np.triu(np.ones(mat_shape), 1).astype(bool)
        return np.zeros(mat_shape).astype(bool)

    def _extract_features_and_labels(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract features and labels for PCA modeling.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing log10 
            transformed features and corresponding sample labels.
        """
        act_lbl = self.attrs["sample_dict"]["Actual sample"]
        qc_lbl = self.attrs["sample_dict"]["QC sample"]
        
        valid_samples = self.columns.get_level_values(
            self.attrs["sample_type"]
        ).isin([act_lbl, qc_lbl])
        
        met_df = self.loc[:, valid_samples].transpose()
        met_df = np.log10(met_df.replace({0: np.nan}))
        met_df = met_df.fillna(met_df.min().min())
        
        labels = met_df.index.to_frame().reset_index(drop=True)
        feat_cols = list(set(met_df.index.names) - {self.attrs["sample_name"]})
        features = met_df.reset_index(feat_cols, drop=True)
        
        return features, labels

    @cached_property
    def pca_results(self) -> Dict[str, Any]:
        """Execute PCA based on the preprocessed intensity matrix.

        Returns:
            Dict[str, Any]: Dictionary containing PCA scatter coordinates,
            explained variance, and detected statistical outliers.
        """
        features, labels = self._extract_features_and_labels()
        multi_idx = pd.MultiIndex.from_frame(labels)
        
        scaler = StandardScaler()
        scaled_feat = pd.DataFrame(
            scaler.fit_transform(features),
            index=features.index, columns=features.columns
        )
        
        with iu.HiddenPrints():
            model = pca(n_components=2)
            res = model.fit_transform(scaled_feat)

        pca_scatter = res["PC"].set_axis(multi_idx, axis=0)
        pca_var = pd.Series(
            res["variance_ratio"], index=pca_scatter.columns
        ).rename("Variance")

        # Organize outliers
        raw_outliers = res["outliers"].set_axis(multi_idx, axis=0)
        spe_outliers = raw_outliers.loc[:, ["y_score_spe", "y_bool_spe"]].copy()
        spe_outliers.columns = ["SPE-DModX", "Outliers (SPE-DModX)"]
        
        ht2_outliers = raw_outliers.loc[:, [
            "p_raw", "y_proba", "y_score", "y_bool"
        ]].copy()
        ht2_outliers.columns = [
            "Raw P-value", "Adj P-value", "Hotelling T2 Score", "Outliers (HT2)"
        ]

        outliers = pd.concat(
            [spe_outliers, ht2_outliers], axis=1, keys=["SPE-DModX", "HT2"]
        )
        outliers = outliers.rename_axis(columns=["Method", "Parameter"])
        
        stat_flags = outliers.loc[:, [
            ("HT2", "Outliers (HT2)"), 
            ("SPE-DModX", "Outliers (SPE-DModX)")
        ]]
        outlier_pct = stat_flags.sum(axis=1) / stat_flags.shape[1]
        
        outliers[("Statistics Outliers", "Outliers (Any)")] = outlier_pct != 0
        outliers[("Statistics Outliers", "Outliers (All)")] = outlier_pct == 1

        return {
            "pca_scatter": pca_scatter,
            "pca_variance": pca_var,
            "outliers": outliers
        }

    def calculate_boundaries(
        self, x: np.ndarray, boundary_type: str = "IQR"
    ) -> Tuple[float, float, float]:
        """Calculate statistical boundaries of a 1-dimensional array.

        Args:
            x: Input numpy array.
            boundary_type: Method to calculate boundaries ('IQR' or 'sigma').

        Returns:
            Tuple[float, float, float]: Central line, lower limit, upper limit.
        """
        if boundary_type in ("mean-std", "sigma"):
            solid = float(np.nanmean(x))
            std_val = float(np.nanstd(x, ddof=1))
            return solid, solid - 3 * std_val, solid + 3 * std_val
            
        elif boundary_type == "IQR":
            solid = float(np.nanmedian(x))
            q1 = float(np.nanquantile(x, 0.25))
            q3 = float(np.nanquantile(x, 0.75))
            iqr = q3 - q1
            return solid, q1 - 1.5 * iqr, q3 + 1.5 * iqr
            
        return 0.0, 0.0, 0.0

    @iu._exe_time
    def execute_qa(self, output_dir: str) -> None:
        """Execute the entire QA workflow, save tables, and render plots.

        Args:
            output_dir: Target directory path for reports.
        """
        iu._check_dir_exists(dir_path=output_dir, handle="makedirs")
        mode = self.attrs.get("mode", "POS")

        # 1. Save Tabular Results
        scatter_path = os.path.join(output_dir, f"PCA_Scatter_{mode}.csv")
        self.pca_results["pca_scatter"].to_csv(
            scatter_path, encoding="utf-8-sig", na_rep="NA"
        )
        
        outliers_path = os.path.join(output_dir, f"PCA_Outliers_{mode}.csv")
        self.pca_results["outliers"].to_csv(
            outliers_path, encoding="utf-8-sig", na_rep="NA"
        )

        # 2. Delegate to Visualizer
        vis = MetaboVisualizerQA(self)
        
        hm_fig = vis.plot_qc_correlation_heatmap(value_max=1.0)
        hm_fig.savefig(
            os.path.join(output_dir, f"QC_Corr_Heatmap_{mode}.pdf"),
            bbox_inches="tight", dpi=300
        )

        pca_fig = vis.plot_pca_scatter()
        pca_fig.savefig(
            os.path.join(output_dir, f"QC_AS_PCA_Scatter_{mode}.pdf"),
            bbox_inches="tight", dpi=300
        )

        stat_fig = vis.plot_statistical_outliers()
        stat_fig.savefig(
            os.path.join(output_dir, f"Outlier_Barplot_{mode}.pdf"),
            bbox_inches="tight", dpi=300
        )

        rsd_fig = vis.plot_qc_rsd_distribution()
        rsd_fig.savefig(
            os.path.join(output_dir, f"QC_RSD_Barplot_{mode}.pdf"),
            bbox_inches="tight", dpi=300
        )

        if len(self.valid_is) > 0:
            sh_fig = vis.plot_shewhart_control_chart()
            sh_fig.savefig(
                os.path.join(output_dir, f"Shewhart_IS_{mode}.pdf"),
                bbox_inches="tight", dpi=300
            )


class MetaboVisualizerQA(visualizer_classes.BaseMetaboVisualizer):
    """Visualization suite for metabolomics data quality assessment."""

    def __init__(self, qa_obj: MetaboIntQA) -> None:
        """Initialize with a computed MetaboIntQA object.

        Args:
            qa_obj: A MetaboIntQA instance with pre-computed statistics.
        """
        self.qa = qa_obj
        self.attrs = qa_obj.attrs

    def _format_heatmap_ticks(
        self, hm: mpl.axes.Axes, tick_color_dict: Dict[str, str]
    ) -> None:
        """Format labels and colors for heatmap ticks."""
        def rename_tick(label_text: str) -> str:
            parts = re.split("-", label_text)
            if len(parts) > 4:
                return "-".join([parts[0]] + parts[4:])
            return label_text

        hm.set_xticklabels([rename_tick(e._text) for e in hm.get_xticklabels()])
        hm.set_yticklabels([rename_tick(e._text) for e in hm.get_yticklabels()])

        for ax_labels in (hm.get_xticklabels(), hm.get_yticklabels()):
            for label in ax_labels:
                for batch, color in tick_color_dict.items():
                    if re.match(pattern=f"^{batch}", string=label._text):
                        label.set_color(color)
                        break

    def plot_qc_correlation_heatmap(
        self, value_max: float = 1.0, title: str = "QC Samples Correlation"
    ) -> plt.Figure:
        """Plot correlation matrix heatmap of QC samples."""
        batch_col = self.attrs["batch"]
        batch_list = self.qa.columns.get_level_values(batch_col).unique()
        
        custom_cmap = pu.custom_linear_cmap(["white", "tab:red"], 100)
        tick_colors = pu.extract_linear_cmap(
            custom_cmap, 0.5, 1.0, len(batch_list)
        )
        tick_color_dict = dict(zip(batch_list, tick_colors))
        color_map = pu.custom_linear_cmap(["white", "tab:red"], 100)

        n_samples = self.qa.qc_corr_matrix.shape[0]
        fig, ax = plt.subplots(1, 1, figsize=(n_samples * 0.2, n_samples * 0.2))
        
        method_title = self.attrs.get("corr_method", "spearman").title()
        
        hm = sns.heatmap(
            self.qa.qc_corr_matrix,
            mask=self.qa.qc_corr_mask,
            vmin=0.85, vmax=value_max,
            cmap=color_map, annot=False,
            linewidths=0.25, linecolor="white", square=True,
            cbar_kws={
                "label": f"{method_title} Correlation",
                "location": "right", "shrink": 0.65, "pad": 0.015,
                "aspect": (n_samples) / 15 + 30, "drawedges": False
            },
            ax=ax
        )
        
        self._apply_standard_format(
            ax=ax, title="QC Samples Correlation", 
            title_fontsize=16, label_fontsize=6, tick_fontsize=6
        )
        self._format_heatmap_ticks(hm=hm, tick_color_dict=tick_color_dict)
        plt.close(fig=fig)
        return fig

    def plot_pca_scatter(
        self, x: str = "PC1", y: str = "PC2", draw_ce: bool = True
    ) -> plt.Figure:
        """Plot PCA scatter plot with confidence ellipses."""
        pca_res = self.qa.pca_results
        pca_var = pca_res["pca_variance"]
        pca_df = pca_res["pca_scatter"].reset_index()
        
        st_col = self.attrs["sample_type"]
        bt_col = self.attrs["batch"]
        qc_lbl = self.attrs["sample_dict"]["QC sample"]
        act_lbl = self.attrs["sample_dict"]["Actual sample"]

        pca_df[st_col] = pca_df[st_col].astype("category")
        pca_df = pca_df.sort_values(by=st_col, ascending=False)
        
        pal_dict = {qc_lbl: "tab:red", act_lbl: "tab:gray"}
        
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.despine(ax=ax)
        
        sns.scatterplot(
            data=pca_df, x=x, y=y, hue=st_col, style=bt_col,
            s=50, edgecolor="k", palette=pal_dict, linewidth=0.5,
            ax=ax, hue_order=[qc_lbl, act_lbl]
        )
        
        if draw_ce:
            for group in (self.qc_lbl, self.act_lbl):
                sub_df = pca_df[pca_df[self.st_col] == group]
                pu.confidence_ellipse(
                    x=sub_df[x], y=sub_df[y], ax=ax, n_std=3.0, alpha=0.1,
                    facecolor=self.pal[group], edgecolor=self.pal[group]
                )
                
        self._apply_standard_format(
            ax=ax, xlabel=f"{x} ({100 * pca_var.loc[x]:.1f}%)", 
            ylabel=f"{y} ({100 * pca_var.loc[y]:.1f}%)"
        )
        ax.legend().remove()
        plt.close(fig=fig)
        return fig

    def plot_statistical_outliers(self) -> plt.Figure:
        """Plot outlier detection results via bar charts."""
        outliers = self.qa.pca_results["outliers"]
        act_lbl = self.attrs["sample_dict"]["Actual sample"]
        st_col = self.attrs["sample_type"]
        
        outliers = outliers[outliers.index.get_level_values(st_col) == act_lbl]
        
        idx_df = outliers.index.to_frame()
        new_idx = idx_df[self.attrs["batch"]] + "-" + \
                  idx_df[self.attrs["sample_name"]]
        outliers.index = new_idx
        outliers = outliers.rename_axis(index=["Sample ID"])
        
        stat_param = self.attrs.get("stat_outlier", "All")
        target_col = ("Statistics Outliers", f"Outliers ({stat_param})")
        outlier_samples = outliers[outliers[target_col] != 0].index.tolist()

        fig, (ax1, ax2) = plt.subplots(
            nrows=2, ncols=1, figsize=(outliers.shape[0] * 0.1 + 1, 8), 
            sharex=True
        )
        sns.despine(fig=fig)

        pal = {True: "tab:red", False: "tab:gray"}
        
        sns.barplot(
            ax=ax1, data=outliers["SPE-DModX"].reset_index(),
            x="Sample ID", y="SPE-DModX", edgecolor="k", linewidth=0.25,
            hue="Outliers (SPE-DModX)", palette=pal
        )
        sns.barplot(
            ax=ax2, data=outliers["HT2"].reset_index(),
            x="Sample ID", y="Hotelling T2 Score", edgecolor="k",
            linewidth=0.25, hue="Outliers (HT2)", palette=pal
        )
        
        pu.change_axis_rotation(ax=ax2, rotation=90, axis="x")
        for ax in (ax1, ax2):
            self._apply_standard_format(
                ax=ax, title_fontsize=12, label_fontsize=12, tick_fontsize=6)
            ax.legend(ncol=2, frameon=True, shadow=True, fontsize=10)
            
        for xlabel in ax2.get_xticklabels():
            if xlabel._text in outlier_samples:
                xlabel.set_color(c="tab:red")
                
        plt.tight_layout()
        plt.close(fig=fig)
        return fig

    def plot_qc_rsd_distribution(self) -> plt.Figure:
        """Plot the distribution of QC relative standard deviations."""
        qc_data = self.qa._qc
        rsd = (qc_data.std(axis=1, ddof=1) / qc_data.mean(axis=1)).rename("RSD")
        
        bins = [-np.inf, 0.1, 0.2, 0.3, np.inf]
        labels = ["0-10%", "10-20%", "20-30%", ">30%"]
        rsd_range = pd.cut(x=rsd, bins=bins, labels=labels, right=False)
        rsd_df = rsd_range.value_counts().rename("Counts").sort_index(
            ).reset_index()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        sns.barplot(
            data=rsd_df, x="RSD", y="Counts", width=0.6, edgecolor="k", 
            color="tab:gray", ax=ax)
        
        for i, text in enumerate(iterable=ax.get_xticklabels()):
            if text._text != ">30%": ax.patches[i].set_facecolor(c="tab:red")
                
        pu.show_values_on_bars(
            axs=ax, show_percentage=True, fontsize=9, position="outside", 
            value_format="{:.0f}")
        self._apply_standard_format(ax=ax)
        plt.close(fig=fig)
        return fig

    def plot_shewhart_control_chart(self) -> Optional[plt.Figure]:
        """Plot Shewhart control charts for internal standards."""
        if not self.qa.valid_is: return None

        plot_data = self.qa.int_order_info(feat_type="IS").reset_index()
        plot_data[self.st_col] = plot_data[self.st_col].astype(dtype="category")
        plot_data[self.bat_col] = plot_data[self.bat_col].astype(dtype="category")
        plot_data = plot_data.sort_values(by=self.st_col, ascending=False)
        
        ncols = 2
        nrows = int(np.ceil(a=len(self.qa.valid_is) / ncols))
        fig = plt.figure(figsize=(7.5 * ncols, 3 * nrows), layout="constrained")
        bound_type = self.attrs.get("boundary", "IQR")
        
        for n, feat in enumerate(iterable=self.qa.valid_is):
            ax = plt.subplot(nrows, ncols, n + 1)
            sns.scatterplot(
                ax=ax, data=plot_data, x=self.io_col, y=feat, s=40, edgecolor="k", 
                linewidth=0.5, style=self.bat_col, palette=self.pal,
                hue=self.st_col, hue_order=[self.qc_lbl, self.act_lbl]
            )

            int_data = self.qa.int_order_info(feat_type="IS")[feat]
            solid, lower, upper = self.qa.calculate_boundaries(
                x=int_data, boundary_type=bound_type)
            
            ax.axhline(y=solid, color="k", linestyle="-", linewidth=1.5)
            ax.axhline(y=lower, color="k", linestyle="--", linewidth=1.5)
            ax.axhline(y=upper, color="k", linestyle="--", linewidth=1.5)
            
            self._apply_standard_format(
                ax=ax, xlabel=self.io_col, ylabel=feat, tick_fontsize=11,
                title_fontsize=13)
            pu.change_axis_format(
                ax=ax, axis_format="scientific notation", axis="y")

            if n == len(self.qa.valid_is) - 1:
                self._format_complex_legend(fig=fig, ax=ax)
            elif ax.get_legend():
                ax.legend().remove()
                        
        plt.suptitle(
            t="Shewhart Control Chart of IS", fontsize=14, weight="bold")
        plt.close(fig=fig)
        return fig