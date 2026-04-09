"""
Purpose of script: Data normalization module for MetaboInt.
"""

import os
import copy
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# import matplotlib.lines as mlines

from scipy import stats
from scipy.optimize import minimize
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from . import core_classes
from . import visualizer_classes
from . import io_utils as iu
from . import plot_utils as pu


class MetaboIntNormalizer(core_classes.MetaboInt):
    """Normalization class for metabolomics intensity data.

    This class handles three categories of normalization exclusively on
    target samples. Blank samples are officially dropped from the dataset
    at this stage as their lifecycle (noise filtering) is complete.

    1. Sample-wise (Column) normalization: TIC, Median, and PQN.
    2. Feature-wise (Row) normalization: VSN, Auto scaling, and Pareto.
    3. Standalone Quantile normalization: NaN-compatible interpolation.

    Attributes:
        attrs (Dict[str, Any]): Dictionary storing custom metadata.
    """

    _metadata: List[str] = ["attrs"]

    def __init__(
        self,
        *args: Any,
        pipeline_params: Optional[Dict[str, Any]] = None,
        col_norm: str = "PQN",
        row_norm: str = "VSN",
        quantile_norm: bool = False,
        **kwargs: Any
    ) -> None:
        """Initialize MetaboIntNormalizer with hierarchical parameter loading.

        Args:
            *args: Variable length arguments passed to pandas DataFrame.
            pipeline_params: Global settings for the pipeline classes.
            col_norm: Column normalization method to apply.
            row_norm: Row normalization method to apply.
            quantile_norm: Whether to apply quantile normalization.
            **kwargs: Extra arguments passed to pandas DataFrame.
        """
        super().__init__(*args, pipeline_params=pipeline_params, **kwargs)

        norm_configs: Dict[str, Any] = {
            "col_norm": col_norm,
            "row_norm": row_norm,
            "quantile_norm": quantile_norm
        }

        if pipeline_params and "MetaboIntNormalizer" in pipeline_params:
            norm_configs.update(pipeline_params["MetaboIntNormalizer"])

        self.attrs.update(norm_configs)

    @property
    def _constructor(self) -> type:
        """Override pandas constructor to return current subclass type."""
        return MetaboIntNormalizer

    def __finalize__(
        self, other: Any, method: Optional[str] = None, **kwargs: Any
    ) -> "MetaboIntNormalizer":
        """Explicitly deepcopy custom attributes during object creation.

        Args:
            other: The original object from which to copy metadata.
            method: The pandas internal method name calling finalize.
            **kwargs: Additional keyword arguments.

        Returns:
            The newly created MetaboIntNormalizer instance.
        """
        super().__finalize__(other, method=method, **kwargs)
        if hasattr(other, "attrs"):
            self.attrs = copy.deepcopy(getattr(other, "attrs", {}))
        return self

    # ====================================================================
    # Data Transformation & Assessment Utilities
    # ====================================================================

    def get_log2_target(self) -> pd.DataFrame:
        """Calculate log2-transformed matrix excluding Blank samples.

        Returns:
            pd.DataFrame: Log2 matrix of targets with NaNs for 0 values.
        """
        blank_cols = self._blank.columns
        target_cols = self.columns.difference(blank_cols)
        df_target = self[target_cols].replace({0: np.nan})
        return np.log2(df_target)

    def calc_rle(self) -> pd.DataFrame:
        """Calculate Relative Log Expression (RLE) matrix.

        Returns:
            pd.DataFrame: RLE matrix with feature medians subtracted.
        """
        df_log = self.get_log2_target()
        feat_medians = df_log.median(axis="columns")
        return df_log.sub(feat_medians, axis="index")

    def calc_ma(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate A and M values exclusively for target samples.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Flattened valid A and M arrays.
        """
        df_log = self.get_log2_target()
        a_vals = df_log.mean(axis="columns")
        m_df = df_log.sub(a_vals, axis="index")

        a_flat = np.repeat(a_vals.values, m_df.shape[1])
        m_flat = m_df.values.flatten()

        valid = ~np.isnan(a_flat) & ~np.isnan(m_flat)
        return a_flat[valid], m_flat[valid]

    # ====================================================================
    # Category 1: Column (Sample) Dimension Normalization
    # ====================================================================

    def _col_norm_tic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Total Ion Current (TIC) normalization column-wise."""
        col_sums = df.sum(axis="index")
        return df.div(col_sums, axis="columns") * col_sums.median()

    def _col_norm_median(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Median normalization column-wise."""
        col_medians = df.median(axis="index")
        return df.div(col_medians, axis="columns") * col_medians.median()

    def _col_norm_pqn(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Probabilistic Quotient Normalization (PQN) column-wise."""
        df_safe = df.replace({0: np.nan})
        ref_spectrum = df_safe.median(axis="columns")
        quotients = df_safe.div(ref_spectrum, axis="index")
        median_quotients = quotients.median(axis="index")
        return df_safe.div(median_quotients, axis="columns").fillna(0)

    # ====================================================================
    # Category 2: Row (Feature) Dimension Normalization
    # ====================================================================

    def _estimate_vsn_params(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Estimate VSN params via mathematically exact Profile Likelihood."""
        rows, cols = data.shape

        def log_likelihood(params: np.ndarray) -> float:
            a_vec = params[:-1]
            b = params[-1]

            z = a_vec + b * data
            transformed = np.arcsinh(z)

            log_jacobian = np.log(b) - 0.5 * np.log1p(z**2)
            row_means = np.nanmean(transformed, axis=1, keepdims=True)
            residuals_sq = (transformed - row_means) ** 2

            valid_mask = ~np.isnan(residuals_sq)
            n_valid = np.sum(valid_mask)

            if n_valid == 0:
                return 1e10

            sigma_sq = np.nanmean(residuals_sq)
            if sigma_sq <= 1e-16:
                return 1e10

            ll = np.nansum(log_jacobian) - (n_valid / 2.0) * np.log(sigma_sq)
            return -ll

        a_init = np.zeros(cols)
        b_init = 1.0 / np.nanmedian(data)
        x0 = np.concatenate([a_init, [b_init]])
        bounds = [(None, None)] * cols + [(1e-12, None)]

        res = minimize(
            log_likelihood, x0=x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-9}
        )
        return res.x[:-1], res.x[-1]

    def _row_norm_vsn(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply pure VSN matching Bioconductor vsn2 behavior."""
        data_arr = df.to_numpy(dtype=np.float64)
        a_vec, b = self._estimate_vsn_params(data_arr)

        shift_constant = np.log2(2 * b)
        normed_arr = (
            np.arcsinh(a_vec + b * data_arr) / np.log(2)
        ) - shift_constant

        log2_data = np.log2(np.where(data_arr > 0, data_arr, np.nan))
        valid = ~np.isnan(log2_data) & ~np.isnan(normed_arr)

        pure_shift = 0.0
        if np.any(valid):
            y_val = normed_arr[valid]
            x_val = log2_data[valid]

            threshold = np.percentile(x_val, 50)
            high_mask = x_val > threshold

            pure_shift = np.median(x_val[high_mask] - y_val[high_mask])
            normed_arr += pure_shift

        self.attrs["vsn_offsets"] = a_vec.tolist()
        self.attrs["vsn_scale"] = float(b)
        self.attrs["vsn_shift"] = float(pure_shift)

        res_df = df.copy()
        res_df.iloc[:, :] = normed_arr
        return res_df

    def _row_norm_auto(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Auto Scaling (Z-score) row-wise."""
        row_means = df.mean(axis="columns")
        row_stds = df.std(axis="columns", ddof=1)
        return df.sub(row_means, axis="index").div(row_stds, axis="index")

    def _row_norm_pareto(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Pareto Scaling row-wise."""
        row_means = df.mean(axis="columns")
        row_stds = np.sqrt(df.std(axis="columns", ddof=1))
        return df.sub(row_means, axis="index").div(row_stds, axis="index")

    # ====================================================================
    # Category 3: Quantile Normalization (NaN Compatible)
    # ====================================================================

    def _quantile_norm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Quantile normalizes a DataFrame with missing value handling."""
        origin_arr = df.to_numpy(dtype=np.float64)
        rows, cols = origin_arr.shape

        sorted_arr = np.sort(origin_arr, axis=0)
        non_nas = rows - np.isnan(sorted_arr).sum(axis=0)

        row_means = np.zeros(rows, dtype=np.float64)
        x_target = np.linspace(0, 1, rows)

        for j in range(cols):
            non_na = non_nas[j]
            if non_na == 0:
                continue
            y = sorted_arr[:non_na, j]
            x = np.linspace(0, 1, non_na)
            row_means += np.interp(x_target, x, y)

        row_means /= float(cols)

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

    # ====================================================================
    # Core Execution Logic
    # ====================================================================

    def apply_normalization(self) -> "MetaboIntNormalizer":
        """Execute normalizations and drop Blanks permanently.

        Returns:
            MetaboIntNormalizer: Normalized object with targets only.
        """
        blank_cols = self._blank.columns
        target_cols = self.columns.difference(blank_cols)

        df_target = self[target_cols].copy()

        if df_target.empty:
            raise ValueError("No target samples available for normalization.")

        if self.attrs.get("quantile_norm", False):
            df_target = self._quantile_norm(df_target)
        else:
            col_norm = self.attrs.get("col_norm")
            if col_norm in ("TIC", "tic"):
                df_target = self._col_norm_tic(df_target)
            elif col_norm in ("Median", "median"):
                df_target = self._col_norm_median(df_target)
            elif col_norm in ("PQN", "pqn"):
                df_target = self._col_norm_pqn(df_target)

            row_norm = self.attrs.get("row_norm")
            if row_norm in ("VSN", "vsn"):
                df_target = self._row_norm_vsn(df_target)
            elif row_norm in ("Auto", "auto"):
                df_target = self._row_norm_auto(df_target)
            elif row_norm in ("Pareto", "pareto"):
                df_target = self._row_norm_pareto(df_target)

        return self._constructor(df_target).__finalize__(self)

    @iu._exe_time
    def execute_norm(self, output_dir: str) -> "MetaboIntNormalizer":
        """Execute normalization workflow and save outputs to disk.

        Args:
            output_dir: Directory path to save normalized data results.

        Returns:
            MetaboIntNormalizer: The normalized metabolomics data object.
        """
        iu._check_dir_exists(dir_path=output_dir, handle="makedirs")

        mode = self.attrs.get("mode", "POS")
        is_quantile = self.attrs.get("quantile_norm", False)

        blank_count = len(self._blank.columns)
        if blank_count > 0:
            logger.info(
                f"Permanently dropping {blank_count} Blank samples."
            )

        if is_quantile:
            suffix = f"Quantile_{mode}"
            logger.info("Applying Standalone Quantile Normalization.")
        else:
            cn = self.attrs.get("col_norm", "None")
            rn = self.attrs.get("row_norm", "None")
            suffix = f"Col_{cn}_Row_{rn}_{mode}"
            logger.info(f"Applying Normalization | Col: {cn} | Row: {rn}")

        normalized_data = self.apply_normalization()

        file_path = os.path.join(output_dir, f"Normalized_Data_{suffix}.csv")
        normalized_data.to_csv(
            path_or_buf=file_path,
            na_rep="NA",
            encoding="utf-8-sig"
        )

        vis = MetaboVisualizerNormalizer(
            raw_obj=self, norm_obj=normalized_data
        )

        fig_rle = vis.plot_rle_boxplot()
        vis.save_and_close_fig(
            fig=fig_rle, 
            file_path=os.path.join(output_dir, f"Norm_RLE_Boxplot_{suffix}.pdf")
        )

        fig_ma = vis.plot_ma_scatter()
        vis.save_and_close_fig(
            fig=fig_ma, 
            file_path=os.path.join(output_dir, f"Norm_MA_Scatter_{suffix}.pdf")
        )

        fig_kde = vis.plot_density_kde()
        vis.save_and_close_fig(
            fig=fig_kde, 
            file_path=os.path.join(output_dir, f"Norm_Density_KDE_{suffix}.pdf")
        )        

        return normalized_data


class MetaboVisualizerNormalizer(visualizer_classes.BaseMetaboVisualizer):
    """Visualization suite for data normalization evaluation."""

    def __init__(
        self, raw_obj: "MetaboIntNormalizer", norm_obj: "MetaboIntNormalizer"
    ) -> None:
        """Initialize with both raw and normalized datasets.

        Args:
            raw_obj: The unnormalized data object.
            norm_obj: The normalized data object.
        """
        super().__init__(metabo_obj=norm_obj)
        self.raw_obj = raw_obj
        self.norm_obj = norm_obj

    def plot_rle_boxplot(self) -> plt.Figure:
        """Plot pre/post RLE boxplot combinations for target samples.

        Returns:
            plt.Figure: The generated matplotlib figure.
        """
        rle_raw = self.raw_obj.calc_rle()
        rle_norm = self.norm_obj.calc_rle()

        def _melt_to_long(df_rle: pd.DataFrame, label: str) -> pd.DataFrame:
            df_t = df_rle.T
            df_flat = df_t.reset_index()
            id_cols = list(df_rle.columns.names)
            df_long = df_flat.melt(
                id_vars=id_cols, var_name="Feature", value_name="RLE"
            )
            df_long["Status"] = label
            return df_long

        df_raw_long = _melt_to_long(rle_raw, "Before")
        df_norm_long = _melt_to_long(rle_norm, "After")

        plot_df = pd.concat([df_raw_long, df_norm_long], ignore_index=True)
        plot_df = plot_df.dropna(subset=["RLE"])

        fig, axes = plt.subplots(
            nrows=1, ncols=2, figsize=(9, 4), sharey=True
        )

        qc_color = self.pal.get(self.qc_lbl, "tab:red")
        act_color = self.pal.get(self.act_lbl, "tab:gray")

        configs = [
            (axes[0], self.qc_lbl, "Pooled QC Samples", qc_color),
            (axes[1], self.act_lbl, "Measured Samples", act_color)
        ]

        statuses = ["Before", "After"]

        for ax, target_type, title, color in configs:
            sub_df = plot_df[plot_df[self.st_col] == target_type]
            curr_palette = {status: color for status in statuses}

            if sub_df.empty:
                self._apply_standard_format(ax=ax, title=f"{title} (No Data)")
                continue

            # Standard boxplot omitting outliers to fix Y-axis scale issues
            sns.boxplot(
                data=sub_df, x="Status", y="RLE", hue="Status",
                width=0.5, dodge=False, palette=curr_palette,
                showfliers=False, ax=ax, linewidth=1.5
            )

            if ax.legend_ is not None:
                ax.legend_.remove()

            y_label = "Relative Log Expression" if ax == axes[0] else ""
            self._apply_standard_format(
                ax=ax, title=title, xlabel="", ylabel=y_label
            )
        return fig

    def plot_ma_scatter(self) -> plt.Figure:
        """Plot pre/post MA scatter exclusively for target samples.

        Returns:
            plt.Figure: The generated matplotlib figure.
        """
        a_raw, m_raw = self.raw_obj.calc_ma()
        a_norm, m_norm = self.norm_obj.calc_ma()

        fig, axes = plt.subplots(
            nrows=1, ncols=2, figsize=(9, 4), sharey=True, sharex=True
        )

        custom_cmap = pu.custom_linear_cmap(["white", "tab:red"], 100)

        hb0 = axes[0].hexbin(
            x=a_raw, y=m_raw, gridsize=50,
            cmap=custom_cmap, mincnt=1, bins="log"
        )
        hb1 = axes[1].hexbin(
            x=a_norm, y=m_norm, gridsize=50,
            cmap=custom_cmap, mincnt=1, bins="log"
        )

        titles = ["Raw Data MA (Excl. Blanks)", "Normalized MA"]
        for ax, title, hb in zip(axes, titles, [hb0, hb1]):
            ax.axhline(0, color="k", linestyle="--", linewidth=1.5)
            self._apply_standard_format(
                ax=ax, title=title,
                xlabel="Average Log2 Intensity (A)",
                ylabel="Log2 Fold Change from Mean (M)"
            )
            cb = fig.colorbar(hb, ax=ax)
            cb.set_label("Log10(Count)")
            ax.autoscale()
        return fig

    def plot_density_kde(self) -> plt.Figure:
        """Plot KDE overlay, comparing pre/post normalization.

        Returns:
            plt.Figure: The generated matplotlib figure.
        """
        log_raw = self.raw_obj.get_log2_target()
        log_norm = self.norm_obj.get_log2_target()

        # Reverted to single subplot with (4, 4) figsize
        fig, ax = plt.subplots(figsize=(4, 4))

        # Plot raw data lines in red
        for col in log_raw.columns:
            data_to_plot = log_raw[col].dropna()
            if data_to_plot.nunique() > 1:
                sns.kdeplot(
                    data=data_to_plot, ax=ax,
                    color="tab:red", alpha=0.15, linewidth=0.5
                )

        # Plot normalized data lines in blue
        for col in log_norm.columns:
            data_to_plot = log_norm[col].dropna()
            if data_to_plot.nunique() > 1:
                sns.kdeplot(
                    data=data_to_plot, ax=ax,
                    color="tab:blue", alpha=0.15, linewidth=0.5
                )

        self._apply_standard_format(
            ax=ax, title="Density Overlay",
            xlabel="Log2 Intensity", ylabel="Density"
        )

        # legend_elements = [
        #     mlines.Line2D([0], [0], color="tab:red", lw=2, label="Before"),
        #     mlines.Line2D([0], [0], color="tab:blue", lw=2, label="After")
        # ]
        
        # # Place legend outside to prevent blocking curves in a compact plot
        # ax.legend(
        #     handles=legend_elements,
        #     loc="upper left",
        #     bbox_to_anchor=(1.05, 1.0),
        #     **self.LEGEND_KWARGS
        # )
        
        ax.plot([], [], color="tab:red", lw=2, label="Raw Data")
        ax.plot([], [], color="tab:blue", lw=2, label="Normalized")
        self._format_single_legend(fig=fig, ax=ax)
        
        return fig