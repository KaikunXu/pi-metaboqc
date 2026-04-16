# src/pimqc/normalization.py
"""
Purpose of script: Data normalization module for MetaboInt.
"""

import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.optimize import minimize
from loguru import logger

from . import core_classes
from . import visualizer_classes
from . import io_utils as iu
from . import plot_utils as pu


class MetaboIntNormalizer(core_classes.MetaboInt):
    """Normalization class for metabolomics intensity data.

    Handles Sample-wise (Column), Feature-wise (Row), and standalone 
    Quantile normalization. Blank samples are permanently dropped here.
    """

    _metadata = ["attrs"]

    def __init__(
        self,
        *args,
        pipeline_params=None,
        col_norm="PQN",
        row_norm="VSN",
        quantile_norm=False,
        **kwargs
    ):
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

        norm_configs = {
            "col_norm": col_norm,
            "row_norm": row_norm,
            "quantile_norm": quantile_norm
        }

        if pipeline_params and "MetaboIntNormalizer" in pipeline_params:
            norm_configs.update(pipeline_params["MetaboIntNormalizer"])

        self.attrs.update(norm_configs)

    @property
    def _constructor(self):
        """Override pandas constructor to return current subclass type."""
        return MetaboIntNormalizer

    def __finalize__(self, other, method=None, **kwargs):
        """Explicitly deepcopy custom attributes during object creation."""
        super().__finalize__(other, method=method, **kwargs)
        if hasattr(other, "attrs"):
            self.attrs = copy.deepcopy(getattr(other, "attrs", {}))
        return self

    # ====================================================================
    # Data Transformation & Matrix Extractors
    # ====================================================================

# ====================================================================
    # Data Transformation & Matrix Extractors (Static Methods)
    # ====================================================================

    @staticmethod
    def calc_log2_transform(df):
        """Perform log2 transformation after replacing zeros with NaNs."""
        # Ensure data is float and handle zeros to avoid -inf in log2
        return np.log2(df.astype(float).replace({0: np.nan}))

    @staticmethod
    def calc_rle_matrix(df_log):
        """Calculate Relative Log Expression (RLE) matrix."""
        feat_medians = df_log.median(axis="columns")
        return df_log.sub(feat_medians, axis="index")

    @staticmethod
    def calc_ma_arrays(df_log):
        """Calculate flattened A and M values for MA-plot visualization."""
        # A = average of log-intensities per feature
        a_vals = df_log.mean(axis="columns")
        # M = difference from the average
        m_df = df_log.sub(a_vals, axis="index")
        
        # Flatten and align for scattering
        a_flat = np.repeat(a_vals.values, m_df.shape[1])
        m_flat = m_df.values.flatten()
        
        # Drop NaN pairs for visualization safety
        valid = ~np.isnan(a_flat) & ~np.isnan(m_flat)
        return a_flat[valid], m_flat[valid]

    # ====================================================================
    # Category 1: Column (Sample) Dimension Normalization
    # ====================================================================
    @staticmethod
    def calc_tic_normalization(df):
        """Apply Total Ion Current (TIC) normalization column-wise."""
        col_sums = df.sum(axis="index")
        return df.div(col_sums, axis="columns") * col_sums.median()

    @staticmethod
    def calc_median_normalization(df):
        """Apply Median normalization column-wise."""
        col_medians = df.median(axis="index")
        return df.div(col_medians, axis="columns") * col_medians.median()
    
    @staticmethod
    def calc_pqn_normalization(df):
        """Apply Probabilistic Quotient Normalization (PQN) column-wise."""
        df_safe = df.replace({0: np.nan})
        ref_spectrum = df_safe.median(axis="columns")
        quotients = df_safe.div(ref_spectrum, axis="index")
        median_quotients = quotients.median(axis="index")
        return df_safe.div(median_quotients, axis="columns").fillna(0)

    # ====================================================================
    # Category 2: Row (Feature) Dimension Normalization
    # ====================================================================
    @staticmethod
    def _estimate_vsn_params(data):
        """Estimate VSN params via mathematically exact Profile Likelihood.

        Args:
            data: Numpy array of raw intensity data (Features x Samples).

        Returns:
            tuple: (a_vec offsets, b scale factor).
        """
        rows, cols = data.shape

        def log_likelihood(params):
            a_vec = params[:-1]
            b = params[-1]
            z = a_vec + b * data
            transformed = np.arcsinh(z)
            log_jacobian = np.log(b) - 0.5 * np.log1p(z**2)
            row_means = np.nanmean(transformed, axis=1, keepdims=True)
            residuals_sq = (transformed - row_means) ** 2
            n_valid = np.sum(~np.isnan(residuals_sq))
            if n_valid == 0: return 1e10
            sigma_sq = np.nanmean(residuals_sq)
            if sigma_sq <= 1e-16: return 1e10
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

    @staticmethod
    def calc_vsn_normalization(df):
        """Apply VSN matching Bioconductor vsn2 behavior.

        The transformation follows the inverse hyperbolic sine:
        $$h(x) = \text{arcsinh}(a + bx)$$

        Args:
            df: Input intensity DataFrame.

        Returns:
            tuple: (Normalized DataFrame, Dictionary of VSN parameters).
        """
        data_arr = df.to_numpy(dtype=np.float64)
        a_vec, b = MetaboIntNormalizer._estimate_vsn_params(data_arr)

        shift_constant = np.log2(2 * b)
        normed_arr = (
            np.arcsinh(a_vec + b * data_arr) / np.log(2)
        ) - shift_constant

        log2_data = np.log2(np.where(data_arr > 0, data_arr, np.nan))
        valid = ~np.isnan(log2_data) & ~np.isnan(normed_arr)

        pure_shift = 0.0
        if np.any(valid):
            y_val, x_val = normed_arr[valid], log2_data[valid]
            high_mask = x_val > np.percentile(x_val, 50)
            pure_shift = np.median(x_val[high_mask] - y_val[high_mask])
            normed_arr += pure_shift

        res_df = df.copy()
        res_df.iloc[:, :] = normed_arr
        
        vsn_meta = {
            "vsn_offsets": a_vec.tolist(),
            "vsn_scale": float(b),
            "vsn_shift": float(pure_shift)
        }
        return res_df, vsn_meta
    
    @staticmethod
    def calc_auto_scaling(df):
        """Apply Auto Scaling (Z-score) row-wise."""
        row_means = df.mean(axis="columns")
        row_stds = df.std(axis="columns", ddof=1)
        return df.sub(row_means, axis="index").div(row_stds, axis="index")
    
    @staticmethod
    def calc_pareto_scaling(df):
        """Apply Pareto Scaling row-wise."""
        row_means = df.mean(axis="columns")
        row_stds = np.sqrt(df.std(axis="columns", ddof=1))
        return df.sub(row_means, axis="index").div(row_stds, axis="index")

    # ====================================================================
    # Category 3: Quantile Normalization (NaN Compatible)
    # ====================================================================
    @staticmethod
    def calc_quantile_normalization(df):
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

    def apply_normalization(
        self, col_norm=None, row_norm=None, quantile_norm=None
    ):
        """Execute normalizations and return intermediate & final objects.

        This method coordinates the execution of static normalization 
        algorithms. It manages the transition from sample-wise (column) 
        to feature-wise (row) scaling.

        Args:
            col_norm: Override for column normalization method.
            row_norm: Override for row normalization method.
            quantile_norm: Override for quantile normalization flag.
        
        Returns:
            tuple: (Column-normalized object or None, Final-normalized object).
        """
        blank_cols = self._blank.columns
        target_cols = self.columns.difference(blank_cols)

        # Slice the data to work only on non-blank samples
        df_target = self[target_cols].copy()

        if df_target.empty:
            raise ValueError("No target samples available for normalization.")

        # Resolve parameters from arguments or object attributes
        c_norm = col_norm or self.attrs.get("col_norm")
        r_norm = row_norm or self.attrs.get("row_norm")
        is_qnt = quantile_norm
        if is_qnt is None:
            is_qnt = self.attrs.get("quantile_norm", False)

        if is_qnt:
            # Quantile normalization is a global non-linear transformation
            df_target = self.calc_quantile_normalization(df_target)
            obj_col = None
        else:
            # Stage 1: Column (Sample-wise) Normalization
            if c_norm in ("TIC", "tic"):
                df_target = self.calc_tic_normalization(df_target)
            elif c_norm in ("Median", "median"):
                df_target = self.calc_median_normalization(df_target)
            elif c_norm in ("PQN", "pqn"):
                df_target = self.calc_pqn_normalization(df_target)
            
            # Encapsulate intermediate column-normalized state
            df_col = df_target.copy()
            obj_col = self._constructor(df_col).__finalize__(self)

            # Stage 2: Row (Feature-wise) Scaling/Normalization
            if r_norm in ("VSN", "vsn"):
                # VSN static method returns (DataFrame, Parameter_Dict)
                df_target, vsn_meta = self.calc_vsn_normalization(df_target)
                self.attrs.update(vsn_meta)
            elif r_norm in ("Auto", "auto"):
                df_target = self.calc_auto_scaling(df_target)
            elif r_norm in ("Pareto", "pareto"):
                df_target = self.calc_pareto_scaling(df_target)

        # Encapsulate the final comprehensive normalized data
        obj_final = self._constructor(df_target).__finalize__(self)
        
        return obj_col, obj_final

    @iu._exe_time
    def execute_normalization(
        self, output_dir, col_norm=None, row_norm=None, quantile_norm=None
    ):
        """Execute normalization workflow, save outputs, and return objects.

        This method handles the high-level orchestration: running the
        normalization, exporting CSV files, and generating diagnostic plots.

        Args:
            output_dir: Directory path to save output files.
            col_norm: Optional column normalization method override.
            row_norm: Optional row normalization method override.
            quantile_norm: Optional quantile normalization flag override.

        Returns:
            tuple: (Column-normalized MetaboInt, Final-normalized MetaboInt).
        """
        iu._check_dir_exists(dir_path=output_dir, handle="makedirs")

        mode = self.attrs.get("mode", "POS")
        st_col = self.attrs.get("sample_type", "Sample Type")
        
        # Resolve sample labels from metadata
        sample_dict = self.attrs.get("sample_dict", {})
        qc_lbl = sample_dict.get("QC sample", "QC")
        act_lbl = sample_dict.get("Actual sample", "Sample")
        
        c_norm = col_norm or self.attrs.get("col_norm", "None")
        r_norm = row_norm or self.attrs.get("row_norm", "None")
        is_qnt = quantile_norm
        if is_qnt is None:
            is_qnt = self.attrs.get("quantile_norm", False)

        # Logging the start of the process
        blank_count = len(self._blank.columns)
        if blank_count > 0:
            logger.info(f"Permanently dropping {blank_count} Blank samples.")

        if is_qnt:
            suffix = f"Quantile_{mode}"
            logger.info("Applying Standalone Quantile Normalization.")
        else:
            suffix = f"Col_{c_norm}_Row_{r_norm}_{mode}"
            logger.info(
                f"Applying Normalization | Col: {c_norm} | Row: {r_norm}"
            )

        # Core logic execution
        col_data, final_data = self.apply_normalization(
            col_norm=c_norm, row_norm=r_norm, quantile_norm=is_qnt
        )

        # Export intermediate CSV if applicable
        if not is_qnt and col_data is not None:
            col_suffix = f"Col_Only_{c_norm}_{mode}"
            file_path_col = os.path.join(
                output_dir, f"Normalized_Data_{col_suffix}.csv"
            )
            col_data.attrs["pipeline_stage"] = "Normalization_Col_Only"
            col_data.to_csv(
                path_or_buf=file_path_col, na_rep="NA", encoding="utf-8-sig"
            )

        # Export Final normalized matrix
        file_path_final = os.path.join(
            output_dir, f"Normalized_Data_{suffix}.csv"
        )
        final_data.attrs["pipeline_stage"] = "Normalization_Final"
        final_data.to_csv(
            path_or_buf=file_path_final, na_rep="NA", encoding="utf-8-sig"
        )

        # Initialize visualizer and generate diagnostic plots
        vis = MetaboVisualizerNormalizer(raw_obj=self, norm_obj=final_data)

        # Plot 1: Relative Log Expression (RLE) Boxplot
        fig_rle = vis.plot_rle_boxplot(
            st_col=st_col, qc_lbl=qc_lbl, act_lbl=act_lbl
        )
        vis.save_and_close_fig(
            fig=fig_rle, 
            file_path=os.path.join(output_dir, f"Norm_RLE_{suffix}.pdf")
        )

        # Plot 2: Minus-Average (MA) Scatter Plot
        fig_ma = vis.plot_ma_scatter()
        vis.save_and_close_fig(
            fig=fig_ma, 
            file_path=os.path.join(output_dir, f"Norm_MA_{suffix}.pdf")
        )

        # Plot 3: Density KDE Overlay
        fig_kde = vis.plot_density_kde()
        vis.save_and_close_fig(
            fig=fig_kde, 
            file_path=os.path.join(output_dir, f"Norm_KDE_{suffix}.pdf")
        )
        
        # Plot 4: Combined Summary Grid (using patchworklib)
        fig_grid = vis.plot_normalization_summary_grid(
            st_col=st_col, qc_lbl=qc_lbl, act_lbl=act_lbl
        )
        if fig_grid:
            grid_path = os.path.join(
                output_dir, f"Normalizer_Summary_Grid_{mode}.pdf"
            )
            vis.save_and_show_pw(pw_obj=fig_grid, file_path=grid_path)

        # Plot 5: Variance Stabilization Validation (Pre vs Post)
        fig_var = vis.plot_variance_stabilization_grid()
        if fig_var is not None:
            var_path = os.path.join(
                output_dir, f"Norm_Variance_Check_{suffix}.pdf"
            )
            vis.save_and_show_pw(pw_obj=fig_var, file_path=var_path)
            
        logger.success("Data normalization completed successfully.")

        return col_data, final_data


class MetaboVisualizerNormalizer(visualizer_classes.BaseMetaboVisualizer):
    """Visualization suite for data normalization evaluation."""

    def __init__(self, raw_obj, norm_obj):
        """Initialize the visualizer with raw and normalized datasets.

        Args:
            raw_obj: MetaboInt object representing the data before normalization.
            norm_obj: MetaboInt object representing the data after normalization.
        """
        super().__init__(metabo_obj=norm_obj)
        self.raw_obj = raw_obj
        self.norm_obj = norm_obj

    def _get_log_target(self, obj, is_norm_obj=False):
        """Extract target columns and conditionally apply log2 transform."""
        target_cols = obj.columns.difference(obj._blank.columns)
        
        # VSN data is already in a generalized log2 scale; avoid double-logging
        if is_norm_obj and obj.attrs.get("row_norm") in ("VSN", "vsn"):
            return obj[target_cols].astype(float)
            
        return obj.calc_log2_transform(obj[target_cols])

    # ====================================================================
    # Plotting Methods
    # ====================================================================

    def plot_rle_boxplot(self, st_col, qc_lbl, act_lbl, ax=None):
        """Plot pre/post RLE boxplot combining QC and target samples."""
        log_raw = self._get_log_target(self.raw_obj, is_norm_obj=False)
        log_norm = self._get_log_target(self.norm_obj, is_norm_obj=True)
        
        rle_raw = self.raw_obj.calc_rle_matrix(log_raw)
        rle_norm = self.norm_obj.calc_rle_matrix(log_norm)

        def _melt_to_long(df_rle, label):
            df_t = df_rle.T
            df_flat = df_t.reset_index()
            id_cols = list(df_rle.columns.names)
            
            # Clear attrs to prevent Pandas ValueError during melt
            df_flat.attrs = {} 
            
            df_long = df_flat.melt(
                id_vars=id_cols, var_name="Feature", value_name="RLE"
            )
            df_long["Status"] = label
            return df_long

        df_raw_long = _melt_to_long(rle_raw, "Before")
        df_norm_long = _melt_to_long(rle_norm, "After")

        plot_df = pd.concat([df_raw_long, df_norm_long], ignore_index=True)
        plot_df = plot_df.dropna(subset=["RLE"])
        
        valid_types = [qc_lbl, act_lbl]
        plot_df = plot_df[plot_df[st_col].isin(valid_types)]

        if ax is None:
            fig, current_ax = plt.subplots(figsize=(6, 5))
        else:
            current_ax = ax
            fig = current_ax.figure

        qc_color = self.pal.get(qc_lbl, "tab:red")
        act_color = self.pal.get(act_lbl, "tab:gray")
        curr_palette = {qc_lbl: qc_color, act_lbl: act_color}
        
        plot_df["Status"] = pd.Categorical(
            plot_df["Status"], categories=["Before", "After"], ordered=True
        )

        if plot_df.empty:
            self._apply_standard_format(
                ax=current_ax, title="Relative Log Expression (No Data)"
            )
            return fig if ax is None else current_ax

        sns.boxplot(
            data=plot_df, x="Status", y="RLE", hue=st_col,
            hue_order=[qc_lbl, act_lbl], width=0.6, dodge=True, 
            palette=curr_palette, showfliers=False, ax=current_ax, 
            linewidth=1.5
        )

        self._apply_standard_format(
            ax=current_ax, title="Relative Log Expression (RLE)", 
            xlabel="", ylabel="Relative Log Expression"
        )
        
        if current_ax.get_legend():
            self._format_single_legend(ax=current_ax, title="Sample Type")

        if ax is None:
            return fig
        return current_ax

    def plot_ma_scatter(self, ax_list=None, cax=None):
        """Plot pre/post MA scatter with optional external colorbar axis."""
        log_raw = self._get_log_target(self.raw_obj, is_norm_obj=False)
        log_norm = self._get_log_target(self.norm_obj, is_norm_obj=True)
        
        a_raw, m_raw = self.raw_obj.calc_ma_arrays(log_raw)
        a_norm, m_norm = self.norm_obj.calc_ma_arrays(log_norm)

        if ax_list is None:
            fig, axes = plt.subplots(
                nrows=1, ncols=2, figsize=(9, 4), sharey=True, sharex=True
            )
        else:
            axes = ax_list
            fig = axes[0].figure

        g_min_x = min(a_raw.min(), a_norm.min())
        g_max_x = max(a_raw.max(), a_norm.max())
        g_min_y = min(m_raw.min(), m_norm.min())
        g_max_y = max(m_raw.max(), m_norm.max())
        
        margin_x = (g_max_x - g_min_x) * 0.05
        margin_y = (g_max_y - g_min_y) * 0.05
        
        extent = (
            g_min_x - margin_x, g_max_x + margin_x, 
            g_min_y - margin_y, g_max_y + margin_y
        )

        custom_cmap = pu.custom_linear_cmap(["white", "tab:red"], 100)

        hb0 = axes[0].hexbin(
            x=a_raw, y=m_raw, gridsize=50, extent=extent,
            cmap=custom_cmap, mincnt=1, bins="log"
        )
        hb1 = axes[1].hexbin(
            x=a_norm, y=m_norm, gridsize=50, extent=extent,
            cmap=custom_cmap, mincnt=1, bins="log"
        )

        titles = ["MA before Normalization", "MA after Normalization"]
        for ax, title in zip(axes, titles):
            ax.axhline(0, color="k", linestyle="--", linewidth=1.5)
            self._apply_standard_format(
                ax=ax, title=title,
                xlabel="Average Log2 Intensity (A)",
                ylabel="Log2 Fold Change from Mean (M)"
            )
            
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

        if cax is not None:
            cb = fig.colorbar(hb1, cax=cax)
        elif ax_list is None:
            cb = fig.colorbar(hb1, ax=axes.tolist())
        else:
            cb = fig.colorbar(hb1, ax=axes[-1])
        cb.set_label("Log10(Count)")
            
        if ax_list is None:
            return fig
        return axes

    def plot_density_kde(self, ax=None):
        """Plot KDE overlay, comparing pre/post normalization."""
        log_raw = self._get_log_target(self.raw_obj, is_norm_obj=False)
        log_norm = self._get_log_target(self.norm_obj, is_norm_obj=True)

        if ax is None:
            fig, current_ax = plt.subplots(figsize=(4, 4))
        else:
            current_ax = ax
            fig = current_ax.figure

        for col in log_raw.columns:
            data_to_plot = log_raw[col].dropna()
            if data_to_plot.nunique() > 1:
                sns.kdeplot(
                    data=data_to_plot, ax=current_ax,
                    color="tab:red", alpha=0.15, linewidth=0.5
                )

        for col in log_norm.columns:
            data_to_plot = log_norm[col].dropna()
            if data_to_plot.nunique() > 1:
                sns.kdeplot(
                    data=data_to_plot, ax=current_ax,
                    color="tab:gray", alpha=0.15, linewidth=0.5
                )

        self._apply_standard_format(
            ax=current_ax, title="Density Overlay",
            xlabel="Log2 Intensity", ylabel="Density"
        )
        
        current_ax.plot([], [], color="tab:red", lw=2, label="Raw Data")
        current_ax.plot([], [], color="tab:gray", lw=2, label="Normalized")
        self._format_single_legend(ax=current_ax)
        
        if ax is None:
            return fig
        return current_ax

    def plot_mean_variance_dependency(self, target_obj, title_suffix="", ax=None):
        """Plot Mean vs Standard Deviation to validate variance stabilization.

        In raw mass spectrometry data, standard deviation typically increases
        with the mean intensity (heteroscedasticity). A successful VSN or log 
        transformation will flatten this dependency curve (homoscedasticity).
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))
        
        # 1. Manually identify target columns for extraction
        target_cols = target_obj.columns.difference(target_obj._blank.columns)
        
        # 2. Extract data in log/generalized-log scale to prevent double logging
        if target_obj.attrs.get("row_norm") == "VSN":
            log_df = target_obj[target_cols].astype(float)
        else:
            log_df = target_obj.calc_log2_transform(target_obj[target_cols])
            
        # 3. Calculate Mean and Standard Deviation across samples per feature
        feat_means = log_df.mean(axis="columns").values
        feat_sds = log_df.std(axis="columns", ddof=1).values
        
        # Drop NaNs generated by missing values to ensure safe plotting
        valid_mask = ~np.isnan(feat_means) & ~np.isnan(feat_sds)
        x_val = feat_means[valid_mask]
        y_val = feat_sds[valid_mask]
        
        if len(x_val) == 0:
            self._apply_standard_format(
                ax=ax, title=f"Mean-Variance Dependency {title_suffix}"
            )
            ax.text(0.5, 0.5, "Insufficient valid data", ha="center")
            return ax

        # 4. Plot scatter points and LOWESS trend line
        ax.scatter(
            x_val, y_val, color="tab:gray", alpha=0.3, s=5, label="Features"
        )
        sns.regplot(
            x=x_val, y=y_val, scatter=False, lowess=True, 
            color="tab:red", line_kws={"linewidth": 2}, 
            label="LOWESS Trend", ax=ax
        )
        
        self._apply_standard_format(
            ax=ax, title=f"Mean-Variance Dependency {title_suffix}",
            xlabel="Feature Mean (Log2-Scale)", 
            ylabel="Standard Deviation"
        )
        self._format_single_legend(ax=ax)
        
        if ax is None:
            return fig
        return ax
    def plot_variance_stabilization_grid(self):
        """Combine pre- and post-normalization variance diagnostic plots.
        
        Returns:
            patchworklib.Brick: Combined 1x2 grid object, or None if failed.
        """
        try:
            import patchworklib as pw
        except ImportError:
            # Import explicitly within the method to avoid global namespace pollution
            from loguru import logger
            logger.warning("patchworklib not found. Skipping variance grid.")
            return None

        pw.clear()

        ax_var_pre = pw.Brick(figsize=(4, 4), label="VAR_PRE")
        self.plot_mean_variance_dependency(
            target_obj=self.raw_obj, title_suffix="(Raw)", ax=ax_var_pre
        )

        ax_var_post = pw.Brick(figsize=(4, 4), label="VAR_POST")
        self.plot_mean_variance_dependency(
            target_obj=self.norm_obj, title_suffix="(Norm)", ax=ax_var_post
        )

        return ax_var_pre | ax_var_post

    def plot_normalization_summary_grid(self, st_col, qc_lbl, act_lbl):
        """Combine RLE, KDE, and MA plots into a global grid."""
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping summary grid.")
            return None

        pw.clear()

        ax_rle = pw.Brick(figsize=(4, 4), label="RLE")
        self.plot_rle_boxplot(
            st_col=st_col, qc_lbl=qc_lbl, act_lbl=act_lbl, ax=ax_rle
        )

        ax_kde = pw.Brick(figsize=(4, 4), label="KDE")
        self.plot_density_kde(ax=ax_kde)

        ax_spacer = pw.Brick(figsize=(0.2, 4), label="Spacer")
        ax_spacer.axis("off")

        ax_ma_pre = pw.Brick(figsize=(4, 4), label="MA_PRE")
        ax_ma_post = pw.Brick(figsize=(4, 4), label="MA_POST")
        ax_ma_cb = pw.Brick(figsize=(0.2, 4), label="MA_CB")
        
        self.plot_ma_scatter(ax_list=[ax_ma_pre, ax_ma_post], cax=ax_ma_cb)

        row1 = ax_rle | ax_kde | ax_spacer
        row2 = ax_ma_pre | ax_ma_post | ax_ma_cb

        return row1 / row2