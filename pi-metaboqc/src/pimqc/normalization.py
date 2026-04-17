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
    # Data Transformation & Matrix Extractors (Static Methods)
    # ====================================================================

    @staticmethod
    def calc_log2_transform(df):
        """Perform log2 transformation after replacing zeros with NaNs."""
        return np.log2(df.astype(float).replace({0: np.nan}))

    @staticmethod
    def calc_rle_matrix(df_log):
        """Calculate Relative Log Expression (RLE) matrix."""
        feat_medians = df_log.median(axis="columns")
        return df_log.sub(feat_medians, axis="index")

    @staticmethod
    def calc_ma_arrays(df_log):
        """Calculate flattened A and M values for MA-plot visualization."""
        a_vals = df_log.mean(axis="columns")
        m_df = df_log.sub(a_vals, axis="index")
        
        a_flat = np.repeat(a_vals.values, m_df.shape[1])
        m_flat = m_df.values.flatten()
        
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

        Args:
            col_norm: Override for column normalization method.
            row_norm: Override for row normalization method.
            quantile_norm: Override for quantile normalization flag.
        
        Returns:
            tuple: (Column-normalized object or None, Final-normalized object).
        """
        blank_cols = self._blank.columns
        target_cols = self.columns.difference(blank_cols)

        df_target = self[target_cols].copy()

        if df_target.empty:
            raise ValueError("No target samples available for normalization.")

        c_norm = col_norm or self.attrs.get("col_norm")
        r_norm = row_norm or self.attrs.get("row_norm")
        is_qnt = quantile_norm
        if is_qnt is None:
            is_qnt = self.attrs.get("quantile_norm", False)

        if is_qnt:
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
            
            df_col = df_target.copy()
            obj_col = self._constructor(df_col).__finalize__(self)

            # Stage 2: Row (Feature-wise) Scaling/Normalization
            if r_norm in ("VSN", "vsn"):
                df_target, vsn_meta = self.calc_vsn_normalization(df_target)
                self.attrs.update(vsn_meta)
            elif r_norm in ("Auto", "auto"):
                df_target = self.calc_auto_scaling(df_target)
            elif r_norm in ("Pareto", "pareto"):
                df_target = self.calc_pareto_scaling(df_target)

        obj_final = self._constructor(df_target).__finalize__(self)
        
        return obj_col, obj_final

    @iu._exe_time
    def execute_normalization(
        self, output_dir, col_norm=None, row_norm=None, quantile_norm=None
    ):
        """Execute normalization workflow, save outputs, and generate plots.

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
        
        sample_dict = self.attrs.get("sample_dict", {})
        qc_lbl = sample_dict.get("QC sample", "QC")
        act_lbl = sample_dict.get("Actual sample", "Sample")
        
        c_norm = col_norm or self.attrs.get("col_norm", "None")
        r_norm = row_norm or self.attrs.get("row_norm", "None")
        is_qnt = quantile_norm
        if is_qnt is None:
            is_qnt = self.attrs.get("quantile_norm", False)

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

        col_data, final_data = self.apply_normalization(
            col_norm=c_norm, row_norm=r_norm, quantile_norm=is_qnt
        )

        if not is_qnt and col_data is not None:
            col_suffix = f"Col_Only_{c_norm}_{mode}"
            file_path_col = os.path.join(
                output_dir, f"Normalized_Data_{col_suffix}.csv"
            )
            col_data.attrs["pipeline_stage"] = "Column normalization"
            col_data.to_csv(
                path_or_buf=file_path_col, na_rep="NA", encoding="utf-8-sig"
            )

        file_path_final = os.path.join(
            output_dir, f"Normalized_Data_{suffix}.csv"
        )
        final_data.attrs["pipeline_stage"] = "Final normalization"
        final_data.to_csv(
            path_or_buf=file_path_final, na_rep="NA", encoding="utf-8-sig"
        )

        # Build visualizer utilizing 3-stage objects
        vis = MetaboVisualizerNormalizer(
            raw_obj=self, col_obj=col_data, norm_obj=final_data
        )

        fig_rle = vis.plot_rle_boxplot(
            st_col=st_col, qc_lbl=qc_lbl, act_lbl=act_lbl
        )
        vis.save_and_close_fig(
            fig=fig_rle, 
            file_path=os.path.join(output_dir, f"Norm_RLE_3Stage_{suffix}.pdf")
        )

        fig_ma = vis.plot_ma_scatter()
        vis.save_and_close_fig(
            fig=fig_ma, 
            file_path=os.path.join(output_dir, f"Norm_MA_3Stage_{suffix}.pdf")
        )

        fig_kde = vis.plot_density_kde()
        vis.save_and_close_fig(
            fig=fig_kde, 
            file_path=os.path.join(output_dir, f"Norm_KDE_Split_{suffix}.pdf")
        )
        
        fig_grid = vis.plot_normalization_summary_grid(
            st_col=st_col, qc_lbl=qc_lbl, act_lbl=act_lbl
        )
        if fig_grid:
            grid_path = os.path.join(
                output_dir, f"Normalizer_Summary_Grid_{mode}.pdf"
            )
            vis.save_and_show_pw(pw_obj=fig_grid, file_path=grid_path)

        fig_var = vis.plot_variance_stabilization_grid()
        if fig_var is not None:
            var_path = os.path.join(
                output_dir, f"Norm_Variance_Check_{suffix}.pdf"
            )
            vis.save_and_show_pw(pw_obj=fig_var, file_path=var_path)
            
        logger.success("Data normalization completed successfully.")

        return col_data, final_data


class MetaboVisualizerNormalizer(visualizer_classes.BaseMetaboVisualizer):
    """Visualization suite for evaluating 3-stage data normalization."""

    def __init__(self, raw_obj, col_obj, norm_obj):
        """Initialize the visualizer with raw, col-normalized, and final data.

        Args:
            raw_obj: MetaboInt object representing data before normalization.
            col_obj: MetaboInt object representing data after column norm.
            norm_obj: MetaboInt object representing fully normalized data.
        """
        super().__init__(metabo_obj=norm_obj)
        self.raw_obj = raw_obj
        self.col_obj = col_obj
        self.norm_obj = norm_obj

    def _get_log_target(self, obj, check_vsn=False):
        """Extract target columns and conditionally apply log2 transform.
        
        Args:
            obj: The MetaboInt object to process.
            check_vsn: If True, blocks double-logging for VSN-scaled objects.
            
        Returns:
            pd.DataFrame: Log-scaled target data, or None if obj is None.
        """
        if obj is None:
            return None
            
        target_cols = obj.columns.difference(obj._blank.columns)
        
        if check_vsn and obj.attrs.get("row_norm") in ("VSN", "vsn"):
            return obj[target_cols].astype(float)
            
        return obj.calc_log2_transform(obj[target_cols])

    # ====================================================================
    # Plotting Methods
    # ====================================================================

    def plot_rle_boxplot(self, st_col, qc_lbl, act_lbl, ax=None):
        """Plot multi-stage RLE boxplot combining QC and target samples."""
        stages = [
            (self.raw_obj, "Before"),
            (self.col_obj, "Col-Norm"),
            (self.norm_obj, "Final")
        ]
        
        long_dfs = []
        for obj, label in stages:
            if obj is None: 
                continue
            
            # VSN checking is strictly assigned to the 'Final' stage
            log_data = self._get_log_target(obj, check_vsn=(label == "Final"))
            rle = obj.calc_rle_matrix(log_data)
            
            df_flat = rle.T.reset_index()
            df_flat.attrs = {} 
            df_long = df_flat.melt(
                id_vars=list(rle.columns.names), 
                var_name="Feature", value_name="RLE"
            )
            df_long["Stage"] = label
            long_dfs.append(df_long)

        plot_df = pd.concat(long_dfs, ignore_index=True)
        plot_df = plot_df.dropna(subset=["RLE"])
        
        valid_types = [qc_lbl, act_lbl]
        plot_df = plot_df[plot_df[st_col].isin(valid_types)]

        if ax is None:
            fig, current_ax = plt.subplots(figsize=(8, 5))
        else:
            current_ax = ax
            fig = current_ax.figure

        curr_palette = {
            qc_lbl: self.pal.get(qc_lbl, "tab:red"), 
            act_lbl: self.pal.get(act_lbl, "tab:gray")
        }
        
        plot_df["Stage"] = pd.Categorical(
            plot_df["Stage"], 
            categories=["Before", "Col-Norm", "Final"], 
            ordered=True
        )

        sns.boxplot(
            data=plot_df, x="Stage", y="RLE", hue=st_col,
            hue_order=[qc_lbl, act_lbl], width=0.6, dodge=True, 
            palette=curr_palette, showfliers=False, ax=current_ax, 
            linewidth=1.2
        )

        self._apply_standard_format(
            ax=current_ax, title="Relative Log Expression Across Stages", 
            xlabel="", ylabel="Relative Log Expression", append_stage=False
        )
        
        if current_ax.get_legend():
            current_ax.legend(loc="best")
            self._format_single_legend(ax=current_ax, title="Sample Type")

        if ax is None:
            return fig
        return current_ax

    def plot_ma_scatter(self, ax_list=None, cax=None):
        """Plot multi-stage MA scatter to assess intensity bias reduction."""
        stages = [
            (self._get_log_target(self.raw_obj), "Before"),
            (self._get_log_target(self.col_obj), "Col-Norm"),
            (self._get_log_target(self.norm_obj, True), "Final")
        ]
        
        active_stages = [(obj, lbl) for obj, lbl in stages if obj is not None]
        n_stages = len(active_stages)

        if ax_list is None:
            fig, axes = plt.subplots(
                nrows=1, ncols=n_stages, 
                figsize=(4.5 * n_stages, 4), sharey=True
            )
            if n_stages == 1:
                axes = [axes]
        else:
            axes = ax_list
            fig = axes[0].figure

        # Compute global bounds for consistency across stages
        all_a_vals = np.concatenate(
            [self.raw_obj.calc_ma_arrays(log_df)[0] for log_df, _ in active_stages]
        )
        a_min, a_max = np.nanmin(all_a_vals), np.nanmax(all_a_vals)
        margin_x = (a_max - a_min) * 0.08
        extent = (a_min - margin_x, a_max + margin_x, -5, 5)
        
        color_map = pu.custom_linear_cmap(
            color_list=["white", "tab:red"], n_colors=256, cmin=0.1, cmax=1.0)

        hb = None
        for i, (log_df, label) in enumerate(active_stages):
            a_vals, m_vals = self.raw_obj.calc_ma_arrays(log_df)
            hb = axes[i].hexbin(
                x=a_vals, y=m_vals, gridsize=40, extent=extent,
                cmap=color_map, mincnt=1, # bins="log"
            )
            axes[i].axhline(0, color="k", linestyle="--", linewidth=1.5)
            self._apply_standard_format(
                ax=axes[i], title=label,
                xlabel="Average Log2 Intensity (A)",
                ylabel="Log2 Fold Change (M)", append_stage=False
            )
            axes[i].set_xlim(extent[0], extent[1])
            axes[i].set_ylim(extent[2], extent[3])

            # sample_size = min(len(a_vals), 2000)
            # idx = np.random.choice(len(a_vals), sample_size, replace=False)

            # sns.regplot(
            #     x=a_vals[idx], y=m_vals[idx], 
            #     scatter=False, lowess=True, 
            #     color="k", line_kws={
            #         "linewidth": 2, "zorder": 5, "linestyle":"dashdot"}, 
            #     ax=axes[i])

        if cax is not None and hb is not None:
            cb = fig.colorbar(hb, cax=cax)
            cb.set_label("Log10(Count)")
        elif ax_list is None and hb is not None:
            cb = fig.colorbar(hb, ax=axes) # pyright: ignore
            cb.set_label("Log10(Count)")
            
        if ax_list is None:
            return fig
        return axes

    def plot_density_kde(self, ax_qc=None, ax_sample=None):
        """Plot KDE overlay split into QC and Sample subplots across stages."""
        if ax_qc is None or ax_sample is None:
            fig, (ax_qc, ax_sample) = plt.subplots(1, 2, figsize=(10, 4))
            return_fig = True
        else:
            fig = ax_qc.figure
            return_fig = False

        stages = [
            (self.raw_obj, "Before", "tab:gray"),
            (self.col_obj, "Col-Norm", "tab:blue"),
            (self.norm_obj, "Final", "tab:red")
        ]

        for obj, label, color in stages:
            if obj is None: 
                continue
                
            log_df = self._get_log_target(obj, check_vsn=(label == "Final"))
            
            qc_cols = obj._qc.columns
            exclude = qc_cols.union(obj._blank.columns)
            sam_cols = obj.columns.difference(exclude)
            
            splits = [
                (qc_cols, ax_qc, "QC"), 
                (sam_cols, ax_sample, "Sample")
            ]
            
            for cols, ax, group_name in splits:
                if cols.empty: 
                    continue
                    
                data_to_plot = log_df[cols].values.flatten()
                data_to_plot = data_to_plot[~np.isnan(data_to_plot)]
                
                if len(data_to_plot) > 1:
                    sns.kdeplot(
                        data=data_to_plot, ax=ax, color=color, 
                        label=label, lw=2, alpha=0.7
                    )
                    
                self._apply_standard_format(
                    ax=ax, title=f"Density Overlay ({group_name})",
                    xlabel="Log2 Intensity", ylabel="Density", 
                    append_stage=False
                )
                
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(loc="best")
                    self._format_single_legend(ax=ax, title="Stage")

        if return_fig:
            plt.tight_layout()
            return fig # pyright: ignore
            
        return ax_qc, ax_sample

    def plot_mean_variance_dependency(self, target_obj, label, ax=None):
        """Plot Mean vs Standard Deviation to validate variance stabilization.

        Homoscedasticity is a key goal of normalization. This plot checks
        if the standard deviation remains constant across intensity levels.

        Args:
            target_obj: MetaboInt object (raw, col-norm, or finalized).
            label: Stage name to display in the title (e.g., 'Before').
            ax: Matplotlib axis object.

        Returns:
            plt.Axes: The axis containing the diagnostic plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))
        
        # Use the safe wrapper to handle VSN vs Log2 logic
        is_norm = (label == "Final")
        log_df = self._get_log_target(target_obj, check_vsn=is_norm)
        
        if log_df is None:
            return ax
            
        feat_means = log_df.mean(axis="columns").values
        feat_sds = log_df.std(axis="columns", ddof=1).values
        
        valid_mask = ~np.isnan(feat_means) & ~np.isnan(feat_sds)
        x_val = feat_means[valid_mask]
        y_val = feat_sds[valid_mask]
        
        if len(x_val) == 0:
            self._apply_standard_format(ax=ax, title=f"Mean-Variance ({label})")
            return ax

        ax.scatter(
            x_val, y_val, color="tab:gray", alpha=0.3, s=5, label="Features"
        )
        # Apply LOWESS smoothing to visualize the mean-variance trend
        sns.regplot(
            x=x_val, y=y_val, scatter=False, lowess=True, 
            color="tab:red", line_kws={"linewidth": 2}, 
            label="LOWESS Trend", ax=ax
        )
        
        self._apply_standard_format(
            ax=ax, title=f"Mean-Variance ({label})",
            xlabel="Feature Mean (Log2-Scale)", 
            ylabel="Standard Deviation", append_stage=False
        )
        
        ax.legend(loc="best")
        self._format_single_legend(ax=ax)
        
        return ax

    def plot_variance_stabilization_grid(self):
        """Combine 3-stage Mean-Variance plots into a diagnostic grid.
        
        Returns:
            patchworklib.Brick: Combined grid object or None.
        """
        try:
            import patchworklib as pw
        except ImportError:
            return None

        pw.clear()

        # Define active stages based on the normalization workflow
        stages = [(self.raw_obj, "Before")]
        if self.col_obj is not None:
            stages.append((self.col_obj, "Col-Norm"))
        stages.append((self.norm_obj, "Final"))

        bricks = []
        for obj, label in stages:
            brick = pw.Brick(figsize=(4, 4), label=f"var_{label}")
            self.plot_mean_variance_dependency(
                target_obj=obj, label=label, ax=brick
            )
            bricks.append(brick)

        # Assemble the grid horizontally (1x2 or 1x3)
        res_grid = bricks[0]
        for b in bricks[1:]:
            res_grid = res_grid | b
            
        return res_grid

    def plot_normalization_summary_grid(self, st_col, qc_lbl, act_lbl):
        """Combine RLE, KDE, and MA plots into a dynamic global grid."""
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping summary grid.")
            return None

        pw.clear()

        # Top Row: RLE | KDE QC | KDE Sample
        ax_rle = pw.Brick(figsize=(4, 4), label="RLE")
        self.plot_rle_boxplot(
            st_col=st_col, qc_lbl=qc_lbl, act_lbl=act_lbl, ax=ax_rle
        )

        ax_qc = pw.Brick(figsize=(4, 4), label="KDE_QC")
        ax_sam = pw.Brick(figsize=(4, 4), label="KDE_SAM")
        self.plot_density_kde(ax_qc=ax_qc, ax_sample=ax_sam)
        
        row1 = ax_rle | ax_qc | ax_sam

        # Bottom Row: MA Scatters (Variable length based on active stages)
        n_stages = 2 if self.col_obj is None else 3
        ax_ma_list = [
            pw.Brick(figsize=(4, 4), label=f"MA_{i}") for i in range(n_stages)
        ]
        ax_cb = pw.Brick(figsize=(0.2, 4), label="MA_CB")
        
        self.plot_ma_scatter(ax_list=ax_ma_list, cax=ax_cb)
        
        row2 = ax_ma_list[0]
        for ax in ax_ma_list[1:]:
            row2 = row2 | ax
        row2 = row2 | ax_cb
        
        # Pad row2 if needed to match row1 width visually
        if n_stages == 2:
            ax_spacer = pw.Brick(figsize=(4, 4), label="Spacer")
            ax_spacer.axis("off")
            row2 = row2 | ax_spacer

        return row1 / row2