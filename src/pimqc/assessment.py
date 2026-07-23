# src/pimqc/assessment.py
"""
Script purpose: Run the quality-assessment checkpoint for MetaboInt objects.

The public execute_assessment() method creates the assessment output folder,
materializes cached QC correlation matrices, batch correlations, PCA results,
and RSD summaries, then exports the joined outlier table. It evaluates IS and
ORF reference features, merges their flags into PCA outlier diagnostics, and
routes all figures through MetaboVisualizerAssessor.
Generated assets include QC and batch heatmaps, PCA score plots, SD/OD
outlier maps, RSD bars, optional IS/ORF Shewhart charts, and a summary
dashboard used by the final report.
"""

import os
import re
import copy
import warnings

import numpy as np
import pandas as pd
from functools import cached_property

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import seaborn as sns

from loguru import logger
from typing import Dict, Any, Optional, Union

from . import io_utils as iu
from . import plot_utils as pu
from . import core_classes
from . import visualizer_classes
from . import pca_utils

warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=RuntimeWarning)


class MetaboIntAssessor(core_classes.MetaboInt):
    """Data quality assessment computational class for metabolomics."""

    # Register "stats" for pandas metadata propagation
    _metadata = ["attrs", "stats"]

    def __init__(
        self,
        *args: object,
        pipeline_params: Optional[Dict[str, Any]] = None,
        corr_method: Optional[str] = None,
        scaling_method: Optional[str] = None,
        is_outlier_threshold: Optional[Union[float, int]] = None,
        orf_outlier_threshold: Optional[Union[float, int]] = None,
        **kwargs: object,
    ) -> None:
        """Initialize the data quality assessment class.

        Args:
            *args: Variable arguments passed to pandas DataFrame.
            pipeline_params: Global configuration dictionary.
            corr_method: Method for correlation (e.g., 'Spearman').
            scaling_method: Scaling method for PCA (e.g., 'Pareto-scaling').
            is_outlier_threshold: Threshold for Internal Standard outliers.
            orf_outlier_threshold: Threshold for Outlier Reference Features.
            **kwargs: Extra arguments passed to pandas DataFrame.
        """
        super().__init__(*args, pipeline_params=pipeline_params, **kwargs)

        # Initialize local state cache for heavy computational results
        if not hasattr(self, "stats"):
            self.stats = {}

        # 1. Base defaults
        configs = {
            "corr_method": "Spearman",
            "scaling_method": "Pareto-scaling",
            "is_outlier_threshold": 0.75,
            "orf_outlier_threshold": 0.5,
        }

        # 2. TOML configuration overrides
        if pipeline_params and "MetaboIntAssessor" in pipeline_params:
            configs.update(pipeline_params["MetaboIntAssessor"])

        # 3. Explicit kwargs override TOML (Highest priority)
        if corr_method is not None:
            configs["corr_method"] = corr_method
        if scaling_method is not None:
            configs["scaling_method"] = scaling_method
        if is_outlier_threshold is not None:
            configs["is_outlier_threshold"] = is_outlier_threshold
        if orf_outlier_threshold is not None:
            configs["orf_outlier_threshold"] = orf_outlier_threshold

        # 4. Flatten strictly into lifecycle attributes (SSOT)
        self.attrs.update(configs)

    @property
    def _constructor(self) -> type["MetaboIntAssessor"]:
        """Override constructor to return MetaboIntAssessor."""
        return MetaboIntAssessor

    def __finalize__(
        self,
        other: object,
        method: Optional[str] = None,
        **kwargs: object,
    ) -> "MetaboIntAssessor":
        """Explicitly preserve custom attributes and state during operations."""
        super().__finalize__(other, method=method, **kwargs)
        for name in self._metadata:
            if hasattr(other, name):
                setattr(self, name, copy.deepcopy(getattr(other, name)))
        return self

    # =========================================================================
    # Core Statistical Calculations (Refactored to Cached Properties)
    # =========================================================================

    @cached_property
    def qc_corr_matrix(self) -> pd.DataFrame:
        """
        Calculates and natively caches the QC sample correlation matrix.
        Relies on internal instance state to avoid unhashable DataFrame args.
        """
        method = self.attrs.get("corr_method", "Spearman")
        qc_data = self._qc

        if qc_data.empty:
            return pd.DataFrame()

        return qc_data.corr(method=method.lower())

    @cached_property
    def batch_qc_corr_matrix(self) -> pd.DataFrame:
        """Aggregates QC correlation matrix into a batch-level median matrix."""
        corr_mat = self.qc_corr_matrix

        if corr_mat.empty:
            return pd.DataFrame()

        batch = self.attrs.get("batch", "Batch")
        batches = self._qc.columns.get_level_values(batch)

        # Compress rows and columns sequentially to extract medians
        batch_corr = corr_mat.groupby(batches).median()
        batch_corr = batch_corr.transpose().groupby(batches).median().transpose()

        return batch_corr

    @cached_property
    def rsd_distribution(self) -> dict[str, dict[str, int]]:
        """Calculates and caches the RSD distribution for QA reporting."""
        sample_type = self.attrs.get("sample_type", "Sample Type")
        actual_label = self.attrs.get("sample_dict", {}).get("Actual sample", "Sample")

        def _get_dist(data: pd.DataFrame) -> dict[str, int]:
            labels = ["0-10%", "10-20%", "20-30%", ">30%"]
            if data.empty:
                return {label: 0 for label in labels}

            # ==========================================================
            # State-Aware Pseudo-linearization (The Magic Trick)
            # Restore exponential distribution to calculate meaningful RSD
            # ==========================================================
            if self.attrs.get("is_logged", False):
                # Use exp2 to reverse both robust_log and approximate VSN glog.
                # Since RSD(C*X) == RSD(X), the constants don't affect the ratio.
                linear_data = np.exp2(data.astype(float)) - 1.0
                # Prevent negative intensities caused by LOD offsets or VSN
                linear_data = linear_data.clip(lower=1e-9)
            else:
                linear_data = data.astype(float)

            # Prevent division by zero if a feature is completely blank
            means = linear_data.mean(axis=1).replace(0, 1e-9)
            stds = linear_data.std(axis=1, ddof=1)

            rsd = stds / means

            # Binning logic
            bins = [-np.inf, 0.1, 0.2, 0.3, np.inf]
            counts = pd.cut(rsd, bins=bins, labels=labels, right=False)
            dist_dict = counts.value_counts(sort=False).to_dict()

            return {label: int(dist_dict.get(label, 0)) for label in labels}

        actual_sample_mask = self.columns.get_level_values(sample_type) == actual_label

        return {
            "qc": _get_dist(self._qc),
            "actual": _get_dist(self.loc[:, actual_sample_mask]),
        }

    @cached_property
    def pca_res(self) -> dict[str, Any]:
        """Execute PCA workflow, outlier detection, and diagnostic metrics."""
        sample_type = self.attrs.get("sample_type", "Sample Type")
        sample_name = self.attrs.get("sample_name", "Sample Name")
        batch = self.attrs.get("batch", "Batch")

        sample_dict = self.attrs.get("sample_dict", {})
        qc_label = sample_dict.get("QC sample", "QC")
        actual_label = sample_dict.get("Actual sample", "Sample")

        # Extract scaling method from Assessor configurations
        s_method = self.attrs.get("scaling_method", "Pareto-scaling")

        # Automatically extract features from internal state with JIT scaling
        features, labels = pca_utils.PCAEngine.extract_features(
            metabo_obj=self,
            sample_type=sample_type,
            sample_name=sample_name,
            actual_label=actual_label,
            qc_label=qc_label,
            scaling_method=s_method,
        )

        # Initialize PCA engine with strict statistical bounds
        _seed = self.attrs.get("global_seed", 123)
        engine = pca_utils.PCAEngine(
            n_components=2, alpha=0.05, od_method="box", global_seed=_seed
        )
        res = engine.run_pca_workflow(features)

        multi_idx = pd.MultiIndex.from_frame(labels)
        pca_scatter = pd.DataFrame(
            res["scores"], index=multi_idx, columns=["PC1", "PC2"]
        )
        pca_var = pd.Series(res["variance"], index=["PC1", "PC2"], name="Variance")
        metrics_df = res["metrics"]
        metrics_df.index = multi_idx

        outliers = pd.DataFrame(
            {
                ("SPE-DModX", "SPE-DModX"): metrics_df["OD"],
                ("SPE-DModX", "Outliers (SPE-DModX)"): metrics_df["is_od_outlier"],
                ("HT2", "Hotelling T2 Score"): metrics_df["SD"],
                ("HT2", "Outliers (HT2)"): metrics_df["is_sd_outlier"],
            },
            index=multi_idx,
        )

        coords = pca_scatter[["PC1", "PC2"]].values
        types = pca_scatter.index.get_level_values(sample_type).values
        batches = pca_scatter.index.get_level_values(batch).values

        relative_dispersion = pca_utils.PCAEngine.calc_relative_dispersion(
            coords, types, qc_label, actual_label
        )
        silhouette_score = pca_utils.PCAEngine.calc_qc_batch_silhouette(
            coords, types, batches, qc_label
        )
        shift_res = pca_utils.PCAEngine.calc_qc_centrality_shift(
            coords, types, qc_label, actual_label
        )

        return {
            "pca_scatter": pca_scatter,
            "pca_variance": pca_var,
            "outliers": outliers,
            "metrics_df": metrics_df,
            "sd_limit": res["sd_limit"],
            "od_limit": res["od_limit"],
            "diagnostics": {
                "relative_dispersion": relative_dispersion,
                "batch_silhouette": silhouette_score,
                "centrality_shift": shift_res["relative_shift"],
            },
        }

    def evaluate_reference_features(self, feat_type: str = "IS") -> pd.DataFrame:
        """Calculate boundaries and construct a reference outlier matrix.

        Dynamically throttled by TOML configuration thresholds. Supports
        both internal standards (IS) and outlier reference features (ORF).
        Iterates over features to calculate robust boundaries individually,
        then evaluates failure ratios per sample.
        """
        feat_type_lower = feat_type.lower()
        feat_type_upper = feat_type.upper()
        valid_feats = getattr(self, f"valid_{feat_type_lower}", [])

        if not valid_feats:
            return pd.DataFrame()

        # Extract subset intensity matrix using the inherited method
        df_ref = self.int_order_info(feat_type=feat_type)
        bound_type = self.attrs.get("boundary", "IQR")

        # Dynamically retrieve threshold with type-specific default fallbacks
        default_thresh = 0.75 if feat_type_lower == "is" else 0.5
        threshold_key = f"{feat_type_lower}_outlier_threshold"
        raw_threshold = self.attrs.get(threshold_key, default_thresh)

        # 1. Evaluate boundaries per individual reference feature
        res_dict = {}
        for feat in valid_feats:
            # Perfectly leveraging your inherited static method
            solid, lower, upper = self.calculate_boundaries(
                x=df_ref[feat].values, boundary_type=bound_type
            )
            res_dict[f"Outliers ({feat})"] = (df_ref[feat] < lower) | (
                df_ref[feat] > upper
            )

        df_eval = pd.DataFrame(res_dict, index=df_ref.index)
        df_eval[f"{feat_type_upper}_Outliers_Count"] = df_eval.sum(axis=1)
        df_eval[f"{feat_type_upper}_Total_Count"] = len(valid_feats)

        # 2. Dynamic Cutoff Resolution (Ratio vs Absolute)
        total_feats = len(valid_feats)

        if isinstance(raw_threshold, float) and 0.0 <= raw_threshold <= 1.0:
            cutoff = total_feats * raw_threshold
            effective_cutoff = max(1, int(np.ceil(cutoff)))
        elif isinstance(raw_threshold, int) and raw_threshold >= 1:
            effective_cutoff = raw_threshold
        else:
            logger.warning(
                f"Invalid threshold '{raw_threshold}' for {feat_type}. "
                "Reverting to 0.5."
            )
            effective_cutoff = max(1, int(np.ceil(total_feats * 0.5)))

        # 3. Final Adjudication
        df_eval[f"{feat_type_upper}_Outlier_Flag"] = (
            df_eval[f"{feat_type_upper}_Outliers_Count"] >= effective_cutoff
        )

        return df_eval

    # =========================================================================
    # Pipeline Execution Method
    # =========================================================================

    @iu._exe_time
    def execute_assessment(self, output_dir: str) -> None:
        """Execute the entire QA workflow, save tables, and render plots."""
        if self.empty:
            logger.warning(
                "Empty matrix detected. Terminating QA assessment execution."
            )
            return

        # Configuration metadata extraction (Single Source of Truth)
        sample_type = self.attrs.get("sample_type", "Sample Type")
        batch = self.attrs.get("batch", "Batch")
        inject_order = self.attrs.get("inject_order", "Inject Order")
        sample_name = self.attrs.get("sample_name", "Sample Name")

        sample_dict = self.attrs.get("sample_dict", {})
        qc_label = sample_dict.get("QC sample", "QC")
        actual_label = sample_dict.get("Actual sample", "Sample")

        corr_method = self.attrs.get("corr_method", "Spearman")
        bound_type = self.attrs.get("boundary", "IQR")
        mask_flag = True

        iu._check_dir_exists(dir_path=output_dir, handle="makedirs")

        qc_data = self._qc

        # Directly access the cached properties
        corr_mat = self.qc_corr_matrix
        batch_corr = self.batch_qc_corr_matrix
        pca_res = self.pca_res
        rsd_data = self.rsd_distribution

        # Evaluate reference features for both symmetrical dimensions
        is_eval = self.evaluate_reference_features(feat_type="IS")
        orf_eval = self.evaluate_reference_features(feat_type="ORF")
        outliers_export = pca_res["outliers"].copy()

        # Align and join Internal Standard assessment results
        if not is_eval.empty:
            is_eval_multi = is_eval.copy()
            is_eval_multi.columns = pd.MultiIndex.from_product(
                [["Internal Standard"], is_eval.columns]
            )
            is_eval_align = is_eval_multi.copy()
            is_eval_align.index = is_eval_align.index.get_level_values(sample_name)
            is_eval_align = is_eval_align[~is_eval_align.index.duplicated(keep="first")]
            outliers_export = outliers_export.join(
                is_eval_align, on=sample_name, how="left"
            )

        # Align and join Outlier Reference Feature assessment results
        if not orf_eval.empty:
            orf_eval_multi = orf_eval.copy()
            orf_eval_multi.columns = pd.MultiIndex.from_product(
                [["Outlier Reference Feature"], orf_eval.columns]
            )
            orf_eval_align = orf_eval_multi.copy()
            orf_eval_align.index = orf_eval_align.index.get_level_values(sample_name)
            orf_eval_align = orf_eval_align[
                ~orf_eval_align.index.duplicated(keep="first")
            ]
            outliers_export = outliers_export.join(
                orf_eval_align, on=sample_name, how="left"
            )

        out_path = os.path.join(output_dir, "QA_Diagnostics_Outliers.csv")
        outliers_export.to_csv(out_path, encoding="utf-8-sig", na_rep="NA")

        # Extract independent boolean masks for precise plot mapping
        is_flags = None
        if ("Internal Standard", "IS_Outlier_Flag") in outliers_export.columns:
            is_flags = (
                outliers_export[("Internal Standard", "IS_Outlier_Flag")]
                .fillna(False)
                .astype(bool)
            )

        orf_flags = None
        if ("Outlier Reference Feature", "ORF_Outlier_Flag") in outliers_export.columns:
            orf_flags = (
                outliers_export[("Outlier Reference Feature", "ORF_Outlier_Flag")]
                .fillna(False)
                .astype(bool)
            )

        # Initialize Visualizer and generate plots
        vis = MetaboVisualizerAssessor(self)
        batches = qc_data.columns.get_level_values(batch).unique()
        qc_mask = (
            np.triu(np.ones_like(corr_mat, dtype=bool), k=1) if mask_flag else None
        )

        vis.save_and_close_fig(
            fig=vis.plot_qc_corr_heatmap(
                corr_matrix=corr_mat,
                corr_mask=qc_mask,
                batches=batches,
                method=corr_method,
                cluster="none",
            ),
            file_path=os.path.join(output_dir, "QC_Correlation_Heatmap"),
        )

        vis.save_and_close_fig(
            fig=vis.plot_batch_corr_heatmap(
                batch_corr_matrix=batch_corr, method=corr_method
            ),
            file_path=os.path.join(output_dir, "Batch_Correlation_Heatmap"),
        )

        vis.save_and_close_fig(
            fig=vis.plot_pca_scatter(
                pca_df=pca_res["pca_scatter"],
                pca_var=pca_res["pca_variance"],
                pca_diagnostics=pca_res["diagnostics"],
                sample_type=sample_type,
                batch=batch,
                qc_label=qc_label,
                actual_label=actual_label,
            ),
            file_path=os.path.join(output_dir, "PCA_Scatter_QC_Sample"),
        )

        vis.save_and_close_fig(
            fig=vis.plot_sd_od_scatter(
                metrics_df=pca_res["metrics_df"],
                sd_limit=pca_res["sd_limit"],
                od_limit=pca_res["od_limit"],
                is_flags=is_flags,
                orf_flags=orf_flags,
            ),
            file_path=os.path.join(output_dir, "Outlier_Scatter"),
        )

        vis.save_and_close_fig(
            fig=vis.plot_rsd_bar(
                rsd_data=rsd_data, qc_label=qc_label, actual_label=actual_label
            ),
            file_path=os.path.join(output_dir, "RSD_Barplot"),
        )

        # Symmetrical execution of control chart visualization factory (PW Mode)
        if len(self.valid_is) > 0:
            is_data = self.int_order_info(feat_type="IS")
            is_grid = vis.plot_ref_shewhart_chart(
                ref_data=is_data,
                valid_feats=self.valid_is,
                sample_type=sample_type,
                batch=batch,
                inject_order=inject_order,
                qc_label=qc_label,
                actual_label=actual_label,
                bound_type=bound_type,
                ref_type="IS",
            )
            vis.save_and_show_pw(
                pw_obj=is_grid,
                show_plot=False,
                file_path=os.path.join(output_dir, "IS_Shewhart_Chart"),
            )

        if len(self.valid_orf) > 0:
            orf_data = self.int_order_info(feat_type="ORF")
            orf_grid = vis.plot_ref_shewhart_chart(
                ref_data=orf_data,
                valid_feats=self.valid_orf,
                sample_type=sample_type,
                batch=batch,
                inject_order=inject_order,
                qc_label=qc_label,
                actual_label=actual_label,
                bound_type=bound_type,
                ref_type="ORF",
            )
            vis.save_and_show_pw(
                pw_obj=orf_grid,
                show_plot=False,
                file_path=os.path.join(output_dir, "ORF_Shewhart_Chart"),
            )

        fig_summary = vis.plot_assessor_summary_grid(
            pca_res=pca_res,
            rsd_data=rsd_data,
            batch_corr=batch_corr,
            corr_mat=corr_mat,
            qc_mask=qc_mask,
            batches=batches,
            method=corr_method,
            sample_type=sample_type,
            batch=batch,
            qc_label=qc_label,
            actual_label=actual_label,
            is_flags=is_flags,
            orf_flags=orf_flags,
        )

        grid_path = os.path.join(output_dir, "QA_Summary_Dashboard.svg")
        vis.save_and_show_pw(pw_obj=fig_summary, file_path=grid_path)

        logger.info(f"Assessor summary dashboard saved as: {grid_path}")
        logger.success("Data quality assessment completed.")

    def _extract_correlation_metrics(
        self,
        qc_corr_mat: pd.DataFrame | None,
        batch_qc_corr_mat: pd.DataFrame | None,
        qc_batch_labels: pd.Index | np.ndarray | list[object],
    ) -> dict[str, Any]:
        """Extracts partitioned correlation metrics (inner vs cross batch)."""
        import numpy as np

        metrics = {
            "method": self.attrs.get("corr_method", "Spearman"),
            "sample_level": {},
            "batch_level": {"is_multi_batch": False},
        }

        # --- 1. Sample-level: Inner-batch vs Cross-batch ---
        if qc_corr_mat is not None and not qc_corr_mat.empty:
            batch_array = np.array(qc_batch_labels)
            is_same_batch = batch_array[:, None] == batch_array[None, :]
            is_diff_batch = batch_array[:, None] != batch_array[None, :]
            np.fill_diagonal(is_same_batch, False)
            upper_tri = np.triu(np.ones(qc_corr_mat.shape, dtype=bool), k=1)

            inner_vals = qc_corr_mat.values[is_same_batch & upper_tri]
            cross_vals = qc_corr_mat.values[is_diff_batch & upper_tri]

            metrics["sample_level"] = {
                "inner_batch_median": (
                    float(np.median(inner_vals)) if len(inner_vals) > 0 else "N/A"
                ),
                "cross_batch_median": (
                    float(np.median(cross_vals)) if len(cross_vals) > 0 else "N/A"
                ),
            }

        # --- 2. Batch-level: Qualitative Diagnostic ---
        if batch_qc_corr_mat is not None and len(batch_qc_corr_mat) > 1:
            metrics["batch_level"]["is_multi_batch"] = True

            upper_tri = np.triu(np.ones(batch_qc_corr_mat.shape, dtype=bool), k=1)
            masked_mat = batch_qc_corr_mat.values.copy()
            masked_mat[~upper_tri] = 100.0

            min_idx = np.unravel_index(np.argmin(masked_mat), batch_qc_corr_mat.shape)
            batch_names = batch_qc_corr_mat.columns
            worst_pair = f"{batch_names[min_idx[0]]} vs {batch_names[min_idx[1]]}"

            metrics["batch_level"]["worst_batch_pair"] = worst_pair
            metrics["batch_level"]["worst_correlation"] = float(
                batch_qc_corr_mat.iloc[min_idx]
            )

        return metrics

    @cached_property
    def assessment_metrics(self) -> dict:
        """
        Extracts and caches global QA metrics for reporting.
        Aggregates partitioned correlation medians, PCA variance, outlier
        counts, and RSD distribution.
        """
        if self.empty:
            return {}

        sample_type = self.attrs.get("sample_type", "Sample Type")
        batch = self.attrs.get("batch", "Batch")
        sample_name = self.attrs.get("sample_name", "Sample Name")

        sample_dict = self.attrs.get("sample_dict", {})
        actual_label = sample_dict.get("Actual sample", "Sample")

        metrics = {"correlation": {}, "pca": {}, "outliers": {}, "rsd_distribution": {}}

        # 1. Pooled QC Correlation Metrics
        qc_data = self._qc
        if not qc_data.empty:
            corr_mat = self.qc_corr_matrix
            batch_corr = self.batch_qc_corr_matrix
            qc_batch_labels = qc_data.columns.get_level_values(batch)

            metrics["correlation"] = self._extract_correlation_metrics(
                qc_corr_mat=corr_mat,
                batch_qc_corr_mat=batch_corr,
                qc_batch_labels=qc_batch_labels,
            )
        else:
            metrics["correlation"]["method"] = self.attrs.get("corr_method", "Spearman")

        # 2. PCA and Multivariate Diagnostics
        try:
            res = self.pca_res
            diag = res.get("diagnostics", {})

            def _safe_float(val: object) -> float | None:
                return float(val) if pd.notna(val) else None

            metrics["pca"] = {
                "scaling_method": self.attrs.get("scaling_method", "Auto-scaling"),
                "pc1_variance": float(res["pca_variance"]["PC1"]),
                "pc2_variance": float(res["pca_variance"]["PC2"]),
                "relative_dispersion": _safe_float(diag.get("relative_dispersion")),
                "batch_silhouette": _safe_float(diag.get("batch_silhouette")),
                "centrality_shift": _safe_float(diag.get("centrality_shift")),
            }

            # 3. Statistical Outlier Statistics
            metrics_df = res["metrics_df"]
            actual_sample_mask = (
                metrics_df.index.get_level_values(sample_type) == actual_label
            )
            act_metrics = metrics_df[actual_sample_mask]

            cat_counts = act_metrics["Category"].value_counts()

            extreme_outlier_mask = act_metrics["Category"] == "Extreme Outlier"
            extreme_samples = (
                act_metrics[extreme_outlier_mask]
                .index.get_level_values(sample_name)
                .tolist()
            )

            metrics["outliers"] = {
                "total_tested": int(actual_sample_mask.sum()),
                "normal_count": int(cat_counts.get("Normal", 0)),
                "strong_count": int(cat_counts.get("Strong Outlier", 0)),
                "orthogonal_count": int(cat_counts.get("Orthogonal Outlier", 0)),
                "extreme_count": int(cat_counts.get("Extreme Outlier", 0)),
                "extreme_samples": extreme_samples,
            }

            # 4. Reference Features Outlier Metrics (IS & ORF)
            # --- Evaluate Internal Standards (IS) ---
            is_df = self.evaluate_reference_features(feat_type="IS")
            if not is_df.empty:
                is_out_mask = is_df["IS_Outlier_Flag"]

                valid_is = getattr(self, "valid_is", [])
                total_is = len(valid_is)
                is_raw_threshold = self.attrs.get("is_outlier_threshold", 0.75)

                if isinstance(is_raw_threshold, float) and (
                    0.0 <= is_raw_threshold <= 1.0
                ):
                    is_thr_display = f"{is_raw_threshold * 100:.0f}% of markers"
                    is_cutoff = max(1, int(np.ceil(total_is * is_raw_threshold)))
                elif isinstance(is_raw_threshold, int) and is_raw_threshold >= 1:
                    is_thr_display = f"{is_raw_threshold} absolute marker(s)"
                    is_cutoff = is_raw_threshold
                else:
                    is_thr_display = "50% of markers"
                    is_cutoff = max(1, int(np.ceil(total_is * 0.5)))

                is_out_df = is_df[is_out_mask]
                is_samples = is_out_df.index.get_level_values(sample_name).tolist()

                metrics["internal_standard_qc"] = {
                    "configured_threshold": is_thr_display,
                    "total_samples_tested": int(is_out_mask.count()),
                    "flagged_outliers_count": int(is_out_mask.sum()),
                    "is_outlier_samples": is_samples,
                    "is_outlier_standard": f">={is_cutoff}/{total_is}",
                }

            # --- Evaluate Outlier Reference Features (ORF) ---
            orf_df = self.evaluate_reference_features(feat_type="ORF")
            if not orf_df.empty:
                orf_out_mask = orf_df["ORF_Outlier_Flag"]

                valid_orf = getattr(self, "valid_orf", [])
                total_orf = len(valid_orf)
                orf_raw_threshold = self.attrs.get("orf_outlier_threshold", 0.5)

                if isinstance(orf_raw_threshold, float) and (
                    0.0 <= orf_raw_threshold <= 1.0
                ):
                    orf_thr_display = f"{orf_raw_threshold * 100:.0f}% of markers"
                    orf_cutoff = max(1, int(np.ceil(total_orf * orf_raw_threshold)))
                elif isinstance(orf_raw_threshold, int) and orf_raw_threshold >= 1:
                    orf_thr_display = f"{orf_raw_threshold} absolute marker(s)"
                    orf_cutoff = orf_raw_threshold
                else:
                    orf_thr_display = "50% of markers"
                    orf_cutoff = max(1, int(np.ceil(total_orf * 0.5)))

                orf_out_df = orf_df[orf_out_mask]
                orf_samples = orf_out_df.index.get_level_values(sample_name).tolist()

                metrics["orf_qc"] = {
                    "configured_threshold": orf_thr_display,
                    "total_samples_tested": int(orf_out_mask.count()),
                    "flagged_outliers_count": int(orf_out_mask.sum()),
                    "orf_outlier_samples": orf_samples,
                    "orf_outlier_standard": f">={orf_cutoff}/{total_orf}",
                }

        except Exception as e:
            logger.warning(f"QA metrics extraction encountered an error: {e}")

        # 5. Feature RSD Distribution Statistics
        metrics["rsd_distribution"] = self.rsd_distribution
        return metrics


class MetaboVisualizerAssessor(visualizer_classes.BaseMetaboVisualizer):
    """Visualization suite for metabolomics data quality assessment."""

    def __init__(self, qa_obj: MetaboIntAssessor) -> None:
        """Initialize with a computed MetaboIntAssessor object."""
        super().__init__(metabo_obj=qa_obj)

    # =========================================================================
    # Matrix Heatmaps and Systemic Trends Plots
    # =========================================================================
    def _format_heatmap_ticks(
        self, hm: plt.Axes, tick_color_dict: dict[object, str]
    ) -> None:
        """Format labels and assign specific colors for heatmap ticks."""

        def rename_tick(label_text: str) -> str:
            parts = re.split("-", label_text)
            if len(parts) > 4:
                return "-".join([parts[0]] + parts[4:])
            return label_text

        # Update tick label text to a shortened format for readability
        hm.set_xticklabels([rename_tick(e._text) for e in hm.get_xticklabels()])
        hm.set_yticklabels([rename_tick(e._text) for e in hm.get_yticklabels()])

        # Apply specific batch colors to tick labels for group separation
        for ax_labels in (hm.get_xticklabels(), hm.get_yticklabels()):
            for label in ax_labels:
                for batch, color in tick_color_dict.items():
                    if re.match(pattern=f"^{batch}", string=label._text):
                        label.set_color(color)
                        break

    @staticmethod
    def _draw_visible_heatmap_cell_edges(
        ax: plt.Axes,
        visible_mask: np.ndarray,
        linewidth: float,
        edgecolor: str = "k",
        zorder: float = 4,
    ) -> None:
        """Draw each visible heatmap cell edge once for uniform vector output."""
        edge_segments = set()
        for row_idx, col_idx in np.argwhere(visible_mask):
            x0, x1 = float(col_idx), float(col_idx + 1)
            y0, y1 = float(row_idx), float(row_idx + 1)
            for edge in (
                ((x0, y0), (x1, y0)),
                ((x0, y1), (x1, y1)),
                ((x0, y0), (x0, y1)),
                ((x1, y0), (x1, y1)),
            ):
                edge_segments.add(edge)

        if not edge_segments:
            return

        ax.add_collection(
            LineCollection(
                list(edge_segments),
                colors=edgecolor,
                linewidths=linewidth,
                zorder=zorder,
                clip_on=False,
            )
        )

    def plot_qc_corr_heatmap(
        self,
        corr_matrix: pd.DataFrame,
        corr_mask: Optional[np.ndarray],
        batches: list[object] | pd.Index | np.ndarray,
        method: str = "spearman",
        vmin: float = 0.85,
        vmax: float = 1.0,
        cluster: str = "within-group",
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Plot sample-level correlation matrix with rigorous clustering forests.

        Features:
        1. Cluster modes: 'total' (global), 'within-group' (batch-isolated forests), or 'none'.
        2. Absolute mathematical alignment using GridSpec (bypassing bounding-box drift).
        3. Standardized formatting via _apply_standard_format with dynamic tick scaling.
        4. Complete eradication of auto-generated axis labels to preserve grid cleanliness.

        Args:
            corr_matrix (pd.DataFrame): Correlation matrix of QC samples.
            corr_mask (Optional[np.ndarray]): (Ignored, hardcoded to lower-triangle geometrically).
            batches (list): List of unique batch identifiers.
            method (str): Correlation metric used.
            vmin/vmax (float): Colormap bounds.
            cluster (str): 'total', 'within-group', or 'none'.
            ax (Optional[Any]): Matplotlib Axes for constrained plotting.

        Returns:
            matplotlib.figure.Figure: The fully assembled figure object.
        """
        import scipy.cluster.hierarchy as sch
        from scipy.spatial.distance import squareform
        import matplotlib.patches as mpatches

        n_samples = corr_matrix.shape[0]
        is_multi_idx = isinstance(corr_matrix.index, pd.MultiIndex)

        # =====================================================================
        # 1. Clustering Strategy Routing
        # =====================================================================
        Z_list = []
        n_list = []
        new_order = []

        cluster_mode = str(cluster).lower().strip()

        if cluster_mode == "total":
            sub_corr = corr_matrix.values.astype(float)
            np.fill_diagonal(sub_corr, 1.0)
            dist_mat = np.sqrt(np.clip(1.0 - sub_corr, 0.0, 2.0))
            dist_mat = (dist_mat + dist_mat.T) / 2.0
            condensed = squareform(dist_mat, checks=False)

            Z = sch.linkage(condensed, method="ward")
            leaf_order = sch.leaves_list(Z)

            Z_list.append(Z)
            n_list.append(n_samples)
            new_order = list(leaf_order)

        elif cluster_mode == "within-group":
            for b in batches:
                if is_multi_idx:
                    b_mask = corr_matrix.index.get_level_values(self.bat_col) == b
                else:
                    b_mask = corr_matrix.index.str.startswith(f"{b}")

                idx_b = np.where(b_mask)[0]
                n_b = len(idx_b)

                if n_b > 1:
                    sub_corr = corr_matrix.iloc[idx_b, idx_b].values.astype(float)
                    np.fill_diagonal(sub_corr, 1.0)
                    dist_mat = np.sqrt(np.clip(1.0 - sub_corr, 0.0, 2.0))
                    dist_mat = (dist_mat + dist_mat.T) / 2.0
                    condensed = squareform(dist_mat, checks=False)

                    Z_b = sch.linkage(condensed, method="ward")
                    leaf_order = sch.leaves_list(Z_b)

                    Z_list.append(Z_b)
                    n_list.append(n_b)
                    new_order.extend(idx_b[leaf_order])
                elif n_b == 1:
                    Z_list.append(None)
                    n_list.append(n_b)
                    new_order.extend(idx_b)

            missing = list(set(range(n_samples)) - set(new_order))
            if missing:
                new_order.extend(missing)
                Z_list.append(None)
                n_list.append(len(missing))

        else:  # "none"
            new_order = list(range(n_samples))

        if cluster_mode in ["total", "within-group"]:
            corr_matrix = corr_matrix.iloc[new_order, new_order]

        # =====================================================================
        # 2. Batch Colors Preparation
        # =====================================================================
        custom_cmap = pu.custom_linear_cmap(["white", pu.PRIMARY_ACCENT_COLOR], 100)
        color_map = mcolors.ListedColormap(
            pu.extract_linear_cmap(custom_cmap, cmin=0.2, cmax=1.0)
        )
        color_map.set_bad(color="white", alpha=1.0)

        tick_colors = pu.extract_linear_cmap(
            cmap=custom_cmap, cmin=0.5, cmax=1.0, n_colors=len(batches)
        )
        tick_color_dict = dict(zip(batches, tick_colors))

        if is_multi_idx:
            ordered_batches = corr_matrix.index.get_level_values(self.bat_col).values
        else:
            ordered_batches = [str(x).split("-")[0] for x in corr_matrix.index]

        # =====================================================================
        # 3. Canvas & Layout Integration (Matplotlib GridSpec)
        # =====================================================================
        hm_size = max(5.0, n_samples * 0.2 + 1.0)

        annot_fmt = ".3f" if n_samples <= 15 else ".2f"
        show_annot = n_samples <= 15
        annot_size = max(5.0, min(12.0, 84.0 / max(1, n_samples)))
        cell_edge_lw = 1.0

        if ax is None:
            fig = plt.figure(
                figsize=(hm_size * 1.25, hm_size * 1.25), constrained_layout=True
            )
            if cluster_mode in ["total", "within-group"]:
                gs = fig.add_gridspec(2, 2, width_ratios=[1, 6], height_ratios=[6, 1])
                ax_heatmap = fig.add_subplot(gs[0, 1])
                ax_dendro_left = fig.add_subplot(gs[0, 0])
                ax_dendro_bottom = fig.add_subplot(gs[1, 1])
            else:
                ax_heatmap = fig.add_subplot(111)
        else:
            fig = ax.figure if hasattr(ax, "figure") else plt.gcf()
            ax_heatmap = ax
            if cluster_mode in ["total", "within-group"]:
                ax_dendro_left = ax_heatmap.inset_axes([-0.15, 0, 0.12, 1.0])
                ax_dendro_bottom = ax_heatmap.inset_axes([0, -0.15, 1.0, 0.12])

        heatmap_w, heatmap_h = pu.axis_size_inches(ax_heatmap)
        raw_x_labels = pu.index_to_tick_labels(corr_matrix.columns)
        raw_y_labels = pu.index_to_tick_labels(corr_matrix.index)
        needs_dense_ticks = pu.tick_labels_need_compaction(
            labels=raw_x_labels + raw_y_labels,
            n_items=n_samples,
            axis_inches=min(heatmap_w, heatmap_h),
            default_size=pu.DEFAULT_AXIS_TICK_FONTSIZE,
        )
        max_tick_chars = pu.dense_label_char_limit(n_samples)
        x_tick_labels = (
            [pu.compact_tick_label(label, max_tick_chars) for label in raw_x_labels]
            if needs_dense_ticks
            else raw_x_labels
        )
        y_tick_labels = (
            [pu.compact_tick_label(label, max_tick_chars) for label in raw_y_labels]
            if needs_dense_ticks
            else raw_y_labels
        )
        max_tick_len = max(
            [len(label) for label in x_tick_labels + y_tick_labels] or [1]
        )
        x_rot = (
            90 if needs_dense_ticks and (n_samples > 12 or max_tick_len > 14) else 45
        )
        x_tick_size = pu.dense_tick_fontsize(
            n_items=n_samples,
            axis_inches=heatmap_w,
            default_size=pu.DEFAULT_AXIS_TICK_FONTSIZE,
            max_size=pu.DEFAULT_AXIS_TICK_FONTSIZE,
            min_size=1.2,
            fill_ratio=0.62 if x_rot == 90 else 0.48,
            force_dense=needs_dense_ticks,
        )
        y_tick_size = pu.dense_tick_fontsize(
            n_items=n_samples,
            axis_inches=heatmap_h,
            default_size=pu.DEFAULT_AXIS_TICK_FONTSIZE,
            max_size=pu.DEFAULT_AXIS_TICK_FONTSIZE,
            min_size=1.2,
            fill_ratio=0.62,
            force_dense=needs_dense_ticks,
        )
        tick_size = min(x_tick_size, y_tick_size)

        # ---------------------------------------------------------------------
        # Dendrogram Renderer Engine
        # ---------------------------------------------------------------------
        def _draw_shifted_dendrograms(
            Z_lst: list[np.ndarray | None],
            n_lst: list[int],
            target_ax: plt.Axes,
            orientation: str,
        ) -> None:
            offset = 0
            max_dist = 0.0

            for Z, n in zip(Z_lst, n_lst):
                if Z is not None:
                    max_dist = max(max_dist, np.max(Z[:, 2]))
                    n_coll = len(target_ax.collections)
                    n_lines = len(target_ax.lines)

                    sch.dendrogram(
                        Z, ax=target_ax, orientation=orientation, no_labels=True
                    )

                    shift = offset * 10

                    for coll in target_ax.collections[n_coll:]:
                        for path in coll.get_paths():
                            if orientation == "bottom":
                                path.vertices[:, 0] += shift
                            else:
                                path.vertices[:, 1] += shift
                        coll.set_linewidth(1.0)
                        coll.set_color("#334155")

                    for line in target_ax.lines[n_lines:]:
                        if orientation == "bottom":
                            line.set_xdata(line.get_xdata() + shift)
                        else:
                            line.set_ydata(line.get_ydata() + shift)
                        line.set_linewidth(1.0)
                        line.set_color("#334155")

                offset += n

            if orientation == "bottom":
                target_ax.set_xlim(0, offset * 10)
                target_ax.set_ylim(max_dist * 1.05, 0)
            else:
                target_ax.set_xlim(max_dist * 1.05, 0)
                target_ax.set_ylim(offset * 10, 0)

            target_ax.axis("off")

        if cluster_mode in ["total", "within-group"]:
            _draw_shifted_dendrograms(Z_list, n_list, ax_dendro_left, "left")
            _draw_shifted_dendrograms(Z_list, n_list, ax_dendro_bottom, "bottom")

        # =====================================================================
        # 4. Main Lower-Triangle Heatmap
        # =====================================================================
        geom_mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        cbar_ax = ax_heatmap.inset_axes([0.60, 0.88, 0.35, 0.035])

        with sns.axes_style("white"):
            sns.heatmap(
                corr_matrix,
                mask=geom_mask,
                vmin=vmin if vmin else corr_matrix.min().min(),
                vmax=vmax,
                cmap=color_map,
                annot=show_annot,
                fmt=annot_fmt,
                linewidths=0,
                linecolor="none",
                square=False,
                xticklabels=1,
                yticklabels=1,
                ax=ax_heatmap,
                cbar_ax=cbar_ax,
                annot_kws={"size": annot_size},
                cbar_kws={
                    "label": f"{method.title()} Corr",
                    "orientation": "horizontal",
                },
            )
            cbar_ax.xaxis.set_ticks_position("top")
            cbar_ax.xaxis.set_label_position("top")
            for spine in cbar_ax.spines.values():
                spine.set_visible(False)

        self._draw_visible_heatmap_cell_edges(
            ax=ax_heatmap,
            visible_mask=np.ones_like(geom_mask, dtype=bool),
            linewidth=cell_edge_lw,
            edgecolor="white",
            zorder=3,
        )
        self._draw_visible_heatmap_cell_edges(
            ax=ax_heatmap,
            visible_mask=~geom_mask,
            linewidth=cell_edge_lw,
            edgecolor="k",
            zorder=4,
        )
        for spine in ax_heatmap.spines.values():
            spine.set_visible(False)

        # Color Patches
        thickness = max(0.4, n_samples * 0.015)
        gap = max(0.1, n_samples * 0.005)

        for i, b in enumerate(ordered_batches):
            c = tick_color_dict.get(b, "tab:gray")
            ax_heatmap.add_patch(
                plt.Rectangle(
                    (i, n_samples + gap),
                    1,
                    thickness,
                    facecolor=c,
                    edgecolor="k",
                    linewidth=0.5,
                    clip_on=False,
                )
            )
            ax_heatmap.add_patch(
                plt.Rectangle(
                    (-thickness - gap, i),
                    thickness,
                    1,
                    facecolor=c,
                    edgecolor="k",
                    linewidth=0.5,
                    clip_on=False,
                )
            )

        # Padding math
        pt_per_unit = (hm_size * 72) / n_samples
        patch_width_in_pt = thickness * pt_per_unit
        pad_amount = max(15, int(patch_width_in_pt + 15))

        ax_heatmap.tick_params(axis="x", pad=pad_amount)
        ax_heatmap.tick_params(axis="y", pad=pad_amount)

        # Remove Seaborn-injected DataFrame axis names before final formatting.
        # This prevents 'inject_order' from being squeezed between ticks and dendrograms.
        ax_heatmap.set_xlabel("")
        ax_heatmap.set_ylabel("")

        # =====================================================================
        # 5. Standardized Formatting & Inset Legend Injection
        # =====================================================================
        # Route to standardized formatting pipeline seamlessly, injecting the optimized tick_size
        self._apply_standard_format(
            ax=ax_heatmap,
            title=f"Pooled QCs Correlation\n[{self.attrs.get('pipeline_stage', '')}]",
            xlabel="",
            ylabel="",
            title_fontsize=pu.DEFAULT_TITLE_FONTSIZE,
            label_fontsize=pu.DEFAULT_AXIS_LABEL_FONTSIZE,
            tick_fontsize=tick_size,
            append_stage=False,  # Stage is already integrated into the title
        )

        tick_positions = np.arange(n_samples) + 0.5
        ax_heatmap.set_xticks(tick_positions)
        ax_heatmap.set_yticks(tick_positions)
        ax_heatmap.set_yticklabels(
            y_tick_labels,
            rotation=0,
            ha="right",
            va="center",
            fontsize=y_tick_size,
        )
        ax_heatmap.set_xticklabels(
            x_tick_labels,
            rotation=x_rot,
            ha="right",
            va="top",
            fontsize=x_tick_size,
            rotation_mode="anchor",
        )
        ax_heatmap.tick_params(axis="x", pad=pad_amount + 3, length=2)
        ax_heatmap.tick_params(axis="y", pad=pad_amount, length=2)
        pu.apply_batch_tick_colors(ax_heatmap.get_xticklabels(), tick_color_dict)
        pu.apply_batch_tick_colors(ax_heatmap.get_yticklabels(), tick_color_dict)

        legend_handles = [
            mpatches.Patch(facecolor=c, edgecolor="k", linewidth=0.5, label=str(b))
            for b, c in tick_color_dict.items()
        ]

        ax_heatmap.legend(
            handles=legend_handles,
            title="Batch",
            loc="upper right",
            bbox_to_anchor=(0.95, 0.82),
            frameon=True,
            edgecolor="k",
        )

        self._format_single_legend(
            ax=ax_heatmap,
            group_title="Batch",
            loc="upper right",
            bbox_to_anchor=(0.95, 0.82),
        )

        # Defensive property re-assignment post-standardization
        if ax_heatmap.get_legend() is not None:
            ax_heatmap.get_legend().set_title("Batch")
            ax_heatmap.get_legend().get_title().set_fontweight("bold")
            ax_heatmap.get_legend().get_title().set_fontsize(11)
            ax_heatmap.get_legend().set_bbox_to_anchor((0.95, 0.82))

        return fig

    def plot_batch_corr_heatmap(
        self,
        batch_corr_matrix: pd.DataFrame,
        method: str,
        vmin: float = 0.85,
        vmax: float = 1.0,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Plot inter-batch QC correlation heatmap using median aggregation.

        Dynamically adapts annotation visibility and tick rotations based on
        the rendering context (standalone figure vs. rigid patchwork grid)
        and the total number of analytical batches.
        """
        n_batches = batch_corr_matrix.shape[0]

        # =====================================================================
        # 1. Context-Aware Dynamic Sizing & Annotation Logic
        # =====================================================================
        is_constrained = ax is not None

        if not is_constrained:
            # Standalone Mode: Dynamically expand figure size for many batches.
            # Ensures enough physical space for both the cells and the text.
            fig_w = max(5.0, n_batches * 0.7 + 2.0)
            fig_h = max(4.0, n_batches * 0.7 + 1.0)
            fig, current_ax = plt.subplots(figsize=(fig_w, fig_h))

            # Text easily fits even for large cohorts in standalone mode
            show_annot = n_batches <= 20
        else:
            # Grid Mode (Patchwork): Size is strictly locked by the parent brick.
            current_ax = ax
            fig = current_ax.figure

            # Disable annotations if batches exceed 6 to prevent text overlapping
            # inside the compact 4x4 inch patchwork container.
            show_annot = n_batches <= 6

        # =====================================================================
        # 2. Heatmap Rendering
        # =====================================================================
        mask = np.triu(np.ones_like(batch_corr_matrix, dtype=bool), k=1)
        custom_cmap = pu.custom_linear_cmap(["white", pu.PRIMARY_ACCENT_COLOR], 100)
        color_map = mcolors.ListedColormap(
            pu.extract_linear_cmap(custom_cmap, cmin=0.2, cmax=1.0)
        )
        color_map.set_bad(color="white", alpha=1.0)

        cbar_ax = current_ax.inset_axes([1.05, 0.1, 0.05, 0.8])

        # Dynamically adjust decimal format to save space for medium cohorts
        annot_fmt = ".3f" if n_batches <= 10 else ".2f"
        annot_size = max(5.0, min(12.0, 90.0 / max(1, n_batches)))
        cell_edge_lw = 1.0

        with sns.axes_style("white"):
            sns.heatmap(
                batch_corr_matrix,
                mask=mask,
                annot=show_annot,
                fmt=annot_fmt,
                vmin=vmin if vmin else batch_corr_matrix.min().min(),
                vmax=vmax,
                cmap=color_map,
                linewidths=0,
                linecolor="none",
                square=True,
                ax=current_ax,
                cbar_ax=cbar_ax,
                annot_kws={"size": annot_size},
                cbar_kws={"label": f"{method.title()} Correlation", "format": "%.2f"},
            )

        for spine in cbar_ax.spines.values():
            spine.set_visible(False)

        self._draw_visible_heatmap_cell_edges(
            ax=current_ax,
            visible_mask=np.ones_like(mask, dtype=bool),
            linewidth=cell_edge_lw,
            edgecolor="white",
            zorder=3,
        )
        self._draw_visible_heatmap_cell_edges(
            ax=current_ax,
            visible_mask=~mask,
            linewidth=cell_edge_lw,
            edgecolor="k",
            zorder=4,
        )
        for spine in current_ax.spines.values():
            spine.set_visible(False)

        # =====================================================================
        # 3. Dense Tick Layout
        # =====================================================================
        ax_w, ax_h = pu.axis_size_inches(current_ax)
        raw_x_labels = pu.index_to_tick_labels(batch_corr_matrix.columns)
        raw_y_labels = pu.index_to_tick_labels(batch_corr_matrix.index)
        needs_dense_ticks = pu.tick_labels_need_compaction(
            labels=raw_x_labels + raw_y_labels,
            n_items=n_batches,
            axis_inches=min(ax_w, ax_h),
            default_size=pu.DEFAULT_AXIS_TICK_FONTSIZE,
        )
        max_tick_chars = pu.dense_label_char_limit(n_batches)
        x_tick_labels = (
            [pu.compact_tick_label(label, max_tick_chars) for label in raw_x_labels]
            if needs_dense_ticks
            else raw_x_labels
        )
        y_tick_labels = (
            [pu.compact_tick_label(label, max_tick_chars) for label in raw_y_labels]
            if needs_dense_ticks
            else raw_y_labels
        )
        max_tick_len = max(
            [len(label) for label in x_tick_labels + y_tick_labels] or [1]
        )
        x_rot = (
            90 if needs_dense_ticks and (n_batches > 10 or max_tick_len > 14) else 45
        )
        x_tick_size = pu.dense_tick_fontsize(
            n_items=n_batches,
            axis_inches=ax_w,
            default_size=pu.DEFAULT_AXIS_TICK_FONTSIZE,
            max_size=pu.DEFAULT_AXIS_TICK_FONTSIZE,
            min_size=1.6,
            fill_ratio=0.70 if x_rot == 90 else 0.50,
            force_dense=needs_dense_ticks,
        )
        y_tick_size = pu.dense_tick_fontsize(
            n_items=n_batches,
            axis_inches=ax_h,
            default_size=pu.DEFAULT_AXIS_TICK_FONTSIZE,
            max_size=pu.DEFAULT_AXIS_TICK_FONTSIZE,
            min_size=1.6,
            fill_ratio=0.70,
            force_dense=needs_dense_ticks,
        )
        tick_size = min(x_tick_size, y_tick_size)

        # =====================================================================
        # 4. Standard Formatting
        # =====================================================================
        self._apply_standard_format(
            ax=current_ax,
            title="Inter-Batch Pooled QC Correlation",
            xlabel="Batch ID",
            ylabel="Batch ID",
            append_stage=True,
            title_fontsize=pu.DEFAULT_TITLE_FONTSIZE,
            label_fontsize=pu.DEFAULT_AXIS_LABEL_FONTSIZE,
            tick_fontsize=tick_size,
        )
        tick_positions = np.arange(n_batches) + 0.5
        current_ax.set_xticks(tick_positions)
        current_ax.set_yticks(tick_positions)
        current_ax.set_yticklabels(
            y_tick_labels,
            rotation=0,
            ha="right",
            va="center",
            fontsize=y_tick_size,
        )
        current_ax.set_xticklabels(
            x_tick_labels,
            rotation=x_rot,
            ha="right",
            va="top",
            fontsize=x_tick_size,
            rotation_mode="anchor",
        )
        current_ax.tick_params(axis="x", pad=5, length=2)
        current_ax.tick_params(axis="y", pad=2, length=2)

        return fig

    # =========================================================================
    # Dimensionality Reduction and Outlier Plots
    # =========================================================================
    @staticmethod
    def _place_pca_annotation(
        ax: plt.Axes,
        text_artist: object,
        occupancy_arrays: list[np.ndarray],
    ) -> None:
        """Place PCA diagnostics text in the least occupied plot corner."""
        occupancy_arrays = [
            values[np.all(np.isfinite(values), axis=1)]
            for values in occupancy_arrays
            if values.size
        ]
        if not occupancy_arrays:
            return

        try:
            ax.figure.canvas.draw()
            renderer = ax.figure.canvas.get_renderer()
            text_bbox = text_artist.get_window_extent(renderer=renderer).transformed(
                ax.transAxes.inverted()
            )
            text_width = min(0.62, max(0.32, float(text_bbox.width)))
            text_height = min(0.30, max(0.14, float(text_bbox.height)))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            text_width = 0.46
            text_height = 0.18

        data_points = np.vstack(occupancy_arrays)
        axes_points = ax.transAxes.inverted().transform(
            ax.transData.transform(data_points)
        )
        axes_points = axes_points[np.all(np.isfinite(axes_points), axis=1)]
        if axes_points.size == 0:
            return

        candidates = [
            {"xy": (0.96, 0.02), "ha": "right", "va": "bottom"},
            {"xy": (0.96, 0.98), "ha": "right", "va": "top"},
            {"xy": (0.04, 0.02), "ha": "left", "va": "bottom"},
            {"xy": (0.04, 0.98), "ha": "left", "va": "top"},
        ]
        best_candidate = candidates[0]
        best_score = float("inf")

        for rank, candidate in enumerate(candidates):
            x_anchor, y_anchor = candidate["xy"]
            if candidate["ha"] == "right":
                x0, x1 = x_anchor - text_width - 0.02, x_anchor + 0.02
            else:
                x0, x1 = x_anchor - 0.02, x_anchor + text_width + 0.02

            if candidate["va"] == "top":
                y0, y1 = y_anchor - text_height - 0.02, y_anchor + 0.02
            else:
                y0, y1 = y_anchor - 0.02, y_anchor + text_height + 0.02

            in_box = (
                (axes_points[:, 0] >= x0)
                & (axes_points[:, 0] <= x1)
                & (axes_points[:, 1] >= y0)
                & (axes_points[:, 1] <= y1)
            )
            out_of_bounds_penalty = (
                max(0.0, -x0) + max(0.0, x1 - 1.0) + max(0.0, -y0) + max(0.0, y1 - 1.0)
            )
            score = float(np.count_nonzero(in_box)) + out_of_bounds_penalty * 1000

            if score < best_score or (np.isclose(score, best_score) and rank == 0):
                best_score = score
                best_candidate = candidate

        text_artist.set_position(best_candidate["xy"])
        text_artist.set_ha(best_candidate["ha"])
        text_artist.set_va(best_candidate["va"])

    def plot_pca_scatter(
        self,
        pca_df: pd.DataFrame,
        pca_var: pd.Series,
        pca_diagnostics: dict[str, Any],
        sample_type: str,
        batch: str,
        qc_label: str,
        actual_label: str,
        x_pc: str = "PC1",
        y_pc: str = "PC2",
        draw_ce: bool = True,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Plot PCA scatter plot with confidence ellipses and QA metrics."""
        plot_df = pca_df.reset_index().copy()
        plot_df[sample_type] = plot_df[sample_type].astype("category")
        plot_df = plot_df.sort_values(by=sample_type, ascending=False)
        palette_dict = {qc_label: pu.PRIMARY_ACCENT_COLOR, actual_label: pu.NEUTRAL_COLOR}

        if ax is None:
            fig, current_ax = plt.subplots(figsize=(4, 4))
        else:
            current_ax = ax
            fig = current_ax.figure

        sns.despine(ax=current_ax)

        relative_dispersion = pca_diagnostics.get("relative_dispersion")
        silhouette_score = pca_diagnostics.get("batch_silhouette")
        centrality_shift = pca_diagnostics.get("centrality_shift")

        rd_str = (
            f"{relative_dispersion:.4f}" if pd.notna(relative_dispersion) else "N/A"
        )
        sil_str = f"{silhouette_score:.4f}" if pd.notna(silhouette_score) else "N/A"
        shift_str = f"{centrality_shift:.4f}" if pd.notna(centrality_shift) else "N/A"

        annot_text = (
            f"Relative Dispersion: {rd_str}\n"
            f"Batch Silhouette: {sil_str}\n"
            f"Centrality Shift: {shift_str}"
        )

        sns.scatterplot(
            data=plot_df,
            x=x_pc,
            y=y_pc,
            hue=sample_type,
            style=batch,
            s=50,
            edgecolor="k",
            palette=palette_dict,
            linewidth=0.5,
            ax=current_ax,
            hue_order=[qc_label, actual_label],
            style_order=self.all_batches,
            markers=self.style_map,
        )

        plot_bounds = []
        occupancy_arrays = []
        finite_x = pd.to_numeric(plot_df[x_pc], errors="coerce").to_numpy(dtype=float)
        finite_y = pd.to_numeric(plot_df[y_pc], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(finite_x) & np.isfinite(finite_y)
        if np.any(finite_mask):
            occupancy_arrays.append(
                np.column_stack((finite_x[finite_mask], finite_y[finite_mask]))
            )
            plot_bounds.append(
                (
                    float(np.nanmin(finite_x[finite_mask])),
                    float(np.nanmax(finite_x[finite_mask])),
                    float(np.nanmin(finite_y[finite_mask])),
                    float(np.nanmax(finite_y[finite_mask])),
                )
            )

        if draw_ce:
            for group in (qc_label, actual_label):
                sub_df = plot_df[plot_df[sample_type] == group]
                if not sub_df.empty:
                    ellipse = pu.confidence_ellipse(
                        x=sub_df[x_pc],
                        y=sub_df[y_pc],
                        ax=current_ax,
                        n_std=3.0,
                        facecolor=mcolors.to_rgba(palette_dict[group], alpha=0.12),
                        edgecolor=palette_dict[group],
                        linewidth=1.2,
                        zorder=3,
                    )
                    pu.mark_preserve_alpha(ellipse)
                    try:
                        vertices = ellipse.get_path().vertices
                        display_vertices = ellipse.get_transform().transform(vertices)
                        data_vertices = current_ax.transData.inverted().transform(
                            display_vertices
                        )
                        plot_bounds.append(
                            (
                                float(np.nanmin(data_vertices[:, 0])),
                                float(np.nanmax(data_vertices[:, 0])),
                                float(np.nanmin(data_vertices[:, 1])),
                                float(np.nanmax(data_vertices[:, 1])),
                            )
                        )
                        occupancy_arrays.append(data_vertices)
                    except (AttributeError, TypeError, ValueError):
                        pass

        annot_artist = current_ax.text(
            0.96,
            0.02,
            annot_text,
            transform=current_ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
            clip_on=False,
            zorder=10,
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor="white", edgecolor="none", alpha=1.0
            ),
        )

        var_x = pca_var.loc[x_pc] * 100
        var_y = pca_var.loc[y_pc] * 100
        self._apply_standard_format(
            ax=current_ax,
            xlabel=f"{x_pc} ({var_x:.1f}%)",
            ylabel=f"{y_pc} ({var_y:.1f}%)",
            append_stage=True,
            title="Pooled QC & Sample PCA Scatter",
        )

        self._format_multi_legends(
            ax=current_ax,
            group_titles=[sample_type, batch],
            loc="upper left",
            start_bbox=(1.05, 1.0),
            row_gap=0.04,
            layout_cols=1,
            sublegend_cols=1,
        )

        if plot_bounds:
            x_min = min(bounds[0] for bounds in plot_bounds)
            x_max = max(bounds[1] for bounds in plot_bounds)
            y_min = min(bounds[2] for bounds in plot_bounds)
            y_max = max(bounds[3] for bounds in plot_bounds)
            x_span = max(x_max - x_min, 1.0)
            y_span = max(y_max - y_min, 1.0)
            current_ax.set_xlim(x_min - 0.08 * x_span, x_max + 0.08 * x_span)
            current_ax.set_ylim(y_min - 0.16 * y_span, y_max + 0.12 * y_span)
        else:
            current_ax.autoscale()

        self._place_pca_annotation(
            ax=current_ax,
            text_artist=annot_artist,
            occupancy_arrays=occupancy_arrays,
        )
        return fig

    # =========================================================================
    # Intra-Run Stability Validation Plots
    # =========================================================================
    def plot_rsd_bar(
        self,
        rsd_data: dict[str, dict[str, int]],
        qc_label: str,
        actual_label: str,
        ax: plt.Axes | None = None,
        **kwargs: object,
    ) -> plt.Figure:
        """Plots the RSD distribution using explicitly provided data.

        Converts the pre-calculated RSD dictionary into a format suitable
        for seaborn. Applies custom RGBA alpha blending, container styling,
        and removes zero-height patches to prevent annotation artifacts.

        Args:
            rsd_data (dict): Pre-calculated RSD distribution dictionary.
            qc_label (str): Label for QC samples.
            actual_label (str): Label for actual samples.
            ax (matplotlib.axes.Axes, optional): The target axes object.
            **kwargs: Additional formatting parameters.

        Returns:
            matplotlib.figure.Figure: The rendered figure object.
        """
        # Initialize axes hierarchy
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(5.5, 4))
        else:
            current_ax = ax
            fig = current_ax.figure

        labels = ["0-10%", "10-20%", "20-30%", ">30%"]
        x_pos = np.arange(len(labels), dtype=float)
        bar_width = 0.36
        qc_counts = np.asarray(
            [rsd_data.get("qc", {}).get(label, 0) for label in labels]
        )
        actual_counts = np.asarray(
            [rsd_data.get("actual", {}).get(label, 0) for label in labels]
        )
        qc_bar_colors = [
            pu.get_equivalent_hex(
                pu.PRIMARY_ACCENT_COLOR if label == ">30%" else pu.NEUTRAL_COLOR
            )
            for label in labels
        ]
        actual_bar_colors = [
            pu.get_equivalent_hex(
                pu.PRIMARY_ACCENT_COLOR if label == ">30%" else pu.NEUTRAL_COLOR,
                alpha=0.4,
            )
            for label in labels
        ]

        current_ax.bar(
            x_pos - bar_width / 2,
            qc_counts,
            width=bar_width,
            color=qc_bar_colors,
            edgecolor="black",
            linewidth=1.0,
            linestyle="-",
            label=qc_label,
        )
        current_ax.bar(
            x_pos + bar_width / 2,
            actual_counts,
            width=bar_width,
            color=actual_bar_colors,
            edgecolor="black",
            linewidth=1.0,
            linestyle="--",
            label=actual_label,
        )

        # Update axis limit and annotate after removing empty patches
        max_count = max(
            float(np.nanmax(qc_counts)), float(np.nanmax(actual_counts)), 1.0
        )
        current_ax.set_ylim(0, max_count * 1.3)
        pu.show_values_on_bars(
            axs=current_ax,
            show_percentage=True,
            value_format="{:.0f}",
            pct_type="group",
            fontsize=8,
        )

        # Manually construct legend to ensure correct style mapping
        h_type = [
            Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=1.0),
                edgecolor="black",
                linestyle="-",
                linewidth=1.0,
                label=qc_label,
            ),
            Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=0.4),
                edgecolor="black",
                linestyle="--",
                linewidth=1.0,
                label=actual_label,
            ),
        ]

        # Bypass auto-formatters using global LEGEND_KWARGS
        legend_kwargs = getattr(self, "LEGEND_KWARGS", {}).copy()
        legend_kwargs.update({"title": "Sample Type", "loc": "best"})
        current_ax.legend(handles=h_type, **legend_kwargs)
        current_ax.set_xticks(x_pos)
        current_ax.set_xticklabels(labels)

        # Execute standardized axis formatting
        if hasattr(self, "_apply_standard_format"):
            self._apply_standard_format(
                ax=current_ax,
                title="Feature RSD Distribution",
                xlabel="RSD Bin",
                ylabel="Feature Count",
            )

        return fig

    # =========================================================================
    # Outlier Detection
    # =========================================================================
    def plot_sd_od_scatter(
        self,
        metrics_df: pd.DataFrame,
        sd_limit: float,
        od_limit: float,
        is_flags: pd.Series | None = None,
        orf_flags: pd.Series | None = None,
        ax: plt.Axes | None = None,
        show_legend: bool = True,
    ) -> plt.Figure:
        """Plot SD-OD diagnostic scatter with multi-dimensional overlays."""
        accent_solid = pu.PRIMARY_ACCENT_COLOR
        accent_alpha = pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=0.5)

        custom_pal = {
            "Normal": "tab:gray",
            "Strong Outlier": accent_alpha,
            "Orthogonal Outlier": accent_alpha,
            "Extreme Outlier": accent_solid,
        }
        custom_markers = {
            "Normal": "o",
            "Strong Outlier": "^",
            "Orthogonal Outlier": "s",
            "Extreme Outlier": "X",
        }

        if ax is None:
            fig, current_ax = plt.subplots(figsize=(4, 4))
        else:
            current_ax = ax
            fig = current_ax.figure

        sns.scatterplot(
            data=metrics_df,
            x="SD",
            y="OD",
            hue="Category",
            style="Category",
            palette=custom_pal,
            markers=custom_markers,
            s=50,
            edgecolor="k",
            linewidth=0.5,
            ax=current_ax,
            zorder=2,
        )

        # Overlay halo effect for analytical IS outliers (accent dashed circle)
        if is_flags is not None and is_flags.any():
            outlier_idx = is_flags[is_flags].index.intersection(metrics_df.index)
            subset = metrics_df.loc[outlier_idx]
            if not subset.empty:
                current_ax.scatter(
                    subset["SD"],
                    subset["OD"],
                    s=150,
                    facecolors="none",
                    edgecolors=pu.PRIMARY_ACCENT_COLOR,
                    linewidths=2.0,
                    linestyle="--",
                    zorder=3,
                    label="IS Outlier",
                )

        # Overlay halo effect for analytical ORF outliers (Orange dash-dot circle)
        if orf_flags is not None and orf_flags.any():
            outlier_idx = orf_flags[orf_flags].index.intersection(metrics_df.index)
            subset = metrics_df.loc[outlier_idx]
            if not subset.empty:
                current_ax.scatter(
                    subset["SD"],
                    subset["OD"],
                    s=180,
                    facecolors="none",
                    edgecolors="tab:orange",
                    linewidths=2.0,
                    linestyle="-.",
                    zorder=4,
                    label="ORF Outlier",
                )

        threshold_color = pu.get_equivalent_hex("k", alpha=0.6)
        current_ax.axvline(x=sd_limit, color=threshold_color, linestyle="--")
        current_ax.axhline(y=od_limit, color=threshold_color, linestyle="--")

        self._apply_standard_format(
            ax=current_ax,
            xlabel="Score Distance (Hotelling's T2)",
            ylabel="Orthogonal Distance (SPE / DModX)",
            append_stage=True,
            title="Integrated Outlier Diagnostics",
        )

        if show_legend:
            handles, labels = current_ax.get_legend_handles_labels()
            dummy_cat = mlines.Line2D([], [], color="none", label="Category")
            dummy_thr = mlines.Line2D([], [], color="none", label="Thresholds")

            line_handles = [
                mlines.Line2D(
                    [],
                    [],
                    color=threshold_color,
                    ls="--",
                    label=f"SD Limit ({sd_limit:.2f})",
                ),
                mlines.Line2D(
                    [],
                    [],
                    color=threshold_color,
                    ls="--",
                    label=f"OD Limit ({od_limit:.2f})",
                ),
            ]

            full_handles = [dummy_cat] + handles + [dummy_thr] + line_handles
            full_labels = (
                ["Category"]
                + labels
                + [
                    "Thresholds",
                    f"SD Limit ({sd_limit:.2f})",
                    f"OD Limit ({od_limit:.2f})",
                ]
            )

            current_ax.legend(full_handles, full_labels)

            self._format_multi_legends(
                ax=current_ax,
                group_titles=["Category", "Thresholds"],
                loc="upper left",
                start_bbox=(1.05, 1.0),
                row_gap=0.04,
                layout_cols=1,
                sublegend_cols=1,
            )
        elif current_ax.get_legend():
            current_ax.get_legend().remove()

        current_ax.autoscale()
        return fig

    def _plot_stat_outliers_bar(
        self,
        outliers_df: pd.DataFrame,
        sample_type: str,
        batch: str,
        sample_name: str,
        actual_label: str,
        target_param: str = "both",
        sd_limit: float | None = None,
        od_limit: float | None = None,
        show_normal: bool = False,
        is_flags: pd.Series | None = None,
        orf_flags: pd.Series | None = None,
        ax1: plt.Axes | None = None,
        ax2: plt.Axes | None = None,
        show_legend: bool = True,
    ) -> plt.Figure | None:
        """Plot outlier results with symmetrical reference flag encodings."""
        sample_types = outliers_df.index.get_level_values(sample_type)
        mask = sample_types == actual_label
        out_df = outliers_df[mask].copy()

        if out_df.empty:
            return None

        def _get_category(row: pd.Series) -> str:
            spe = row[("SPE-DModX", "Outliers (SPE-DModX)")]
            ht2 = row[("HT2", "Outliers (HT2)")]
            if spe and ht2:
                return "Extreme Outlier"
            elif spe:
                return "Orthogonal Outlier"
            elif ht2:
                return "Strong Outlier"
            return "Normal"

        cats = out_df.apply(_get_category, axis=1)

        if not show_normal:
            outlier_mask = cats != "Normal"

            if is_flags is not None:
                is_sub_mask = is_flags.loc[out_df.index].fillna(False).values
                outlier_mask = outlier_mask | is_sub_mask

            if orf_flags is not None:
                orf_sub_mask = orf_flags.loc[out_df.index].fillna(False).values
                outlier_mask = outlier_mask | orf_sub_mask

            out_df = out_df[outlier_mask].copy()
            cats = cats[outlier_mask].copy()

        if out_df.empty:
            return None

        io_col = getattr(self, "io_col", "Inject Order")
        if io_col in out_df.index.names:
            out_df = out_df.sort_index(level=io_col)
            cats = cats.loc[out_df.index]

        idx_df = out_df.index.to_frame()
        batch_str = idx_df[batch].astype(str)
        name_str = idx_df[sample_name].astype(str)
        new_idx = (batch_str + "-" + name_str).values

        # Symmetrically fetch reference flags with strict length checks
        if is_flags is not None:
            is_sub = is_flags.loc[out_df.index].fillna(False).values
        else:
            is_sub = np.zeros(len(out_df), dtype=bool)

        if orf_flags is not None:
            orf_sub = orf_flags.loc[out_df.index].fillna(False).values
        else:
            orf_sub = np.zeros(len(out_df), dtype=bool)

        labeled_idx = []
        for name, f_is, f_orf in zip(new_idx, is_sub, orf_sub):
            if f_is and f_orf:
                labeled_idx.append(f"{name} *#")
            elif f_is:
                labeled_idx.append(f"{name} *")
            elif f_orf:
                labeled_idx.append(f"{name} #")
            else:
                labeled_idx.append(name)
        new_idx = np.array(labeled_idx)

        out_df.index = new_idx
        cats.index = new_idx
        out_df = out_df.rename_axis(index=["Sample ID"])

        accent_solid = pu.PRIMARY_ACCENT_COLOR
        accent_alpha = pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=0.5)
        gray_col = pu.NEUTRAL_COLOR

        palette_spe = {
            "Extreme Outlier": accent_solid,
            "Orthogonal Outlier": accent_alpha,
            "Strong Outlier": gray_col,
            "Normal": gray_col,
        }
        palette_ht2 = {
            "Extreme Outlier": accent_solid,
            "Orthogonal Outlier": gray_col,
            "Strong Outlier": accent_alpha,
            "Normal": gray_col,
        }

        hatch_styles = {
            "Extreme Outlier": "",
            "Orthogonal Outlier": "///",
            "Strong Outlier": r"\\\\",
            "Normal": "",
        }
        cat_order = [
            "Extreme Outlier",
            "Orthogonal Outlier",
            "Strong Outlier",
            "Normal",
        ]

        if ax1 is None or ax2 is None:
            fig, (current_ax1, current_ax2) = plt.subplots(
                nrows=2, ncols=1, figsize=(out_df.shape[0] * 0.3 + 2, 7), sharex=True
            )
        else:
            current_ax1, current_ax2 = ax1, ax2
            fig = current_ax1.figure

        axes_list = [current_ax1, current_ax2]
        metrics = ["SPE-DModX", "HT2"]
        cols = ["SPE-DModX", "Hotelling T2 Score"]
        palettes = [palette_spe, palette_ht2]

        n_samples = out_df.shape[0]
        ax_w, _ = pu.axis_size_inches(current_ax2)
        raw_tick_labels = new_idx.tolist()
        needs_dense_ticks = pu.tick_labels_need_compaction(
            labels=raw_tick_labels,
            n_items=n_samples,
            axis_inches=ax_w,
            default_size=pu.DEFAULT_AXIS_TICK_FONTSIZE,
        )
        max_tick_chars = pu.dense_label_char_limit(n_samples)
        display_idx = (
            [pu.compact_tick_label(label, max_tick_chars) for label in raw_tick_labels]
            if needs_dense_ticks
            else raw_tick_labels
        )
        dynamic_tick_size = pu.dense_tick_fontsize(
            n_items=n_samples,
            axis_inches=ax_w,
            default_size=pu.DEFAULT_AXIS_TICK_FONTSIZE,
            max_size=pu.DEFAULT_AXIS_TICK_FONTSIZE,
            min_size=1.4,
            fill_ratio=0.78,
            force_dense=needs_dense_ticks,
        )
        max_full_ticks = max(40, int(ax_w * 14))
        show_all_ticks = (not needs_dense_ticks) or n_samples <= max_full_ticks
        max_sparse_ticks = max(45, int(ax_w * 10))
        step = max(1, int(np.ceil(n_samples / max_sparse_ticks)))

        for i, (ax, metric, col, pal) in enumerate(
            zip(axes_list, metrics, cols, palettes)
        ):
            df_plot = out_df[metric].reset_index()
            df_plot["Category"] = cats.values
            present_cats = [c for c in cat_order if c in cats.values]

            sns.barplot(
                ax=ax,
                data=df_plot,
                x="Sample ID",
                y=col,
                hue="Category",
                palette=pal,
                hue_order=present_cats,
                dodge=False,
            )

            for j, cat in enumerate(present_cats):
                if j < len(ax.containers):
                    for bar in ax.containers[j]:
                        bar.set_facecolor(pal[cat])
                        bar.set_edgecolor("black")
                        bar.set_linewidth(0.8)
                        bar.set_hatch(hatch_styles[cat])

            if i == 0 and od_limit is not None:
                ax.axhline(
                    y=od_limit,
                    color="k",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                    zorder=2,
                )
            elif i == 1 and sd_limit is not None:
                ax.axhline(
                    y=sd_limit,
                    color="k",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                    zorder=2,
                )

            if i == 0:
                self._apply_standard_format(
                    ax=ax,
                    title="Integrated Outlier Barplot",
                    title_fontsize=pu.DEFAULT_TITLE_FONTSIZE,
                    label_fontsize=pu.DEFAULT_AXIS_LABEL_FONTSIZE,
                    append_stage=True,
                    ylabel="Orthogonal Distance\n(SPE / DModX)",
                )
            else:
                self._apply_standard_format(
                    ax=ax,
                    title="",
                    xlabel="Sample ID",
                    title_fontsize=pu.DEFAULT_TITLE_FONTSIZE,
                    label_fontsize=pu.DEFAULT_AXIS_LABEL_FONTSIZE,
                    append_stage=False,
                    ylabel="Score Distance\n(Hotelling's T2)",
                )

            if ax.get_legend():
                ax.get_legend().remove()

        current_ax1.set_xlabel("")
        current_ax1.tick_params(axis="x", bottom=False, labelbottom=False)
        current_ax2.tick_params(axis="x", bottom=True, labelbottom=True)
        current_ax2.set_xlabel("Sample ID")
        current_ax2.xaxis.label.set_fontsize(pu.DEFAULT_AXIS_LABEL_FONTSIZE)
        current_ax2.xaxis.label.set_fontweight(pu.DEFAULT_AXIS_LABEL_WEIGHT)

        current_ax2.set_xticks(np.arange(n_samples))
        current_ax2.set_xticklabels(
            display_idx,
            rotation=90,
            ha="right",
            va="center",
            fontsize=dynamic_tick_size,
            rotation_mode="anchor",
        )
        current_ax2.tick_params(axis="x", pad=1, length=2)

        cat_values = cats.values
        for idx, label in enumerate(current_ax2.xaxis.get_ticklabels()):
            full_text = str(new_idx[idx]) if idx < len(new_idx) else label.get_text()
            is_extreme = idx < len(cat_values) and cat_values[idx] == "Extreme Outlier"
            if "*" in full_text or "#" in full_text or is_extreme:
                label.set_color(pu.PRIMARY_ACCENT_COLOR)

        if not show_all_ticks:
            visible_indices = {0, n_samples - 1}
            for i in range(step, n_samples - 1, step):
                if (n_samples - 1 - i) > (step * 0.7):
                    visible_indices.add(i)

            for idx, label in enumerate(current_ax2.xaxis.get_ticklabels()):
                label.set_visible(idx in visible_indices)

        return fig

    def plot_outlier_standalone_legend(
        self,
        metrics_df: pd.DataFrame,
        sd_limit: float,
        od_limit: float,
        ax: plt.Axes | None = None,
        is_flags: pd.Series | None = None,
        orf_flags: pd.Series | None = None,
    ) -> plt.Figure | plt.Axes:
        """Create a comprehensive unified legend for all outlier diagnostics."""
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(2.0, 4.0))
        else:
            current_ax = ax
            fig = current_ax.figure

        accent_solid = pu.PRIMARY_ACCENT_COLOR
        accent_alpha = pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=0.5)
        gray_col = pu.NEUTRAL_COLOR

        cat_styles = {
            "Extreme Outlier": {"color": accent_solid, "marker": "X", "hatch": ""},
            "Orthogonal Outlier": {"color": accent_alpha, "marker": "s", "hatch": "///"},
            "Strong Outlier": {"color": accent_alpha, "marker": "^", "hatch": r"\\\\"},
            "Normal": {"color": gray_col, "marker": "o", "hatch": ""},
        }

        present_categories = metrics_df["Category"].unique()
        legend_handles, legend_labels = [], []
        group_titles = ["Scatter Diagnostics", "Bar Diagnostics", "Thresholds"]

        # --- Group A: Scatter Diagnostics (Markers) ---
        legend_handles.append(
            mlines.Line2D([], [], color="none", label="Scatter Diagnostics")
        )
        legend_labels.append("Scatter Diagnostics")

        for label, style in cat_styles.items():
            if label in present_categories:
                h = mlines.Line2D(
                    [],
                    [],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle="none",
                    markersize=7,
                    markeredgecolor="k",
                    markeredgewidth=0.5,
                    label=label,
                )
                legend_handles.append(h)
                legend_labels.append(label)

        if is_flags is not None and is_flags.any():
            halo_handle = mlines.Line2D(
                [],
                [],
                color="none",
                markeredgecolor=pu.PRIMARY_ACCENT_COLOR,
                marker="o",
                markersize=9,
                markeredgewidth=2.0,
                linestyle="--",
                label="IS Outlier",
            )
            legend_handles.append(halo_handle)
            legend_labels.append("IS Outlier")

        if orf_flags is not None and orf_flags.any():
            orf_halo_handle = mlines.Line2D(
                [],
                [],
                color="none",
                markeredgecolor="tab:orange",
                marker="o",
                markersize=11,
                markeredgewidth=2.0,
                linestyle="-.",
                label="ORF Outlier",
            )
            legend_handles.append(orf_halo_handle)
            legend_labels.append("ORF Outlier")

        # --- Group B: Bar Diagnostics (Hatch Styles) ---
        legend_handles.append(
            mlines.Line2D([], [], color="none", label="Bar Diagnostics")
        )
        legend_labels.append("Bar Diagnostics")

        for label, style in cat_styles.items():
            if label in present_categories and label != "Normal":
                h = mpatches.Patch(
                    facecolor=style["color"],
                    edgecolor="black",
                    linewidth=0.8,
                    hatch=style["hatch"],
                    label=label,
                )
                legend_handles.append(h)
                legend_labels.append(label)

        if is_flags is not None and is_flags.any():
            star_handle = mlines.Line2D(
                [],
                [],
                color="none",
                markerfacecolor=pu.PRIMARY_ACCENT_COLOR,
                markeredgecolor=pu.PRIMARY_ACCENT_COLOR,
                marker=r"$\ast$",
                markersize=10,
                linestyle="none",
                label="IS Outlier",
            )
            legend_handles.append(star_handle)
            legend_labels.append("IS Outlier")

        if orf_flags is not None and orf_flags.any():
            hash_handle = mlines.Line2D(
                [],
                [],
                color="none",
                markerfacecolor="tab:orange",
                markeredgecolor="tab:orange",
                marker=r"$\#$",
                markersize=10,
                linestyle="none",
                label="ORF Outlier",
            )
            legend_handles.append(hash_handle)
            legend_labels.append("ORF Outlier")

        # --- Group C: Thresholds (Lines) ---
        legend_handles.append(mlines.Line2D([], [], color="none", label="Thresholds"))
        legend_labels.append("Thresholds")

        if sd_limit is not None:
            legend_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color="k",
                    ls="--",
                    alpha=0.8,
                    lw=1.5,
                    label=f"HT2 Limit ({sd_limit:.2f})",
                )
            )
            legend_labels.append(f"HT2 Limit ({sd_limit:.2f})")

        if od_limit is not None:
            legend_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color="k",
                    ls="--",
                    alpha=0.8,
                    lw=1.5,
                    label=f"SPE Limit ({od_limit:.2f})",
                )
            )
            legend_labels.append(f"SPE Limit ({od_limit:.2f})")

        current_ax.legend(legend_handles, legend_labels)

        self._format_multi_legends(
            ax=current_ax,
            group_titles=group_titles,
            loc="upper left",
            start_bbox=(0.0, 1.0),
            row_gap=0.04,
            layout_cols=1,
            sublegend_cols=1,
        )
        current_ax.axis("off")

        return fig if ax is None else current_ax

    def plot_ref_shewhart_chart(
        self,
        ref_data: pd.DataFrame,
        valid_feats: list[str],
        sample_type: str,
        batch: str,
        inject_order: str,
        qc_label: str,
        actual_label: str,
        bound_type: str,
        ref_type: str = "IS",
    ) -> object | None:
        """Plot Shewhart control charts with adaptive PW panels and one legend."""
        try:
            import patchworklib as pw
        except ImportError:
            return None

        if not valid_feats:
            return None

        pw.clear()
        ref_type_upper = ref_type.upper()
        plot_df = ref_data.reset_index().copy()
        plot_df[sample_type] = pd.Categorical(
            plot_df[sample_type], categories=[actual_label, qc_label], ordered=True
        )
        plot_df[batch] = plot_df[batch].astype("category")
        plot_df = plot_df.sort_values(by=sample_type, ascending=True)

        # Symmetrical visual markers mapping from previous specification
        v_color = pu.PRIMARY_ACCENT_COLOR if ref_type_upper == "IS" else "tab:orange"
        v_ls = "--" if ref_type_upper == "IS" else "-."

        # Step 1: Generate analytical control chart bricks sequentially
        plot_bricks = []
        panel_cols = 1 if len(valid_feats) == 1 else 2
        panel_size = (6.5, 2.0)
        for feat_idx, feat in enumerate(valid_feats):
            brick = pw.Brick(
                figsize=panel_size, label=f"{ref_type_upper}_shewhart_{feat_idx}"
            )

            sns.scatterplot(
                ax=brick,
                data=plot_df,
                x=inject_order,
                y=feat,
                s=40,
                edgecolor="k",
                linewidth=0.5,
                style=batch,
                palette={qc_label: pu.PRIMARY_ACCENT_COLOR, actual_label: pu.NEUTRAL_COLOR},
                hue=sample_type,
                hue_order=[qc_label, actual_label],
                markers=self.style_map,
            )

            solid, lower, upper = core_classes.MetaboInt.calculate_boundaries(
                x=ref_data[feat].values, boundary_type=bound_type
            )

            is_out = (plot_df[feat] < lower) | (plot_df[feat] > upper)
            outliers_data = plot_df[is_out]
            if not outliers_data.empty:
                brick.scatter(
                    outliers_data[inject_order],
                    outliers_data[feat],
                    facecolors="none",
                    edgecolors=v_color,
                    s=150,
                    linewidths=2.0,
                    linestyle=v_ls,
                    zorder=0,
                )

            brick.axhline(y=solid, color="k", linestyle="-", linewidth=1.5)
            brick.axhline(y=lower, color="k", linestyle="--", linewidth=1.5)
            brick.axhline(y=upper, color="k", linestyle="--", linewidth=1.5)

            self._apply_standard_format(
                ax=brick,
                title=feat,
                xlabel=inject_order,
                ylabel="Intensity",
                append_stage=True,
            )
            pu.change_axis_format(ax=brick, axis_format="sci", axis="y")
            pu.change_fontsize(ax=brick, axis="y")
            pu.change_weight(ax=brick, axis="y")
            offset_text = brick.yaxis.get_offset_text()
            offset_text.set_fontsize(pu.DEFAULT_AXIS_TICK_FONTSIZE)
            offset_text.set_weight(pu.DEFAULT_AXIS_TICK_WEIGHT)

            if brick.get_legend():
                brick.get_legend().remove()

            plot_bricks.append(brick)

        # Step 2: Construct the standalone comprehensive master legend brick
        row_bricks = []
        for row_start in range(0, len(plot_bricks), panel_cols):
            row_items = plot_bricks[row_start : row_start + panel_cols]
            if panel_cols == 2 and len(row_items) == 1:
                spacer = pw.Brick(
                    figsize=panel_size,
                    label=f"{ref_type_upper}_shewhart_spacer_{row_start}",
                )
                spacer.axis("off")
                row_items.append(spacer)

            row = row_items[0]
            for item in row_items[1:]:
                row = row | item
            row_bricks.append(row)

        plot_grid = row_bricks[0]
        for row in row_bricks[1:]:
            plot_grid = plot_grid / row

        legend_height = max(panel_size[1], len(row_bricks) * panel_size[1])
        leg_brick = pw.Brick(
            figsize=(2.5, legend_height), label=f"{ref_type_upper}_shewhart_legend"
        )
        leg_brick.axis("off")

        legend_handles = []
        legend_labels = []

        # Consolidate groups by merging Outlier Status directly into Sample Type
        group_titles = [sample_type, batch, "Control Limits"]

        # Group A: Sample Type & Outlier Status (Unified Dimension)
        legend_handles.append(mlines.Line2D([], [], color="none", label=sample_type))
        legend_labels.append(sample_type)
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=pu.PRIMARY_ACCENT_COLOR,
                marker="o",
                linestyle="none",
                markersize=6,
                markeredgecolor="k",
                markeredgewidth=0.5,
                label=qc_label,
            )
        )
        legend_labels.append(qc_label)
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color="tab:gray",
                marker="o",
                linestyle="none",
                markersize=6,
                markeredgecolor="k",
                markeredgewidth=0.5,
                label=actual_label,
            )
        )
        legend_labels.append(actual_label)

        # Append the hollow halo indicator directly inside the Sample Type group
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color="none",
                markeredgecolor=v_color,
                marker="o",
                markersize=10,
                markeredgewidth=2.0,
                linestyle=v_ls,
                label=f"{ref_type_upper} Outlier",
            )
        )
        legend_labels.append(f"{ref_type_upper} Outlier")

        # Group B: Chronological Batch Configurations (Aligned linewidth)
        legend_handles.append(mlines.Line2D([], [], color="none", label=batch))
        legend_labels.append(batch)
        for b_val in self.all_batches:
            m_style = self.style_map.get(b_val, "o")
            legend_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color="tab:gray",
                    marker=m_style,
                    linestyle="none",
                    markersize=6,
                    markeredgecolor="k",
                    markeredgewidth=0.5,
                    label=str(b_val),
                )
            )
            legend_labels.append(str(b_val))

        # Group C: Statistical Boundary Thresholds
        legend_handles.append(
            mlines.Line2D([], [], color="none", label="Control Limits")
        )
        legend_labels.append("Control Limits")

        if str(bound_type).upper() == "IQR":
            solid_label, low_label, up_label = (
                "Median",
                "Q1 - 1.5 IQR",
                "Q3 + 1.5 IQR",
            )
        else:
            solid_label, low_label, up_label = ("Mean", "Mean - 3 Std", "Mean + 3 Std")

        legend_handles.append(
            mlines.Line2D([], [], color="k", ls="-", lw=1.5, label=solid_label)
        )
        legend_labels.append(solid_label)
        legend_handles.append(
            mlines.Line2D([], [], color="k", ls="--", lw=1.5, label=low_label)
        )
        legend_labels.append(low_label)
        legend_handles.append(
            mlines.Line2D([], [], color="k", ls="--", lw=1.5, label=up_label)
        )
        legend_labels.append(up_label)

        leg_brick.legend(legend_handles, legend_labels)

        # Format layout into 2 parallel columns for optimized space distribution
        self._format_multi_legends(
            ax=leg_brick,
            group_titles=group_titles,
            loc="upper left",
            start_bbox=(0.05, 0.95),
            row_gap=0.04,
            layout_cols=1,
            column_gap=0.1,
            sublegend_cols=1,
        )

        if hasattr(leg_brick.figure, "legends"):
            for leg in list(leg_brick.figure.legends):
                leg_brick.add_artist(leg)
            leg_brick.figure.legends.clear()

        return plot_grid | leg_brick

    def plot_assessor_summary_grid(
        self,
        pca_res: dict[str, Any],
        rsd_data: dict[str, dict[str, int]],
        batch_corr: pd.DataFrame,
        corr_mat: pd.DataFrame,
        qc_mask: np.ndarray | None,
        batches: list[object] | pd.Index | np.ndarray,
        method: str,
        sample_type: str,
        batch: str,
        qc_label: str,
        actual_label: str,
        is_flags: pd.Series | None = None,
        orf_flags: pd.Series | None = None,
        sample_name: str = "Sample Name",
        target_param: str = "both",
    ) -> object | None:
        """Refactored assessment summary grid with robust flag handling."""
        try:
            import patchworklib as pw
        except ImportError:
            return None

        pw.clear()

        def _bind_legends_to_axes(ax: plt.Axes | None) -> None:
            if ax is not None and hasattr(ax.figure, "legends"):
                for leg in list(ax.figure.legends):
                    ax.add_artist(leg)
                ax.figure.legends.clear()

        # Row 1 Assembly
        ax1 = pw.Brick(figsize=(4.8, 4))
        ax1.axis("off")
        ax_corr = ax1.inset_axes([0.0, 0.0, 0.83, 1.0])

        n_batches = batch_corr.shape[0] if batch_corr is not None else 0
        if n_batches <= 1:
            self.plot_qc_corr_heatmap(
                corr_matrix=corr_mat,
                corr_mask=qc_mask,
                batches=batches,
                method=method,
                cluster="none",
                ax=ax_corr,
            )
        else:
            self.plot_batch_corr_heatmap(
                batch_corr_matrix=batch_corr, method=method, ax=ax_corr
            )
        _bind_legends_to_axes(ax_corr)

        ax2 = pw.Brick(figsize=(4.0, 4))
        self.plot_rsd_bar(
            rsd_data=rsd_data, qc_label=qc_label, actual_label=actual_label, ax=ax2
        )
        _bind_legends_to_axes(ax2)

        ax3 = pw.Brick(figsize=(5.2, 4))
        ax3.axis("off")
        ax_pca = ax3.inset_axes([0.0, 0.0, 0.77, 1.0])

        self.plot_pca_scatter(
            pca_df=pca_res["pca_scatter"],
            pca_var=pca_res["pca_variance"],
            pca_diagnostics=pca_res["diagnostics"],
            sample_type=sample_type,
            batch=batch,
            qc_label=qc_label,
            actual_label=actual_label,
            ax=ax_pca,
        )
        _bind_legends_to_axes(ax_pca)

        # Row 2 Assembly
        ax4 = pw.Brick(figsize=(4.0, 4))
        self.plot_sd_od_scatter(
            metrics_df=pca_res["metrics_df"],
            sd_limit=pca_res["sd_limit"],
            od_limit=pca_res["od_limit"],
            ax=ax4,
            show_legend=False,
            is_flags=is_flags,
            orf_flags=orf_flags,
        )

        ax5 = pw.Brick(figsize=(8.8, 4))
        ax5.axis("off")
        ax5_top = ax5.inset_axes([0.0, 0.52, 1.0, 0.48])
        ax5_bot = ax5.inset_axes([0.0, 0.0, 1.0, 0.48], sharex=ax5_top)

        self._plot_stat_outliers_bar(
            outliers_df=pca_res["outliers"],
            sample_type=sample_type,
            batch=batch,
            sample_name=sample_name,
            actual_label=actual_label,
            target_param=target_param,
            sd_limit=pca_res["sd_limit"],
            od_limit=pca_res["od_limit"],
            ax1=ax5_top,
            ax2=ax5_bot,
            show_legend=False,
            is_flags=is_flags,
            orf_flags=orf_flags,
        )

        ax6 = pw.Brick(figsize=(1.2, 4))
        self.plot_outlier_standalone_legend(
            metrics_df=pca_res["metrics_df"],
            sd_limit=pca_res["sd_limit"],
            od_limit=pca_res["od_limit"],
            ax=ax6,
            is_flags=is_flags,
            orf_flags=orf_flags,
        )
        _bind_legends_to_axes(ax6)

        return (ax1 | ax2 | ax3) / (ax4 | ax5 | ax6)
