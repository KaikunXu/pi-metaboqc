# src/pimqc/assessment.py
"""
Purpose of script: Data quality assessment module for MetaboInt.
Provides PCA diagnostics, Correlation Heatmaps, and RSD metrics.
"""

import os
import re
import copy
import warnings

import numpy as np
import pandas as pd
from functools import cached_property

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
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
    
    # Register "stats" for pandas metadata propagation
    _metadata = ["attrs", "stats"]
    
    def __init__(self, *args, pipeline_params=None, **kwargs):
        """
        Initialize the data quality assessment class.

        Args:
            *args: Variable length arguments passed to pandas DataFrame.
            pipeline_params: Configuration dictionary for the pipeline.
            **kwargs: Extra arguments passed to pandas DataFrame.
        """
        super().__init__(
            *args, pipeline_params=pipeline_params, **kwargs
        )

        # Initialize local state cache for heavy computational results
        if not hasattr(self, "stats"):
            self.stats = {}

        if pipeline_params is not None:
            self.attrs["pipeline_parameters"] = pipeline_params

    @property
    def assess_params(self) -> dict:
        """Safely extract Assessor specific parameters from TOML."""
        params = self.attrs.get("pipeline_parameters", {})
        return params.get("MetaboIntAssessor", {})

    @property
    def base_params(self) -> dict:
        """Safely extract global Base parameters from TOML."""
        params = self.attrs.get("pipeline_parameters", {})
        return params.get("MetaboInt", {})

    @property
    def _constructor(self):
        """Override constructor to return MetaboIntAssessor."""
        return MetaboIntAssessor

    def __finalize__(self, other, method=None, **kwargs):
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
    def qc_corr_matrix(self):
        """
        Calculates and natively caches the QC sample correlation matrix.
        Relies on internal instance state to avoid unhashable DataFrame args.
        """
        method = self.assess_params.get("corr_method", "Spearman")
        qc_data = self._qc
        
        if qc_data.empty:
            return pd.DataFrame()
            
        return qc_data.corr(method=method.lower())

    @cached_property
    def batch_qc_corr_matrix(self):
        """Aggregates QC correlation matrix into a batch-level median matrix."""
        corr_mat = self.qc_corr_matrix
        
        if corr_mat.empty:
            return pd.DataFrame()
            
        batch = self.base_params.get("batch", "Batch")
        batches = self._qc.columns.get_level_values(batch)
        
        # Compress rows and columns sequentially to extract medians
        batch_corr = corr_mat.groupby(batches).median()
        batch_corr = batch_corr.transpose().groupby(
            batches
        ).median().transpose()
        
        return batch_corr

    @cached_property
    def rsd_distribution(self):
        """Calculates and caches the RSD distribution for QA reporting."""
        sample_type = self.base_params.get("sample_type", "Sample Type")
        actual_label = self.base_params.get("sample_dict", {}).get(
            "Actual sample", "Sample"
        )

        def _get_dist(data: pd.DataFrame):
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

        actual_sample_mask = self.columns.get_level_values(
            sample_type) == actual_label
        
        return {
            "qc": _get_dist(self._qc),
            "actual": _get_dist(self.loc[:, actual_sample_mask])
        }

    @cached_property
    def pca_res(self):
        """Execute PCA workflow, outlier detection, and diagnostic metrics."""
        sample_type = self.base_params.get("sample_type", "Sample Type")
        sample_name = self.base_params.get("sample_name", "Sample Name")
        batch = self.base_params.get("batch", "Batch")
        
        sample_dict = self.base_params.get("sample_dict", {})
        qc_label = sample_dict.get("QC sample", "QC")
        actual_label = sample_dict.get("Actual sample", "Sample")

        # Extract scaling method from Assessor configurations
        s_method = self.assess_params.get("scaling_method", "Pareto-scaling")

        # Automatically extract features from internal state with JIT scaling
        features, labels = pca_utils.PCAEngine.extract_features(
            metabo_obj=self, sample_type=sample_type, sample_name=sample_name, 
            actual_label=actual_label, qc_label=qc_label,
            scaling_method=s_method
        )

        # Initialize PCA engine with strict statistical bounds
        _seed = self.base_params.get("global_seed", 123)
        engine = pca_utils.PCAEngine(
            n_components=2, alpha=0.05, od_method="box", 
            global_seed=_seed
        )
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
        
        outliers = pd.DataFrame({
            ("SPE-DModX", "SPE-DModX"): metrics_df["OD"],
            ("SPE-DModX", "Outliers (SPE-DModX)"): metrics_df["is_od_outlier"],
            ("HT2", "Hotelling T2 Score"): metrics_df["SD"],
            ("HT2", "Outliers (HT2)"): metrics_df["is_sd_outlier"]
        }, index=multi_idx)

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
                "centrality_shift": shift_res["relative_shift"]
            }
        }

    def evaluate_reference_features(
        self, feat_type: str = "IS"
    ) -> pd.DataFrame:
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
        bound_type = self.base_params.get("boundary", "IQR")
        
        # Dynamically retrieve threshold with type-specific default fallbacks
        default_thresh = 0.75 if feat_type_lower == "is" else 0.5
        threshold_key = f"{feat_type_lower}_outlier_threshold"
        raw_threshold = self.assess_params.get(threshold_key, default_thresh)
        
        # 1. Evaluate boundaries per individual reference feature
        res_dict = {}
        for feat in valid_feats:
            # Perfectly leveraging your inherited static method
            solid, lower, upper = self.calculate_boundaries(
                x=df_ref[feat].values, boundary_type=bound_type
            )
            res_dict[f"Outliers ({feat})"] = (df_ref[feat] < lower) | (
                df_ref[feat] > upper)
            
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
        df_eval[f"{feat_type_upper}_Outlier_Flag"] = \
            df_eval[f"{feat_type_upper}_Outliers_Count"] >= effective_cutoff
        
        return df_eval
    
    # =========================================================================
    # Pipeline Execution Method
    # =========================================================================

    @iu._exe_time
    def execute_assessment(self, output_dir):
        """Execute the entire QA workflow, save tables, and render plots."""
        if self.empty:
            logger.warning(
                "Empty matrix detected. Terminating QA assessment execution."
            )
            return

        # Configuration metadata extraction (Single Source of Truth)
        sample_type = self.base_params.get("sample_type", "Sample Type")
        batch = self.base_params.get("batch", "Batch")
        inject_order = self.base_params.get("inject_order", "Inject Order")
        sample_name = self.base_params.get("sample_name", "Sample Name")
        
        sample_dict = self.base_params.get("sample_dict", {})
        qc_label = sample_dict.get("QC sample", "QC")
        actual_label = sample_dict.get("Actual sample", "Sample")
        
        corr_method = self.assess_params.get("corr_method", "Spearman")
        bound_type = self.base_params.get("boundary", "IQR")
        
        stat_outlier = "both"
        mask_flag = True

        iu._check_dir_exists(dir_path=output_dir, handle="makedirs")

        qc_data = self._qc
        actual_sample_mask = (
            self.columns.get_level_values(sample_type) == actual_label
        )
        
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
            is_eval_align.index = is_eval_align.index.get_level_values(
                sample_name
            )
            is_eval_align = is_eval_align[
                ~is_eval_align.index.duplicated(keep="first")
            ]
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
            orf_eval_align.index = orf_eval_align.index.get_level_values(
                sample_name
            )
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
            is_flags = outliers_export[
                ("Internal Standard", "IS_Outlier_Flag")
            ].fillna(False).astype(bool)

        orf_flags = None
        if (
            "Outlier Reference Feature", "ORF_Outlier_Flag"
        ) in outliers_export.columns:
            orf_flags = outliers_export[
                ("Outlier Reference Feature", "ORF_Outlier_Flag")
            ].fillna(False).astype(bool)

        # Initialize Visualizer and generate plots
        vis = MetaboVisualizerAssessor(self)
        batches = qc_data.columns.get_level_values(batch).unique()
        qc_mask = (
            np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
            if mask_flag else None)
            
        vis.save_and_close_fig(
            fig=vis.plot_qc_corr_heatmap(
                corr_matrix=corr_mat, corr_mask=qc_mask, batches=batches, 
                method=corr_method),
            file_path=os.path.join(output_dir, "QC_Correlation_Heatmap"))

        vis.save_and_close_fig(
            fig=vis.plot_batch_corr_heatmap(
                batch_corr_matrix=batch_corr, method=corr_method),
            file_path=os.path.join(output_dir, "Batch_Correlation_Heatmap"))

        vis.save_and_close_fig(
            fig=vis.plot_pca_scatter(
                pca_df=pca_res["pca_scatter"], 
                pca_var=pca_res["pca_variance"],
                pca_diagnostics=pca_res["diagnostics"],
                sample_type=sample_type, batch=batch, 
                qc_label=qc_label, actual_label=actual_label),
            file_path=os.path.join(output_dir, "PCA_Scatter_QC_Sample"))
        
        vis.save_and_close_fig(
            fig=vis.plot_sd_od_scatter(
                metrics_df=pca_res["metrics_df"], 
                sd_limit=pca_res["sd_limit"], od_limit=pca_res["od_limit"],
                is_flags=is_flags, orf_flags=orf_flags),
            file_path=os.path.join(output_dir, "Outlier_Scatter"))

        vis.save_and_close_fig(
            fig=vis.plot_rsd_bar(
                rsd_data=rsd_data, qc_label=qc_label, actual_label=actual_label),
            file_path=os.path.join(output_dir, "RSD_Barplot"))

        # Symmetrical execution of control chart visualization factory (PW Mode)
        if len(self.valid_is) > 0:
            is_data = self.int_order_info(feat_type="IS")
            is_grid = vis.plot_ref_shewhart_chart(
                ref_data=is_data, valid_feats=self.valid_is, 
                sample_type=sample_type, batch=batch, 
                inject_order=inject_order, qc_label=qc_label,
                actual_label=actual_label, bound_type=bound_type,
                ref_type="IS")
            vis.save_and_show_pw(
                pw_obj=is_grid, show_plot=False,
                file_path=os.path.join(output_dir, "IS_Shewhart_Chart"))

        if len(self.valid_orf) > 0:
            orf_data = self.int_order_info(feat_type="ORF")
            orf_grid = vis.plot_ref_shewhart_chart(
                ref_data=orf_data, valid_feats=self.valid_orf, 
                sample_type=sample_type, batch=batch, 
                inject_order=inject_order, qc_label=qc_label,
                actual_label=actual_label, bound_type=bound_type,
                ref_type="ORF")
            vis.save_and_show_pw(
                pw_obj=orf_grid, show_plot=False,
                file_path=os.path.join(output_dir, "ORF_Shewhart_Chart"))

        fig_summary = vis.plot_assessor_summary_grid(
            pca_res=pca_res, rsd_data=rsd_data, batch_corr=batch_corr, 
            corr_mat=corr_mat, qc_mask=qc_mask, batches=batches, 
            method=corr_method, sample_type=sample_type, batch=batch,
            qc_label=qc_label, actual_label=actual_label,
            is_flags=is_flags, orf_flags=orf_flags)
        
        grid_path = os.path.join(output_dir, "QA_Summary_Dashboard.svg")
        vis.save_and_show_pw(pw_obj=fig_summary, file_path=grid_path)
        
        logger.info(f"Assessor summary dashboard saved as: {grid_path}")
        logger.success("Data quality assessment completed.")

    def _extract_correlation_metrics(
        self, qc_corr_mat, batch_qc_corr_mat, qc_batch_labels
    ) -> dict:
        """Extracts partitioned correlation metrics (inner vs cross batch)."""
        import numpy as np
        
        metrics = {
            "method": self.assess_params.get("corr_method", "Spearman"),
            "sample_level": {},
            "batch_level": {"is_multi_batch": False}
        }

        # --- 1. Sample-level: Inner-batch vs Cross-batch ---
        if qc_corr_mat is not None and not qc_corr_mat.empty:
            batch_array = np.array(qc_batch_labels)
            is_same_batch = (batch_array[:, None] == batch_array[None, :])
            is_diff_batch = (batch_array[:, None] != batch_array[None, :])
            np.fill_diagonal(is_same_batch, False)
            upper_tri = np.triu(np.ones(qc_corr_mat.shape, dtype=bool), k=1)
            
            inner_vals = qc_corr_mat.values[is_same_batch & upper_tri]
            cross_vals = qc_corr_mat.values[is_diff_batch & upper_tri]
            
            metrics["sample_level"] = {
                "inner_batch_median": float(np.median(inner_vals)) 
                if len(inner_vals) > 0 else "N/A",
                "cross_batch_median": float(np.median(cross_vals)) 
                if len(cross_vals) > 0 else "N/A"
            }

        # --- 2. Batch-level: Qualitative Diagnostic ---
        if batch_qc_corr_mat is not None and len(batch_qc_corr_mat) > 1:
            metrics["batch_level"]["is_multi_batch"] = True
            
            upper_tri = np.triu(
                np.ones(batch_qc_corr_mat.shape, dtype=bool), k=1
            )
            masked_mat = batch_qc_corr_mat.values.copy()
            masked_mat[~upper_tri] = 100.0  
            
            min_idx = np.unravel_index(
                np.argmin(masked_mat), batch_qc_corr_mat.shape
            )
            batch_names = batch_qc_corr_mat.columns
            worst_pair = f"{
                batch_names[min_idx[0]]} vs {batch_names[min_idx[1]]}"
            
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
            
        sample_type = self.base_params.get("sample_type", "Sample Type")
        batch = self.base_params.get("batch", "Batch")
        sample_name = self.base_params.get("sample_name", "Sample Name")
        
        sample_dict = self.base_params.get("sample_dict", {})
        actual_label = sample_dict.get("Actual sample", "Sample")
        
        metrics = {
            "correlation": {}, "pca": {}, 
            "outliers": {}, "rsd_distribution": {}
        }
        
        # 1. Pooled QC Correlation Metrics
        qc_data = self._qc
        if not qc_data.empty:
            corr_mat = self.qc_corr_matrix
            batch_corr = self.batch_qc_corr_matrix
            qc_batch_labels = qc_data.columns.get_level_values(batch)
            
            metrics["correlation"] = self._extract_correlation_metrics(
                qc_corr_mat=corr_mat, 
                batch_qc_corr_mat=batch_corr, 
                qc_batch_labels=qc_batch_labels
            )
        else:
            metrics["correlation"]["method"] = self.assess_params.get(
                "corr_method", "Spearman"
            )
        
        # 2. PCA and Multivariate Diagnostics
        try:
            res = self.pca_res
            diag = res.get("diagnostics", {})
            
            def _safe_float(val):
                return float(val) if pd.notna(val) else None
                
            metrics["pca"] = {
                "scaling_method": self.assess_params.get(
                    "scaling_method", "Auto-scaling"),
                "pc1_variance": float(res["pca_variance"]["PC1"]),
                "pc2_variance": float(res["pca_variance"]["PC2"]),
                "relative_dispersion": _safe_float(
                    diag.get("relative_dispersion")
                ),
                "batch_silhouette": _safe_float(diag.get("batch_silhouette")),
                "centrality_shift": _safe_float(diag.get("centrality_shift"))
            }
            
            # 3. Statistical Outlier Statistics
            metrics_df = res["metrics_df"]
            actual_sample_mask = metrics_df.index.get_level_values(
                sample_type
            ) == actual_label
            act_metrics = metrics_df[actual_sample_mask]
            
            cat_counts = act_metrics["Category"].value_counts()
            
            extreme_outlier_mask = act_metrics["Category"] == "Extreme Outlier"
            extreme_samples = act_metrics[
                extreme_outlier_mask].index.get_level_values(
                sample_name).tolist()
            
            metrics["outliers"] = {
                "total_tested": int(actual_sample_mask.sum()),
                "normal_count": int(cat_counts.get("Normal", 0)),
                "strong_count": int(cat_counts.get("Strong Outlier", 0)),
                "orthogonal_count": int(
                    cat_counts.get("Orthogonal Outlier", 0)
                ),
                "extreme_count": int(cat_counts.get("Extreme Outlier", 0)),
                "extreme_samples": extreme_samples
            }
            
            # 4. Reference Features Outlier Metrics (IS & ORF)
            # --- Evaluate Internal Standards (IS) ---
            is_df = self.evaluate_reference_features(feat_type="IS")
            if not is_df.empty:
                is_out_mask = is_df["IS_Outlier_Flag"]
                
                valid_is = getattr(self, "valid_is", [])
                total_is = len(valid_is)
                is_raw_threshold = self.assess_params.get(
                    "is_outlier_threshold", 0.75
                )
                
                if isinstance(is_raw_threshold, float) and (
                    0.0 <= is_raw_threshold <= 1.0
                ):
                    is_thr_display = f"{is_raw_threshold * 100:.0f}% of markers"
                    is_cutoff = max(
                        1, int(np.ceil(total_is * is_raw_threshold))
                    )
                elif isinstance(is_raw_threshold, int) and is_raw_threshold >= 1:
                    is_thr_display = f"{is_raw_threshold} absolute marker(s)"
                    is_cutoff = is_raw_threshold
                else:
                    is_thr_display = "50% of markers"
                    is_cutoff = max(1, int(np.ceil(total_is * 0.5)))
                    
                is_out_df = is_df[is_out_mask]
                is_samples = is_out_df.index.get_level_values(
                    sample_name).tolist()
                    
                metrics["internal_standard_qc"] = {
                    "configured_threshold": is_thr_display,
                    "total_samples_tested": int(is_out_mask.count()),
                    "flagged_outliers_count": int(is_out_mask.sum()),
                    "is_outlier_samples": is_samples,
                    "is_outlier_standard": f">={is_cutoff}/{total_is}"
                }

            # --- Evaluate Outlier Reference Features (ORF) ---
            orf_df = self.evaluate_reference_features(feat_type="ORF")
            if not orf_df.empty:
                orf_out_mask = orf_df["ORF_Outlier_Flag"]
                
                valid_orf = getattr(self, "valid_orf", [])
                total_orf = len(valid_orf)
                orf_raw_threshold = self.assess_params.get(
                    "orf_outlier_threshold", 0.5
                )
                
                if isinstance(orf_raw_threshold, float) and (
                    0.0 <= orf_raw_threshold <= 1.0
                ):
                    orf_thr_display = f"{orf_raw_threshold * 100:.0f}% of markers"
                    orf_cutoff = max(
                        1, int(np.ceil(total_orf * orf_raw_threshold))
                    )
                elif isinstance(orf_raw_threshold, int) and orf_raw_threshold >= 1:
                    orf_thr_display = f"{orf_raw_threshold} absolute marker(s)"
                    orf_cutoff = orf_raw_threshold
                else:
                    orf_thr_display = "50% of markers"
                    orf_cutoff = max(1, int(np.ceil(total_orf * 0.5)))
                
                orf_out_df = orf_df[orf_out_mask]
                orf_samples = orf_out_df.index.get_level_values(
                    sample_name).tolist()
                    
                metrics["orf_qc"] = {
                    "configured_threshold": orf_thr_display,
                    "total_samples_tested": int(orf_out_mask.count()),
                    "flagged_outliers_count": int(orf_out_mask.sum()),
                    "orf_outlier_samples": orf_samples,
                    "orf_outlier_standard": f">={orf_cutoff}/{total_orf}"
                }
            
        except Exception as e:
            logger.warning(f"QA metrics extraction encountered an error: {e}")
            
        # 5. Feature RSD Distribution Statistics
        metrics["rsd_distribution"] = self.rsd_distribution
        return metrics


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

        # Apply specific batch colors to tick labels for group separation
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
            tick_fontsize=2.5, append_stage=True
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
        
        cbar_ax = current_ax.inset_axes([1.05, 0.1, 0.05, 0.8])
        
        with sns.axes_style("white"):
            sns.heatmap(
                batch_corr_matrix, mask=mask, annot=True, fmt=".4f", 
                vmin=vmin if vmin else batch_corr_matrix.min().min(), 
                vmax=vmax,
                cmap=color_map, linewidths=0.25, linecolor="white", 
                square=True, ax=current_ax, cbar_ax=cbar_ax,
                cbar_kws={
                    "label": f"{method.title()} Correlation",
                    "format": "%.2f"})

        self._apply_standard_format(
            ax=current_ax, title="Inter-Batch Pooled QC Correlation",
            xlabel="Batch ID", ylabel="Batch ID", append_stage=True,
            title_fontsize=14, label_fontsize=12, tick_fontsize=10
        )
        return fig

    # =========================================================================
    # Dimensionality Reduction and Outlier Plots
    # =========================================================================
    def plot_pca_scatter(
        self, pca_df, pca_var, pca_diagnostics, sample_type, batch, 
        qc_label, actual_label, x_pc="PC1", y_pc="PC2", 
        draw_ce=True, ax=None
    ):
        """Plot PCA scatter plot with confidence ellipses and QA metrics."""
        plot_df = pca_df.reset_index().copy()
        plot_df[sample_type] = plot_df[sample_type].astype("category")
        plot_df = plot_df.sort_values(by=sample_type, ascending=False)
        palette_dict = {qc_label: "tab:red", actual_label: "tab:gray"}
        
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(4, 4))
        else:
            current_ax = ax
            fig = current_ax.figure
            
        sns.despine(ax=current_ax)

        relative_dispersion = pca_diagnostics.get("relative_dispersion")
        silhouette_score = pca_diagnostics.get("batch_silhouette")
        centrality_shift = pca_diagnostics.get("centrality_shift")

        rd_str = f"{relative_dispersion:.4f}" if pd.notna(
            relative_dispersion) else "N/A"
        sil_str = f"{silhouette_score:.4f}" if pd.notna(
            silhouette_score) else "N/A"
        shift_str = f"{centrality_shift:.4f}" if pd.notna(
            centrality_shift) else "N/A"
        
        annot_text = (
            f"Relative Dispersion: {rd_str}\n"
            f"Batch Silhouette: {sil_str}\n"
            f"Centrality Shift: {shift_str}"
        )

        sns.scatterplot(
            data=plot_df, x=x_pc, y=y_pc, hue=sample_type, style=batch,
            s=50, edgecolor="k", palette=palette_dict, linewidth=0.5,
            ax=current_ax, hue_order=[qc_label, actual_label], 
            style_order=self.all_batches, markers=self.style_map
        )
        
        if draw_ce:
            for group in (qc_label, actual_label):
                sub_df = plot_df[plot_df[sample_type] == group]
                if not sub_df.empty:
                    pu.confidence_ellipse(
                        x=sub_df[x_pc], y=sub_df[y_pc], ax=current_ax, 
                        n_std=3.0, alpha=0.1, facecolor=palette_dict[group], 
                        edgecolor=palette_dict[group]
                    )

        current_ax.text(
            0.96, 0.02, annot_text, transform=current_ax.transAxes,
            fontsize=10, verticalalignment="bottom", 
            horizontalalignment="right", clip_on=False,
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor="white", edgecolor="none",
                alpha=0.6
            )
        ) 
                
        var_x = pca_var.loc[x_pc] * 100
        var_y = pca_var.loc[y_pc] * 100
        self._apply_standard_format(
            ax=current_ax, xlabel=f"{x_pc} ({var_x:.1f}%)", 
            ylabel=f"{y_pc} ({var_y:.1f}%)", append_stage=True,
            title="Pooled QC & Sample PCA Scatter"
        )
        
        self._format_multi_legends(
            ax=current_ax, 
            group_titles=[sample_type, batch],
            loc="upper left", start_bbox=(1.05, 1.0),
            group_pad=0.04, ncols=1
        )
        
        current_ax.autoscale()
        return fig

    # =========================================================================
    # Intra-Run Stability Validation Plots
    # =========================================================================
    def plot_rsd_bar(
        self, rsd_data, qc_label, actual_label, ax=None, **kwargs
    ):
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
        # Reconstruct seaborn-compatible DataFrame dynamically
        records = []
        for r_bin, count in rsd_data.get("qc", {}).items():
            records.append({"RSD": r_bin, "Counts": count, "Type": qc_label})
        for r_bin, count in rsd_data.get("actual", {}).items():
            records.append({"RSD": r_bin, "Counts": count, "Type": actual_label})
        plot_df = pd.DataFrame(records)

        # Initialize axes hierarchy
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(5.5, 4))
        else:
            current_ax = ax
            fig = current_ax.figure

        # Initialize barplot layout with strict explicit ordering
        labels = ["0-10%", "10-20%", "20-30%", ">30%"]
        sns.barplot(
            data=plot_df, x="RSD", y="Counts", hue="Type",
            ax=current_ax, hue_order=[qc_label, actual_label], order=labels
        )
        
        # Apply aesthetics utilizing RGBA for PDF backend stability
        for i, container in enumerate(current_ax.containers):
            # QC (i=0): Solid-like | Actual (i=1): Ghost-like, dashed
            line_style = "-" if i == 0 else "--"
            alpha_val = 1.0 if i == 0 else 0.4

            for j, bar in enumerate(container):
                base_color = "tab:red" if j == 3 else "tab:gray"
                rgba_color = mcolors.to_rgba(base_color, alpha=alpha_val)
                
                bar.set_facecolor(rgba_color)
                bar.set_edgecolor("black")
                bar.set_linestyle(line_style)
                bar.set_linewidth(1.0)

        # Physically remove zero-height bars to kill ghost labels
        for p in list(current_ax.patches):
            if p.get_height() <= 0:
                p.remove()

        # Update axis limit and annotate after removing empty patches
        max_count = plot_df["Counts"].max()
        current_ax.set_ylim(0, max_count * 1.3)
        pu.show_values_on_bars(
            axs=current_ax, show_percentage=True,
            value_format="{:.0f}", pct_type="group", fontsize=8
        )

        # Manually construct legend to ensure correct style mapping
        h_type = [
            Patch(
                facecolor=mcolors.to_rgba("tab:gray", alpha=0.9),
                edgecolor="black", linestyle="-", linewidth=1.0,
                label=qc_label
            ),
            Patch(
                facecolor=mcolors.to_rgba("tab:gray", alpha=0.4),
                edgecolor="black", linestyle="--", linewidth=1.0,
                label=actual_label
            )
        ]

        # Bypass auto-formatters using global LEGEND_KWARGS
        legend_kwargs = getattr(self, "LEGEND_KWARGS", {}).copy()
        legend_kwargs.update({"title": "Sample Type", "loc": "best"})
        current_ax.legend(handles=h_type, **legend_kwargs)

        # Execute standardized axis formatting
        if hasattr(self, "_apply_standard_format"):
            self._apply_standard_format(
                ax=current_ax,
                title="Feature RSD Distribution",
                xlabel="RSD Bin",
                ylabel="Feature Count"
            )

        return fig
    
    # =========================================================================
    # Outlier Detection
    # =========================================================================
    def plot_sd_od_scatter(
        self, metrics_df, sd_limit, od_limit, is_flags=None, orf_flags=None,
        ax=None, show_legend=True
    ):
        """Plot SD-OD diagnostic scatter with multi-dimensional overlays."""
        red_solid = "tab:red"
        red_alpha = pu.get_equivalent_hex("tab:red", alpha=0.5)
        
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
        
        # Overlay halo effect for analytical IS outliers (Red dashed circle)
        if is_flags is not None and is_flags.any():
            outlier_idx = (
                is_flags[is_flags].index.intersection(metrics_df.index)
            )
            subset = metrics_df.loc[outlier_idx]
            if not subset.empty:
                current_ax.scatter(
                    subset["SD"], subset["OD"], 
                    s=150, facecolors="none", edgecolors="tab:red",
                    linewidths=2.0, linestyle="--", zorder=0,
                    label="IS Outlier"
                )
                
        # Overlay halo effect for analytical ORF outliers (Orange dash-dot circle)
        if orf_flags is not None and orf_flags.any():
            outlier_idx = (
                orf_flags[orf_flags].index.intersection(metrics_df.index)
            )
            subset = metrics_df.loc[outlier_idx]
            if not subset.empty:
                current_ax.scatter(
                    subset["SD"], subset["OD"], 
                    s=180, facecolors="none", edgecolors="tab:orange",
                    linewidths=2.0, linestyle="-.", zorder=0,
                    label="ORF Outlier"
                )
        
        current_ax.axvline(x=sd_limit, color="k", linestyle="--", alpha=0.6)
        current_ax.axhline(y=od_limit, color="k", linestyle="--", alpha=0.6)
        
        self._apply_standard_format(
            ax=current_ax, xlabel="Score Distance (Hotelling's T2)", 
            ylabel="Orthogonal Distance (SPE / DModX)", append_stage=True,
            title="Integrated Outlier Diagnostics"
        )

        if show_legend:
            handles, labels = current_ax.get_legend_handles_labels()
            dummy_cat = mlines.Line2D([], [], color="none", label="Category")
            dummy_thr = mlines.Line2D([], [], color="none", label="Thresholds")
            
            line_handles = [
                mlines.Line2D(
                    [], [], color="k", ls="--", alpha=0.6,
                    label=f"SD Limit ({sd_limit:.2f})"),
                mlines.Line2D(
                    [], [], color="k", ls="--", alpha=0.6,
                    label=f"OD Limit ({od_limit:.2f})")
            ]
            
            full_handles = [dummy_cat] + handles + [dummy_thr] + line_handles
            full_labels = ["Category"] + labels + [
                "Thresholds", f"SD Limit ({sd_limit:.2f})",
                f"OD Limit ({od_limit:.2f})"
            ]
            
            current_ax.legend(full_handles, full_labels)
            
            self._format_multi_legends(
                ax=current_ax, 
                group_titles=["Category", "Thresholds"],
                loc="upper left", start_bbox=(1.05, 1.0),
                group_pad=0.04, ncols=1
            )
        elif current_ax.get_legend():
            current_ax.get_legend().remove()
        
        current_ax.autoscale()
        return fig

    def _plot_stat_outliers_bar(
        self, outliers_df, sample_type, batch, sample_name, actual_label,
        target_param, sd_limit=None, od_limit=None, show_normal=False,
        is_flags=None, orf_flags=None, ax1=None, ax2=None, show_legend=True
    ):
        """Plot outlier results with symmetrical reference flag encodings."""
        mask = outliers_df.index.get_level_values(sample_type) == actual_label
        out_df = outliers_df[mask].copy()

        if out_df.empty:
            return None

        def _get_category(row):
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
            outlier_mask = (cats != "Normal")

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
        new_idx = (
            idx_df[batch].astype(str) + "-" + idx_df[sample_name].astype(str)
        ).values
        
        # Parallel vector evaluation for multi-dimensional flag appending
        is_sub = (
            is_flags.loc[out_df.index].fillna(False).values 
            if is_flags is not None else np.zeros(len(out_df), dtype=bool)
        )
        orf_sub = (
            orf_flags.loc[out_df.index].fillna(False).values 
            if orf_flags is not None else np.zeros(len(out_df), dtype=bool)
        )
        
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

        red_solid = "tab:red"
        red_alpha = pu.get_equivalent_hex("tab:red", alpha=0.5)
        gray_col = "tab:gray"

        palette_spe = {
            "Extreme Outlier": red_solid, "Orthogonal Outlier": red_alpha,
            "Strong Outlier": gray_col, "Normal": gray_col
        }
        palette_ht2 = {
            "Extreme Outlier": red_solid, "Orthogonal Outlier": gray_col,
            "Strong Outlier": red_alpha, "Normal": gray_col
        }

        hatch_styles = {
            "Extreme Outlier": "", "Orthogonal Outlier": "///",
            "Strong Outlier": r"\\\\", "Normal": ""
        }
        cat_order = [
            "Extreme Outlier", "Orthogonal Outlier", "Strong Outlier", "Normal"
        ]

        if ax1 is None or ax2 is None:
            fig, (current_ax1, current_ax2) = plt.subplots(
                nrows=2, ncols=1, figsize=(out_df.shape[0] * 0.3 + 2, 7),
                sharex=True
            )
        else:
            current_ax1, current_ax2 = ax1, ax2
            fig = current_ax1.figure

        axes_list = [current_ax1, current_ax2]
        metrics = ["SPE-DModX", "HT2"]
        cols = ["SPE-DModX", "Hotelling T2 Score"]
        palettes = [palette_spe, palette_ht2]
        
        n_samples = out_df.shape[0]
        dynamic_tick_size = max(
            2.0, min(6.0, 100.0 / n_samples)
        ) if n_samples > 0 else 6.0

        for i, (ax, metric, col, pal) in enumerate(
            zip(axes_list, metrics, cols, palettes)
        ):
            df_plot = out_df[metric].reset_index()
            df_plot["Category"] = cats.values
            present_cats = [c for c in cat_order if c in cats.values]

            sns.barplot(
                ax=ax, data=df_plot, x="Sample ID", y=col,
                hue="Category", palette=pal, hue_order=present_cats,
                dodge=False
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
                    y=od_limit, color="k", linestyle="--", linewidth=1.5,
                    alpha=0.8, zorder=2
                )
            elif i == 1 and sd_limit is not None:
                ax.axhline(
                    y=sd_limit, color="k", linestyle="--", linewidth=1.5,
                    alpha=0.8, zorder=2
                )

            if i == 0:
                self._apply_standard_format(
                    ax=ax, title="Integrated Outlier Barplot",
                    title_fontsize=12, label_fontsize=12,
                    tick_fontsize=dynamic_tick_size, append_stage=True,
                    ylabel="Orthogonal Distance\n(SPE / DModX)"
                )
            else:
                self._apply_standard_format(
                    ax=ax, title="", xlabel="Sample ID",
                    title_fontsize=12, label_fontsize=12,
                    tick_fontsize=dynamic_tick_size, append_stage=False,
                    ylabel="Score Distance\n(Hotelling's T2)"
                )

            if ax.get_legend():
                ax.get_legend().remove()

        current_ax1.set_xlabel("")
        current_ax1.tick_params(axis='x', bottom=False, labelbottom=False)
        current_ax2.tick_params(axis='x', bottom=True, labelbottom=True)
        current_ax2.set_xlabel("Sample ID", fontweight="bold")

        pu.change_axis_rotation(ax=current_ax2, rotation=90, axis="x")
        
        for xlabel in current_ax2.get_xticklabels():
            text = xlabel.get_text()
            is_extreme = (cats == "Extreme Outlier").get(text, False)
            if "*" in text or "#" in text or is_extreme:
                xlabel.set_color("tab:red")

        return fig

    def plot_outlier_standalone_legend(
        self, metrics_df, sd_limit, od_limit, ax=None,
        is_flags=None, orf_flags=None
    ):
        """Create a comprehensive unified legend for all outlier diagnostics."""
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(2.0, 4.0))
        else:
            current_ax = ax
            fig = current_ax.figure

        red_solid = "tab:red"
        red_alpha = pu.get_equivalent_hex("tab:red", alpha=0.5)
        gray_col = "tab:gray"

        cat_styles = {
            "Extreme Outlier": {
                "color": red_solid, "marker": "X", "hatch": ""
            },
            "Orthogonal Outlier": {
                "color": red_alpha, "marker": "s", "hatch": "///"
            },
            "Strong Outlier": {
                "color": red_alpha, "marker": "^", "hatch": r"\\\\"
            },
            "Normal": {
                "color": gray_col, "marker": "o", "hatch": ""
            }
        }
        
        present_categories = metrics_df['Category'].unique()
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
                    [], [], color=style["color"], marker=style["marker"], 
                    linestyle='none', markersize=7, markeredgecolor='k', 
                    markeredgewidth=0.5, label=label
                )
                legend_handles.append(h)
                legend_labels.append(label)
                
        if is_flags is not None and is_flags.any():
            halo_handle = mlines.Line2D(
                [], [], color="none", markeredgecolor="tab:red", marker="o", 
                markersize=9, markeredgewidth=2.0, linestyle="--", 
                label="IS Outlier"
            )
            legend_handles.append(halo_handle)
            legend_labels.append("IS Outlier")

        if orf_flags is not None and orf_flags.any():
            orf_halo_handle = mlines.Line2D(
                [], [], color="none", markeredgecolor="tab:orange", marker="o", 
                markersize=11, markeredgewidth=2.0, linestyle="-.", 
                label="ORF Outlier"
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
                    facecolor=style["color"], edgecolor="black", 
                    linewidth=0.8, hatch=style["hatch"], label=label
                )
                legend_handles.append(h)
                legend_labels.append(label)
                
        if is_flags is not None and is_flags.any():
            star_handle = mlines.Line2D(
                [], [], color="none", markerfacecolor="tab:red", 
                markeredgecolor="tab:red", marker=r"$\ast$", 
                markersize=10, linestyle="none", label="IS Outlier"
            )
            legend_handles.append(star_handle)
            legend_labels.append("IS Outlier")

        if orf_flags is not None and orf_flags.any():
            hash_handle = mlines.Line2D(
                [], [], color="none", markerfacecolor="tab:orange", 
                markeredgecolor="tab:orange", marker=r"$\#$", 
                markersize=10, linestyle="none", label="ORF Outlier"
            )
            legend_handles.append(hash_handle)
            legend_labels.append("ORF Outlier")

        # --- Group C: Thresholds (Lines) ---
        legend_handles.append(
            mlines.Line2D([], [], color="none", label="Thresholds")
        )
        legend_labels.append("Thresholds")
        
        if sd_limit is not None:
            legend_handles.append(
                mlines.Line2D(
                    [], [], color="k", ls="--", alpha=0.8, lw=1.5, 
                    label=f"HT2 Limit ({sd_limit:.2f})"
                )
            )
            legend_labels.append(f"HT2 Limit ({sd_limit:.2f})")
            
        if od_limit is not None:
            legend_handles.append(
                mlines.Line2D(
                    [], [], color="k", ls="--", alpha=0.8, lw=1.5, 
                    label=f"SPE Limit ({od_limit:.2f})"
                )
            )
            legend_labels.append(f"SPE Limit ({od_limit:.2f})")
            
        current_ax.legend(legend_handles, legend_labels)
        
        self._format_multi_legends(
            ax=current_ax, group_titles=group_titles, 
            loc="upper left", start_bbox=(0.0, 1.0),
            group_pad=0.04, ncols=1
        )
        current_ax.axis("off")
        
        return fig if ax is None else current_ax

    def plot_ref_shewhart_chart(
        self, ref_data, valid_feats, sample_type, batch, inject_order, 
        qc_label, actual_label, bound_type, ref_type="IS"
    ):
        """Plot Shewhart control charts using patchworklib mosaic layout."""
        try:
            import patchworklib as pw
        except ImportError:
            return None
        
        pw.clear()
        ref_type_upper = ref_type.upper()
        plot_df = ref_data.reset_index().copy()
        plot_df[sample_type] = plot_df[sample_type].astype("category")
        plot_df[batch] = plot_df[batch].astype("category")
        plot_df = plot_df.sort_values(by=sample_type, ascending=False)
        
        # Symmetrical visual markers mapping from previous specification
        v_color = "tab:red" if ref_type_upper == "IS" else "tab:orange"
        v_ls = "--" if ref_type_upper == "IS" else "-."
        
        # Step 1: Generate analytical control chart bricks sequentially
        bricks = []
        for feat in valid_feats:
            # Enforce 6.5x3 aspect ratio for clean time-series visualization
            brick = pw.Brick(figsize=(6.5, 3.0))
            
            sns.scatterplot(
                ax=brick, data=plot_df, x=inject_order, y=feat, s=40, 
                edgecolor="k", linewidth=0.5, style=batch, 
                palette={qc_label: "tab:red", actual_label: "tab:gray"},
                hue=sample_type, hue_order=[qc_label, actual_label],
                markers=self.style_map
            )

            solid, lower, upper = (
                core_classes.MetaboInt.calculate_boundaries(
                    x=ref_data[feat].values, boundary_type=bound_type
                )
            )
            
            is_out = (plot_df[feat] < lower) | (plot_df[feat] > upper)
            outliers_data = plot_df[is_out]
            if not outliers_data.empty:
                brick.scatter(
                    outliers_data[inject_order], outliers_data[feat],
                    facecolors="none", edgecolors=v_color, s=150, 
                    linewidths=2.0, linestyle=v_ls, zorder=0
                )
            
            brick.axhline(y=solid, color="k", linestyle="-", linewidth=1.5)
            brick.axhline(y=lower, color="k", linestyle="--", linewidth=1.5)
            brick.axhline(y=upper, color="k", linestyle="--", linewidth=1.5)
            
            self._apply_standard_format(
                ax=brick, title=feat, xlabel=inject_order, ylabel="Intensity",
                append_stage=True
            )
            pu.change_axis_format(ax=brick, axis_format="sci", axis="y")
            
            if brick.get_legend():
                brick.get_legend().remove()
                
            bricks.append(brick)

        # Step 2: Construct the standalone comprehensive master legend brick
        leg_brick = pw.Brick(figsize=(6.5, 3.0))
        leg_brick.axis("off")
        
        legend_handles = []
        legend_labels = []
        
        # Consolidate groups by merging Outlier Status directly into Sample Type
        group_titles = [sample_type, batch, "Control Limits"]
        
        # Group A: Sample Type & Outlier Status (Unified Dimension)
        legend_handles.append(
            mlines.Line2D([], [], color="none", label=sample_type)
        )
        legend_labels.append(sample_type)
        legend_handles.append(
            mlines.Line2D(
                [], [], color="tab:red", marker="o", linestyle="none", 
                markersize=6, markeredgecolor="k", markeredgewidth=0.5,
                label=qc_label
            )
        )
        legend_labels.append(qc_label)
        legend_handles.append(
            mlines.Line2D(
                [], [], color="tab:gray", marker="o", linestyle="none", 
                markersize=6, markeredgecolor="k", markeredgewidth=0.5,
                label=actual_label
            )
        )
        legend_labels.append(actual_label)
        
        # Append the hollow halo indicator directly inside the Sample Type group
        legend_handles.append(
            mlines.Line2D(
                [], [], color="none", markeredgecolor=v_color, marker="o", 
                markersize=10, markeredgewidth=2.0, linestyle=v_ls, 
                label=f"{ref_type_upper} Outlier"
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
                    [], [], color="tab:gray", marker=m_style, linestyle="none", 
                    markersize=6, markeredgecolor="k", markeredgewidth=0.5,
                    label=str(b_val)
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
                "Median", "Q1 - 1.5 IQR", "Q3 + 1.5 IQR"
            )
        else:
            solid_label, low_label, up_label = (
                "Mean", "Mean - 3 Std", "Mean + 3 Std"
            )
            
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
            ax=leg_brick, group_titles=group_titles,
            loc="upper left", start_bbox=(0.05, 0.95),
            group_pad=0.04, ncols=1, col_pad=0.1
        )
        
        if hasattr(leg_brick.figure, "legends"):
            for leg in list(leg_brick.figure.legends):
                leg_brick.add_artist(leg)
            leg_brick.figure.legends.clear()
            
        bricks.append(leg_brick)

        # Step 3: Compile bricks into standard double-column rows
        rows = []
        for i in range(0, len(bricks), 2):
            row_bricks = bricks[i:i+2]
            if len(row_bricks) == 2:
                rows.append(row_bricks[0] | row_bricks[1])
            else:
                spacer = pw.Brick(figsize=(6.5, 3.0))
                spacer.axis("off")
                rows.append(row_bricks[0] | spacer)
                
        master_grid = rows[0]
        for row in rows[1:]:
            master_grid = master_grid / row
            
        return master_grid
    
    def plot_assessor_summary_grid(
        self, pca_res, rsd_data, batch_corr, corr_mat, qc_mask, batches,
        method, sample_type, batch, qc_label, actual_label,
        is_flags=None, orf_flags=None, sample_name="Sample Name",
        target_param="both"
    ):
        """Refactored assessment summary grid with robust flag handling."""
        try:
            import patchworklib as pw
        except ImportError:
            return None
        
        pw.clear()

        def _bind_legends_to_axes(ax):
            if ax is not None and hasattr(ax.figure, "legends"):
                for leg in list(ax.figure.legends):
                    ax.add_artist(leg)
                ax.figure.legends.clear()
        
        # Row 1 Assembly
        ax1 = pw.Brick(figsize=(4.8, 4))
        ax1.axis('off')
        ax_corr = ax1.inset_axes([0.0, 0.0, 0.83, 1.0])
        
        n_batches = batch_corr.shape[0] if batch_corr is not None else 0
        if n_batches <= 1:
            self.plot_qc_corr_heatmap(
                corr_matrix=corr_mat, corr_mask=qc_mask, batches=batches, 
                method=method, ax=ax_corr
            )
        else:
            self.plot_batch_corr_heatmap(
                batch_corr_matrix=batch_corr, method=method, ax=ax_corr
            )
        _bind_legends_to_axes(ax_corr)

        ax2 = pw.Brick(figsize=(4.0, 4))
        self.plot_rsd_bar(
            rsd_data=rsd_data, qc_label=qc_label,
            actual_label=actual_label, ax=ax2
        )
        _bind_legends_to_axes(ax2)
        
        ax3 = pw.Brick(figsize=(5.2, 4))
        ax3.axis('off')
        ax_pca = ax3.inset_axes([0.0, 0.0, 0.77, 1.0])
        
        self.plot_pca_scatter(
            pca_df=pca_res["pca_scatter"], pca_var=pca_res["pca_variance"], 
            pca_diagnostics=pca_res["diagnostics"], sample_type=sample_type, 
            batch=batch, qc_label=qc_label, actual_label=actual_label,
            ax=ax_pca
        )
        _bind_legends_to_axes(ax_pca)

        # Row 2 Assembly
        ax4 = pw.Brick(figsize=(4.0, 4))
        self.plot_sd_od_scatter(
            metrics_df=pca_res["metrics_df"], sd_limit=pca_res["sd_limit"], 
            od_limit=pca_res["od_limit"], ax=ax4, show_legend=False,
            is_flags=is_flags, orf_flags=orf_flags
        )
        
        ax5 = pw.Brick(figsize=(8.8, 4))
        ax5.axis('off')
        ax5_top = ax5.inset_axes([0.0, 0.52, 1.0, 0.48])
        ax5_bot = ax5.inset_axes([0.0, 0.0, 1.0, 0.48], sharex=ax5_top)
        
        self._plot_stat_outliers_bar(
            outliers_df=pca_res["outliers"], sample_type=sample_type, 
            batch=batch, sample_name=sample_name, actual_label=actual_label, 
            target_param=target_param, sd_limit=pca_res["sd_limit"], 
            od_limit=pca_res["od_limit"], ax1=ax5_top, ax2=ax5_bot,
            show_legend=False, is_flags=is_flags, orf_flags=orf_flags
        )
        
        ax6 = pw.Brick(figsize=(1.2, 4))
        self.plot_outlier_standalone_legend(
            metrics_df=pca_res["metrics_df"], sd_limit=pca_res["sd_limit"], 
            od_limit=pca_res["od_limit"], ax=ax6,
            is_flags=is_flags, orf_flags=orf_flags
        )
        _bind_legends_to_axes(ax6)

        return (ax1 | ax2 | ax3) / (ax4 | ax5 | ax6)