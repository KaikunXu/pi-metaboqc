"""Stage-wise quality-assessment calculations for MetaboInt datasets.

MetaboIntAssessor computes QC and batch consistency, RSD distributions, PCA
structure, multivariate outliers, and internal-standard or reference-feature
flags. It materializes reusable assessment metrics and tables for raw data and
for each processed stage before their visual summaries are generated.
"""

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

from ...config import resolve_stage_config
from ...constants import DEFAULT_RANDOM_SEED
from ...core import model
from ...runtime import log_execution_time
from ...statistics import pca as pca_utils
from ..stage import StageResult


@dataclass(frozen=True)
class AssessmentDiagnostics:
    """Hold computed QA artifacts independently of filesystem output.

    Assessment is observational: unlike filtering or normalization, it does
    not produce a replacement intensity matrix. The diagnostic tables and
    plot inputs are therefore the data payload carried by its ``StageResult``.
    """

    qc_correlation: pd.DataFrame = field(default_factory=pd.DataFrame)
    batch_qc_correlation: pd.DataFrame = field(default_factory=pd.DataFrame)
    pca: dict[str, Any] = field(default_factory=dict)
    rsd_distribution: dict[str, dict[str, int]] = field(default_factory=dict)
    internal_standard_evaluation: pd.DataFrame = field(
        default_factory=pd.DataFrame
    )
    outlier_reference_evaluation: pd.DataFrame = field(
        default_factory=pd.DataFrame
    )
    outliers: pd.DataFrame = field(default_factory=pd.DataFrame)
    internal_standard_flags: pd.Series | None = None
    outlier_reference_flags: pd.Series | None = None
    internal_standard_data: pd.DataFrame | None = None
    outlier_reference_data: pd.DataFrame | None = None


class MetaboIntAssessor(model.MetaboInt):
    """Data quality assessment computational class for metabolomics."""

    # Register "stats" for pandas metadata propagation
    _metadata = ["attrs", "stats"]
    _RUNTIME_CONFIG_KEYS = frozenset(
        {
            "corr_method",
            "scaling_method",
            "is_outlier_threshold",
            "orf_outlier_threshold",
        }
    )

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

        configs = resolve_stage_config(
            pipeline_params,
            "MetaboIntAssessor",
            {
                "corr_method": "Spearman",
                "scaling_method": "Pareto-scaling",
                "is_outlier_threshold": 0.75,
                "orf_outlier_threshold": 0.5,
            },
            {
                "corr_method": corr_method,
                "scaling_method": scaling_method,
                "is_outlier_threshold": is_outlier_threshold,
                "orf_outlier_threshold": orf_outlier_threshold,
            },
        )

        # Flatten strictly into lifecycle attributes (SSOT)
        self.attrs.update(configs)

    @property
    def _constructor(self) -> type["MetaboIntAssessor"]:
        """Override constructor to return MetaboIntAssessor."""
        return MetaboIntAssessor

    # =========================================================================
    # Cached Statistical Calculations
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
        batch_corr = (
            batch_corr.transpose().groupby(batches).median().transpose()
        )

        return batch_corr

    @cached_property
    def rsd_distribution(self) -> dict[str, dict[str, int]]:
        """Calculates and caches the RSD distribution for QA reporting."""
        sample_type = self.attrs.get("sample_type", "Sample Type")
        actual_label = self.attrs.get("sample_dict", {}).get(
            "Actual sample", "Sample"
        )

        def _get_dist(data: pd.DataFrame) -> dict[str, int]:
            labels = ["0-10%", "10-20%", "20-30%", ">30%"]
            if data.empty:
                return {label: 0 for label in labels}

            # =================================================================
            # State-Aware Pseudo-linearization (The Magic Trick)
            # Restore exponential distribution to calculate meaningful RSD
            # =================================================================
            if self.attrs.get("is_logged", False):
                # Use exp2 to reverse both robust_log and approximate VSN glog.
                # Since RSD(C*X) == RSD(X), the constants don't affect the
                # ratio.
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

        actual_sample_mask = (
            self.columns.get_level_values(sample_type) == actual_label
        )

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
        _seed = self.attrs.get("global_seed", DEFAULT_RANDOM_SEED)
        engine = pca_utils.PCAEngine(
            n_components=2, alpha=0.05, od_method="box", global_seed=_seed
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

        outliers = pd.DataFrame(
            {
                ("SPE-DModX", "SPE-DModX"): metrics_df["OD"],
                ("SPE-DModX", "Outliers (SPE-DModX)"): metrics_df[
                    "is_od_outlier"
                ],
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
        bound_type = self.attrs.get("boundary", "IQR")

        # Dynamically retrieve threshold with type-specific default fallbacks
        default_thresh = 0.75 if feat_type_lower == "is" else 0.5
        threshold_key = f"{feat_type_lower}_outlier_threshold"
        raw_threshold = self.attrs.get(threshold_key, default_thresh)

        # Evaluate boundaries per individual reference feature
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

        # Dynamic Cutoff Resolution (Ratio vs Absolute)
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

        # Final Adjudication
        df_eval[f"{feat_type_upper}_Outlier_Flag"] = (
            df_eval[f"{feat_type_upper}_Outliers_Count"] >= effective_cutoff
        )

        return df_eval

    # =========================================================================
    # Assessment Execution
    # =========================================================================

    def compute_assessment(
        self,
    ) -> StageResult[AssessmentDiagnostics]:
        """Compute QA diagnostics without writing files or rendering figures."""
        if self.empty:
            logger.warning(
                "Empty matrix detected. Terminating QA assessment execution."
            )
            return StageResult(
                data=AssessmentDiagnostics(),
                metrics={},
                candidates=pd.DataFrame(),
                metadata={"skipped": True},
            )

        sample_type = self.attrs.get("sample_type", "Sample Type")
        batch = self.attrs.get("batch", "Batch")
        inject_order = self.attrs.get("inject_order", "Inject Order")
        sample_name = self.attrs.get("sample_name", "Sample Name")

        sample_dict = self.attrs.get("sample_dict", {})
        qc_label = sample_dict.get("QC sample", "QC")
        actual_label = sample_dict.get("Actual sample", "Sample")

        corr_method = self.attrs.get("corr_method", "Spearman")
        bound_type = self.attrs.get("boundary", "IQR")
        qc_data = self._qc

        corr_mat = self.qc_corr_matrix
        batch_corr = self.batch_qc_corr_matrix
        pca_result = self.pca_res
        rsd_data = self.rsd_distribution

        is_eval = self.evaluate_reference_features(feat_type="IS")
        orf_eval = self.evaluate_reference_features(feat_type="ORF")
        outliers_export = pca_result["outliers"].copy()

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

        is_flags = None
        if ("Internal Standard", "IS_Outlier_Flag") in outliers_export.columns:
            is_flags = (
                outliers_export[("Internal Standard", "IS_Outlier_Flag")]
                .fillna(False)
                .astype(bool)
            )

        orf_flags = None
        if (
            "Outlier Reference Feature",
            "ORF_Outlier_Flag",
        ) in outliers_export.columns:
            orf_flags = (
                outliers_export[
                    ("Outlier Reference Feature", "ORF_Outlier_Flag")
                ]
                .fillna(False)
                .astype(bool)
            )

        valid_is = tuple(self.valid_is)
        valid_orf = tuple(self.valid_orf)
        diagnostics = AssessmentDiagnostics(
            qc_correlation=corr_mat,
            batch_qc_correlation=batch_corr,
            pca=pca_result,
            rsd_distribution=rsd_data,
            internal_standard_evaluation=is_eval,
            outlier_reference_evaluation=orf_eval,
            outliers=outliers_export,
            internal_standard_flags=is_flags,
            outlier_reference_flags=orf_flags,
            internal_standard_data=(
                self.int_order_info(feat_type="IS") if valid_is else None
            ),
            outlier_reference_data=(
                self.int_order_info(feat_type="ORF") if valid_orf else None
            ),
        )
        qc_mask = np.triu(
            np.ones_like(corr_mat, dtype=bool),
            k=1,
        )
        return StageResult(
            data=diagnostics,
            metrics=self.assessment_metrics,
            candidates=outliers_export,
            metadata={
                "skipped": False,
                "sample_type": sample_type,
                "batch": batch,
                "inject_order": inject_order,
                "qc_label": qc_label,
                "actual_label": actual_label,
                "correlation_method": corr_method,
                "boundary_type": bound_type,
                "qc_batches": qc_data.columns.get_level_values(batch).unique(),
                "qc_mask": qc_mask,
                "valid_is": valid_is,
                "valid_orf": valid_orf,
            },
        )

    @log_execution_time
    def run_assessment(
        self,
        output_dir: str | None = None,
        legend_mode: str = "external",
        **runtime_overrides: object,
    ) -> StageResult[AssessmentDiagnostics]:
        """Compute, export, and render QA diagnostics through a stage runner.

        When ``output_dir`` is omitted, the complete structured result is
        returned without CSV or visualization side effects. Supplying a
        directory preserves the established artifacts while returning the same
        result for downstream in-memory consumers. Call-time settings take
        precedence over pipeline configuration and module defaults.
        """
        from .runner import AssessmentStageRunner

        # Preserve the previous empty-input behavior: validate and compute, but
        # do not create an otherwise empty artifact directory.
        effective_output_dir = None if self.empty else output_dir
        return AssessmentStageRunner(
            self,
            effective_output_dir,
            legend_mode=legend_mode,
            runtime_overrides=runtime_overrides,
            allowed_override_keys=self._RUNTIME_CONFIG_KEYS,
        ).run()

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
                    float(np.median(inner_vals))
                    if len(inner_vals) > 0
                    else "N/A"
                ),
                "cross_batch_median": (
                    float(np.median(cross_vals))
                    if len(cross_vals) > 0
                    else "N/A"
                ),
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
            worst_pair = (
                f"{batch_names[min_idx[0]]} vs {batch_names[min_idx[1]]}"
            )

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

        metrics = {
            "correlation": {},
            "pca": {},
            "outliers": {},
            "rsd_distribution": {},
        }

        # Pooled QC Correlation Metrics
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
            metrics["correlation"]["method"] = self.attrs.get(
                "corr_method", "Spearman"
            )

        # PCA and Multivariate Diagnostics
        try:
            res = self.pca_res
            diag = res.get("diagnostics", {})

            def _safe_float(val: object) -> float | None:
                return float(val) if pd.notna(val) else None

            metrics["pca"] = {
                "scaling_method": self.attrs.get(
                    "scaling_method", "Auto-scaling"
                ),
                "pc1_variance": float(res["pca_variance"]["PC1"]),
                "pc2_variance": float(res["pca_variance"]["PC2"]),
                "relative_dispersion": _safe_float(
                    diag.get("relative_dispersion")
                ),
                "batch_silhouette": _safe_float(diag.get("batch_silhouette")),
                "centrality_shift": _safe_float(diag.get("centrality_shift")),
            }

            # Statistical Outlier Statistics
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
                "orthogonal_count": int(
                    cat_counts.get("Orthogonal Outlier", 0)
                ),
                "extreme_count": int(cat_counts.get("Extreme Outlier", 0)),
                "extreme_samples": extreme_samples,
            }

            # Reference Features Outlier Metrics (IS & ORF)
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
                    is_cutoff = max(
                        1, int(np.ceil(total_is * is_raw_threshold))
                    )
                elif (
                    isinstance(is_raw_threshold, int) and is_raw_threshold >= 1
                ):
                    is_thr_display = f"{is_raw_threshold} absolute marker(s)"
                    is_cutoff = is_raw_threshold
                else:
                    is_thr_display = "50% of markers"
                    is_cutoff = max(1, int(np.ceil(total_is * 0.5)))

                is_out_df = is_df[is_out_mask]
                is_samples = is_out_df.index.get_level_values(
                    sample_name
                ).tolist()

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
                    orf_thr_display = (
                        f"{orf_raw_threshold * 100:.0f}% of markers"
                    )
                    orf_cutoff = max(
                        1, int(np.ceil(total_orf * orf_raw_threshold))
                    )
                elif (
                    isinstance(orf_raw_threshold, int)
                    and orf_raw_threshold >= 1
                ):
                    orf_thr_display = f"{orf_raw_threshold} absolute marker(s)"
                    orf_cutoff = orf_raw_threshold
                else:
                    orf_thr_display = "50% of markers"
                    orf_cutoff = max(1, int(np.ceil(total_orf * 0.5)))

                orf_out_df = orf_df[orf_out_mask]
                orf_samples = orf_out_df.index.get_level_values(
                    sample_name
                ).tolist()

                metrics["orf_qc"] = {
                    "configured_threshold": orf_thr_display,
                    "total_samples_tested": int(orf_out_mask.count()),
                    "flagged_outliers_count": int(orf_out_mask.sum()),
                    "orf_outlier_samples": orf_samples,
                    "orf_outlier_standard": f">={orf_cutoff}/{total_orf}",
                }

        except Exception as e:
            logger.warning(f"QA metrics extraction encountered an error: {e}")

        # Feature RSD Distribution Statistics
        metrics["rsd_distribution"] = self.rsd_distribution
        return metrics
