"""Missing-value triage and low-quality feature filtering calculations.

MetaboIntFilter removes high-missing samples, classifies features as MAR, MNAR,
or invalid, applies biological-group and QC rescue rules, then filters features
by blank-to-QC abundance and QC RSD. It stores retained indices and tracking
tables required by imputation, assessment, and reporting.
"""

import copy
import numpy as np
import pandas as pd
from functools import cached_property


from loguru import logger
from typing import Dict, Any, Optional

from ...runtime import log_execution_time
from ...core import model
from ...config import resolve_stage_config
from ..stage import StageResult
from .runner import (
    MissingValueFilteringStageRunner,
    QualityFilteringStageRunner,
    SampleFilteringStageRunner,
)


class MetaboIntFilter(model.MetaboInt):
    """Filtering engine for metabolomics datasets with QC enforcement."""

    _metadata = ["attrs", "stats"]
    _RUNTIME_CONFIG_KEYS = frozenset(
        {
            "sample_mv_tol",
            "mv_group_tol",
            "mv_qc_tol",
            "mnar_group_mv_tol",
            "mnar_qc_mv_tol",
            "mnar_intensity_pct",
            "qc_rsd_tol",
            "blank_qc_ratio_tol",
        }
    )

    _INVALID_STRS = {
        "unknown",
        "na",
        "n/a",
        "nan",
        "none",
        "null",
        "",
        "unassigned",
        "blank",
        "blk",
        "is",
        "solvent",
        "wash",
        "sst",
        "pool",
        "invalid",
        "unvalid",
    }

    def __init__(
        self,
        data: object | None = None,
        pipeline_params: Optional[Dict[str, Any]] = None,
        sample_mv_tol: Optional[float] = None,
        mv_group_tol: Optional[float] = None,
        mv_qc_tol: Optional[float] = None,
        mnar_group_mv_tol: Optional[float] = None,
        mnar_qc_mv_tol: Optional[float] = None,
        mnar_intensity_pct: Optional[float] = None,
        qc_rsd_tol: Optional[float] = None,
        blank_qc_ratio_tol: Optional[float] = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Initialize the filtering engine with QC enforcement.

        Args:
            data: Input intensity matrix (DataFrame-like).
            pipeline_params: Global configuration dictionary from TOML.
            sample_mv_tol: Max missing rate for sample removal.
            mv_group_tol: Base missing rate tolerance within bio groups.
            mv_qc_tol: Base missing rate tolerance in QC samples.
            mnar_group_mv_tol: Max missing rate for group MNAR rescue.
            mnar_qc_mv_tol: Max missing rate for QC MNAR rescue.
            mnar_intensity_pct: Intensity percentile threshold for MNAR QC.
            qc_rsd_tol: Max relative standard deviation in QC.
            blank_qc_ratio_tol: Max allowable blank to QC intensity ratio.
            *args: Variable arguments passed to pandas DataFrame.
            **kwargs: Extra keyword arguments passed to pandas DataFrame.
        """
        super().__init__(
            data=data, pipeline_params=pipeline_params, *args, **kwargs
        )

        if not hasattr(self, "stats"):
            self.stats = {}

        # Explicitly inherit runtime stats from upstream data object
        input_data = data if data is not None else kwargs.get("data")
        if input_data is None and len(args) > 0:
            input_data = args[0]

        if input_data is not None and hasattr(input_data, "stats"):
            self.stats.update(copy.deepcopy(input_data.stats))

        # Initialize runtime tracking if not inherited
        if "feature_counts" not in self.stats:
            self.stats["feature_counts"] = {}

        filter_configs = resolve_stage_config(
            pipeline_params,
            "MetaboIntFilter",
            {
                "sample_mv_tol": 0.5,
                "mv_group_tol": 0.5,
                "mv_qc_tol": 0.3,
                "mnar_group_mv_tol": 0.8,
                "mnar_qc_mv_tol": 0.2,
                "mnar_intensity_pct": 0.1,
                "qc_rsd_tol": 0.3,
                "blank_qc_ratio_tol": 0.2,
            },
            {
                "sample_mv_tol": sample_mv_tol,
                "mv_group_tol": mv_group_tol,
                "mv_qc_tol": mv_qc_tol,
                "mnar_group_mv_tol": mnar_group_mv_tol,
                "mnar_qc_mv_tol": mnar_qc_mv_tol,
                "mnar_intensity_pct": mnar_intensity_pct,
                "qc_rsd_tol": qc_rsd_tol,
                "blank_qc_ratio_tol": blank_qc_ratio_tol,
            },
        )
        self.attrs.update(filter_configs)
        self.stats.update(
            {
                "mv_group_df": pd.DataFrame(),
                "mv_qc_series": pd.Series(dtype=float),
                "mv_global_series": pd.Series(dtype=float),
                "blank_mean": pd.Series(dtype=float),
                "qc_mean": pd.Series(dtype=float),
                "qc_rsd_all": pd.Series(dtype=float),
                "idx_mar": pd.Index([]),
                "idx_mnar": pd.Index([]),
                "idx_mnar_group": pd.Index([]),
                "idx_mnar_qc": pd.Index([]),
            }
        )

    def _audit_snapshot(self) -> dict[str, Any]:
        """Copy runtime audit values into the explicit stage result."""
        return copy.deepcopy(self.stats)

    @property
    def _constructor(self) -> type["MetaboIntFilter"]:
        """Override constructor to return MetaboIntFilter."""
        return MetaboIntFilter

    # =========================================================================
    # Sample-Level Filtering
    # =========================================================================
    def transform_sample_filtering(self) -> StageResult[pd.DataFrame]:
        """Filter high-MV samples and retain their attrition audit table."""
        batch = self.attrs.get("batch", "Batch")
        inject_order = self.attrs.get("inject_order", "Inject Order")
        sample_mv_tol = self.attrs.get("sample_mv_tol", 0.5)

        # Strictly evaluate only QC and Actual Samples via concatenation
        df_check = pd.concat([self._qc, self._actual_sample], axis=1)
        sample_mv_rates = df_check.isna().mean(axis=0)

        # Determine status
        bad_mask = sample_mv_rates > sample_mv_tol
        bad_samples = sample_mv_rates[bad_mask].index

        # Safe drop: Removes bad samples while retaining Blanks/Others intact
        retained_samples = self.columns.difference(bad_samples)

        # Extract sample types for tracking
        sample_type = self.attrs.get("sample_type", "Sample Type")
        check_types = df_check.columns.get_level_values(sample_type)

        # Build tracking table for diagnostics
        track_df = pd.DataFrame(
            {
                "Sample_ID": sample_mv_rates.index.get_level_values(0),
                "Sample_Type": check_types,
                "MV_Rate_Pct": sample_mv_rates.values * 100,
                "Status": np.where(bad_mask, "Dropped", "Retained"),
            }
        ).set_index("Sample_ID")

        self.stats["sample_tracking"] = track_df
        self.stats["sample_dropped_idx"] = bad_samples

        if not bad_samples.empty:
            logger.warning(
                f"Dropping {len(bad_samples)} samples (MV > {sample_mv_tol})"
            )

        df_filtered = self.loc[:, retained_samples].copy()

        # Physically reorder the remaining columns by injection sequence
        sort_levels = [
            lvl
            for lvl in [batch, inject_order]
            if lvl in df_filtered.columns.names
        ]
        if sort_levels:
            df_filtered = df_filtered.sort_index(axis=1, level=sort_levels)

        return StageResult(
            data=df_filtered,
            candidates=track_df,
            metadata={"dropped_samples": bad_samples},
            audit_tables={"sample_tracking": track_df},
        )

    @log_execution_time
    def run_sample_filtering(
        self,
        output_dir: str | None = None,
        **runtime_overrides: object,
    ) -> StageResult[pd.DataFrame]:
        """Return the structured sample-filtering stage result.

        Named overrides take precedence over pipeline configuration and module
        defaults. This stage accepts ``sample_mv_tol``.
        """
        runner = SampleFilteringStageRunner(
            self,
            output_dir,
            runtime_overrides=runtime_overrides,
            allowed_override_keys=frozenset({"sample_mv_tol"}),
        )
        return runner.run()

    # =========================================================================
    # Feature-Level Missing Value Classification
    # =========================================================================

    def _get_valid_bio_groups(self) -> list[object]:
        """Extract valid biological group names from the column index."""
        bio_group = self.attrs.get("bio_group", "Bio Group")
        sample_dict = self.attrs.get("sample_dict", {})
        qc_label = sample_dict.get("QC sample", "QC")

        valid_bio_groups = []
        if bio_group in self.columns.names:
            raw_groups = self.columns.get_level_values(bio_group).unique()
            for group in raw_groups:
                if pd.isna(group):
                    continue
                group_str = str(group).strip().lower()
                if (
                    group_str in self._INVALID_STRS
                    or group_str == str(qc_label).lower()
                ):
                    continue
                valid_bio_groups.append(group)
        return valid_bio_groups

    @log_execution_time
    def classify_missing_types(self) -> tuple[pd.Index, pd.Index, pd.Index]:
        """Classifies features with strict QC enforcement and dynamic tol."""
        total_missing = self.isna().sum().sum()
        empty_idx = self.index[:0]
        if total_missing == 0:
            logger.warning("No missing values detected. Defaulting to MAR.")
            return self.index, empty_idx, empty_idx

        try:
            bio_group = self.attrs.get("bio_group", "Bio Group")
            sample_type = self.attrs.get("sample_type", "Sample Type")
            sample_dict = self.attrs.get("sample_dict", {})
            qc_label = sample_dict.get("QC sample", "QC")

            mv_group_tol = self.attrs.get("mv_group_tol", 0.5)
            mv_qc_tol = self.attrs.get("mv_qc_tol", 0.3)
            mnar_group_mv_tol = self.attrs.get("mnar_group_mv_tol", 0.8)
            mnar_qc_mv_tol = self.attrs.get("mnar_qc_mv_tol", 0.2)
            mnar_intensity_pct = self.attrs.get("mnar_intensity_pct", 0.1)

            qc_mask = (
                self.columns.get_level_values(sample_type) == qc_label
                if sample_type in self.columns.names
                else np.zeros(self.shape[1], dtype=bool)
            )

            if not qc_mask.any():
                raise ValueError("Fatal: QC samples are required.")

            idx_mnar_group = pd.Index([])
            valid_bio_groups = self._get_valid_bio_groups()

            # Group Rescue (Tier 1 - Optimal)
            if valid_bio_groups:
                na_rate_group = self.isna().T.groupby(level=bio_group).mean().T
                na_rate_valid = na_rate_group[valid_bio_groups]
                cond_group_mnar = (na_rate_valid >= mnar_group_mv_tol).any(
                    axis=1
                ) & (na_rate_valid <= mv_group_tol).any(axis=1)
                idx_mnar_group = self.index[cond_group_mnar]

            # QC Rescue (Enforced QC standard)
            df_qc = self.loc[:, qc_mask]
            qc_na_rate = df_qc.isna().mean(axis=1)
            qc_median = df_qc.median(axis=1)
            int_threshold = qc_median.quantile(mnar_intensity_pct)

            cond_qc_mv = qc_na_rate > mnar_qc_mv_tol
            cond_qc_int = qc_median <= int_threshold

            if valid_bio_groups:
                cond_qc_bio_valid = (na_rate_valid <= mv_group_tol).any(axis=1)
                idx_mnar_qc = self.index[
                    cond_qc_mv & cond_qc_int & cond_qc_bio_valid
                ]
            else:
                idx_mnar_qc = self.index[cond_qc_mv & cond_qc_int]
                logger.warning("No Bio Groups. Falling back to QC rescue.")

            idx_mnar_all = idx_mnar_group.union(idx_mnar_qc)

            # Base Health (Tiered Degradation)
            if valid_bio_groups:
                cond_healthy = (na_rate_valid <= mv_group_tol).any(axis=1)
            else:
                cond_healthy = self.stats["mv_qc_series"] <= mv_qc_tol

            idx_healthy = self.index[cond_healthy]
            idx_mar = idx_healthy.difference(idx_mnar_all)
            idx_dropped = self.index.difference(idx_mar.union(idx_mnar_all))

            self.stats.update(
                {
                    "idx_mar": idx_mar,
                    "idx_mnar": idx_mnar_all,
                    "idx_mnar_group": idx_mnar_group,
                    "idx_mnar_qc": idx_mnar_qc,
                    "idx_dropped_stage1": idx_dropped,
                }
            )
            return idx_mar, idx_mnar_all, idx_dropped

        except Exception as e:
            logger.error(f"Classification failed ({e}). Returning empty.")
            return empty_idx, empty_idx, empty_idx

    # =========================================================================
    # Filtering Execution Flow
    # =========================================================================

    def transform_mv_filtering(self) -> StageResult[pd.DataFrame]:
        """Classify missingness and retain accepted feature classes."""
        # Execute sample filtering before feature-level missingness checks.
        # Update current instance data with sample-filtered dataframe
        sample_result = self.transform_sample_filtering()
        df_clean_samples = sample_result.data
        self._update_inplace(df_clean_samples)

        feature_counts = self.stats["feature_counts"]
        if "raw" not in feature_counts:
            feature_counts["raw"] = self.shape[0]

        sample_type = self.attrs.get("sample_type", "Sample Type")
        sample_dict = self.attrs.get("sample_dict", {})
        qc_label = sample_dict.get("QC sample", "QC")
        valid_groups = self._get_valid_bio_groups()

        qc_mask = (
            self.columns.get_level_values(sample_type) == qc_label
            if sample_type in self.columns.names
            else np.zeros(self.shape[1], dtype=bool)
        )

        if qc_mask.any():
            self.stats["mv_qc_series"] = (
                self.loc[:, qc_mask].isna().mean(axis=1)
            )
        self.stats["mv_global_series"] = self.isna().mean(axis=1)

        if valid_groups:
            bio_group = self.attrs.get("bio_group", "Bio Group")
            group_na = self.isna().T.groupby(level=bio_group).mean().T
            self.stats["mv_group_df"] = group_na[valid_groups]

        idx_mar, idx_mnar, idx_dropped = self.classify_missing_types()

        retained_idx = idx_mar.union(idx_mnar)
        feature_counts["post_stage1"] = len(retained_idx)
        df_final = self.loc[retained_idx].copy(deep=True)

        # Store indices as native Python lists for JSON serialization.
        # serialization bugs in downstream metadata parsers.
        df_final.attrs["idx_mar"] = idx_mar.tolist()
        df_final.attrs["idx_mnar"] = idx_mnar.tolist()

        idx_mnar_group = self.stats.get("idx_mnar_group", pd.Index([]))
        idx_mnar_qc = self.stats.get("idx_mnar_qc", pd.Index([]))
        df_final.attrs["idx_mnar_group"] = idx_mnar_group.tolist()
        df_final.attrs["idx_mnar_qc"] = idx_mnar_qc.tolist()

        df_tracking = self._generate_s1_tracking_table(
            qc_mask, idx_mar, idx_mnar, idx_dropped
        )
        self.stats["stage1_tracking"] = df_tracking

        df_final.attrs["pipeline_stage"] = "High-missing values filtering"
        return StageResult(
            data=df_final,
            metrics=self.mv_filtering_metrics,
            candidates=df_tracking,
            metadata={
                "qc_mask": qc_mask,
                "valid_groups": valid_groups,
                "sample_filtered": df_clean_samples,
                "sample_tracking": sample_result.candidates,
            },
            audit_tables=self._audit_snapshot(),
        )

    @log_execution_time
    def run_mv_filtering(
        self,
        output_dir: str | None = None,
        **runtime_overrides: object,
    ) -> StageResult[pd.DataFrame]:
        """Return the structured missingness-filtering stage result.

        Supported settings are ``sample_mv_tol``, ``mv_group_tol``,
        ``mv_qc_tol``, ``mnar_group_mv_tol``, ``mnar_qc_mv_tol``, and
        ``mnar_intensity_pct``. They override pipeline configuration only for
        the current processor instance.
        """
        runner = MissingValueFilteringStageRunner(
            self,
            output_dir,
            runtime_overrides=runtime_overrides,
            allowed_override_keys=self._RUNTIME_CONFIG_KEYS.difference(
                {"qc_rsd_tol", "blank_qc_ratio_tol"}
            ),
        )
        return runner.run()

    def _generate_s1_tracking_table(
        self,
        qc_mask: np.ndarray,
        idx_mar: pd.Index,
        idx_mnar: pd.Index,
        idx_dropped: pd.Index,
    ) -> pd.DataFrame:
        """Builds a detailed feature status tracking DataFrame."""
        global_mv = self.stats.get("mv_global_series", pd.Series(dtype=float))
        qc_mv = self.stats.get("mv_qc_series", pd.Series(dtype=float))
        group_df = self.stats.get("mv_group_df", pd.DataFrame())

        idx_mnar_group = self.stats.get("idx_mnar_group", pd.Index([]))
        idx_mnar_qc = self.stats.get("idx_mnar_qc", pd.Index([]))

        if qc_mask.any():
            qc_median = self.loc[:, qc_mask].median(axis=1)
        else:
            qc_median = pd.Series(dtype=float)

        track_data = []
        for feat in self.index:
            val_global = global_mv.get(feat, np.nan) * 100
            val_qc = qc_mv.get(feat, np.nan) * 100

            val_group_max = np.nan
            val_group_min = np.nan

            if not group_df.empty and feat in group_df.index:
                val_group_max = group_df.loc[feat].max() * 100
                val_group_min = group_df.loc[feat].min() * 100

            val_intensity = qc_median.get(feat, np.nan)
            log2_int = (
                np.log2(val_intensity + 1)
                if pd.notna(val_intensity)
                else np.nan
            )

            # Use exact string matching for sample-type labels.
            # Downstream visualization relies heavily on .str.contains("Group")
            # and .str.contains("QC"). Abbreviations break the routing logic!
            if feat in idx_dropped:
                status, reason, sort_order = "INVALID", "Fail MV rules", 0
            elif feat in idx_mar:
                status, reason, sort_order = "MAR", "Health passed", 1
            elif feat in idx_mnar_group and feat in idx_mnar_qc:
                status, reason, sort_order = (
                    "MNAR (Group & QC)",
                    "Dual rescue",
                    2,
                )
            elif feat in idx_mnar_group:
                status, reason, sort_order = "MNAR (Group)", "Group pass", 3
            elif feat in idx_mnar_qc:
                status, reason, sort_order = "MNAR (QC)", "QC pass", 4
            else:
                status, reason, sort_order = "Unknown", "Logic gap", 5

            track_data.append(
                {
                    "Feature_ID": feat,
                    "Global_MV_Pct": round(val_global, 2),
                    "QC_MV_Pct": round(val_qc, 2),
                    "Min_Group_MV_Pct": round(val_group_min, 2),
                    "Max_Group_MV_Pct": round(val_group_max, 2),
                    "Log2_Intensity": round(log2_int, 4),
                    "Stage1_Status": status,
                    "Reference_Basis": reason,
                    "_sort": sort_order,
                }
            )

        df_tracking = pd.DataFrame(track_data).set_index("Feature_ID")
        return df_tracking.sort_values(by="_sort").drop(columns=["_sort"])

    @cached_property
    def mv_filtering_metrics(self) -> Dict[str, Any]:
        """
        Extracts metrics from Stage-1 missing value filtering.
        Unifies results into 'sample_wise' and 'feature_wise' dimensions.
        """
        feature_counts = self.stats.get("feature_counts", {})

        # =====================================================================
        # Sample-wise Metrics Extraction
        # =====================================================================
        track_df = self.stats.get("sample_tracking", pd.DataFrame())
        sample_metrics = {}
        if not track_df.empty:
            sample_total = len(track_df)
            sample_dropped = sum(track_df["Status"] == "Dropped")
            sample_retained = sample_total - sample_dropped
            sample_retention_rate = (
                round(sample_retained / sample_total * 100, 2)
                if sample_total
                else 0.0
            )

            sample_metrics = {
                "thresholds": {
                    "sample_mv_tol": self.attrs.get("sample_mv_tol", 0.5)
                },
                "feature_retention": {  # Kept naming style consistent
                    "total_checked": sample_total,
                    "retained_count": sample_retained,
                    "dropped_count": sample_dropped,
                    "retention_rate_pct": sample_retention_rate,
                },
            }

        # =====================================================================
        # Feature-wise Metrics Extraction
        # =====================================================================
        valid_groups = self._get_valid_bio_groups()
        sample_type = self.attrs.get("sample_type", "Sample Type")
        sample_dict = self.attrs.get("sample_dict", {})
        qc_label = sample_dict.get("QC sample", "QC")

        has_qc = False
        if sample_type in self.columns.names:
            has_qc = (
                self.columns.get_level_values(sample_type) == qc_label
            ).any()

        if valid_groups:
            filter_level = "Group"
        elif has_qc:
            filter_level = "QC"
        else:
            filter_level = "Global"

        idx_mar = self.stats.get("idx_mar", pd.Index([]))
        idx_mnar = self.stats.get("idx_mnar", pd.Index([]))
        idx_mnar_group = self.stats.get("idx_mnar_group", pd.Index([]))
        idx_mnar_qc = self.stats.get("idx_mnar_qc", pd.Index([]))

        retained_idx = idx_mar.union(idx_mnar)

        feature_raw_count = int(feature_counts.get("raw", 0))
        feature_retained_count = len(retained_idx)
        feature_dropped_count = max(
            0, feature_raw_count - feature_retained_count
        )
        feature_retention_rate = (
            round((feature_retained_count / feature_raw_count) * 100, 2)
            if (feature_raw_count > 0)
            else 0.0
        )

        feature_metrics = {
            "filtering_level": filter_level,
            "thresholds": {
                "mv_group_tol": self.attrs.get("mv_group_tol", 0.5),
                "mv_qc_tol": self.attrs.get("mv_qc_tol", 0.3),
                "mnar_group_mv_tol": self.attrs.get("mnar_group_mv_tol", 0.8),
                "mnar_qc_mv_tol": self.attrs.get("mnar_qc_mv_tol", 0.2),
            },
            "missing_classification": {
                "mar_count": int(len(idx_mar)),
                "mnar_total": int(len(idx_mnar)),
                "mnar_group": int(len(idx_mnar_group)),
                "mnar_qc": int(len(idx_mnar_qc)),
            },
            "feature_retention": {
                "pre_mv_filter_count": feature_raw_count,
                "after_mv_filter_count": feature_retained_count,
                "dropped_count": feature_dropped_count,
                "retention_rate_pct": feature_retention_rate,
            },
        }

        # =====================================================================
        # Unified Metric Assembly
        # =====================================================================
        return {"sample_wise": sample_metrics, "feature_wise": feature_metrics}

    def transform_quality_filtering(
        self,
        idx_mar: pd.Index | list[object] | None = None,
        idx_mnar: pd.Index | list[object] | None = None,
    ) -> StageResult[pd.DataFrame]:
        """Apply blank-ratio and QC-RSD feature-quality rules."""
        if idx_mar is None:
            idx_mar = self.attrs.get("idx_mar")
        if idx_mnar is None:
            idx_mnar = self.attrs.get("idx_mnar")

        if idx_mar is None or idx_mnar is None:
            logger.warning("MAR/MNAR indices missing. Recomputing natively.")
            idx_mar, idx_mnar, _ = self.classify_missing_types()

        if not isinstance(idx_mar, pd.Index):
            idx_mar = pd.Index(idx_mar)
        if not isinstance(idx_mnar, pd.Index):
            idx_mnar = pd.Index(idx_mnar)

        self.stats["idx_mnar_group"] = self.attrs.get(
            "idx_mnar_group", pd.Index([])
        )
        self.stats["idx_mnar_qc"] = self.attrs.get("idx_mnar_qc", pd.Index([]))
        self.stats["idx_mar"] = idx_mar
        self.stats["idx_mnar"] = idx_mnar

        feature_counts = self.stats["feature_counts"]

        sample_type = self.attrs.get("sample_type", "Sample Type")
        sample_dict = self.attrs.get("sample_dict", {})
        qc_label = sample_dict.get("QC sample", "QC")
        blank_label = sample_dict.get("Blank sample", "Blank")

        qc_rsd_tol = self.attrs.get("qc_rsd_tol", 0.3)
        blank_qc_ratio_tol = self.attrs.get("blank_qc_ratio_tol", 0.2)

        qc_mask = self.columns.get_level_values(sample_type) == qc_label
        blank_mask = self.columns.get_level_values(sample_type) == blank_label
        current_idx = self.index
        logger.info(f"Features before filtering: {len(current_idx)}")

        # Blank Ratio Quality Check
        if blank_mask.any() and qc_mask.any():
            qc_mean = self.loc[:, qc_mask].mean(axis=1)
            blank_mean = self.loc[:, blank_mask].mean(axis=1)

            self.stats.update({"qc_mean": qc_mean, "blank_mean": blank_mean})

            blank_mean_safe = blank_mean.fillna(0)
            qc_safe = qc_mean.replace(0, np.finfo(float).eps)

            ratio_series = blank_mean_safe / qc_safe
            pass_blank = ratio_series[ratio_series <= blank_qc_ratio_tol].index

            next_idx = current_idx.intersection(pass_blank)
            self.stats["idx_dropped_blank"] = current_idx.difference(next_idx)
            current_idx = next_idx
            logger.info(f"Features after Blank/QC check: {len(current_idx)}")
        else:
            self.stats["idx_dropped_blank"] = pd.Index([])

        feature_counts["post_stage2_blank"] = len(current_idx)

        # QC RSD Quality Check
        if qc_mask.any():
            df_qc = self.loc[current_idx, qc_mask]
            std_qc = df_qc.std(axis=1, ddof=1)
            mean_qc = df_qc.mean(axis=1)

            self.stats["qc_rsd_all"] = std_qc / mean_qc
            pass_mar = self.stats["qc_rsd_all"].loc[
                idx_mar.intersection(current_idx)
            ]
            final_idx = pass_mar[pass_mar <= qc_rsd_tol].index.union(
                idx_mnar.intersection(current_idx)
            )

            self.stats["idx_dropped_rsd"] = current_idx.difference(final_idx)
            logger.info(f"Features after QC RSD check: {len(final_idx)}")
        else:
            final_idx = current_idx
            self.stats["idx_dropped_rsd"] = pd.Index([])

        self.stats["idx_retained_stage2"] = final_idx
        feature_counts["post_stage2_rsd"] = len(final_idx)
        df_final = self.loc[final_idx].copy()

        # Intersect to remove features dropped in Stage 2,
        # then explicitly convert to native lists for safe propagation.
        df_final.attrs["idx_mar"] = idx_mar.intersection(final_idx).tolist()
        df_final.attrs["idx_mnar"] = idx_mnar.intersection(final_idx).tolist()

        idx_mnar_group = pd.Index(self.stats.get("idx_mnar_group", []))
        idx_mnar_qc = pd.Index(self.stats.get("idx_mnar_qc", []))
        df_final.attrs["idx_mnar_group"] = idx_mnar_group.intersection(
            final_idx
        ).tolist()
        df_final.attrs["idx_mnar_qc"] = idx_mnar_qc.intersection(
            final_idx
        ).tolist()

        # Generate detailed tracking table via private helper
        start_idx = idx_mar.union(idx_mnar).intersection(self.index)
        df_tracking = self._generate_s2_tracking_table(
            start_idx=start_idx, idx_mar=idx_mar, idx_mnar=idx_mnar
        )
        self.stats["stage2_tracking"] = df_tracking

        df_final.attrs["pipeline_stage"] = "Low-quality filtering"
        return StageResult(
            data=df_final,
            metrics=df_final.quality_filtering_metrics,
            candidates=df_tracking,
            audit_tables=self._audit_snapshot(),
        )

    @log_execution_time
    def run_quality_filtering(
        self,
        idx_mar: pd.Index | list[object] | None = None,
        idx_mnar: pd.Index | list[object] | None = None,
        output_dir: str | None = None,
        **runtime_overrides: object,
    ) -> StageResult[pd.DataFrame]:
        """Return the structured quality-filtering stage result.

        This stage accepts ``qc_rsd_tol`` and ``blank_qc_ratio_tol``. MAR and
        MNAR indices remain explicit data inputs rather than configuration.
        """
        runner = QualityFilteringStageRunner(
            self,
            output_dir,
            idx_mar=idx_mar,
            idx_mnar=idx_mnar,
            runtime_overrides=runtime_overrides,
            allowed_override_keys=frozenset(
                {"qc_rsd_tol", "blank_qc_ratio_tol"}
            ),
        )
        return runner.run()

    def _generate_s2_tracking_table(
        self, start_idx: pd.Index, idx_mar: pd.Index, idx_mnar: pd.Index
    ) -> pd.DataFrame:
        """Builds a detailed feature status tracking DataFrame for Stage 2."""
        idx_dropped_blank = self.stats.get("idx_dropped_blank", pd.Index([]))
        idx_dropped_rsd = self.stats.get("idx_dropped_rsd", pd.Index([]))

        blank_mean = self.stats.get("blank_mean", pd.Series(dtype=float))
        qc_mean = self.stats.get("qc_mean", pd.Series(dtype=float))
        qc_rsd = self.stats.get("qc_rsd_all", pd.Series(dtype=float))

        track_data = []
        for feat in start_idx:
            base_type = "MNAR" if feat in idx_mnar else "MAR"
            val_blank = blank_mean.get(feat, np.nan)
            val_qc = qc_mean.get(feat, np.nan)
            ratio_val = np.nan

            if pd.notna(val_qc):
                if val_qc > 1e-9:
                    safe_val_blank = 0.0 if pd.isna(val_blank) else val_blank
                    ratio_val = round(safe_val_blank / val_qc, 4)
                else:
                    ratio_val = "QC Mean <= 0"

            val_rsd = qc_rsd.get(feat, np.nan)
            rsd_val = round(val_rsd, 4) if pd.notna(val_rsd) else np.nan

            if feat in idx_dropped_blank:
                (
                    blank_check_status,
                    rsd_check_status,
                    stage2_status,
                    sort_order,
                ) = (
                    "Failed",
                    "Skipped",
                    "Drop",
                    0,
                )
            elif feat in idx_dropped_rsd:
                (
                    blank_check_status,
                    rsd_check_status,
                    stage2_status,
                    sort_order,
                ) = (
                    "Passed",
                    "Failed",
                    "Drop",
                    1,
                )
            else:
                (
                    blank_check_status,
                    rsd_check_status,
                    stage2_status,
                    sort_order,
                ) = (
                    "Passed",
                    "Passed",
                    "Keep",
                    2,
                )

            if base_type == "MNAR" and feat not in idx_dropped_blank:
                rsd_check_status = "Exempted (MNAR)"

            track_data.append(
                {
                    "Feature_ID": feat,
                    "Base_Type": base_type,
                    "Ratio_Value": ratio_val,
                    "Ratio_Check": blank_check_status,
                    "RSD_Value": rsd_val,
                    "RSD_Check": rsd_check_status,
                    "Stage2_Status": stage2_status,
                    "_sort": sort_order,
                }
            )

        df_tracking = pd.DataFrame(track_data).set_index("Feature_ID")
        df_tracking = df_tracking.sort_values(by=["_sort", "Base_Type"])
        return df_tracking.drop(columns=["_sort"])

    @cached_property
    def quality_filtering_metrics(self) -> dict:
        """Extracts metrics from Stage-2 low-quality feature filtering."""
        feature_counts = self.stats.get("feature_counts", {})

        idx_dropped_blank = self.stats.get("idx_dropped_blank", pd.Index([]))
        idx_dropped_rsd = self.stats.get("idx_dropped_rsd", pd.Index([]))
        idx_mar = self.stats.get("idx_mar", pd.Index([]))
        idx_mnar = self.stats.get("idx_mnar", pd.Index([]))

        blank_drop_mar = len(idx_dropped_blank.intersection(idx_mar))
        blank_drop_mnar = len(idx_dropped_blank.intersection(idx_mnar))
        rsd_drop_mar = len(idx_dropped_rsd.intersection(idx_mar))
        rsd_drop_mnar = len(idx_dropped_rsd.intersection(idx_mnar))

        pre_mar = len(idx_mar)
        pre_mnar = len(idx_mnar)

        post_blank_mar = pre_mar - blank_drop_mar
        post_blank_mnar = pre_mnar - blank_drop_mnar
        post_rsd_mar = post_blank_mar - rsd_drop_mar
        post_rsd_mnar = post_blank_mnar - rsd_drop_mnar

        metrics = {
            "thresholds": {
                "blank_qc_ratio_tol": self.attrs.get("blank_qc_ratio_tol", 0.2),
                "qc_rsd_tol": self.attrs.get("qc_rsd_tol", 0.3),
            },
            "feature_retention": {
                "pre_stage2": {
                    "total": feature_counts.get("post_stage1", 0),
                    "mar_count": pre_mar,
                    "mnar_count": pre_mnar,
                },
                "post_blank_check": {
                    "total": feature_counts.get("post_stage2_blank", 0),
                    "mar_count": post_blank_mar,
                    "mnar_count": post_blank_mnar,
                },
                "post_rsd_check": {
                    "total": feature_counts.get("post_stage2_rsd", 0),
                    "mar_count": post_rsd_mar,
                    "mnar_count": post_rsd_mnar,
                },
            },
            "filtering_breakdown": {
                "dropped_by_blank": {
                    "total": len(idx_dropped_blank),
                    "mar_count": blank_drop_mar,
                    "mnar_count": blank_drop_mnar,
                },
                "dropped_by_rsd": {
                    "total": len(idx_dropped_rsd),
                    "mar_count": rsd_drop_mar,
                    "mnar_count": rsd_drop_mnar,
                },
            },
        }
        return metrics
