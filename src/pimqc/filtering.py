# src/pimqc/filtering.py
"""
Script purpose: Execute sample and feature filtering in two pipeline stages.

execute_mv_filtering() first calls execute_sample_filtering(), removing
QC/actual samples above the missing-value tolerance while preserving other
sample types. It then calculates QC/global/group missingness, classifies
features as MAR, MNAR, or dropped, applies biological-group and QC rescue
rules, exports tracking tables, and writes the Stage 1 dashboard.
execute_quality_filtering() consumes the MAR/MNAR labels, removes features
that fail blank-to-QC abundance checks or QC RSD thresholds, preserves eligible
MNAR features, updates retained indices, and exports Stage 2 diagnostics.
"""

import os
import copy
import numpy as np
import pandas as pd
from functools import cached_property

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.lines as mlines

from loguru import logger
from typing import Dict, Any, Optional

from . import io_utils as iu
from . import plot_utils as pu
from . import core_classes
from . import visualizer_classes


class MetaboIntFilter(core_classes.MetaboInt):
    """Filtering engine for metabolomics datasets with QC enforcement."""

    _metadata = ["attrs", "stats"]

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
        super().__init__(data=data, pipeline_params=pipeline_params, *args, **kwargs)

        if not hasattr(self, "stats"):
            self.stats = {}

        # 1. Explicitly inherit runtime stats from upstream data object
        input_data = data if data is not None else kwargs.get("data")
        if input_data is None and len(args) > 0:
            input_data = args[0]

        if input_data is not None and hasattr(input_data, "stats"):
            self.stats.update(copy.deepcopy(input_data.stats))

        # 2. Initialize runtime tracking if not inherited
        if "feature_counts" not in self.stats:
            self.stats["feature_counts"] = {}

        # 3. Base defaults
        filter_configs = {
            "sample_mv_tol": 0.5,
            "mv_group_tol": 0.5,
            "mv_qc_tol": 0.3,
            "mnar_group_mv_tol": 0.8,
            "mnar_qc_mv_tol": 0.2,
            "mnar_intensity_pct": 0.1,
            "qc_rsd_tol": 0.3,
            "blank_qc_ratio_tol": 0.2,
        }

        # 4. TOML configuration overrides
        if pipeline_params and "MetaboIntFilter" in pipeline_params:
            filter_configs.update(pipeline_params["MetaboIntFilter"])

        # 5. Explicit kwargs override TOML (Highest priority)
        local_args = locals()
        explicit_params = [
            "sample_mv_tol",
            "mv_group_tol",
            "mv_qc_tol",
            "mnar_group_mv_tol",
            "mnar_qc_mv_tol",
            "mnar_intensity_pct",
            "qc_rsd_tol",
            "blank_qc_ratio_tol",
        ]
        for param in explicit_params:
            if local_args[param] is not None:
                filter_configs[param] = local_args[param]

        # 6. Flatten strictly into lifecycle attributes (SSOT)
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

    @property
    def _constructor(self) -> type["MetaboIntFilter"]:
        """Override constructor to return MetaboIntFilter."""
        return MetaboIntFilter

    def __finalize__(
        self,
        other: object,
        method: Optional[str] = None,
        **kwargs: object,
    ) -> "MetaboIntFilter":
        """Deepcopy custom attributes during pandas operations."""
        self = super().__finalize__(other, method=method, **kwargs)
        for name in self._metadata:
            if hasattr(other, name):
                setattr(self, name, copy.deepcopy(getattr(other, name)))
        return self

    # =========================================================================
    # 1. Sample-Level Filtering
    # =========================================================================
    @iu._exe_time
    def execute_sample_filtering(self, output_dir: str = None) -> pd.DataFrame:
        """
        Filters out high-MV samples by evaluating only QC and Actual Samples.
        Generates tracking tables for sample-level attrition.
        """
        batch = self.attrs.get("batch", "Batch")
        inject_order = self.attrs.get("inject_order", "Inject Order")
        sample_mv_tol = self.attrs.get("sample_mv_tol", 0.5)

        # 1. Strictly evaluate only QC and Actual Samples via concatenation
        df_check = pd.concat([self._qc, self._actual_sample], axis=1)
        sample_mv_rates = df_check.isna().mean(axis=0)

        # 2. Determine status
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

        # 3. Physically reorder the remaining columns by injection sequence
        sort_levels = [
            lvl for lvl in [batch, inject_order] if lvl in df_filtered.columns.names
        ]
        if sort_levels:
            df_filtered = df_filtered.sort_index(axis=1, level=sort_levels)

        if output_dir:
            iu._check_dir_exists(output_dir, handle="makedirs")
            df_filtered.to_csv(
                os.path.join(output_dir, "Filtered_Data_High-MV_Samples.csv")
            )
            track_df.to_csv(
                os.path.join(output_dir, "Filtering_Tracking_High-MV_Samples.csv")
            )

        return df_filtered

    # =========================================================================
    # 2. Feature-Level Missing Value Classification
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

    @iu._exe_time
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

            # 1. Group Rescue (Tier 1 - Optimal)
            if valid_bio_groups:
                na_rate_group = self.isna().T.groupby(level=bio_group).mean().T
                na_rate_valid = na_rate_group[valid_bio_groups]
                cond_group_mnar = (na_rate_valid >= mnar_group_mv_tol).any(axis=1) & (
                    na_rate_valid <= mv_group_tol
                ).any(axis=1)
                idx_mnar_group = self.index[cond_group_mnar]

            # 2. QC Rescue (Enforced QC standard)
            df_qc = self.loc[:, qc_mask]
            qc_na_rate = df_qc.isna().mean(axis=1)
            qc_median = df_qc.median(axis=1)
            int_threshold = qc_median.quantile(mnar_intensity_pct)

            cond_qc_mv = qc_na_rate > mnar_qc_mv_tol
            cond_qc_int = qc_median <= int_threshold

            if valid_bio_groups:
                cond_qc_bio_valid = (na_rate_valid <= mv_group_tol).any(axis=1)
                idx_mnar_qc = self.index[cond_qc_mv & cond_qc_int & cond_qc_bio_valid]
            else:
                idx_mnar_qc = self.index[cond_qc_mv & cond_qc_int]
                logger.warning("No Bio Groups. Falling back to QC rescue.")

            idx_mnar_all = idx_mnar_group.union(idx_mnar_qc)

            # 3. Base Health (Tiered Degradation)
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
    # 3. Filtering Execution Flow
    # =========================================================================

    @iu._exe_time
    def execute_mv_filtering(self, output_dir: str = None) -> pd.DataFrame:
        """Orchestrates Stage-1 MV filtering and exports diagnostics."""
        # --- NEW: Step 0 - Execute Sample Filtering First ---
        # Update current instance data with sample-filtered dataframe
        df_clean_samples = self.execute_sample_filtering(output_dir)
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
            self.stats["mv_qc_series"] = self.loc[:, qc_mask].isna().mean(axis=1)
        self.stats["mv_global_series"] = self.isna().mean(axis=1)

        if valid_groups:
            bio_group = self.attrs.get("bio_group", "Bio Group")
            group_na = self.isna().T.groupby(level=bio_group).mean().T
            self.stats["mv_group_df"] = group_na[valid_groups]

        idx_mar, idx_mnar, idx_dropped = self.classify_missing_types()

        retained_idx = idx_mar.union(idx_mnar)
        feature_counts["post_stage1"] = len(retained_idx)
        df_final = self.loc[retained_idx].copy(deep=True)

        # [FIX]: Store indices as native Python lists to prevent JSON
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

        if output_dir:
            iu._check_dir_exists(output_dir, handle="makedirs")
            df_final.attrs["pipeline_stage"] = "High-missing values filtering"
            df_final.to_csv(
                os.path.join(output_dir, "Filtered_Data_High-MV_Features.csv")
            )
            df_tracking.to_csv(
                os.path.join(output_dir, "Filtering_Tracking_High-MV_Features.csv")
            )

            self._execute_s1_visualization(
                output_dir, df_tracking, qc_mask, valid_groups
            )

        logger.success("High-missing value feature filtering completed.")
        return df_final

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
            log2_int = np.log2(val_intensity + 1) if pd.notna(val_intensity) else np.nan

            # CRITICAL FIX: Restored explicitly exact string matching.
            # Downstream visualization relies heavily on .str.contains("Group")
            # and .str.contains("QC"). Abbreviations break the routing logic!
            if feat in idx_dropped:
                status, reason, sort_order = "INVALID", "Fail MV rules", 0
            elif feat in idx_mar:
                status, reason, sort_order = "MAR", "Health passed", 1
            elif feat in idx_mnar_group and feat in idx_mnar_qc:
                status, reason, sort_order = "MNAR (Group & QC)", "Dual rescue", 2
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

    def _execute_s1_visualization(
        self,
        output_dir: str,
        df_tracking: pd.DataFrame,
        qc_mask: np.ndarray,
        valid_groups: list[object],
    ) -> None:
        """Helper for orchestrating dashboard generation."""
        vis = MetaboVisualizerFilter(self)

        active_base_tol = self.attrs.get(
            "mv_group_tol" if valid_groups else "mv_qc_tol", 0.5
        )

        mnar_int_threshold = None
        if qc_mask.any():
            mnar_intensity_pct = self.attrs.get("mnar_intensity_pct", 0.1)
            raw_threshold = (
                self.loc[:, qc_mask].median(axis=1).quantile(mnar_intensity_pct)
            )
            mnar_int_threshold = np.log2(raw_threshold + 1)

        fig_grid = vis.plot_mv_filtering_summary_grid(
            tracking_df=df_tracking,
            active_base_tol=active_base_tol,
            mnar_group_mv_tol=self.attrs.get("mnar_group_mv_tol", 0.8),
            mnar_qc_mv_tol=self.attrs.get("mnar_qc_mv_tol", 0.2),
            mnar_int_threshold=mnar_int_threshold,
        )
        if fig_grid:
            grid_path = os.path.join(output_dir, "MV_Classification_Dashboard.svg")
            vis.save_and_show_pw(pw_obj=fig_grid, file_path=grid_path)
            logger.info(f"High-MV Filter summary dashboard saved as: {grid_path}")

    @cached_property
    def mv_filtering_metrics(self) -> Dict[str, Any]:
        """
        Extracts metrics from Stage-1 missing value filtering.
        Unifies results into 'sample_wise' and 'feature_wise' dimensions.
        """
        feature_counts = self.stats.get("feature_counts", {})

        # =====================================================================
        # 1. Sample-wise Metrics Extraction
        # =====================================================================
        track_df = self.stats.get("sample_tracking", pd.DataFrame())
        sample_metrics = {}
        if not track_df.empty:
            sample_total = len(track_df)
            sample_dropped = sum(track_df["Status"] == "Dropped")
            sample_retained = sample_total - sample_dropped
            sample_retention_rate = (
                round(sample_retained / sample_total * 100, 2) if sample_total else 0.0
            )

            sample_metrics = {
                "thresholds": {"sample_mv_tol": self.attrs.get("sample_mv_tol", 0.5)},
                "feature_retention": {  # Kept naming style consistent
                    "total_checked": sample_total,
                    "retained_count": sample_retained,
                    "dropped_count": sample_dropped,
                    "retention_rate_pct": sample_retention_rate,
                },
            }

        # =====================================================================
        # 2. Feature-wise Metrics Extraction
        # =====================================================================
        valid_groups = self._get_valid_bio_groups()
        sample_type = self.attrs.get("sample_type", "Sample Type")
        sample_dict = self.attrs.get("sample_dict", {})
        qc_label = sample_dict.get("QC sample", "QC")

        has_qc = False
        if sample_type in self.columns.names:
            has_qc = (self.columns.get_level_values(sample_type) == qc_label).any()

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
        feature_dropped_count = max(0, feature_raw_count - feature_retained_count)
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
        # 3. Unified Export
        # =====================================================================
        return {"sample_wise": sample_metrics, "feature_wise": feature_metrics}

    @iu._exe_time
    def execute_quality_filtering(
        self,
        idx_mar: pd.Index | list[object] | None = None,
        idx_mnar: pd.Index | list[object] | None = None,
        output_dir: str | None = None,
    ) -> pd.DataFrame:
        """Executes Stage-2 quality filter (Blank Ratio & QC RSD)."""
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

        self.stats["idx_mnar_group"] = self.attrs.get("idx_mnar_group", pd.Index([]))
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

        # 1. Blank Ratio Quality Check
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

        # 2. QC RSD Quality Check
        if qc_mask.any():
            df_qc = self.loc[current_idx, qc_mask]
            std_qc = df_qc.std(axis=1, ddof=1)
            mean_qc = df_qc.mean(axis=1)

            self.stats["qc_rsd_all"] = std_qc / mean_qc
            pass_mar = self.stats["qc_rsd_all"].loc[idx_mar.intersection(current_idx)]
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

        # [FIX]: Intersect to remove features dropped in Stage 2,
        # then explicitly convert to native lists for safe propagation.
        df_final.attrs["idx_mar"] = idx_mar.intersection(final_idx).tolist()
        df_final.attrs["idx_mnar"] = idx_mnar.intersection(final_idx).tolist()

        idx_mnar_group = pd.Index(self.stats.get("idx_mnar_group", []))
        idx_mnar_qc = pd.Index(self.stats.get("idx_mnar_qc", []))
        df_final.attrs["idx_mnar_group"] = idx_mnar_group.intersection(
            final_idx
        ).tolist()
        df_final.attrs["idx_mnar_qc"] = idx_mnar_qc.intersection(final_idx).tolist()

        # 3. Generate detailed tracking table via private helper
        start_idx = idx_mar.union(idx_mnar).intersection(self.index)
        df_tracking = self._generate_s2_tracking_table(
            start_idx=start_idx, idx_mar=idx_mar, idx_mnar=idx_mnar
        )
        self.stats["stage2_tracking"] = df_tracking

        # 4. Handle output and visualizations
        if output_dir:
            iu._check_dir_exists(dir_path=output_dir, handle="makedirs")

            df_final.attrs["pipeline_stage"] = "Low-quality filtering"
            csv_path = os.path.join(
                output_dir, "Filtered_Data_Low-quality_Features.csv"
            )
            df_final.to_csv(csv_path, encoding="utf-8-sig", na_rep="NA")
            logger.info(
                f"Data after low-quality features filtering saved as: {csv_path}"
            )

            trk_path2 = os.path.join(
                output_dir, "Filtering_Tracking_Low-quality_Features.csv"
            )
            df_tracking.to_csv(trk_path2, na_rep="N/A")

            # Execute visualization routing
            self._execute_s2_visualization(output_dir, df_final)

        logger.success("Low-quality features filtering completed.")
        return df_final

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
                    ratio_val = "QC Mean ≈ 0"

            val_rsd = qc_rsd.get(feat, np.nan)
            rsd_val = round(val_rsd, 4) if pd.notna(val_rsd) else np.nan

            if feat in idx_dropped_blank:
                blank_check_status, rsd_check_status, stage2_status, sort_order = (
                    "Failed",
                    "Skipped",
                    "Drop",
                    0,
                )
            elif feat in idx_dropped_rsd:
                blank_check_status, rsd_check_status, stage2_status, sort_order = (
                    "Passed",
                    "Failed",
                    "Drop",
                    1,
                )
            else:
                blank_check_status, rsd_check_status, stage2_status, sort_order = (
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

    def _execute_s2_visualization(
        self, output_dir: str, df_final: pd.DataFrame
    ) -> None:
        """Helper for orchestrating Stage 2 dashboard generation."""
        vis = MetaboVisualizerFilter(engine=df_final)

        try:
            fig_grid = vis.plot_quality_filtering_summary_grid()
            if fig_grid:
                grid_path = os.path.join(
                    output_dir, "Low-quality_Filtering_Dashboard.svg"
                )
                vis.save_and_show_pw(pw_obj=fig_grid, file_path=grid_path)
                logger.info(
                    f"Low-quality Filter summary dashboard saved as: {grid_path}"
                )
        except Exception as e:
            logger.error(
                f"Grid of low-quality features filtering generation failed: {e}"
            )

    @cached_property
    def quality_filtering_metrics(self) -> dict:
        """Extracts metrics from Stage-2 low-quality feature filtering."""
        feature_counts = self.attrs.get("feature_counts", {})

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


class MetaboVisualizerFilter(visualizer_classes.BaseMetaboVisualizer):
    """Visualization suite for metabolomics filtering results."""

    def __init__(self, engine: MetaboIntFilter) -> None:
        """Initialize with the filtering engine."""
        super().__init__(metabo_obj=engine)
        self.engine = engine

    # =========================================================================
    # High-Misssing Values Samples Filtering
    # =========================================================================
    def _plot_sample_mv_stripplot(
        self,
        track_df: pd.DataFrame,
        tol: float,
        ax: plt.Axes | None = None,
    ) -> plt.Figure | plt.Axes | None:
        """Plot sample missing rates using a stripplot, annotating outliers."""
        if track_df.empty:
            return None if ax is None else ax

        if ax is None:
            fig, current_ax = plt.subplots(figsize=(4, 4))
        else:
            current_ax = ax
            fig = current_ax.figure

        df_plot = track_df.copy()

        color_dict = {
            t: "tab:red" if "QC" in str(t).upper() else "tab:gray"
            for t in df_plot["Sample_Type"].unique()
        }

        sns.stripplot(
            data=df_plot,
            x="Sample_Type",
            y="MV_Rate_Pct",
            hue="Sample_Type",
            palette=color_dict,
            jitter=True,
            size=7,
            alpha=0.8,
            edgecolor="k",
            linewidth=0.5,
            ax=current_ax,
            legend=False,
        )

        # Threshold line
        tol_pct = tol * 100
        current_ax.axhline(
            tol_pct,
            color="k",
            linestyle="--",
            linewidth=1.5,
            label=f"Sample MV Tol: {tol_pct:.0f}%",
        )

        # Extract unique categories natively to match x-axis indices
        cat_order = df_plot["Sample_Type"].unique().tolist()

        # Smart Annotation for Outliers
        outliers = df_plot[df_plot["MV_Rate_Pct"] > tol_pct]
        for idx, row in outliers.iterrows():
            x_pos = cat_order.index(row["Sample_Type"])
            y_pos = row["MV_Rate_Pct"]
            current_ax.annotate(
                str(idx),
                (x_pos, y_pos),
                xytext=(12, 0),
                textcoords="offset points",
                fontsize=8,
                color="darkred",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="k", alpha=0.9),
                arrowprops=dict(arrowstyle="-", color="k", lw=1.0),
            )

        self._apply_standard_format(
            ax=current_ax,
            title="Sample Missing Value Ratios",
            xlabel="Sample Type",
            ylabel="Missing Rate (%)",
            append_stage=False,
        )

        self._format_single_legend(
            ax=current_ax, loc="upper right", bbox_to_anchor=None
        )

        # Adjust Y-limit slightly to prevent top annotations from clipping
        actual_y_max = max(df_plot["MV_Rate_Pct"].max(), tol_pct)
        padding = actual_y_max * 0.15 if actual_y_max > 0 else 10
        current_ax.set_ylim(-5, actual_y_max + padding)

        return fig if ax is None else current_ax

    # =========================================================================
    # High-MV Features Filtering Unified Summary Dashboard (1+N Layout)
    # =========================================================================
    def _plot_group_rescue_scatter(
        self,
        df: pd.DataFrame,
        max_col: str,
        min_col: str,
        mnar_group_mv_tol: float,
        active_base_tol: float,
        ax: plt.Axes,
        title: str,
    ) -> None:
        """Scatter plot visualizing the 2D logic of Group MNAR rescue."""

        if df.empty:
            ax.axis("off")
            return

        color_mnar = pu.get_equivalent_hex("tab:red", alpha=0.5)
        color_pending = "tab:gray"

        df_plot = df.copy()

        df_plot["Step_Status"] = np.where(
            df_plot["Stage1_Status"].str.contains("Group"), "MNAR (Group)", "Pending"
        )

        df_plot = df_plot.sort_values(by="Step_Status", ascending=False)

        sns.scatterplot(
            data=df_plot,
            x=max_col,
            y=min_col,
            hue="Step_Status",
            palette={"MNAR (Group)": color_mnar, "Pending": color_pending},
            style="Step_Status",
            markers={"MNAR (Group)": "X", "Pending": "o"},
            alpha=0.8,
            ax=ax,
            legend=False,
            edgecolor="k",
        )

        tol_max_pct = mnar_group_mv_tol * 100
        tol_min_pct = active_base_tol * 100

        ax.plot([0, 100], [0, 100], color="gray", linestyle="-.", alpha=0.5, zorder=1)

        ax.axvline(tol_max_pct, color="k", linestyle="--")
        ax.axhline(tol_min_pct, color="k", linestyle=":")

        ax.fill_between(
            [tol_max_pct, 105], -5, tol_min_pct, color=color_mnar, alpha=0.2, zorder=0
        )

        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)

        handles = [
            mlines.Line2D([], [], color="none", label="Status"),
            mlines.Line2D(
                [],
                [],
                color=color_mnar,
                marker="X",
                linestyle="",
                label="MNAR (Group)",
                markeredgecolor="k",
            ),
            mlines.Line2D(
                [],
                [],
                color=color_pending,
                marker="o",
                linestyle="",
                label="Pending",
                markeredgecolor="k",
            ),
            mlines.Line2D([], [], color="none", label="Thresholds"),
            mlines.Line2D([], [], color="gray", linestyle="-.", label="y=x Limit"),
            mlines.Line2D(
                [],
                [],
                color="k",
                linestyle="--",
                label=f"Max MV Cutoff ({tol_max_pct:.0f}%)",
            ),
            mlines.Line2D(
                [],
                [],
                color="k",
                linestyle=":",
                label=f"Min MV Cutoff ({tol_min_pct:.0f}%)",
            ),
        ]

        ax.legend(handles=handles)
        self._format_multi_legends(
            ax=ax,
            group_titles=["Status", "Thresholds"],
            loc="upper left",
            start_bbox=(0.05, 1.0),
        )
        self._apply_standard_format(
            ax=ax,
            title=title,
            xlabel="Max Group MV (%)",
            ylabel="Min Group MV (%)",
            append_stage=False,
        )
        ax.title.set_weight("bold")

    def _plot_qc_rescue_scatter(
        self,
        df: pd.DataFrame,
        mnar_qc_mv_tol: float,
        mnar_int_threshold: float | None,
        ax: plt.Axes,
        title: str,
    ) -> None:
        """
        Diagnostic scatter for Step 2 with dual-threshold L-shape.
        Uses advanced 2.5D Bubble mapping:
        Color/Shape -> Rescue Status
        Bubble Size -> Min Group MV Pct (The 3rd continuous metric)
        """

        if df.empty:
            ax.axis("off")
            return

        color_mnar = pu.get_equivalent_hex("tab:red", alpha=0.5)
        color_blocked = "tab:gray"
        color_pending = "tab:gray"

        df_plot = df.copy()

        def _determine_status(row: pd.Series) -> str:
            if "QC" in row["Stage1_Status"]:
                return "MNAR (QC)"
            elif (
                (row["QC_MV_Pct"] > mnar_qc_mv_tol * 100)
                and (mnar_int_threshold is not None)
                and (row["Log2_Intensity"] <= mnar_int_threshold)
            ):
                return "Blocked by Group Valid"
            else:
                return "Pending"

        df_plot["Step_Status"] = df_plot.apply(_determine_status, axis=1)
        df_plot = df_plot.sort_values(by="Step_Status", ascending=False)

        has_group_info = "Min_Group_MV_Pct" in df_plot.columns
        if has_group_info:
            sizes = df_plot["Min_Group_MV_Pct"].fillna(0)
            size_norm = (15, 100)
        else:
            sizes = None
            size_norm = None

        sns.scatterplot(
            data=df_plot,
            x="Log2_Intensity",
            y="QC_MV_Pct",
            hue="Step_Status",
            size=sizes if has_group_info else None,
            sizes=size_norm,
            palette={
                "MNAR (QC)": color_mnar,
                "Blocked by Group Valid": color_blocked,
                "Pending": color_pending,
            },
            style="Step_Status",
            markers={"MNAR (QC)": "X", "Blocked by Group Valid": "v", "Pending": "o"},
            alpha=0.75,
            ax=ax,
            legend=False,
            edgecolor="k",
        )

        ax.axhline(mnar_qc_mv_tol * 100, color="k", linestyle=":", label="MV Cutoff")

        if mnar_int_threshold is not None:
            ax.axvline(
                mnar_int_threshold, color="k", linestyle="--", label="Int Cutoff"
            )
            ax.fill_between(
                [ax.get_xlim()[0], mnar_int_threshold],
                mnar_qc_mv_tol * 100,
                100,
                color=color_mnar,
                alpha=0.15,
                zorder=0,
            )

        handles = [
            mlines.Line2D([], [], color="none", label="Status"),
            mlines.Line2D(
                [],
                [],
                color=color_mnar,
                marker="X",
                linestyle="",
                label="MNAR (QC)",
                markeredgecolor="k",
            ),
            mlines.Line2D(
                [],
                [],
                color=color_blocked,
                marker="v",
                linestyle="",
                label="Blocked",
                markeredgecolor="k",
            ),
            mlines.Line2D(
                [],
                [],
                color=color_pending,
                marker="o",
                linestyle="",
                label="Pending",
                markeredgecolor="k",
            ),
            mlines.Line2D([], [], color="none", label="Thresholds"),
            mlines.Line2D(
                [],
                [],
                color="k",
                linestyle=":",
                label=f"MV > {mnar_qc_mv_tol*100:.0f}%",
            ),
            mlines.Line2D([], [], color="k", linestyle="--", label="Low Int"),
        ]

        group_titles = ["Status", "Thresholds"]

        if has_group_info:
            handles.extend(
                [
                    mlines.Line2D([], [], color="none", label="Size Reference"),
                    mlines.Line2D(
                        [],
                        [],
                        color="none",
                        marker="o",
                        markersize=8,
                        label="Larger =\nHigher Min Group MV",
                        markeredgecolor="gray",
                        linestyle="",
                    ),
                ]
            )
            group_titles.append("Size Reference")

        ax.legend(handles=handles)
        self._format_multi_legends(
            ax=ax, group_titles=group_titles, loc="upper left", start_bbox=(0.50, 1.0)
        )

        self._apply_standard_format(
            ax=ax,
            title=title,
            xlabel="Log2(Median QC Intensity)",
            ylabel="QC Missing Rate (%)",
            append_stage=False,
        )
        ax.title.set_weight("bold")

    def _plot_cutoff_histogram(
        self,
        df: pd.DataFrame,
        x_col: str,
        hue_col: str,
        tol: float,
        palette: dict[str, str],
        hue_order: list[str],
        ax: plt.Axes,
        title: str,
        x_label: str,
    ) -> None:
        """Generic layered histogram using absolute counts."""
        if df.empty:
            ax.axis("off")
            return

        sns.histplot(
            data=df,
            x=x_col,
            hue=hue_col,
            multiple="layer",
            palette=palette,
            hue_order=hue_order,
            bins=np.arange(0, 105, 5),
            edgecolor="k",
            alpha=1,
            ax=ax,
        )

        ax.axvline(tol * 100, color="k", linestyle=":", lw=1.5)

        handles = [mlines.Line2D([], [], color="none", label="Status")]
        handles.extend(
            [
                mpatches.Patch(facecolor=palette[cat], edgecolor="k", label=cat)
                for cat in hue_order
                if cat in palette
            ]
        )

        handles.append(mlines.Line2D([], [], color="none", label="Thresholds"))
        handles.append(
            mlines.Line2D(
                [], [], color="k", linestyle=":", label=f"Cutoff ({tol*100:.0f}%)"
            )
        )

        ax.legend(handles=handles)
        self._format_multi_legends(
            ax=ax,
            group_titles=["Status", "Thresholds"],
            loc="upper left",
            start_bbox=(0.5, 1.0),
        )

        self._apply_standard_format(
            ax=ax,
            title=title,
            xlabel=x_label,
            ylabel="Feature Count",
            append_stage=False,
        )
        ax.title.set_weight("bold")

    def _plot_pipeline_flowchart_atom(
        self,
        df: pd.DataFrame,
        ax: plt.Axes,
        mnar_group_mv_tol: float,
        mnar_qc_mv_tol: float,
        active_base_tol: float,
        has_group_info: bool,
        margin_left: float = 0.0,
        margin_right: float = 0.0,
        margin_top: float = 0.0,
        margin_bottom: float = 0.0,
    ) -> None:
        """
        Horizontal flowchart with strictly QC-anchored logic.
        Dynamically adapts topology (removes Group Rescue nodes completely
        if no bio-group info exists) and re-balances X-axis coordinates.
        """
        total = len(df)
        count_group = sum(df["Stage1_Status"].str.contains("Group"))
        df_s2 = df[~df["Stage1_Status"].str.contains("Group")]
        count_qc = sum(df_s2["Stage1_Status"].str.contains("QC"))
        df_s3 = df_s2[~df_s2["Stage1_Status"].str.contains("QC")]
        count_mar = sum(df_s3["Stage1_Status"] == "MAR")
        count_inv = sum(df_s3["Stage1_Status"] == "INVALID")

        ax.axis("off")

        ax.set_xlim(0 - margin_left, 33 + margin_right)
        ax.set_ylim(0 - margin_bottom, 10 + margin_top)

        color_mar = "tab:red"
        color_mnar = pu.get_equivalent_hex("tab:red", alpha=0.5)
        color_inv = "tab:gray"
        color_pass = "white"
        box_style = "round,pad=0.9"

        def _node(x: float, y: float, text: str, bg: str) -> tuple[float, float]:
            text_color = pu.get_contrast_color(bg)
            bbox = dict(boxstyle=box_style, facecolor=bg, ec="k", lw=1.2)
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=text_color,
                bbox=bbox,
                zorder=3,
            )
            return (x, y)

        def _arrow(
            posA: tuple[float, float],
            posB: tuple[float, float],
            style: str = "horizontal",
        ) -> None:
            x_a, y_a = posA
            x_b, y_b = posB

            kwargs = dict(
                arrowstyle="-|>",
                color="gray",
                lw=2,
                mutation_scale=15,
                zorder=2,
                shrinkA=38,
                shrinkB=38,
            )

            if style in ["horizontal", "vertical"]:
                arrow = mpatches.FancyArrowPatch(
                    posA=(x_a, y_a), posB=(x_b, y_b), **kwargs
                )
            elif style == "step_h":
                mid_x = (x_a + x_b) / 2
                path = mpath.Path(
                    [(x_a, y_a), (mid_x, y_a), (mid_x, y_b), (x_b, y_b)],
                    [
                        mpath.Path.MOVETO,
                        mpath.Path.LINETO,
                        mpath.Path.LINETO,
                        mpath.Path.LINETO,
                    ],
                )
                arrow = mpatches.FancyArrowPatch(path=path, **kwargs)
            ax.add_patch(arrow)

        # =====================================================================
        # Topology A: Full Pipeline (With BioGroups)
        # 4 Logical Columns distributed across X=[2.0, 9.5, 17.0, 24.5, 31.0]
        # =====================================================================
        if has_group_info:
            str_group = (
                f"Max MV ≥ {mnar_group_mv_tol*100:.0f}%\n"
                f"& Min MV ≤ {active_base_tol*100:.0f}%"
            )
            qc_cond = (
                f"QC MV > {mnar_qc_mv_tol*100:.0f}%\n& Low Int\n"
                f"& Min Group\nMV ≤ {active_base_tol*100:.0f}%"
            )

            node_root = _node(2.0, 5, f"Raw Features\n(n={total})", color_pass)
            node_c1 = _node(9.5, 5, f"Group Rescue\n({str_group})", color_pass)
            node_g = _node(9.5, 8.5, f"MNAR Group\n(n={count_group})", color_mnar)
            node_c2 = _node(17.0, 5, f"QC Rescue\n({qc_cond})", color_pass)
            node_q = _node(17.0, 8.5, f"MNAR QC\n(n={count_qc})", color_mnar)
            node_c3 = _node(
                24.5,
                5,
                f"Min Group\nMV Check\n(≤ {active_base_tol*100:.0f}%)",
                color_pass,
            )
            node_mar = _node(31.0, 7.5, f"MAR\n(n={count_mar})", color_mar)
            node_inv = _node(31.0, 2.5, f"INVALID\n(n={count_inv})", color_inv)

            _arrow(node_root, node_c1, "horizontal")
            _arrow(node_c1, node_c2, "horizontal")
            _arrow(node_c1, node_g, "vertical")
            _arrow(node_c2, node_c3, "horizontal")
            _arrow(node_c2, node_q, "vertical")
            _arrow(node_c3, node_mar, "step_h")
            _arrow(node_c3, node_inv, "step_h")

        # =====================================================================
        # Topology B: Simplified Pipeline (No BioGroups)
        # 3 Logical Columns distributed dynamically across X=[3.0, 12.0, 21.0, 30.0]
        # =====================================================================
        else:
            qc_cond = f"QC MV > {mnar_qc_mv_tol*100:.0f}%\n& Low Int"

            node_root = _node(3.0, 5, f"Raw Features\n(n={total})", color_pass)
            node_c2 = _node(12.0, 5, f"QC Rescue\n({qc_cond})", color_pass)
            node_q = _node(12.0, 8.5, f"MNAR QC\n(n={count_qc})", color_mnar)
            node_c3 = _node(
                21.0, 5, f"QC MV Check\n(≤ {active_base_tol*100:.0f}%)", color_pass
            )
            node_mar = _node(30.0, 7.5, f"MAR\n(n={count_mar})", color_mar)
            node_inv = _node(30.0, 2.5, f"INVALID\n(n={count_inv})", color_inv)

            _arrow(node_root, node_c2, "horizontal")
            _arrow(node_c2, node_c3, "horizontal")
            _arrow(node_c2, node_q, "vertical")
            _arrow(node_c3, node_mar, "step_h")
            _arrow(node_c3, node_inv, "step_h")

    def plot_mv_filtering_summary_grid(
        self,
        tracking_df: pd.DataFrame,
        active_base_tol: float,
        mnar_group_mv_tol: float | None = None,
        mnar_qc_mv_tol: float = 0.2,
        mnar_int_threshold: float | None = None,
    ) -> object | None:
        """
        Orchestrates a unified diagnostic dashboard for Stage-1 filtering.
        Dynamically adapts layout based on biological grouping and utilizes
        patchworklib to align subplots precisely via topological rules.
        """
        try:
            import patchworklib as pw
        except ImportError:
            return None

        pw.clear()

        # 1. Initialize data copy and evaluate biological grouping status
        df_curr = tracking_df.copy()
        has_group_info = ("Max_Group_MV_Pct" in df_curr.columns) and (
            df_curr["Max_Group_MV_Pct"].notna().any()
        )

        # 2. Build the universal Sample MV Stripplot Brick
        ax_sample = pw.Brick(figsize=(4, 4), label="sample_mv")
        sample_track = self.engine.stats.get("sample_tracking", pd.DataFrame())
        sample_mv_tol = self.engine.attrs.get("sample_mv_tol", 0.5)
        self._plot_sample_mv_stripplot(sample_track, sample_mv_tol, ax=ax_sample)

        # 3. Dynamic layout assembly based on biological grouping
        if has_group_info:
            # --- Layout A: With Groups (1+2 Top, 1+1+1 Bottom) ---

            # Flowchart ratio is 2 units wide to match the bottom 2 plots
            ax_flow = pw.Brick(figsize=(8, 4), label="flowchart")
            self._plot_pipeline_flowchart_atom(
                df=df_curr,
                ax=ax_flow,
                mnar_group_mv_tol=mnar_group_mv_tol,
                mnar_qc_mv_tol=mnar_qc_mv_tol,
                active_base_tol=active_base_tol,
                has_group_info=True,
                margin_right=0.0,
            )

            # Subplot S1: Group Rescue Scatter
            ax_group_rescue = pw.Brick(figsize=(4, 4), label="s1")
            self._plot_group_rescue_scatter(
                df_curr,
                "Max_Group_MV_Pct",
                "Min_Group_MV_Pct",
                mnar_group_mv_tol,
                active_base_tol,
                ax_group_rescue,
                "Group-level MNAR Rescue",
            )
            # Cascade remaining features downward
            mask_group = df_curr["Stage1_Status"].str.contains("Group")
            df_curr = df_curr[~mask_group]

            # Subplot S2: QC Rescue Scatter
            ax_qc_rescue = pw.Brick(figsize=(4, 4), label="s2")
            self._plot_qc_rescue_scatter(
                df_curr,
                mnar_qc_mv_tol,
                mnar_int_threshold,
                ax_qc_rescue,
                "QC-level MNAR Rescue",
            )
            # Cascade remaining features downward
            mask_qc = df_curr["Stage1_Status"].str.contains("QC")
            df_curr = df_curr[~mask_qc]

            # Subplot S3: Base Threshold Check Histogram
            ax_base_check = pw.Brick(figsize=(4, 4), label="s3")
            self._plot_cutoff_histogram(
                df_curr,
                "Min_Group_MV_Pct",
                "Stage1_Status",
                active_base_tol,
                {"MAR": "tab:red", "INVALID": "tab:gray"},
                ["MAR", "INVALID"],
                ax_base_check,
                "Min Group-level MV Check",
                "Min Group-level MV (%)",
            )

            # Column-first topology binding to enforce strict vertical alignment
            # Prevents width stretching caused by the axis-off flowchart
            col_left = ax_sample / ax_group_rescue
            col_right = ax_flow / (ax_qc_rescue | ax_base_check)
            return col_left | col_right

        else:
            # --- Layout B: No Groups (1 Full-width Top, 1+1+1 Bottom) ---

            # Flowchart ratio is 3 units wide to span the entire top row
            ax_flow = pw.Brick(figsize=(12, 4), label="flowchart")
            self._plot_pipeline_flowchart_atom(
                df=df_curr,
                ax=ax_flow,
                mnar_group_mv_tol=None,
                mnar_qc_mv_tol=mnar_qc_mv_tol,
                active_base_tol=active_base_tol,
                has_group_info=False,
            )

            # Subplot S2: QC Rescue Scatter (Acts as Step 1 here)
            ax_qc_rescue = pw.Brick(figsize=(4, 4), label="s2")
            self._plot_qc_rescue_scatter(
                df_curr,
                mnar_qc_mv_tol,
                mnar_int_threshold,
                ax_qc_rescue,
                "QC-level MNAR Rescue",
            )
            mask_qc = df_curr["Stage1_Status"].str.contains("QC")
            df_curr = df_curr[~mask_qc]

            # Subplot S3: Base Threshold Check Histogram (Acts as Step 2 here)
            ax_base_check = pw.Brick(figsize=(4, 4), label="s3")
            self._plot_cutoff_histogram(
                df_curr,
                "QC_MV_Pct",
                "Stage1_Status",
                active_base_tol,
                {"MAR": "tab:red", "INVALID": "tab:gray"},
                ["MAR", "INVALID"],
                ax_base_check,
                "QC-level MV Check",
                "QC-level MV (%)",
            )

            # Row-first topology binding: Full width top over 3 equal bottom
            row_bottom = ax_sample | ax_qc_rescue | ax_base_check
            return ax_flow / row_bottom

    # =========================================================================
    # Low-quality Features Filtering Unified Summary Dashboard (1+N Layout)
    # =========================================================================
    def _plot_retained_count_steps(
        self, ax: plt.Axes | None = None
    ) -> plt.Figure | plt.Axes:
        """Plot feature attrition cascade stacked bar chart by MAR/MNAR."""

        if ax is None:
            fig, current_ax = plt.subplots(figsize=(4.5, 4))
        else:
            current_ax = ax
            fig = current_ax.figure

        feature_counts = self.engine.stats.get("feature_counts", {})
        stats = self.engine.stats

        # Check if blank samples exist to dynamically adapt the X-axis steps
        blank_mean = stats.get("blank_mean")
        has_blanks = blank_mean is not None and not blank_mean.empty

        step_keys = ["raw", "post_stage1", "post_stage2_blank", "post_stage2_rsd"]
        step_labels = [
            "Raw\nData",
            "High-MV\nCheck",
            "QC/Blank\nCheck",
            "QC RSD\nCheck",
        ]

        # Build valid indices, skipping Blank Check entirely if no blanks exist
        valid_idx = []
        for i, k in enumerate(step_keys):
            if k in feature_counts:
                if k == "post_stage2_blank" and not has_blanks:
                    continue
                valid_idx.append(i)

        if not valid_idx:
            return fig if ax is None else current_ax

        labels = [step_labels[i] for i in valid_idx]

        idx_mar = stats.get("idx_mar", pd.Index([]))
        idx_mnar = stats.get("idx_mnar", pd.Index([]))
        idx_dropped_blank = stats.get("idx_dropped_blank", pd.Index([]))
        idx_dropped_rsd = stats.get("idx_dropped_rsd", pd.Index([]))

        mar_base = len(idx_mar)
        mnar_base = len(idx_mnar)

        blank_drop_mar = len(idx_dropped_blank.intersection(idx_mar))
        blank_drop_mnar = len(idx_dropped_blank.intersection(idx_mnar))
        rsd_drop_mar = len(idx_dropped_rsd.intersection(idx_mar))
        rsd_drop_mnar = len(idx_dropped_rsd.intersection(idx_mnar))

        mar_all = np.array(
            [
                mar_base,
                mar_base,
                mar_base - blank_drop_mar,
                mar_base - blank_drop_mar - rsd_drop_mar,
            ]
        )

        mnar_all = np.array(
            [
                mnar_base,
                mnar_base,
                mnar_base - blank_drop_mnar,
                mnar_base - blank_drop_mnar - rsd_drop_mnar,
            ]
        )

        inv_base = max(0, feature_counts.get("raw", 0) - (mar_base + mnar_base))
        inv_all = np.array([inv_base, 0, 0, 0])

        mar_counts = mar_all[valid_idx]
        mnar_counts = mnar_all[valid_idx]
        inv_counts = inv_all[valid_idx]

        color_mar = "tab:red"
        color_mnar = pu.get_equivalent_hex("tab:red", alpha=0.5)
        color_inv = "tab:gray"

        x = np.arange(len(labels))
        width = 0.55

        # Track dynamic bottoms for stacked bars to prevent empty legend items
        current_bottom = np.zeros(len(labels))

        if mar_base > 0:
            current_ax.bar(
                x,
                mar_counts,
                bottom=current_bottom,
                label="MAR",
                color=color_mar,
                edgecolor="k",
                width=width,
            )
            current_bottom += mar_counts

        if mnar_base > 0:
            current_ax.bar(
                x,
                mnar_counts,
                bottom=current_bottom,
                label="MNAR",
                color=color_mnar,
                edgecolor="k",
                width=width,
            )
            current_bottom += mnar_counts

        if inv_base > 0:
            current_ax.bar(
                x,
                inv_counts,
                bottom=current_bottom,
                label="Invalid",
                color=color_inv,
                edgecolor="k",
                width=width,
            )
            current_bottom += inv_counts

        totals = current_bottom

        current_ax.set_xticks(x)
        current_ax.set_xticklabels(labels)

        pu.show_values_on_bars(
            axs=current_ax,
            value_format="{:.0f}",
            fontsize=8,
            stacked=True,
            skip_zero=True,
            threshold_pct=0.05,
        )

        self._apply_standard_format(
            ax=current_ax,
            title="Feature Retention Across Filtering Steps",
            xlabel="Filtering Steps",
            ylabel="Feature Count",
            append_stage=False,
        )

        self._format_single_legend(
            ax=current_ax, title="Feature Type", loc="upper right", bbox_to_anchor=None
        )

        max_height = totals.max() if len(totals) > 0 else 1
        current_ax.set_ylim(0, max_height * 1.25)

        if ax is None:
            return fig
        return current_ax

    def _plot_qc_blank_scatter(
        self, ax: plt.Axes | None = None
    ) -> plt.Figure | plt.Axes | None:
        """Plots Log2 scatter of QC vs Blank intensities."""
        blank_mean = self.engine.stats.get("blank_mean")
        qc_mean = self.engine.stats.get("qc_mean")

        idx_mnar = pd.Index(self.engine.stats.get("idx_mnar", []))

        if blank_mean is None or qc_mean is None or blank_mean.empty:
            return None if ax is None else ax

        # Prepare data frame for plotting (treat missing blanks as 0)
        blank_safe = blank_mean.fillna(0).astype(float)

        df_plot = pd.DataFrame(
            {"QC": np.log2(qc_mean.astype(float) + 1), "Blank": np.log2(blank_safe + 1)}
        )

        df_plot["Feature Type"] = "MAR"
        valid_mnar = idx_mnar.intersection(df_plot.index)
        if not valid_mnar.empty:
            df_plot.loc[valid_mnar, "Feature Type"] = "MNAR"

        # [CRITICAL FIX]: Use blank_safe for ratio to match engine logic.
        # NaN <= 0.2 evaluates to False, falsely flagging them as Filtered.
        blank_qc_ratio_tol = self.engine.attrs.get("blank_qc_ratio_tol", 0.2)
        qc_safe = qc_mean.replace(0, np.finfo(float).eps).astype(float)

        df_plot["Status"] = np.where(
            blank_safe / qc_safe <= blank_qc_ratio_tol, "Retained", "Filtered"
        )

        # # ====================================================================
        # # [DEBUG MODULE]: Print exact coordinates of MNAR points to console
        # # ====================================================================
        # mnar_df = df_plot[df_plot["Feature Type"] == "MNAR"]
        # logger.info(
        #     f"--- DEBUG: Total MNAR features mapped: {len(mnar_df)} ---")

        # for status in ["Retained", "Filtered"]:
        #     subset = mnar_df[mnar_df["Status"] == status]
        #     logger.info(f"DEBUG | MNAR [{status}]: {len(subset)} features.")
        #     if not subset.empty:
        #         # Print up to 10 coordinates to avoid terminal flooding
        #         logger.info(
        #             f"DEBUG | MNAR [{status}] Coordinates (QC, Blank):\n"
        #             f"{subset[['QC', 'Blank']].head(10)}"
        #         )
        # # ====================================================================

        # [CRITICAL FIX]: Sort DataFrame to ensure MNAR points plot on top.
        # Alphabetical sorting ("MAR" < "MNAR") pushes MNAR to the bottom of
        # the DataFrame, causing seaborn to render them last and on top.
        df_plot = df_plot.sort_values(by="Feature Type", ascending=True)

        if ax is None:
            fig, current_ax = plt.subplots(figsize=(5, 4))
        else:
            current_ax = ax
            fig = current_ax.figure

        sns.scatterplot(
            data=df_plot,
            x="QC",
            y="Blank",
            ax=current_ax,
            hue="Status",
            palette={"Retained": "tab:gray", "Filtered": "tab:red"},
            style="Feature Type",
            markers={"MAR": "o", "MNAR": "X"},
            s=50,
            edgecolor="k",
            linewidth=0.5,
        )

        lims = [
            np.min([current_ax.get_xlim(), current_ax.get_ylim()]),
            np.max([current_ax.get_xlim(), current_ax.get_ylim()]),
        ]
        x_line = np.linspace(max(0, lims[0]), lims[1], 200)
        current_ax.plot(
            x_line,
            np.log2(((2**x_line - 1) * blank_qc_ratio_tol) + 1),
            color="k",
            linestyle="--",
            linewidth=1.5,
            label=f"Ratio={blank_qc_ratio_tol}",
        )

        self._apply_standard_format(
            ax=current_ax,
            title="Blank/QC Check",
            xlabel="Log2(Mean QC + 1)",
            ylabel="Log2(Mean Blank + 1)",
            append_stage=False,
        )

        self._format_multi_legends(
            ax=current_ax,
            group_titles=["Status", "Feature Type"],
            loc="upper left",
            start_bbox=(1.05, 1.0),
        )

        if ax is None:
            return fig
        return current_ax

    def _plot_rsd_dist(
        self, idx_mnar: pd.Index | list[object], ax: plt.Axes | None = None
    ) -> plt.Figure | plt.Axes | None:
        """Plot RSD distribution with consistent bins for MAR and MNAR."""
        qc_rsd_all = self.engine.stats.get("qc_rsd_all")
        if qc_rsd_all is None or qc_rsd_all.empty:
            return None if ax is None else ax

        if not isinstance(idx_mnar, pd.Index):
            idx_mnar = pd.Index(idx_mnar)

        if ax is None:
            fig, current_ax = plt.subplots(figsize=(4, 4))
        else:
            current_ax = ax
            fig = current_ax.figure

        qc_rsd_tol = self.engine.attrs.get("qc_rsd_tol", 0.3)
        idx_mnar_valid = qc_rsd_all.index.intersection(idx_mnar)

        types = pd.Series("MAR", index=qc_rsd_all.index)
        if not idx_mnar_valid.empty:
            types.loc[idx_mnar_valid] = "MNAR"

        df_plot = pd.DataFrame({"RSD": qc_rsd_all, "Feature Type": types})

        max_rsd = float(qc_rsd_all.max())
        bin_edges = np.linspace(0, max_rsd, 50)

        sns.histplot(
            data=df_plot,
            x="RSD",
            hue="Feature Type",
            palette={"MAR": "tab:gray", "MNAR": "tab:red"},
            hue_order=[
                t for t in ["MAR", "MNAR"] if t in df_plot["Feature Type"].values
            ],
            bins=bin_edges,
            kde=True,
            ax=current_ax,
            legend=False,
            edgecolor="k",
            alpha=0.6,
        )

        current_ax.axvline(x=qc_rsd_tol, color="k", linestyle="--", linewidth=1.5)

        handles = []
        if "MAR" in df_plot["Feature Type"].values:
            handles.append(
                mpatches.Patch(
                    facecolor="tab:gray", edgecolor="k", linewidth=1.0, label="MAR"
                )
            )
        if "MNAR" in df_plot["Feature Type"].values:
            handles.append(
                mpatches.Patch(
                    facecolor="tab:red", edgecolor="k", linewidth=1.0, label="MNAR"
                )
            )

        handles.append(
            mlines.Line2D(
                [],
                [],
                color="k",
                linestyle="--",
                label=f"MAR Threshold ({qc_rsd_tol})",
                linewidth=1.0,
            )
        )

        current_ax.legend(
            handles=handles,
            title="Feature Type",
            loc="upper right",
            **getattr(self, "LEGEND_KWARGS", {}),
        )

        self._apply_standard_format(
            ax=current_ax,
            title="QC RSD Check",
            xlabel="RSD",
            ylabel="Feature Count",
            append_stage=False,
        )

        if ax is None:
            return fig
        return current_ax

    def plot_quality_filtering_summary_grid(self) -> object | None:
        """Combine Stage 2 plots into a single figure using patchworklib.

        Dynamically adapts the grid layout: renders a 1x3 grid if Blank
        samples are present, or a 1x2 grid if Blank samples are missing.
        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping summary grid.")
            return None

        pw.clear()

        # 1. Detect if Blank data exists to determine the topology
        blank_mean = self.engine.stats.get("blank_mean")
        has_blanks = blank_mean is not None and not blank_mean.empty

        idx_mnar = self.engine.stats.get("idx_mnar", pd.Index([]))

        # 2. Topology A: 1x3 Grid (Blank samples exist)
        if has_blanks:
            ax1 = pw.Brick(figsize=(4, 4), label="qc_blank")
            ax2 = pw.Brick(figsize=(4, 4), label="qc_rsd")
            ax3 = pw.Brick(figsize=(4, 4), label="retention")

            self._plot_qc_blank_scatter(ax=ax1)
            self._plot_rsd_dist(idx_mnar=idx_mnar, ax=ax2)
            self._plot_retained_count_steps(ax=ax3)

            return ax1 | ax2 | ax3

        # 3. Topology B: 1x2 Grid (No Blank samples)
        else:
            ax2 = pw.Brick(figsize=(4, 4), label="qc_rsd")
            ax3 = pw.Brick(figsize=(4, 4), label="retention")

            self._plot_rsd_dist(idx_mnar=idx_mnar, ax=ax2)
            self._plot_retained_count_steps(ax=ax3)

            return ax2 | ax3
