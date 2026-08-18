"""Dataset construction and validation from metadata and intensity tables.

MetaboIntBuilder reconciles sample identifiers, validates required metadata,
normalizes acquisition order, detects available QC, blank, internal-standard,
and reference-feature channels, then returns an annotated MetaboInt object.
It records audit information and exports construction diagnostics for the first
pipeline stage.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from typing import Optional, Dict, Any

from ..io import ensure_directory
from ..runtime import log_execution_time
from ..core.model import MetaboInt


class MetaboIntBuilder:
    """
    Builder class responsible for data integrity checks, metadata
    alignment, and the construction of the core MetaboInt object.
    Driven completely by the pipeline TOML configuration.
    """

    def __init__(
        self,
        meta_info: pd.DataFrame,
        int_df: pd.DataFrame,
        pipeline_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize builder using global pipeline configuration."""
        self.metadata_dataframe = meta_info.copy()
        self.intensity_dataframe = int_df.copy()
        self.params = pipeline_params or {}

        # Extract configuration attributes smartly
        metabo_parms = self.params.get("MetaboInt", {})

        self.mode = metabo_parms.get("mode", "POS")
        self.batch = metabo_parms.get("batch", "Batch")
        self.sample_type = metabo_parms.get("sample_type", "Sample Type")
        self.bio_group = metabo_parms.get("bio_group", "Bio Group")
        self.sample_name = metabo_parms.get("sample_name", "Sample Name")
        self.inject_order = metabo_parms.get("inject_order", "Inject Order")
        self.resort_strategy = metabo_parms.get("resort_inject_order", "Auto")

        self.unique_batches = []
        self.is_multi_batch = False

    def _resolve_duplicate_features(self) -> None:
        """Average intensities for duplicated metabolite names."""
        if isinstance(self.intensity_dataframe.index, pd.RangeIndex):
            first_column = self.intensity_dataframe.columns[0]
            logger.warning(
                f"RangeIndex detected. Setting '{first_column}' as "
                "feature index."
            )
            self.intensity_dataframe = self.intensity_dataframe.set_index(
                first_column
            )

        self.intensity_dataframe.index.name = "Metabolite"

        # Record original index to identify modified features
        orig_idx = self.intensity_dataframe.index.astype(str)
        new_idx = orig_idx.str.strip()

        # Calculate exactly how many features had leading/trailing whitespaces
        stripped_count = (orig_idx != new_idx).sum()
        self.intensity_dataframe.index = new_idx

        # Logger 1: Output the number of stripped features
        if stripped_count > 0:
            logger.info(
                f"Stripped whitespaces from {stripped_count} feature(s)."
            )

        if self.intensity_dataframe.index.duplicated().any():
            num_duplicates = self.intensity_dataframe.index.duplicated().sum()
            duplicated_features = (
                self.intensity_dataframe.index[
                    self.intensity_dataframe.index.duplicated()
                ]
                .unique()
                .tolist()
            )

            # Logger 2: Output the exact names of features merged by mean
            logger.info(f"Features merged via mean: {duplicated_features}")

            logged_features = (
                duplicated_features[:5] + ["..."]
                if len(duplicated_features) > 5
                else duplicated_features
            )
            logger.warning(
                f"Detected {num_duplicates} duplicate row indices "
                f"(e.g., {logged_features}). Merging their intensities "
                "by averaging (mean)."
            )
            self.intensity_dataframe = self.intensity_dataframe.groupby(
                level=0, sort=False
            ).mean()

    def _check_duplicate_samples(self) -> None:
        """Abort if intensity dataframe contains duplicated sample columns."""
        value_counts = self.intensity_dataframe.columns.value_counts()
        if value_counts.max() > 1:
            duplicated_names = value_counts[value_counts > 1].index.tolist()
            duplicated_details = []

            for name in duplicated_names:
                locations = [
                    i
                    for i, col in enumerate(self.intensity_dataframe.columns)
                    if col == name
                ]
                duplicated_details.append(f'"{name}" (indices: {locations})')

            error_message = (
                "Duplicate sample names in intensity data: "
                f"{', '.join(duplicated_details)}."
            )
            raise AssertionError(error_message)

    def _verify_sample_consistency(self) -> None:
        """Ensure sample names perfectly match between meta and intensity."""
        meta_samples = set(self.metadata_dataframe[self.sample_name])
        intensity_samples = set(self.intensity_dataframe.columns)

        if meta_samples != intensity_samples:
            only_in_meta = sorted(list(meta_samples - intensity_samples))
            only_in_intensity = sorted(list(intensity_samples - meta_samples))

            intersection_size = len(
                meta_samples.intersection(intensity_samples)
            )
            union_size = len(meta_samples.union(intensity_samples))
            jaccard_score = intersection_size / union_size

            message = [
                f"Sample inconsistency (Jaccard Score: {jaccard_score:.4f})."
            ]

            if only_in_meta:
                log_meta = (
                    only_in_meta[:5] + ["..."]
                    if len(only_in_meta) > 5
                    else only_in_meta
                )
                message.append(f"Only in Metadata: {log_meta}")

            if only_in_intensity:
                log_intensity = (
                    only_in_intensity[:5] + ["..."]
                    if len(only_in_intensity) > 5
                    else only_in_intensity
                )
                message.append(f"Only in Intensity: {log_intensity}")

            raise AssertionError(" ".join(message))

    def _verify_metadata_completeness(self) -> None:
        """Abort if critical tracking columns are missing from metadata."""
        required_cols = {
            "Batch": self.batch,
            "Sample Type": self.sample_type,
            "Sample Name": self.sample_name,
            "Inject Order": self.inject_order,
        }

        missing_cols = [
            col_name
            for label, col_name in required_cols.items()
            if col_name not in self.metadata_dataframe.columns
        ]

        if missing_cols:
            raise AssertionError(
                f"Incomplete metadata. Missing columns: {missing_cols}."
            )

        self.unique_batches = self.metadata_dataframe[self.batch].unique()
        self.is_multi_batch = len(self.unique_batches) > 1

    def _manage_injection_orders(self) -> None:
        """Align injection sequences across batches to prevent overlap."""
        if not self.is_multi_batch or (
            self.inject_order not in self.metadata_dataframe
        ):
            return

        if not self.resort_strategy:
            return

        self.metadata_dataframe[self.inject_order] = pd.to_numeric(
            self.metadata_dataframe[self.inject_order], errors="coerce"
        )
        ordered_batches = sorted(
            self.metadata_dataframe[self.batch].dropna().unique().tolist()
        )

        is_overlap = False
        current_max = -float("inf")
        for batch_id in ordered_batches:
            batch_mask = self.metadata_dataframe[self.batch] == batch_id
            if batch_mask.sum() == 0:
                continue
            batch_min = self.metadata_dataframe.loc[
                batch_mask, self.inject_order
            ].min()
            batch_max = self.metadata_dataframe.loc[
                batch_mask, self.inject_order
            ].max()

            if batch_min <= current_max:
                is_overlap = True
                break
            current_max = max(current_max, batch_max)

        trigger = False
        if self.resort_strategy == "Auto" and is_overlap:
            trigger = True
        elif str(self.resort_strategy).lower() in ["force", "true", "always"]:
            trigger = True

        if trigger:
            logger.warning(
                f"Inject orders resort triggered "
                f"(mode: {self.resort_strategy})."
                " Re-numbering sequentially to ensure global continuity."
            )
            previous_max = None
            for batch_id in ordered_batches:
                batch_mask = self.metadata_dataframe[self.batch] == batch_id
                if batch_mask.sum() == 0:
                    continue

                batch_min = self.metadata_dataframe.loc[
                    batch_mask, self.inject_order
                ].min()
                if previous_max is not None:
                    offset = previous_max - batch_min + 1
                    self.metadata_dataframe.loc[
                        batch_mask, self.inject_order
                    ] += offset
                previous_max = self.metadata_dataframe.loc[
                    batch_mask, self.inject_order
                ].max()

    def _audit_dataset_health(self, metabo_obj: MetaboInt) -> None:
        """
        Conducts a comprehensive health audit on the newly built dataset.
        Emits targeted warnings and infos to set user expectations regarding
        pipeline degradation and algorithmic limitations.
        """
        logger.info("Executing dataset health audit...")

        # Blank Samples Check
        if metabo_obj._blank.empty:
            logger.warning(
                "[Audit] No Blank samples detected. Pipeline will skip Stage-2 "
                "Blank/QC ratio filtering and degrade to QC RSD check only."
            )

        # Biological Groups Check
        if not self.bio_group or self.bio_group not in metabo_obj.columns.names:
            logger.warning(
                "[Audit] No Biological Group information detected. "
                "Missing value "
                "imputation and filtering will fall back to QC-only rescue."
            )

        # Internal Standards (IS) Check
        if len(getattr(metabo_obj, "valid_is", [])) == 0:
            logger.warning(
                "[Audit] No Internal Standards (IS) detected. "
                "Analytical outlier "
                "diagnostics will rely solely on global PCA statistics."
            )

        # Outlier Reference Features (ORF) Check
        if len(getattr(metabo_obj, "valid_orf", [])) == 0:
            logger.info(
                "[Audit] No Outlier Reference Features (ORF) detected. This is "
                "normal for untargeted datasets; ORF diagnostics "
                "will be skipped."
            )

        # High-throughput Cohort Check (Batch Count > 15)
        # Threshold set to 15 to match the plotter's MathText threshold
        n_batches = len(metabo_obj.attrs.get("batch_list", []))
        if n_batches > 15:
            logger.info(
                f"[Audit] High batch count (n={n_batches}) detected. "
                "Visualizations will automatically switch to "
                "alphanumeric mode for readability."
            )

        # Critical QC Density Check
        if metabo_obj._qc.empty:
            logger.error(
                "[Audit] FATAL: No QC samples detected! Subsequent correction "
                "and evaluation steps will inevitably fail."
            )
        else:
            qc_counts = metabo_obj._qc.columns.get_level_values(
                self.batch
            ).value_counts()

            # Warn if any batch has fewer than 3 QCs (Minimum required for
            # SVR/RFSC)
            weak_batches = qc_counts[qc_counts < 3].index.tolist()
            if weak_batches:
                logger.warning(
                    f"[Audit] Batches {weak_batches} contain fewer than 3 QCs. "
                    "Step-wise machine learning correction algorithms (e.g., "
                    "QC-SVR) may severely overfit or fail in these batches."
                )

    def execute_build(self, output_dir: Optional[str] = None) -> MetaboInt:
        """Execute the validation pipeline and build the MetaboInt object."""
        self._resolve_duplicate_features()
        self._check_duplicate_samples()
        self._verify_sample_consistency()
        self._verify_metadata_completeness()
        self._manage_injection_orders()

        # Build MultiIndex matrix
        self.intensity_dataframe = self.intensity_dataframe.rename_axis(
            index=["Metabolite"], columns=[self.sample_name]
        )
        column_dataframe = (
            self.intensity_dataframe.columns.to_frame().reset_index(drop=True)
        )

        column_dataframe = pd.merge(
            left=column_dataframe,
            right=self.metadata_dataframe,
            on=self.sample_name,
            how="left",
        )

        has_bio_group = pd.notna(self.bio_group) and (
            self.bio_group in self.metadata_dataframe.columns
        )
        column_order = (
            [
                self.batch,
                self.sample_type,
                self.bio_group,
                self.inject_order,
                self.sample_name,
            ]
            if has_bio_group
            else [
                self.batch,
                self.sample_type,
                self.inject_order,
                self.sample_name,
            ]
        )

        self.intensity_dataframe.columns = pd.MultiIndex.from_frame(
            column_dataframe.loc[:, column_order]
        )

        # Filter out samples lacking names
        valid_mask = self.intensity_dataframe.columns.get_level_values(
            level=self.sample_name
        ).notnull()
        self.intensity_dataframe = self.intensity_dataframe.loc[:, valid_mask]

        # =====================================================================
        # Zero-value detection and conversion
        # =====================================================================
        zero_mask = self.intensity_dataframe == 0
        zero_count = int(zero_mask.sum().sum())

        if zero_count > 0:
            logger.warning(
                f"[Dataset Builder] Detected {zero_count} explicit zero (0) "
                "values in the raw matrix. In mass spectrometry, these "
                "typically represent missing values rather than true zero "
                "intensity. They have been automatically converted to NaN to "
                "ensure pipeline safety."
            )
            # Represent explicit zeros as missing values for downstream steps.
            self.intensity_dataframe = self.intensity_dataframe.replace(
                0, np.nan
            )

        # Use float64 for stable downstream machine-learning assignments.
        self.intensity_dataframe = self.intensity_dataframe.astype(float)

        if output_dir:
            ensure_directory(output_dir)
            output_path = os.path.join(output_dir, "Raw_Data_Intensity.csv")
            self.intensity_dataframe.to_csv(
                output_path, na_rep="NA", encoding="utf-8-sig"
            )
            logger.info(f"MetaboInt raw dataset saved as: {output_path}")

        # Instantiate MetaboInt
        metabo_obj = MetaboInt(
            self.intensity_dataframe,
            pipeline_params=self.params,
            mode=self.mode,
            sample_name=self.sample_name,
            sample_type=self.sample_type,
            bio_group=self.bio_group,
            batch=self.batch,
            inject_order=self.inject_order,
        )

        metabo_obj.attrs["is_multi_batch"] = self.is_multi_batch
        metabo_obj.attrs["batch_list"] = self.unique_batches.tolist()

        logger.info(
            f"MetaboInt object built: {metabo_obj.shape[0]} metabolites, "
            f"{metabo_obj.shape[1]} samples."
        )

        self._audit_dataset_health(metabo_obj)

        return metabo_obj


@log_execution_time
def build_dataset(
    meta_info: pd.DataFrame,
    int_df: pd.DataFrame,
    pipeline_params: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
) -> MetaboInt:
    """
    Factory wrapper for MetaboIntBuilder, driven strictly by configuration.
    Accepts raw dataframes and config, returning a MetaboInt object.
    """
    builder = MetaboIntBuilder(
        meta_info=meta_info, int_df=int_df, pipeline_params=pipeline_params
    )

    metabo_obj = builder.execute_build(output_dir=output_dir)
    if output_dir is not None:
        _render_dataset_dashboard(metabo_obj, output_dir)
    return metabo_obj


def _render_dataset_dashboard(
    metabo_obj: MetaboInt,
    output_dir: str | os.PathLike[str],
) -> None:
    """Render the dataset overview as part of dataset execution.

    Plotting is imported lazily so dataset validation and construction remain
    usable without importing the plotting stack. Notebook display is handled
    by ``save_and_show_pw`` when the same execution runs in Jupyter.
    """
    from ..plotting.dataset import DatasetPlotter

    try:
        plotter = DatasetPlotter(builder_obj=metabo_obj)
        dashboard = plotter.plot_dataset_dashboard()
        if dashboard is None:
            logger.warning(
                "Dataset dashboard was skipped because no plotting backend "
                "was available."
            )
            return

        overview_path = Path(output_dir) / "Global_Acquisition_Overview.svg"
        plotter.save_and_show_pw(
            pw_obj=dashboard,
            file_path=str(overview_path),
        )
        logger.info(
            f"Global acquisition overview plot saved as: {overview_path}"
        )
    except Exception as exc:
        logger.error(f"Dataset dashboard rendering failed: {exc}")
