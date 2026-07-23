# src/pimqc/dataset_builder.py
"""
Script purpose: Convert raw project tables into a validated MetaboInt object.

The execute_build() workflow resolves duplicate feature names, rejects
duplicate or unmatched sample IDs, verifies required metadata, and normalizes
injection-order and batch structure before creating the MultiIndex intensity
matrix. It treats explicit zero intensities as missing values, casts the
matrix to float for downstream ML steps, and saves the raw matrix when an
output directory is provided.
The builder then stamps batch metadata, audits dataset health, and optionally
renders the acquisition overview dashboard.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from loguru import logger
from typing import Optional, Dict, Any

from . import io_utils as iu
from . import plot_utils as pu
from . import visualizer_classes
from .core_classes import MetaboInt


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
                f"RangeIndex detected. Setting '{first_column}' as feature index."
            )
            self.intensity_dataframe = self.intensity_dataframe.set_index(first_column)

        self.intensity_dataframe.index.name = "Metabolite"

        # Record original index to identify modified features
        orig_idx = self.intensity_dataframe.index.astype(str)
        new_idx = orig_idx.str.strip()

        # Calculate exactly how many features had leading/trailing whitespaces
        stripped_count = (orig_idx != new_idx).sum()
        self.intensity_dataframe.index = new_idx

        # Logger 1: Output the number of stripped features
        if stripped_count > 0:
            logger.info(f"Stripped whitespaces from {stripped_count} feature(s).")

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

            intersection_size = len(meta_samples.intersection(intensity_samples))
            union_size = len(meta_samples.union(intensity_samples))
            jaccard_score = intersection_size / union_size

            message = [f"Sample inconsistency (Jaccard Score: {jaccard_score:.4f})."]

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
            batch_min = self.metadata_dataframe.loc[batch_mask, self.inject_order].min()
            batch_max = self.metadata_dataframe.loc[batch_mask, self.inject_order].max()

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
                f"Inject orders resort triggered (mode: {self.resort_strategy})."
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
                    self.metadata_dataframe.loc[batch_mask, self.inject_order] += offset
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

        # 1. Blank Samples Check
        if metabo_obj._blank.empty:
            logger.warning(
                "[Audit] No Blank samples detected. Pipeline will skip Stage-2 "
                "Blank/QC ratio filtering and degrade to QC RSD check only."
            )

        # 2. Biological Groups Check
        if not self.bio_group or self.bio_group not in metabo_obj.columns.names:
            logger.warning(
                "[Audit] No Biological Group information detected. Missing value "
                "imputation and filtering will fall back to QC-only rescue."
            )

        # 3. Internal Standards (IS) Check
        if len(getattr(metabo_obj, "valid_is", [])) == 0:
            logger.warning(
                "[Audit] No Internal Standards (IS) detected. Analytical outlier "
                "diagnostics will rely solely on global PCA statistics."
            )

        # 4. Outlier Reference Features (ORF) Check
        if len(getattr(metabo_obj, "valid_orf", [])) == 0:
            logger.info(
                "[Audit] No Outlier Reference Features (ORF) detected. This is "
                "normal for untargeted datasets; ORF diagnostics will be skipped."
            )

        # 5. High-throughput Cohort Check (Batch Count > 15)
        # Threshold set to 15 to match the visualizer's MathText threshold
        n_batches = len(metabo_obj.attrs.get("batch_list", []))
        if n_batches > 15:
            logger.info(
                f"[Audit] High batch count (n={n_batches}) detected. Visualizations "
                "will automatically switch to alphanumeric mode for readability."
            )

        # 6. Critical QC Density Check
        if metabo_obj._qc.empty:
            logger.error(
                "[Audit] FATAL: No QC samples detected! Subsequent correction "
                "and evaluation steps will inevitably fail."
            )
        else:
            qc_counts = metabo_obj._qc.columns.get_level_values(
                self.batch
            ).value_counts()

            # Warn if any batch has fewer than 3 QCs (Minimum required for SVR/RFSC)
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
        column_dataframe = self.intensity_dataframe.columns.to_frame().reset_index(
            drop=True
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
            else [self.batch, self.sample_type, self.inject_order, self.sample_name]
        )

        self.intensity_dataframe.columns = pd.MultiIndex.from_frame(
            column_dataframe.loc[:, column_order]
        )

        # Filter out samples lacking names
        valid_mask = self.intensity_dataframe.columns.get_level_values(
            level=self.sample_name
        ).notnull()
        self.intensity_dataframe = self.intensity_dataframe.loc[:, valid_mask]

        # =================================================================
        # Zero-value detection and conversion
        # =================================================================
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
            self.intensity_dataframe = self.intensity_dataframe.replace(0, np.nan)

        # Use float64 for stable downstream machine-learning assignments.
        self.intensity_dataframe = self.intensity_dataframe.astype(float)

        if output_dir:
            iu._check_dir_exists(dir_path=output_dir, handle="makedirs")
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

        # Generate Visualizations if output directory is specified
        if output_dir:
            vis = MetaboVisualizerBuilder(builder_obj=metabo_obj)
            try:
                summary_grid = vis.plot_builder_summary_grid()
                if summary_grid:
                    diag_path = os.path.join(
                        output_dir, "Global_Acquisition_Overview.svg"
                    )
                    vis.save_and_show_pw(pw_obj=summary_grid, file_path=diag_path)
                    logger.info(
                        f"Global acquisition overview plot saved as: {diag_path}"
                    )
            except Exception as e:
                logger.error(f"Dataset builder visualization failed: {e}")

        return metabo_obj


class MetaboVisualizerBuilder(visualizer_classes.BaseMetaboVisualizer):
    """
    Visualization suite for dataset construction (Entry-level Diagnostics).
    Generates aligned multi-tier barcodes and missing value charts using
    standardized formatting rules from BaseMetaboVisualizer.
    """

    def __init__(self, builder_obj: MetaboInt) -> None:
        """Initialize with a built MetaboInt dataset object."""
        super().__init__(metabo_obj=builder_obj)
        self.engine = builder_obj

    def _get_plot_metadata(self) -> dict[str, str]:
        """Helper to extract unified metadata for plotting functions."""
        pipeline_parameters = self.engine.attrs.get("pipeline_parameters", {})
        sample_dict = pipeline_parameters.get("MetaboInt", {}).get("sample_dict", {})

        return {
            "batch_column": self.engine.attrs.get("batch", "Batch"),
            "sample_type_column": self.engine.attrs.get("sample_type", "Sample Type"),
            "inject_order_column": self.engine.attrs.get(
                "inject_order", "Inject Order"
            ),
            "qc_label": sample_dict.get("QC sample", "QC"),
            "actual_label": sample_dict.get("Actual sample", "Sample"),
            "blank_label": sample_dict.get("Blank sample", "Blank"),
        }

    def _get_batch_color_map(self) -> dict[object, str]:
        """Helper to generate deterministic colors aligned with assessment.py."""

        plot_metadata = self._get_plot_metadata()
        batch_column = plot_metadata["batch_column"]

        if batch_column in self.engine.columns.names:
            unique_batches = sorted(
                self.engine.columns.get_level_values(batch_column).unique()
            )
            custom_cmap = pu.custom_linear_cmap(["white", "tab:red"], 100)
            palette = pu.extract_linear_cmap(
                cmap=custom_cmap, cmin=0.5, cmax=1.0, n_colors=len(unique_batches)
            )
            return dict(zip(unique_batches, palette))
        return {}

    def _get_batch_boundaries(self) -> list[float]:
        """Helper to compute X-axis boundaries between different batches."""
        plot_metadata = self._get_plot_metadata()
        inject_order_column = plot_metadata["inject_order_column"]
        batch_column = plot_metadata["batch_column"]

        column_dataframe = self.engine.columns.to_frame().reset_index(drop=True)
        plotting_dataframe = column_dataframe.sort_values(
            by=inject_order_column
        ).dropna(subset=[inject_order_column])

        boundaries = []
        batch_array = plotting_dataframe[batch_column].values
        inject_array = plotting_dataframe[inject_order_column].values

        for i in range(1, len(batch_array)):
            if batch_array[i] != batch_array[i - 1]:
                midpoint = (inject_array[i - 1] + inject_array[i]) / 2.0
                boundaries.append(midpoint)
        return boundaries

    def _apply_aligned_x_limits(
        self,
        current_axes: plt.Axes,
        plotting_dataframe: pd.DataFrame,
        inject_order_column: str,
    ) -> None:
        """Helper to apply tight, aligned x-axis limits across stacked plots."""
        min_order = plotting_dataframe[inject_order_column].min()
        max_order = plotting_dataframe[inject_order_column].max()
        padding = (max_order - min_order) * 0.01 if max_order > min_order else 1
        current_axes.set_xlim(min_order - padding, max_order + padding)

    def _get_barcode_df(self) -> pd.DataFrame:
        """Extracts and formats column metadata for barcode visualizations.

        Retrieves the metadata embedded in the MetaboInt MultiIndex columns
        using native plot metadata mappers, and ensures chronological sorting.

        Returns:
            pd.DataFrame: A flat dataframe containing sample metadata.
        """
        plot_metadata = self._get_plot_metadata()
        inject_order_col = plot_metadata["inject_order_column"]

        # Extract metadata from the engine's column MultiIndex
        df = self.engine.columns.to_frame().reset_index(drop=True)

        # Safely convert injection order to numeric for correct X-axis scaling
        if inject_order_col in df.columns:
            df[inject_order_col] = pd.to_numeric(df[inject_order_col], errors="coerce")

        return df

    def _plot_batch_tracking_barcode(
        self, ax: plt.Axes | None = None, clean_xaxis: bool = True
    ) -> plt.Figure | plt.Axes:
        """Plot a 1D event barcode showing batch membership across injection order.

        Args:
            ax (matplotlib.axes.Axes, optional): Target axes.
            clean_xaxis (bool): If True, strips the X-axis for seamless stacking.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 0.5))
        else:
            fig = ax.figure

        plot_metadata = self._get_plot_metadata()
        batch_col = plot_metadata["batch_column"]
        inject_order_col = plot_metadata["inject_order_column"]

        df_plot = self._get_barcode_df()

        # Utilize the native color mapper defined in the class
        batch_color_map = self._get_batch_color_map()
        unique_batches = sorted(df_plot[batch_col].dropna().unique())

        for b_val in unique_batches:
            batch_data = df_plot[df_plot[batch_col] == b_val]
            if not batch_data.empty:
                # Map to the specific batch color or fallback to gray
                color = batch_color_map.get(b_val, "tab:gray")
                ax.eventplot(
                    batch_data[inject_order_col],
                    orientation="horizontal",
                    linewidths=0.6,
                    colors=[color],
                    lineoffsets=1,
                    linelengths=1,
                )

        ax.set_ylabel(batch_col, rotation=0, ha="right", va="center")

        ax.set_yticks([])
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)

        if clean_xaxis:
            ax.set_xticks([])
            ax.spines["bottom"].set_visible(False)
            ax.set_xlabel("")
        else:
            ax.spines["bottom"].set_visible(True)

        return fig if ax is None else ax

    def _plot_run_tracking_barcode(
        self, ax: plt.Axes | None = None, clean_xaxis: bool = True
    ) -> plt.Figure | plt.Axes:
        """Plot a 1D event barcode showing sample type membership.

        Args:
            ax (matplotlib.axes.Axes, optional): Target axes.
            clean_xaxis (bool): If True, strips the X-axis components.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 0.5))
        else:
            fig = ax.figure

        plot_metadata = self._get_plot_metadata()
        sample_type_col = plot_metadata["sample_type_column"]
        inject_order_col = plot_metadata["inject_order_column"]
        qc_lbl = plot_metadata["qc_label"]
        act_lbl = plot_metadata["actual_label"]
        blk_lbl = plot_metadata["blank_label"]

        df_plot = self._get_barcode_df()

        qc_data = df_plot[df_plot[sample_type_col] == qc_lbl]
        sample_data = df_plot[df_plot[sample_type_col] == act_lbl]
        blank_data = df_plot[df_plot[sample_type_col] == blk_lbl]

        render_queue = [
            (sample_data, "tab:gray", 0.6),
            (blank_data, "tab:blue", 0.6),
            (qc_data, "tab:red", 0.8),
        ]

        for data, color, lw in render_queue:
            if not data.empty:
                ax.eventplot(
                    data[inject_order_col],
                    orientation="horizontal",
                    linewidths=lw,
                    colors=[color],
                    lineoffsets=1,
                    linelengths=1,
                )

        ax.set_ylabel(sample_type_col, rotation=0, ha="right", va="center")

        ax.set_yticks([])
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)

        if clean_xaxis:
            ax.set_xticks([])
            ax.spines["bottom"].set_visible(False)
            ax.set_xlabel("")
        else:
            ax.spines["bottom"].set_visible(True)
            self._apply_standard_format(
                ax=ax, xlabel=inject_order_col, append_stage=False
            )
            # Ensure the left spine is forcefully removed again in case
            # standard formatting routines re-enable it.
            ax.spines["left"].set_visible(False)

        return fig if ax is None else ax

    def _plot_missing_value_barplot(
        self, ax: plt.Axes | None = None
    ) -> plt.Figure | plt.Axes:
        """Private: Plot missing value rate (%) mapped to inject order."""
        if ax is None:
            fig, current_axes = plt.subplots(figsize=(10, 2.5))
        else:
            current_axes = ax
            fig = current_axes.figure

        plot_metadata = self._get_plot_metadata()
        inject_order_column = plot_metadata["inject_order_column"]
        sample_type_column = plot_metadata["sample_type_column"]

        missing_value_rates = self.engine.isna().mean(axis=0).values * 100
        column_dataframe = self.engine.columns.to_frame().reset_index(drop=True)
        column_dataframe["MV_Rate"] = missing_value_rates

        plotting_dataframe = column_dataframe.sort_values(
            by=inject_order_column
        ).dropna(subset=[inject_order_column])

        sample_color_map = {
            plot_metadata["qc_label"]: "tab:red",
            plot_metadata["actual_label"]: "tab:gray",
            plot_metadata["blank_label"]: "tab:blue",
        }

        current_axes.bar(
            plotting_dataframe[inject_order_column].values,
            plotting_dataframe["MV_Rate"].values,
            width=0.8,
            color=plotting_dataframe[sample_type_column]
            .map(sample_color_map)
            .fillna("black")
            .map(lambda color: pu.get_equivalent_hex(color, alpha=0.85))
            .values,
            edgecolor="black",
            linewidth=0.3,
            zorder=3,
        )

        for boundary in self._get_batch_boundaries():
            current_axes.axvline(
                x=boundary,
                color=pu.get_equivalent_hex("gray", alpha=0.7),
                linestyle="--",
                linewidth=0.8,
                zorder=0,
            )

        self._apply_standard_format(
            ax=current_axes,
            title="",
            xlabel=f"{inject_order_column} (Sequence)",
            ylabel="Missing Value Rate (%)",
            label_fontsize=10,
            append_stage=False,
        )

        # The barplot usually keeps the standard 90-degree rotation
        # for multi-line text to save horizontal space, but we still
        # push it left to match the barcodes' alignment.

        current_axes.grid(
            axis="y",
            linestyle="--",
            color=pu.get_equivalent_hex("gray", alpha=0.5),
            zorder=0,
        )
        current_axes.set_ylim(0, min(105, plotting_dataframe["MV_Rate"].max() * 1.1))

        for spine in ["top", "right"]:
            current_axes.spines[spine].set_visible(False)

        self._apply_aligned_x_limits(
            current_axes, plotting_dataframe, inject_order_column
        )
        return fig if ax is None else current_axes

    def _plot_standalone_legend(
        self, ax: plt.Axes | None = None
    ) -> plt.Figure | plt.Axes:
        """Create a unified, flat multi-group legend for both Batch and Sample Type.

        Extracts the active analytical batches and sample types present in the
        current dataset, maps them to their respective color palettes, and
        formats the batch legend with an adaptive internal column count while
        keeping a single group title for publication-friendly output.

        Args:
            ax (matplotlib.axes.Axes, optional): The target axis brick.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(1.0, 4.0))
        else:
            fig = ax.figure

        # Disable all axis lines and background ticks for a clean legend canvas
        ax.axis("off")

        # Retrieve configuration metadata and column mappers natively
        plot_metadata = self._get_plot_metadata()
        sample_type_col = plot_metadata["sample_type_column"]
        batch_col = plot_metadata["batch_column"]
        qc_lbl = plot_metadata["qc_label"]
        act_lbl = plot_metadata["actual_label"]
        blk_lbl = plot_metadata["blank_label"]

        df_plot = self._get_barcode_df()

        # =====================================================================
        # Group 1: Batch Legend Configuration (First Column Group)
        # =====================================================================
        batch_color_map = self._get_batch_color_map()
        unique_batches = sorted(df_plot[batch_col].dropna().unique())
        max_item_rows = 6
        legend_handles = [mpatches.Patch(color="none", label=batch_col)]
        legend_labels = [batch_col]

        for b_val in unique_batches:
            color = batch_color_map.get(b_val, "tab:gray")
            legend_handles.append(
                mpatches.Patch(
                    facecolor=color, edgecolor="k", linewidth=0.5, label=str(b_val)
                )
            )
            legend_labels.append(str(b_val))

        # =====================================================================
        # Group 2: Sample Type Legend Configuration (Second Column Group)
        # =====================================================================
        type_color_mapping = {
            act_lbl: "tab:gray",
            blk_lbl: "tab:blue",
            qc_lbl: "tab:red",
        }

        # Only display sample types that actually exist in the current dataframe
        present_types = [
            t
            for t in [act_lbl, blk_lbl, qc_lbl]
            if t in df_plot[sample_type_col].values
        ]

        legend_handles.append(mpatches.Patch(color="none", label=sample_type_col))
        legend_labels.append(sample_type_col)

        for t_val in present_types:
            color = type_color_mapping[t_val]
            legend_handles.append(
                mpatches.Patch(
                    facecolor=color, edgecolor="k", linewidth=0.5, label=t_val
                )
            )
            legend_labels.append(t_val)

        # =====================================================================
        # Render and format via the shared multi-legend layout engine.
        # =====================================================================
        ax.legend(legend_handles, legend_labels)
        self._format_multi_legends(
            ax=ax,
            group_titles=[batch_col, sample_type_col],
            loc="upper left",
            start_bbox=(0.0, 1.0),
            row_gap=0.06,
            layout_cols=1,
            column_gap=0.22,
            max_item_rows=max_item_rows,
            handlelength=1.0,
            handletextpad=0.45,
            columnspacing=0.85,
            borderaxespad=0.0,
        )

        return fig if ax is None else ax

    def _get_acquisition_legend_size(self) -> tuple[float, float]:
        """Return a legend brick size that can display all batch labels."""
        plot_metadata = self._get_plot_metadata()
        batch_col = plot_metadata["batch_column"]

        df_plot = self._get_barcode_df()
        n_batches = df_plot[batch_col].dropna().nunique()
        max_item_rows = 6
        n_batch_columns = max(1, int(np.ceil(n_batches / max_item_rows)))
        n_stacked_groups = 2

        width = 1.2 + max(0, n_batch_columns - 1) * 0.85
        visible_rows = min(n_batches, max_item_rows) + n_stacked_groups + 2
        height = max(1.5, min(3.5, 0.45 + visible_rows * 0.22))
        return width, height

    def plot_builder_summary_grid(self) -> object | None:
        """
        Dynamically assembles the summary dashboard using patchworklib.
        Automatically adapts to 100% complete datasets by rendering an ultra-clean
        barcode-only view, preserving the essential X-axis (Inject order).
        """
        try:
            import patchworklib as pw
        except ImportError:
            return None

        pw.clear()

        total_missing = self.engine.isna().sum().sum()
        has_missing = total_missing > 0

        # =====================================================================
        # Topology A: Full 3-plot layout (Missing values exist)
        # =====================================================================
        if has_missing:
            ax_left = pw.Brick(figsize=(9, 3.5))
            ax_left.axis("off")  # Eliminate parent brick bounding box

            ax_batch = ax_left.inset_axes([0, 0.88, 1, 0.10])
            ax_type = ax_left.inset_axes([0, 0.76, 1, 0.10], sharex=ax_batch)
            ax_bar = ax_left.inset_axes([0, 0.00, 1, 0.74], sharex=ax_batch)

            self._plot_batch_tracking_barcode(ax=ax_batch, clean_xaxis=True)
            self._plot_run_tracking_barcode(ax=ax_type, clean_xaxis=True)
            self._plot_missing_value_barplot(ax=ax_bar)

            # Dynamically calculate labelpad based on y-ticks string length
            y_ticks = ax_bar.get_yticks()
            max_tick_val = max(y_ticks) if len(y_ticks) > 0 else 100.0

            # Format to 1 decimal place to mimic visualizer output strings
            max_char_len = len(f"{max_tick_val:.1f}")

            # Base pad of 5, plus roughly 5 points per character width
            dynamic_pad = 5 + (max_char_len * 5)

            # Apply dynamic pad universally to ensure vertical alignment
            ax_batch.yaxis.labelpad = dynamic_pad
            ax_type.yaxis.labelpad = dynamic_pad
            ax_bar.yaxis.labelpad = dynamic_pad

            legend_width, _ = self._get_acquisition_legend_size()
            ax_right = pw.Brick(figsize=(legend_width, 3.5))
            self._plot_standalone_legend(ax=ax_right)

            return ax_left | ax_right

        # =====================================================================
        # Topology B: Ultra-clean Barcode layout (100% Complete Data)
        # =====================================================================
        else:
            logger.info("No missing values detected. Adapting to Barcode-only layout.")

            ax_left = pw.Brick(figsize=(9, 1.5))
            ax_left.axis("off")  # Eliminate parent brick bounding box

            ax_batch = ax_left.inset_axes([0, 0.55, 1, 0.45])
            ax_type = ax_left.inset_axes([0, 0.10, 1, 0.45], sharex=ax_batch)

            self._plot_batch_tracking_barcode(ax=ax_batch, clean_xaxis=True)
            self._plot_run_tracking_barcode(ax=ax_type, clean_xaxis=False)

            ax_batch.yaxis.labelpad = 15
            ax_type.yaxis.labelpad = 15

            legend_width, legend_height = self._get_acquisition_legend_size()
            ax_right = pw.Brick(figsize=(legend_width, legend_height))
            self._plot_standalone_legend(ax=ax_right)

            return ax_left | ax_right


@iu._exe_time
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

    return builder.execute_build(output_dir=output_dir)
