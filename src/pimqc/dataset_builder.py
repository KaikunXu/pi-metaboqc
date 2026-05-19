# src/pimqc/dataset_builder.py
"""
Purpose of script: 
    Build a standardized MetaboInt object from metadata and intensity matrices.
    Provides strict object-oriented validation, batch continuity management,
    and entry-level diagnostic visualizations. Driven strictly by TOML config.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
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
        pipeline_params: Optional[Dict[str, Any]] = None
    ):
        """Initialize builder using global pipeline configuration."""
        self.metadata_dataframe = meta_info.copy()
        self.intensity_dataframe = int_df.copy()
        self.params = pipeline_params or {}
        
        # Extract configuration attributes smartly
        metabolomics_parameters = self.params.get("MetaboInt", {})
        
        self.mode = metabolomics_parameters.get("mode", "POS")
        self.batch = metabolomics_parameters.get("batch", "Batch")
        self.sample_type = metabolomics_parameters.get(
            "sample_type", "Sample Type"
        )
        self.bio_group = metabolomics_parameters.get("bio_group", "Bio Group")
        self.sample_name = metabolomics_parameters.get(
            "sample_name", "Sample Name"
        )
        self.inject_order = metabolomics_parameters.get(
            "inject_order", "Inject Order"
        )
        self.resort_strategy = metabolomics_parameters.get(
            "resort_inject_order", "Auto"
        )
        
        self.unique_batches = []
        self.is_multi_batch = False

    def _resolve_duplicate_features(self):
        """Sum intensities for duplicated metabolite names to ensure unique."""
        if isinstance(self.intensity_dataframe.index, pd.RangeIndex):
            first_column = self.intensity_dataframe.columns[0]
            logger.warning(
                f"RangeIndex detected. Setting '{first_column}' "
                "as feature index."
            )
            self.intensity_dataframe = self.intensity_dataframe.set_index(
                first_column
            )
        
        self.intensity_dataframe.index.name = "Metabolite"

        if self.intensity_dataframe.index.duplicated().any():
            num_duplicates = self.intensity_dataframe.index.duplicated().sum()
            duplicated_features = self.intensity_dataframe.index[
                self.intensity_dataframe.index.duplicated()
            ].unique().tolist()
            
            logged_features = (
                duplicated_features[:5] + ["..."] 
                if len(duplicated_features) > 5 else duplicated_features
            )
            logger.warning(
                f"Detected {num_duplicates} duplicate row indices "
                f"(e.g., {logged_features}). Merging their intensities "
                "by summation."
            )
            self.intensity_dataframe = self.intensity_dataframe.groupby(
                level=0, sort=False
            ).sum()

    def _check_duplicate_samples(self):
        """Abort if intensity dataframe contains duplicated sample columns."""
        value_counts = self.intensity_dataframe.columns.value_counts()
        if value_counts.max() > 1:
            duplicated_names = value_counts[value_counts > 1].index.tolist()
            duplicated_details = []
            
            for name in duplicated_names:
                locations = [
                    i for i, col in enumerate(self.intensity_dataframe.columns) 
                    if col == name
                ]
                duplicated_details.append(f'"{name}" (indices: {locations})')
                
            error_message = (
                "Duplicate sample names in intensity data: "
                f'{", ".join(duplicated_details)}.'
            )
            raise AssertionError(error_message)

    def _verify_sample_consistency(self):
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
                    if len(only_in_meta) > 5 else only_in_meta
                )
                message.append(f"Only in Metadata: {log_meta}")
                
            if only_in_intensity:
                log_intensity = (
                    only_in_intensity[:5] + ["..."] 
                    if len(only_in_intensity) > 5 else only_in_intensity
                )
                message.append(f"Only in Intensity: {log_intensity}")
                
            raise AssertionError(" ".join(message))

    def _verify_metadata_completeness(self):
        """Abort if critical tracking columns are missing from metadata."""
        required_cols = {
            "Batch": self.batch,
            "Sample Type": self.sample_type,
            "Sample Name": self.sample_name,
            "Inject Order": self.inject_order
        }
        
        missing_cols = [
            col_name for label, col_name in required_cols.items() 
            if col_name not in self.metadata_dataframe.columns
        ]
        
        if missing_cols:
            raise AssertionError(
                f"Incomplete metadata. Missing columns: {missing_cols}."
            )
            
        self.unique_batches = self.metadata_dataframe[self.batch].unique()
        self.is_multi_batch = len(self.unique_batches) > 1

    def _manage_injection_orders(self):
        """Align injection sequences across batches to prevent overlap."""
        if not self.is_multi_batch or \
           self.inject_order not in self.metadata_dataframe:
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
                    self.metadata_dataframe.loc[
                        batch_mask, self.inject_order
                    ] += offset
                previous_max = self.metadata_dataframe.loc[
                    batch_mask, self.inject_order
                ].max()

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
        column_dataframe = self.intensity_dataframe.columns.to_frame(
        ).reset_index(drop=True)
        
        column_dataframe = pd.merge(
            left=column_dataframe, right=self.metadata_dataframe, 
            on=self.sample_name, how="left"
        )
        
        has_bio_group = pd.notna(self.bio_group) and (
            self.bio_group in self.metadata_dataframe.columns
        )
        column_order = (
            [
                self.batch, self.sample_type, self.bio_group, 
                self.inject_order, self.sample_name
            ] if has_bio_group else [
                self.batch, self.sample_type, self.inject_order, 
                self.sample_name
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
        
        if output_dir:
            iu._check_dir_exists(dir_path=output_dir, handle="makedirs")
            output_path = os.path.join(
                output_dir, "Raw_Data_Intensity.csv"
            )
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
            inject_order=self.inject_order)
        
        metabo_obj.attrs["is_multi_batch"] = self.is_multi_batch
        metabo_obj.attrs["batch_list"] = self.unique_batches.tolist()

        logger.info(
            f"MetaboInt object built: {metabo_obj.shape[0]} metabolites, "
            f"{metabo_obj.shape[1]} samples.")

        # Generate Visualizations if output directory is specified
        if output_dir:
            vis = MetaboVisualizerBuilder(builder_obj=metabo_obj)
            try:
                summary_grid = vis.plot_builder_summary_grid()
                if summary_grid:
                    diag_path = os.path.join(
                        output_dir, "Global_Acquisition_Overview.svg")
                    vis.save_and_show_pw(
                        pw_obj=summary_grid, file_path=diag_path)
                    logger.info(
                        f"Global acquisition overview plot saved as: {diag_path}")
            except Exception as e:
                logger.error(f"Dataset builder visualization failed: {e}")

        return metabo_obj


class MetaboVisualizerBuilder(visualizer_classes.BaseMetaboVisualizer):
    """
    Visualization suite for dataset construction (Entry-level Diagnostics).
    Generates aligned multi-tier barcodes and missing value charts using 
    standardized formatting rules from BaseMetaboVisualizer.
    """

    def __init__(self, builder_obj):
        """Initialize with a built MetaboInt dataset object."""
        super().__init__(metabo_obj=builder_obj)
        self.engine = builder_obj

    def _get_plot_metadata(self):
        """Helper to extract unified metadata for plotting functions."""
        pipeline_parameters = self.engine.attrs.get("pipeline_parameters", {})
        sample_dict = pipeline_parameters.get("MetaboInt", {}).get(
            "sample_dict", {})
        
        return {
            "batch_column": self.engine.attrs.get("batch", "Batch"),
            "sample_type_column": self.engine.attrs.get(
                "sample_type", "Sample Type"
            ),
            "inject_order_column": self.engine.attrs.get(
                "inject_order", "Inject Order"
            ),
            "qc_label": sample_dict.get("QC sample", "QC"),
            "actual_label": sample_dict.get("Actual sample", "Sample"),
            "blank_label": sample_dict.get("Blank sample", "Blank")
        }

    def _get_batch_color_map(self):
        """Helper to generate deterministic colors aligned with assessment.py."""
        from . import plot_utils as pu
        
        plot_metadata = self._get_plot_metadata()
        batch_column = plot_metadata["batch_column"]
        
        if batch_column in self.engine.columns.names:
            unique_batches = sorted(
                self.engine.columns.get_level_values(batch_column).unique()
            )
            custom_cmap = pu.custom_linear_cmap(["white", "tab:red"], 100)
            palette = pu.extract_linear_cmap(
                cmap=custom_cmap, cmin=0.5, cmax=1.0, 
                n_colors=len(unique_batches)
            )
            return dict(zip(unique_batches, palette))
        return {}

    def _get_batch_boundaries(self):
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
            if batch_array[i] != batch_array[i-1]:
                midpoint = (inject_array[i-1] + inject_array[i]) / 2.0
                boundaries.append(midpoint)
        return boundaries

    def _apply_aligned_x_limits(
        self, current_axes, plotting_dataframe, inject_order_column
    ):
        """Helper to apply tight, aligned x-axis limits across stacked plots."""
        min_order = plotting_dataframe[inject_order_column].min()
        max_order = plotting_dataframe[inject_order_column].max()
        padding = (max_order - min_order) * 0.01 if max_order > min_order else 1
        current_axes.set_xlim(min_order - padding, max_order + padding)

    def _plot_batch_tracking_barcode(self, ax=None):
        """Private: Plot sequence barcode indicating analytical batches."""
        if ax is None:
            fig, current_axes = plt.subplots(figsize=(10, 0.4))
        else:
            current_axes = ax
            fig = current_axes.figure

        plot_metadata = self._get_plot_metadata()
        batch_column = plot_metadata["batch_column"]
        inject_order_column = plot_metadata["inject_order_column"]

        column_dataframe = self.engine.columns.to_frame().reset_index(drop=True)
        plotting_dataframe = column_dataframe.sort_values(
            by=inject_order_column
        ).dropna(subset=[inject_order_column])

        batch_color_map = self._get_batch_color_map()
        for batch_id, color in batch_color_map.items():
            subset = plotting_dataframe[
                plotting_dataframe[batch_column] == batch_id
            ]
            if not subset.empty:
                current_axes.vlines(
                    x=subset[inject_order_column].values, ymin=0, ymax=1, 
                    colors=[color], linewidth=1.5, alpha=0.9
                )

        for boundary in self._get_batch_boundaries():
            current_axes.axvline(
                x=boundary, color="gray", linestyle="--", 
                linewidth=0.8, alpha=0.7, zorder=0
            )

        # Apply standard formatting but explicitly override the ylabel parameters
        self._apply_standard_format(
            ax=current_axes, title="Global Acquisition Overview", 
            xlabel="", ylabel="Batch", title_fontsize=14, 
            label_fontsize=10, append_stage=False
        )
        
        # Override rotation and alignment directly on the axis label
        current_axes.yaxis.label.set_rotation(0)
        current_axes.yaxis.label.set_ha("right")
        current_axes.yaxis.label.set_va("center")
        
        current_axes.set_yticks([])
        current_axes.set_xticks([])
        # Push the label further left (-0.06 instead of -0.02)
        current_axes.get_yaxis().set_label_coords(-0.01, 0.5)
        
        for spine in ['top', 'right', 'left', 'bottom']:
            current_axes.spines[spine].set_visible(False)

        self._apply_aligned_x_limits(
            current_axes, plotting_dataframe, inject_order_column
        )
        return fig if ax is None else current_axes

    def _plot_run_tracking_barcode(self, ax=None):
        """Private: Plot sequence barcode tracking Sample/QC/Blank types."""
        if ax is None:
            fig, current_axes = plt.subplots(figsize=(10, 0.4))
        else:
            current_axes = ax
            fig = current_axes.figure

        plot_metadata = self._get_plot_metadata()
        inject_order_column = plot_metadata["inject_order_column"]
        sample_type_column = plot_metadata["sample_type_column"]

        column_dataframe = self.engine.columns.to_frame().reset_index(drop=True)
        plotting_dataframe = column_dataframe.sort_values(
            by=inject_order_column
        ).dropna(subset=[inject_order_column])

        sample_color_map = {
            plot_metadata["qc_label"]: "tab:red",
            plot_metadata["actual_label"]: "tab:gray",
            plot_metadata["blank_label"]: "tab:blue"
        }

        for sample_type in [
            plot_metadata["blank_label"], plot_metadata["actual_label"], 
            plot_metadata["qc_label"]
        ]:
            subset = plotting_dataframe[
                plotting_dataframe[sample_type_column] == sample_type
            ]
            if not subset.empty:
                current_axes.vlines(
                    x=subset[inject_order_column].values, ymin=0, ymax=1, 
                    colors=sample_color_map.get(sample_type, "black"), 
                    linewidth=1.5, alpha=0.9
                )

        for boundary in self._get_batch_boundaries():
            current_axes.axvline(
                x=boundary, color="gray", linestyle="--", 
                linewidth=0.8, alpha=0.7, zorder=0
            )

        self._apply_standard_format(
            ax=current_axes, title="", xlabel="", ylabel="Sample Type", 
            label_fontsize=10, append_stage=False
        )
        
        # Override rotation and alignment directly on the axis label
        current_axes.yaxis.label.set_rotation(0)
        current_axes.yaxis.label.set_ha("right")
        current_axes.yaxis.label.set_va("center")
        
        current_axes.set_yticks([])
        current_axes.set_xticks([])
        # Push the label further left (-0.06 instead of -0.02)
        current_axes.get_yaxis().set_label_coords(-0.01, 0.5)
        
        for spine in ['top', 'right', 'left', 'bottom']:
            current_axes.spines[spine].set_visible(False)

        self._apply_aligned_x_limits(
            current_axes, plotting_dataframe, inject_order_column
        )
        return fig if ax is None else current_axes

    def _plot_missing_value_barplot(self, ax=None):
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
            plot_metadata["blank_label"]: "tab:blue"
        }

        current_axes.bar(
            plotting_dataframe[inject_order_column].values, 
            plotting_dataframe["MV_Rate"].values, 
            width=0.8, color=plotting_dataframe[sample_type_column].map(
                sample_color_map).fillna("black").values, 
            edgecolor="black", linewidth=0.3, alpha=0.85, zorder=3
        )

        for boundary in self._get_batch_boundaries():
            current_axes.axvline(
                x=boundary, color="gray", linestyle="--", 
                linewidth=0.8, alpha=0.7, zorder=0
            )

        self._apply_standard_format(
            ax=current_axes, title="", 
            xlabel=f"{inject_order_column} (Sequence)", 
            ylabel="Missing Value Rate (%)", label_fontsize=10, 
            append_stage=False
        )
        
        # The barplot usually keeps the standard 90-degree rotation 
        # for multi-line text to save horizontal space, but we still 
        # push it left to match the barcodes' alignment.
        
        current_axes.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        current_axes.set_ylim(
            0, min(105, plotting_dataframe["MV_Rate"].max()*1.1))
        
        # Align with the barcodes above (-0.06)
        current_axes.get_yaxis().set_label_coords(-0.04, 0.5)
        
        for spine in ['top', 'right']:
            current_axes.spines[spine].set_visible(False)

        self._apply_aligned_x_limits(
            current_axes, plotting_dataframe, inject_order_column
        )
        return fig if ax is None else current_axes

    def _plot_standalone_legend(self, ax=None):
        """Create a blank axes for the formatted legend at absolute top-left."""
        import matplotlib.lines as mlines
        if ax is None:
            fig, current_axes = plt.subplots(figsize=(2, 4.0))
        else:
            current_axes = ax
            fig = current_axes.figure

        plot_metadata = self._get_plot_metadata()
        sample_color_map = {
            plot_metadata["qc_label"]: "tab:red",
            plot_metadata["actual_label"]: "tab:gray",
            plot_metadata["blank_label"]: "tab:blue"
        }
        batch_color_map = self._get_batch_color_map()
        existing_types = self.engine.columns.get_level_values(
            plot_metadata["sample_type_column"]).unique()

        legend_handles, legend_labels = [], []

        if batch_color_map:
            legend_handles.append(
                mlines.Line2D([], [], color="none", label="Batch"))
            legend_labels.append("Batch")
            for batch_id, color in batch_color_map.items():
                legend_handles.append(mpatches.Patch(
                    facecolor=color, edgecolor="black", label=str(batch_id)))
                legend_labels.append(str(batch_id))

        legend_handles.append(
            mlines.Line2D([], [], color="none", label="Sample Type"))
        legend_labels.append("Sample Type")
        for sample_type, color in sample_color_map.items():
            if sample_type in existing_types:
                legend_handles.append(mpatches.Patch(
                    facecolor=color, edgecolor="black", label=sample_type))
                legend_labels.append(sample_type)

        current_axes.legend(legend_handles, legend_labels)
        group_titles = ["Batch ID", "Sample Type"] if (
            batch_color_map) else ["Sample Type"]
        self._format_multi_legends(
            ax=current_axes, group_titles=group_titles,
            loc="upper left", start_bbox=(0, 1)
        )
        current_axes.axis("off")
        return fig if ax is None else current_axes

    def plot_builder_summary_grid(self):
        """Combine aligned Barcodes and Barplot into a diagnostic panel."""
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping builder grid.")
            return None

        pw.clear()
        
        # 1. Create a unified left-side "canvas" Brick container
        ax_left = pw.Brick(figsize=(10, 3.5))
        
        # Hide the parent Brick's borders, ticks, and background 
        # as it acts only as a spatial wrapper
        ax_left.axis('off')
        
        # 2. Utilize Matplotlib's native inset_axes for absolute 
        # physical alignment within the canvas wrapper.
        # inset_axes parameters: [x0_ratio, y0_ratio, width_ratio, height_ratio]
        # Width ratio is set to 1 to enforce 100% identical horizontal alignment,
        # completely eliminating coordinate mismatch across subplots.
        ax_batch = ax_left.inset_axes([0, 0.88, 1, 0.10])
        ax_type  = ax_left.inset_axes([0, 0.76, 1, 0.10])
        ax_bar   = ax_left.inset_axes([0, 0.00, 1, 0.74], sharex=ax_type)
        
        # Note: The bottom of ax_batch is at 0.88, and the top of ax_type 
        # is at 0.76 + 0.12 = 0.88. This ensures exactly 0 physical spacing!
        
        # 3. Render the plots precisely into these native axes
        self._plot_batch_tracking_barcode(ax=ax_batch)
        self._plot_run_tracking_barcode(ax=ax_type)
        self._plot_missing_value_barplot(ax=ax_bar)

        # 4. Create a dedicated right-side Brick for the standalone legend
        ax_right = pw.Brick(figsize=(0.8, 3.5))
        self._plot_standalone_legend(ax=ax_right)

        # 5. Patchworklib assembly: Seamless left container | Right legend
        return ax_left | ax_right
    
    
@iu._exe_time
def build_dataset(
    meta_info: pd.DataFrame,
    int_df: pd.DataFrame,
    pipeline_params: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None
) -> MetaboInt:
    """
    Factory wrapper for MetaboIntBuilder, driven strictly by configuration.
    Accepts raw dataframes and config, returning a MetaboInt object.
    """
    builder = MetaboIntBuilder(
        meta_info=meta_info,
        int_df=int_df,
        pipeline_params=pipeline_params
    )
    
    return builder.execute_build(output_dir=output_dir)