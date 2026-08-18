"""Acquisition and data-quality plots for dataset construction.

DatasetPlotter renders the initial project overview, including sample
composition, acquisition order, batch membership, intensity completeness, and
missing-value distribution. The resulting figures document the inputs and
validation state before filtering or preprocessing begins.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from loguru import logger

from ..core.model import MetaboInt
from . import plot_utils as pu
from .base import BasePlotter
from .colors import build_categorical_palette


class DatasetPlotter(BasePlotter):
    """Plotting suite for dataset construction diagnostics.

    Generates aligned multi-tier barcodes and missing value charts using
    standardized formatting rules from :class:`BasePlotter`.
    """

    def __init__(self, builder_obj: MetaboInt) -> None:
        """Initialize with a built MetaboInt dataset object."""
        super().__init__(metabo_obj=builder_obj)
        self.engine = builder_obj

    def _get_plot_metadata(self) -> dict[str, str]:
        """Helper to extract unified metadata for plotting functions."""
        sample_dict = self.engine.attrs.get("sample_dict", {})

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
            "blank_label": sample_dict.get("Blank sample", "Blank"),
        }

    def _get_batch_color_map(self) -> dict[object, str]:
        """
        Helper to generate deterministic colors aligned with assessment.py.
        """

        plot_metadata = self._get_plot_metadata()
        batch_column = plot_metadata["batch_column"]

        if batch_column in self.engine.columns.names:
            unique_batches = self.engine.columns.get_level_values(
                batch_column
            ).unique()
            return build_categorical_palette(unique_batches)
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
            df[inject_order_col] = pd.to_numeric(
                df[inject_order_col], errors="coerce"
            )

        return df

    def _plot_batch_tracking_barcode(
        self, ax: plt.Axes | None = None, clean_xaxis: bool = True
    ) -> plt.Figure | plt.Axes:
        """
        Plot a 1D event barcode showing batch membership across injection order.

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

        ax.set_ylabel(
            batch_col,
            rotation=0,
            ha="right",
            va="center",
            fontsize=pu.DEFAULT_AXIS_LABEL_FONTSIZE,
            fontweight=pu.DEFAULT_AXIS_LABEL_WEIGHT,
        )

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

        ax.set_ylabel(
            sample_type_col,
            rotation=0,
            ha="right",
            va="center",
            fontsize=pu.DEFAULT_AXIS_LABEL_FONTSIZE,
            fontweight=pu.DEFAULT_AXIS_LABEL_WEIGHT,
        )

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
            label_fontsize=pu.DEFAULT_AXIS_LABEL_FONTSIZE,
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
        current_axes.set_ylim(
            0, min(105, plotting_dataframe["MV_Rate"].max() * 1.1)
        )

        for spine in ["top", "right"]:
            current_axes.spines[spine].set_visible(False)

        self._apply_aligned_x_limits(
            current_axes, plotting_dataframe, inject_order_column
        )
        return fig if ax is None else current_axes

    def _plot_standalone_legend(
        self, ax: plt.Axes | None = None
    ) -> plt.Figure | plt.Axes:
        """
        Create a unified, flat multi-group legend for both Batch and Sample
        Type.

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

        # Batch Legend Configuration
        batch_color_map = self._get_batch_color_map()
        unique_batches = sorted(df_plot[batch_col].dropna().unique())
        max_item_rows = 6
        legend_handles = [mpatches.Patch(color="none", label=batch_col)]
        legend_labels = [batch_col]

        for b_val in unique_batches:
            color = batch_color_map.get(b_val, "tab:gray")
            legend_handles.append(
                mpatches.Patch(
                    facecolor=color,
                    edgecolor="k",
                    linewidth=0.5,
                    label=str(b_val),
                )
            )
            legend_labels.append(str(b_val))

        # Sample-Type Legend Configuration
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

        legend_handles.append(
            mpatches.Patch(color="none", label=sample_type_col)
        )
        legend_labels.append(sample_type_col)

        for t_val in present_types:
            color = type_color_mapping[t_val]
            legend_handles.append(
                mpatches.Patch(
                    facecolor=color, edgecolor="k", linewidth=0.5, label=t_val
                )
            )
            legend_labels.append(t_val)

        # Render and format via the shared multi-legend layout engine.
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

    def plot_dataset_dashboard(self) -> object | None:
        """Assemble the dataset overview dashboard.

        Complete datasets use a compact acquisition-barcode layout. Datasets
        containing missing values additionally include the missingness panel.
        """
        try:
            import patchworklib as pw
        except ImportError:
            return None

        pw.clear()

        total_missing = self.engine.isna().sum().sum()
        has_missing = total_missing > 0

        # Include acquisition tracking and missingness when values are absent.
        if has_missing:
            legend_width, _ = self._get_acquisition_legend_size()
            layout_width = 9.0 + legend_width
            ax_left = pw.Brick(
                figsize=pu.dashboard_brick_size(9.0, 3.5, layout_width)
            )
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

            # Format to one decimal place before estimating label width.
            max_char_len = len(f"{max_tick_val:.1f}")

            # Base pad of 5, plus roughly 5 points per character width
            dynamic_pad = 5 + (max_char_len * 5)

            # Apply dynamic pad universally to ensure vertical alignment
            ax_batch.yaxis.labelpad = dynamic_pad
            ax_type.yaxis.labelpad = dynamic_pad
            ax_bar.yaxis.labelpad = dynamic_pad

            ax_right = pw.Brick(
                figsize=pu.dashboard_brick_size(legend_width, 3.5, layout_width)
            )
            self._plot_standalone_legend(ax=ax_right)

            return ax_left | ax_right

        # Barcode-Only Layout for Complete Data
        else:
            logger.info(
                "No missing values detected. Adapting to Barcode-only layout."
            )

            legend_width, legend_height = self._get_acquisition_legend_size()
            layout_width = 9.0 + legend_width
            ax_left = pw.Brick(
                figsize=pu.dashboard_brick_size(9.0, 1.5, layout_width)
            )
            ax_left.axis("off")  # Eliminate parent brick bounding box

            ax_batch = ax_left.inset_axes([0, 0.55, 1, 0.45])
            ax_type = ax_left.inset_axes([0, 0.10, 1, 0.45], sharex=ax_batch)

            self._plot_batch_tracking_barcode(ax=ax_batch, clean_xaxis=True)
            self._plot_run_tracking_barcode(ax=ax_type, clean_xaxis=False)

            ax_batch.yaxis.labelpad = 15
            ax_type.yaxis.labelpad = 15

            ax_right = pw.Brick(
                figsize=pu.dashboard_brick_size(
                    legend_width, legend_height, layout_width
                )
            )
            self._plot_standalone_legend(ax=ax_right)

            return ax_left | ax_right
