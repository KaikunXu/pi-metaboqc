"""Diagnostic visualizations for stage-wise quality assessment.

MetaboVisualizerAssessor renders correlation heatmaps, RSD summaries, PCA
scores, robust-distance outlier views, control charts, and report-ready
composite dashboards. It consumes assessment calculations without changing
stage data and supports both standalone exports and report-assembly panels.
"""

import re
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.legend import Legend
import seaborn as sns

from typing import Any, Optional

from ...visualization import plot_utils as pu
from ...core import model
from ...visualization import base as visualizer_classes

warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=RuntimeWarning)


from .analysis import MetaboIntAssessor


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
        """
        Draw each visible heatmap cell edge once for uniform vector output.
        """
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
        show_colorbar: bool = True,
        title_mode: str = "full",
    ) -> plt.Figure:
        """
        Plot sample-level correlation matrix with rigorous clustering forests.

        Features:
        1. Cluster modes: 'total' (global), 'within-group' (batch-isolated
            forests), or 'none'.
        2. Absolute mathematical alignment using GridSpec (bypassing
        bounding-box drift).
        3. Standardized formatting via _apply_standard_format with dynamic tick
        scaling.
        4. Complete eradication of auto-generated axis labels to preserve grid
        cleanliness.

        Args:
        corr_matrix (pd.DataFrame): Correlation matrix of QC samples.
        corr_mask (Optional[np.ndarray]): (Ignored, hardcoded to lower-triangle
            geometrically).
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
                    b_mask = (
                        corr_matrix.index.get_level_values(self.bat_col) == b
                    )
                else:
                    b_mask = corr_matrix.index.str.startswith(f"{b}")

                idx_b = np.where(b_mask)[0]
                n_b = len(idx_b)

                if n_b > 1:
                    sub_corr = corr_matrix.iloc[idx_b, idx_b].values.astype(
                        float
                    )
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
        custom_cmap = pu.custom_linear_cmap(
            ["white", pu.PRIMARY_ACCENT_COLOR], 100
        )
        color_map = mcolors.ListedColormap(
            pu.extract_linear_cmap(custom_cmap, cmin=0.2, cmax=1.0)
        )
        color_map.set_bad(color="white", alpha=1.0)

        tick_colors = pu.extract_linear_cmap(
            cmap=custom_cmap, cmin=0.5, cmax=1.0, n_colors=len(batches)
        )
        tick_color_dict = dict(zip(batches, tick_colors))

        if is_multi_idx:
            ordered_batches = corr_matrix.index.get_level_values(
                self.bat_col
            ).values
        else:
            ordered_batches = [str(x).split("-")[0] for x in corr_matrix.index]

        # =====================================================================
        # 3. Canvas & Layout Integration (Matplotlib GridSpec)
        # =====================================================================
        hm_size = max(5.0, n_samples * 0.2 + 1.0)

        annot_fmt = ".3f" if n_samples <= 15 else ".2f"
        show_annot = n_samples <= 15
        cell_edge_lw = pu.DEFAULT_HEATMAP_CELL_LINEWIDTH

        if ax is None:
            compact_hm_size = pu.dashboard_brick_size(4.8, 4.0, 14.0)
            fig = plt.figure(
                figsize=(
                    max(compact_hm_size[0], n_samples * 0.12 + 1.0),
                    max(compact_hm_size[1], n_samples * 0.12 + 1.0),
                ),
                constrained_layout=True,
            )
            if cluster_mode in ["total", "within-group"]:
                gs = fig.add_gridspec(
                    2, 2, width_ratios=[1, 6], height_ratios=[6, 1]
                )
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

        annot_size = pu.heatmap_annotation_fontsize(
            ax=ax_heatmap,
            n_rows=n_samples,
            n_cols=n_samples,
            default_size=pu.DEFAULT_ANNOTATION_FONTSIZE,
            max_size=pu.DEFAULT_ANNOTATION_FONTSIZE,
            min_size=2.5,
        )

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
            [
                pu.compact_tick_label(label, max_tick_chars)
                for label in raw_x_labels
            ]
            if needs_dense_ticks
            else raw_x_labels
        )
        y_tick_labels = (
            [
                pu.compact_tick_label(label, max_tick_chars)
                for label in raw_y_labels
            ]
            if needs_dense_ticks
            else raw_y_labels
        )
        max_tick_len = max(
            [len(label) for label in x_tick_labels + y_tick_labels] or [1]
        )
        x_rot = (
            90
            if needs_dense_ticks and (n_samples > 12 or max_tick_len > 14)
            else 45
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
                        coll.set_linewidth(pu.DEFAULT_GUIDE_LINEWIDTH)
                        coll.set_color("#334155")

                    for line in target_ax.lines[n_lines:]:
                        if orientation == "bottom":
                            line.set_xdata(line.get_xdata() + shift)
                        else:
                            line.set_ydata(line.get_ydata() + shift)
                        line.set_linewidth(pu.DEFAULT_GUIDE_LINEWIDTH)
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
            _draw_shifted_dendrograms(
                Z_list, n_list, ax_dendro_bottom, "bottom"
            )

        # =====================================================================
        # 4. Main Lower-Triangle Heatmap
        # =====================================================================
        geom_mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        cbar_ax = (
            ax_heatmap.inset_axes([0.60, 0.88, 0.35, 0.035])
            if show_colorbar
            else None
        )

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
                cbar=show_colorbar,
                cbar_ax=cbar_ax,
                annot_kws={"size": annot_size},
                cbar_kws={
                    "label": f"{method.title()} Corr",
                    "orientation": "horizontal",
                },
            )
            if cbar_ax is not None:
                cbar_ax.xaxis.set_ticks_position("top")
                cbar_ax.xaxis.set_label_position("top")
                for spine in cbar_ax.spines.values():
                    spine.set_visible(False)
                self._format_colorbar_axes(cbar_ax)

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
        # This prevents 'inject_order' from being squeezed between ticks and
        # dendrograms.
        ax_heatmap.set_xlabel("")
        ax_heatmap.set_ylabel("")

        # =====================================================================
        # 5. Standardized Formatting & Inset Legend Injection
        # =====================================================================
        # Route to standardized formatting pipeline seamlessly, injecting the
        # optimized tick_size
        self._apply_standard_format(
            ax=ax_heatmap,
            title=(
                "Pooled QCs Correlation\n"
                f"[{self.attrs.get('pipeline_stage', '')}]"
                if self._validate_title_mode(title_mode) == "full"
                else self._panel_title("Pooled QCs Correlation", title_mode)[0]
            ),
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
        pu.apply_batch_tick_colors(
            ax_heatmap.get_xticklabels(), tick_color_dict
        )
        pu.apply_batch_tick_colors(
            ax_heatmap.get_yticklabels(), tick_color_dict
        )

        legend_handles = [
            mpatches.Patch(
                facecolor=c, edgecolor="k", linewidth=0.5, label=str(b)
            )
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
            ax_heatmap.get_legend().get_title().set_fontsize(
                pu.DEFAULT_LEGEND_TITLE_FONTSIZE
            )
            ax_heatmap.get_legend().set_bbox_to_anchor((0.95, 0.82))

        return fig

    def plot_batch_corr_heatmap(
        self,
        batch_corr_matrix: pd.DataFrame,
        method: str,
        vmin: float = 0.85,
        vmax: float = 1.0,
        ax: plt.Axes | None = None,
        show_colorbar: bool = True,
        title_mode: str = "full",
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
            base_w, base_h = pu.dashboard_brick_size(4.8, 4.0, 14.0)
            fig_w = max(base_w, n_batches * 0.28 + 0.8)
            fig_h = max(base_h, n_batches * 0.28 + 0.8)
            fig, current_ax = plt.subplots(figsize=(fig_w, fig_h))

            # Text easily fits even for large cohorts in standalone mode
            show_annot = n_batches <= 20
        else:
            # Grid Mode (Patchwork): Size is strictly locked by the parent
            # brick.
            current_ax = ax
            fig = current_ax.figure

            # Disable annotations if batches exceed 6 to prevent text
            # overlapping
            # inside the compact 4x4 inch patchwork container.
            show_annot = n_batches <= 6

        # =====================================================================
        # 2. Heatmap Rendering
        # =====================================================================
        mask = np.triu(np.ones_like(batch_corr_matrix, dtype=bool), k=1)
        custom_cmap = pu.custom_linear_cmap(
            ["white", pu.PRIMARY_ACCENT_COLOR], 100
        )
        color_map = mcolors.ListedColormap(
            pu.extract_linear_cmap(custom_cmap, cmin=0.2, cmax=1.0)
        )
        color_map.set_bad(color="white", alpha=1.0)

        cbar_ax = (
            current_ax.inset_axes([1.05, 0.1, 0.05, 0.8])
            if show_colorbar
            else None
        )

        # Dynamically adjust decimal format to save space for medium cohorts
        annot_fmt = ".3f" if n_batches <= 10 else ".2f"
        annot_size = pu.heatmap_annotation_fontsize(
            ax=current_ax,
            n_rows=n_batches,
            n_cols=n_batches,
            default_size=pu.DEFAULT_ANNOTATION_FONTSIZE,
            max_size=pu.DEFAULT_ANNOTATION_FONTSIZE,
            min_size=3.0,
        )
        cell_edge_lw = pu.DEFAULT_HEATMAP_CELL_LINEWIDTH

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
                cbar=show_colorbar,
                cbar_ax=cbar_ax,
                annot_kws={"size": annot_size},
                cbar_kws={
                    "label": f"{method.title()} Correlation",
                    "format": "%.2f",
                },
            )

        if cbar_ax is not None:
            for spine in cbar_ax.spines.values():
                spine.set_visible(False)
            self._format_colorbar_axes(cbar_ax)

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
            [
                pu.compact_tick_label(label, max_tick_chars)
                for label in raw_x_labels
            ]
            if needs_dense_ticks
            else raw_x_labels
        )
        y_tick_labels = (
            [
                pu.compact_tick_label(label, max_tick_chars)
                for label in raw_y_labels
            ]
            if needs_dense_ticks
            else raw_y_labels
        )
        max_tick_len = max(
            [len(label) for label in x_tick_labels + y_tick_labels] or [1]
        )
        x_rot = (
            90
            if needs_dense_ticks and (n_batches > 10 or max_tick_len > 14)
            else 45
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
        batch_title, append_stage = self._panel_title(
            "Inter-Batch Pooled QC Correlation", title_mode
        )
        self._apply_standard_format(
            ax=current_ax,
            title=batch_title,
            xlabel="Batch ID",
            ylabel="Batch ID",
            append_stage=append_stage,
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

    def plot_correlation_colorbar_legend(
        self,
        method: str,
        vmin: float = 0.85,
        vmax: float = 1.0,
    ) -> plt.Figure:
        """
        Render the shared correlation colorbar used by report heatmap grids.
        """
        fig = plt.figure(figsize=(0.95, 3.0))
        # Keep the color strip narrow; the label and ticks expand the tight
        # export boundary only as much as their text actually requires.
        current_ax = fig.add_axes([0.08, 0.08, 0.16, 0.84])
        custom_cmap = pu.custom_linear_cmap(
            ["white", pu.PRIMARY_ACCENT_COLOR], 100
        )
        color_map = mcolors.ListedColormap(
            pu.extract_linear_cmap(custom_cmap, cmin=0.2, cmax=1.0)
        )
        color_map.set_bad(color="white", alpha=1.0)
        scalar_map = plt.cm.ScalarMappable(
            norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=color_map
        )
        colorbar = fig.colorbar(
            scalar_map,
            cax=current_ax,
            format="%.2f",
            label=f"{method.title()} Correlation",
        )
        self._format_colorbar_axes(colorbar.ax)
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
        pu.place_annotation_in_least_occupied_corner(
            ax=ax,
            text_artist=text_artist,
            occupancy_arrays=occupancy_arrays,
        )

    @staticmethod
    def _reserve_pca_annotation_margin(
        ax: plt.Axes,
        text_artist: object,
        plot_bounds: list[tuple[float, float, float, float]],
    ) -> None:
        """Expand the selected PCA annotation edge beyond all plotted content.

        PCA diagnostics are positioned in axes coordinates, so Matplotlib's
        data autoscaling does not consider the annotation box.  Measure the
        rendered white textbox, then reserve equivalent data-space room above
        or below the scatter/ellipse envelope selected by the corner placer.
        """
        if not plot_bounds:
            return

        try:
            ax.figure.canvas.draw()
            renderer = ax.figure.canvas.get_renderer()
            bbox_patch = getattr(text_artist, "get_bbox_patch", lambda: None)()
            text_bbox = (
                bbox_patch.get_window_extent(renderer=renderer)
                if bbox_patch is not None
                else text_artist.get_window_extent(renderer=renderer)
            )
            axes_bbox = ax.get_window_extent(renderer=renderer)
            text_height = text_bbox.height / max(axes_bbox.height, 1.0)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            text_height = 0.16

        # Include both the rendered box and a small visual gutter.  The upper
        # clamp prevents an unexpectedly long annotation from dominating a PCA
        # panel, while normal three-line diagnostics need about 0.18-0.22.
        reserve_fraction = min(0.42, max(0.10, float(text_height) + 0.04))
        content_ymin = min(bounds[2] for bounds in plot_bounds)
        content_ymax = max(bounds[3] for bounds in plot_bounds)
        y_lower, y_upper = ax.get_ylim()

        if text_artist.get_va() == "bottom":
            # Keep the full envelope above the lower annotation band.
            required_lower = y_upper - (y_upper - content_ymin) / (
                1.0 - reserve_fraction
            )
            ax.set_ylim(min(y_lower, required_lower), y_upper)
        else:
            # Keep the full envelope below the upper annotation band.
            required_upper = y_lower + (content_ymax - y_lower) / (
                1.0 - reserve_fraction
            )
            ax.set_ylim(y_lower, max(y_upper, required_upper))

    @staticmethod
    def _remove_axis_legends(ax: plt.Axes) -> None:
        """
        Remove local and grouped legends from one axes without touching peers.
        """
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        for artist in list(getattr(ax, "artists", [])):
            if isinstance(artist, Legend):
                artist.remove()

    @staticmethod
    def _validate_legend_mode(legend_mode: str) -> str:
        mode = str(legend_mode or "local").strip().lower()
        if mode not in {"local", "external", "none"}:
            raise ValueError(
                "legend_mode must be 'local', 'external', or 'none'."
            )
        return mode

    @staticmethod
    def _validate_title_mode(title_mode: str) -> str:
        mode = str(title_mode or "full").strip().lower()
        if mode not in {"full", "stage"}:
            raise ValueError("title_mode must be 'full' or 'stage'.")
        return mode

    def _panel_title(
        self, full_title: str, title_mode: str
    ) -> tuple[str, bool]:
        """
        Resolve a panel title and whether standard formatting appends stage.
        """
        mode = self._validate_title_mode(title_mode)
        if mode == "full":
            return full_title, True
        stage = str(self.attrs.get("pipeline_stage", "")).strip()
        return (f"[{stage}]" if stage else full_title), False

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
        legend_mode: str = "local",
        title_mode: str = "full",
    ) -> plt.Figure:
        """Plot PCA scatter plot with confidence ellipses and QA metrics."""
        legend_mode = self._validate_legend_mode(legend_mode)
        plot_df = pca_df.reset_index().copy()
        plot_df[sample_type] = plot_df[sample_type].astype("category")
        plot_df = plot_df.sort_values(by=sample_type, ascending=False)
        palette_dict = {
            qc_label: pu.PRIMARY_ACCENT_COLOR,
            actual_label: pu.NEUTRAL_COLOR,
        }

        if ax is None:
            fig, current_ax = plt.subplots(
                figsize=pu.dashboard_brick_size(4.0, 4.0, 14.0)
            )
        else:
            current_ax = ax
            fig = current_ax.figure

        sns.despine(ax=current_ax)

        relative_dispersion = pca_diagnostics.get("relative_dispersion")
        silhouette_score = pca_diagnostics.get("batch_silhouette")
        centrality_shift = pca_diagnostics.get("centrality_shift")

        rd_str = (
            f"{relative_dispersion:.4f}"
            if pd.notna(relative_dispersion)
            else "N/A"
        )
        sil_str = (
            f"{silhouette_score:.4f}" if pd.notna(silhouette_score) else "N/A"
        )
        shift_str = (
            f"{centrality_shift:.4f}" if pd.notna(centrality_shift) else "N/A"
        )

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
            s=pu.DEFAULT_SCATTER_MARKER_AREA,
            edgecolor="k",
            palette=palette_dict,
            linewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
            ax=current_ax,
            hue_order=[qc_label, actual_label],
            style_order=self.all_batches,
            markers=self.style_map,
        )

        plot_bounds = []
        occupancy_arrays = []
        finite_x = pd.to_numeric(plot_df[x_pc], errors="coerce").to_numpy(
            dtype=float
        )
        finite_y = pd.to_numeric(plot_df[y_pc], errors="coerce").to_numpy(
            dtype=float
        )
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
                        facecolor=mcolors.to_rgba(
                            palette_dict[group], alpha=0.12
                        ),
                        edgecolor=palette_dict[group],
                        linewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
                        zorder=3,
                    )
                    pu.mark_preserve_alpha(ellipse)
                    try:
                        vertices = ellipse.get_path().vertices
                        display_vertices = ellipse.get_transform().transform(
                            vertices
                        )
                        data_vertices = (
                            current_ax.transData.inverted().transform(
                                display_vertices
                            )
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
            fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
            verticalalignment="bottom",
            horizontalalignment="right",
            clip_on=False,
            zorder=10,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor="none",
                alpha=1.0,
            ),
        )

        var_x = pca_var.loc[x_pc] * 100
        var_y = pca_var.loc[y_pc] * 100
        pca_title, append_stage = self._panel_title(
            "Pooled QC & Sample PCA Scatter", title_mode
        )

        if legend_mode == "local":
            self._format_multi_legends(
                ax=current_ax,
                group_titles=[sample_type, batch],
                loc="upper left",
                start_bbox=(1.05, 1.0),
                row_gap=0.04,
                layout_cols=1,
                sublegend_cols=1,
                markerscale=0.85,
            )
        else:
            self._remove_axis_legends(current_ax)

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
        self._reserve_pca_annotation_margin(
            ax=current_ax,
            text_artist=annot_artist,
            plot_bounds=plot_bounds,
        )
        # Limit expansion can cause Matplotlib to generate new major tick
        # labels. Apply the project style only once all final limits are set.
        self._apply_standard_format(
            ax=current_ax,
            xlabel=f"{x_pc} ({var_x:.1f}%)",
            ylabel=f"{y_pc} ({var_y:.1f}%)",
            append_stage=append_stage,
            title=pca_title,
        )
        return fig

    def plot_rsd_standalone_legend(
        self,
        qc_label: str,
        actual_label: str,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Render the shared sample-type legend used by exported RSD panels."""
        standalone = ax is None
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(2.2, 1.6))
        else:
            current_ax = ax
            fig = current_ax.figure

        handles = [
            Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=1.0),
                edgecolor="black",
                linestyle="-",
                linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
                label=qc_label,
            ),
            Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=0.4),
                edgecolor="black",
                linestyle="--",
                linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
                label=actual_label,
            ),
        ]
        legend = current_ax.legend(
            handles=handles,
            title="Sample Type",
            loc="center left",
            bbox_to_anchor=(0.0, 0.5),
            **getattr(self, "LEGEND_KWARGS", {}),
        )
        # A figure-level legend lets ``bbox_inches='tight'`` crop to the
        # legend itself instead of retaining the otherwise empty axes canvas.
        if standalone and legend not in fig.legends:
            fig.legends.append(legend)
        if standalone:
            current_ax.set_visible(False)
        return fig

    def plot_pca_standalone_legend(
        self,
        pca_df: pd.DataFrame,
        sample_type: str,
        batch: str,
        qc_label: str,
        actual_label: str,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Render the shared sample-type and batch legend for PCA panels."""
        standalone = ax is None
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(2.4, 3.8))
        else:
            current_ax = ax
            fig = current_ax.figure

        current_ax.set_position([0.0, 0.0, 1.0, 1.0])

        present_batches = set(pca_df.index.get_level_values(batch))
        handles = [
            mlines.Line2D([], [], color="none", label="Sample Type"),
            mlines.Line2D(
                [],
                [],
                color=pu.PRIMARY_ACCENT_COLOR,
                marker="o",
                linestyle="none",
                markeredgecolor="black",
                markeredgewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
                markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                label=qc_label,
            ),
            mlines.Line2D(
                [],
                [],
                color=pu.NEUTRAL_COLOR,
                marker="o",
                linestyle="none",
                markeredgecolor="black",
                markeredgewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
                markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                label=actual_label,
            ),
            mlines.Line2D([], [], color="none", label="Batch"),
        ]
        for batch_id in self.all_batches:
            if batch_id not in present_batches:
                continue
            handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color=pu.NEUTRAL_COLOR,
                    marker=self.style_map[batch_id],
                    linestyle="none",
                    markeredgecolor="black",
                    markeredgewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
                    markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                    label=str(batch_id),
                )
            )

        current_ax.legend(handles=handles)
        self._format_multi_legends(
            ax=current_ax,
            group_titles=["Sample Type", "Batch"],
            loc="upper left",
            start_bbox=(0.0, 0.98),
            row_gap=0.04,
            layout_cols=1,
            sublegend_cols=1,
            markerscale=0.85,
        )
        # _format_multi_legends promotes its sublegends to figure artists.
        # Hide the carrier axes so a tight SVG contains no blank canvas.
        if standalone:
            current_ax.set_visible(False)
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
        legend_mode: str = "local",
        title_mode: str = "full",
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
        legend_mode = self._validate_legend_mode(legend_mode)

        # Initialize axes hierarchy
        if ax is None:
            fig, current_ax = plt.subplots(
                figsize=pu.dashboard_brick_size(4.0, 4.0, 14.0)
            )
        else:
            current_ax = ax
            fig = current_ax.figure

        labels = ["0-10%", "10-20%", "20-30%", ">30%"]
        x_pos = np.arange(len(labels), dtype=float)
        bar_width = 0.42
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
                pu.PRIMARY_ACCENT_COLOR
                if label == ">30%"
                else pu.NEUTRAL_COLOR,
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
            linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
            linestyle="-",
            label=qc_label,
        )
        current_ax.bar(
            x_pos + bar_width / 2,
            actual_counts,
            width=bar_width,
            color=actual_bar_colors,
            edgecolor="black",
            linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
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
            fontsize=pu.DEFAULT_DENSE_BAR_ANNOTATION_FONTSIZE,
        )

        # Manually construct legend to ensure correct style mapping
        h_type = [
            Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=1.0),
                edgecolor="black",
                linestyle="-",
                linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
                label=qc_label,
            ),
            Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=0.4),
                edgecolor="black",
                linestyle="--",
                linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
                label=actual_label,
            ),
        ]

        # Bypass auto-formatters using global LEGEND_KWARGS
        legend_kwargs = getattr(self, "LEGEND_KWARGS", {}).copy()
        legend_kwargs.update({"title": "Sample Type", "loc": "best"})
        if legend_mode == "local":
            current_ax.legend(handles=h_type, **legend_kwargs)
        current_ax.set_xticks(x_pos)
        current_ax.set_xticklabels(labels)

        # Execute standardized axis formatting
        if hasattr(self, "_apply_standard_format"):
            rsd_title, append_stage = self._panel_title(
                "Feature RSD Distribution", title_mode
            )
            self._apply_standard_format(
                ax=current_ax,
                title=rsd_title,
                xlabel="RSD Bin",
                ylabel="Feature Count",
                append_stage=append_stage,
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
        annotate_thresholds: bool = False,
        legend_mode: str = "local",
        title_mode: str = "full",
    ) -> plt.Figure:
        """Plot SD-OD diagnostic scatter with multi-dimensional overlays."""
        legend_mode = self._validate_legend_mode(legend_mode)
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
            fig, current_ax = plt.subplots(
                figsize=pu.dashboard_brick_size(4.0, 4.0, 14.0)
            )
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
            s=pu.DEFAULT_SCATTER_MARKER_AREA,
            edgecolor="k",
            linewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
            ax=current_ax,
            zorder=2,
        )

        # Overlay halo effect for analytical IS outliers (accent dashed circle)
        if is_flags is not None and is_flags.any():
            outlier_idx = is_flags[is_flags].index.intersection(
                metrics_df.index
            )
            subset = metrics_df.loc[outlier_idx]
            if not subset.empty:
                current_ax.scatter(
                    subset["SD"],
                    subset["OD"],
                    s=60,
                    facecolors="none",
                    edgecolors=pu.PRIMARY_ACCENT_COLOR,
                    linewidths=pu.DEFAULT_GUIDE_LINEWIDTH,
                    linestyle="--",
                    zorder=3,
                    label="IS Outlier",
                )

        # Overlay halo effect for analytical ORF outliers (Orange dash-dot
        # circle)
        if orf_flags is not None and orf_flags.any():
            outlier_idx = orf_flags[orf_flags].index.intersection(
                metrics_df.index
            )
            subset = metrics_df.loc[outlier_idx]
            if not subset.empty:
                current_ax.scatter(
                    subset["SD"],
                    subset["OD"],
                    s=72,
                    facecolors="none",
                    edgecolors="tab:orange",
                    linewidths=pu.DEFAULT_GUIDE_LINEWIDTH,
                    linestyle="-.",
                    zorder=4,
                    label="ORF Outlier",
                )

        threshold_color = pu.get_equivalent_hex("k", alpha=0.6)
        current_ax.axvline(
            x=sd_limit,
            color=threshold_color,
            linestyle="--",
            linewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
        )
        current_ax.axhline(
            y=od_limit,
            color=threshold_color,
            linestyle="--",
            linewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
        )

        if annotate_thresholds:
            current_ax.annotate(
                f"SD limit = {sd_limit:.2f}",
                xy=(sd_limit, 0.98),
                xycoords=("data", "axes fraction"),
                xytext=(3, 0),
                textcoords="offset points",
                rotation=90,
                ha="left",
                va="top",
                fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                color=threshold_color,
                clip_on=False,
            )
            current_ax.annotate(
                f"OD limit = {od_limit:.2f}",
                xy=(0.98, od_limit),
                xycoords=("axes fraction", "data"),
                xytext=(0, 3),
                textcoords="offset points",
                rotation=0,
                ha="right",
                va="bottom",
                fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                color=threshold_color,
                clip_on=False,
            )

        outlier_title, append_stage = self._panel_title(
            "Integrated Outlier Diagnostics", title_mode
        )
        self._apply_standard_format(
            ax=current_ax,
            xlabel="Score Distance (Hotelling's T2)",
            ylabel="Orthogonal Distance (SPE / DModX)",
            append_stage=append_stage,
            title=outlier_title,
        )

        if show_legend and legend_mode == "local":
            handles, labels = current_ax.get_legend_handles_labels()
            dummy_cat = mlines.Line2D([], [], color="none", label="Category")
            full_handles = [dummy_cat] + handles
            full_labels = ["Category"] + labels

            current_ax.legend(full_handles, full_labels)

            self._format_multi_legends(
                ax=current_ax,
                group_titles=["Category"],
                loc="upper left",
                start_bbox=(1.05, 1.0),
                row_gap=0.04,
                layout_cols=1,
                sublegend_cols=1,
            )
        else:
            self._remove_axis_legends(current_ax)

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
                nrows=2,
                ncols=1,
                figsize=(out_df.shape[0] * 0.3 + 2, 7),
                sharex=True,
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
            [
                pu.compact_tick_label(label, max_tick_chars)
                for label in raw_tick_labels
            ]
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
                        bar.set_linewidth(pu.DEFAULT_AXIS_LINEWIDTH)
                        bar.set_hatch(hatch_styles[cat])

            if i == 0 and od_limit is not None:
                ax.axhline(
                    y=od_limit,
                    color="k",
                    linestyle="--",
                    linewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
                    alpha=0.8,
                    zorder=2,
                )
            elif i == 1 and sd_limit is not None:
                ax.axhline(
                    y=sd_limit,
                    color="k",
                    linestyle="--",
                    linewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
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
            full_text = (
                str(new_idx[idx]) if idx < len(new_idx) else label.get_text()
            )
            is_extreme = (
                idx < len(cat_values) and cat_values[idx] == "Extreme Outlier"
            )
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
        complete_categories: bool = False,
        include_bar_diagnostics: bool = True,
        include_thresholds: bool = True,
    ) -> plt.Figure | plt.Axes:
        """Create a comprehensive unified legend for all outlier diagnostics."""
        standalone = ax is None
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(2.0, 4.0))
        else:
            current_ax = ax
            fig = current_ax.figure

        accent_solid = pu.PRIMARY_ACCENT_COLOR
        accent_alpha = pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=0.5)
        gray_col = pu.NEUTRAL_COLOR

        cat_styles = {
            "Extreme Outlier": {
                "color": accent_solid,
                "marker": "X",
                "hatch": "",
            },
            "Orthogonal Outlier": {
                "color": accent_alpha,
                "marker": "s",
                "hatch": "///",
            },
            "Strong Outlier": {
                "color": accent_alpha,
                "marker": "^",
                "hatch": r"\\\\",
            },
            "Normal": {"color": gray_col, "marker": "o", "hatch": ""},
        }

        def _reference_features_available(
            flags: pd.Series | None, feature_attr: str
        ) -> bool:
            """
            Return whether a reference-feature channel exists for this QA run.

            A boolean flag series can be present even when every sample is
            normal. The assessor's validated feature list is the authoritative
            availability check, so configured-but-invalid feature names do not
            create misleading legend entries.

            """
            source_obj = getattr(self, "obj", None)
            if source_obj is None:
                return flags is not None
            try:
                return bool(getattr(source_obj, feature_attr))
            except (AttributeError, KeyError, TypeError, ValueError):
                return False

        has_is_features = _reference_features_available(is_flags, "valid_is")
        has_orf_features = _reference_features_available(orf_flags, "valid_orf")

        present_categories = set(metrics_df["Category"].unique())
        legend_handles, legend_labels = [], []
        group_titles = ["Scatter Diagnostics"]
        if include_bar_diagnostics:
            group_titles.append("Bar Diagnostics")
        if include_thresholds:
            group_titles.append("Thresholds")

        # --- Group A: Scatter Diagnostics (Markers) ---
        legend_handles.append(
            mlines.Line2D([], [], color="none", label="Scatter Diagnostics")
        )
        legend_labels.append("Scatter Diagnostics")

        for label, style in cat_styles.items():
            if complete_categories or label in present_categories:
                h = mlines.Line2D(
                    [],
                    [],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle="none",
                    markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                    markeredgecolor="k",
                    markeredgewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
                    label=label,
                )
                legend_handles.append(h)
                legend_labels.append(label)

        if has_is_features:
            halo_handle = mlines.Line2D(
                [],
                [],
                color="none",
                markeredgecolor=pu.PRIMARY_ACCENT_COLOR,
                marker="o",
                markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                markeredgewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
                linestyle="--",
                label="IS Outlier",
            )
            legend_handles.append(halo_handle)
            legend_labels.append("IS Outlier")

        if has_orf_features:
            orf_halo_handle = mlines.Line2D(
                [],
                [],
                color="none",
                markeredgecolor="tab:orange",
                marker="o",
                markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                markeredgewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
                linestyle="-.",
                label="ORF Outlier",
            )
            legend_handles.append(orf_halo_handle)
            legend_labels.append("ORF Outlier")

        if include_bar_diagnostics:
            # --- Group B: Bar Diagnostics (Hatch Styles) ---
            legend_handles.append(
                mlines.Line2D([], [], color="none", label="Bar Diagnostics")
            )
            legend_labels.append("Bar Diagnostics")

            for label, style in cat_styles.items():
                if (
                    complete_categories or label in present_categories
                ) and label != "Normal":
                    h = mpatches.Patch(
                        facecolor=style["color"],
                        edgecolor="black",
                        linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
                        hatch=style["hatch"],
                        label=label,
                    )
                    legend_handles.append(h)
                    legend_labels.append(label)

            if has_is_features:
                legend_handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        color="none",
                        markerfacecolor=pu.PRIMARY_ACCENT_COLOR,
                        markeredgecolor=pu.PRIMARY_ACCENT_COLOR,
                        marker=r"$\ast$",
                        markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                        linestyle="none",
                        label="IS Outlier",
                    )
                )
                legend_labels.append("IS Outlier")

            if has_orf_features:
                legend_handles.append(
                    mlines.Line2D(
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
                )
                legend_labels.append("ORF Outlier")

        if include_thresholds:
            # --- Group C: Thresholds (Lines) ---
            legend_handles.append(
                mlines.Line2D([], [], color="none", label="Thresholds")
            )
            legend_labels.append("Thresholds")

            if sd_limit is not None:
                legend_handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        color="k",
                        ls="--",
                        alpha=0.8,
                        lw=pu.DEFAULT_GUIDE_LINEWIDTH,
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
                        lw=pu.DEFAULT_GUIDE_LINEWIDTH,
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
        # The grouped legends are figure artists after _format_multi_legends;
        # excluding the carrier axes keeps the standalone SVG compact.
        current_ax.axis("off")
        if standalone:
            current_ax.set_visible(False)

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
        """
        Plot Shewhart control charts with adaptive PW panels and one legend.
        """
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
            plot_df[sample_type],
            categories=[actual_label, qc_label],
            ordered=True,
        )
        plot_df[batch] = plot_df[batch].astype("category")
        plot_df = plot_df.sort_values(by=sample_type, ascending=True)

        # Symmetrical visual markers mapping from previous specification
        v_color = (
            pu.PRIMARY_ACCENT_COLOR if ref_type_upper == "IS" else "tab:orange"
        )
        v_ls = "--" if ref_type_upper == "IS" else "-."

        # Step 1: Generate analytical control chart bricks sequentially
        plot_bricks = []
        panel_cols = 1 if len(valid_feats) == 1 else 2
        shewhart_source_width = 13.0
        shewhart_legend_source_width = 2.5
        shewhart_layout_width = (
            shewhart_source_width + shewhart_legend_source_width
        )
        panel_size = pu.dashboard_brick_size(6.5, 2.0, shewhart_layout_width)
        for feat_idx, feat in enumerate(valid_feats):
            brick = pw.Brick(
                figsize=panel_size,
                label=f"{ref_type_upper}_shewhart_{feat_idx}",
            )

            sns.scatterplot(
                ax=brick,
                data=plot_df,
                x=inject_order,
                y=feat,
                s=pu.DEFAULT_SCATTER_MARKER_AREA,
                edgecolor="k",
                linewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
                style=batch,
                palette={
                    qc_label: pu.PRIMARY_ACCENT_COLOR,
                    actual_label: pu.NEUTRAL_COLOR,
                },
                hue=sample_type,
                hue_order=[qc_label, actual_label],
                markers=self.style_map,
            )

            solid, lower, upper = model.MetaboInt.calculate_boundaries(
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
                    s=60,
                    linewidths=pu.DEFAULT_GUIDE_LINEWIDTH,
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

        legend_height_source = max(2.0, len(row_bricks) * 2.0)
        leg_brick = pw.Brick(
            figsize=pu.dashboard_brick_size(
                shewhart_legend_source_width,
                legend_height_source,
                shewhart_layout_width,
            ),
            label=f"{ref_type_upper}_shewhart_legend",
        )
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
                [],
                [],
                color=pu.PRIMARY_ACCENT_COLOR,
                marker="o",
                linestyle="none",
                markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                markeredgecolor="k",
                markeredgewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
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
                markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                markeredgecolor="k",
                markeredgewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
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
                    markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                    markeredgecolor="k",
                    markeredgewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
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
            solid_label, low_label, up_label = (
                "Mean",
                "Mean - 3 Std",
                "Mean + 3 Std",
            )

        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color="k",
                ls="-",
                lw=pu.DEFAULT_GUIDE_LINEWIDTH,
                label=solid_label,
            )
        )
        legend_labels.append(solid_label)
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color="k",
                ls="--",
                lw=pu.DEFAULT_GUIDE_LINEWIDTH,
                label=low_label,
            )
        )
        legend_labels.append(low_label)
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color="k",
                ls="--",
                lw=pu.DEFAULT_GUIDE_LINEWIDTH,
                label=up_label,
            )
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
        layout_width = 14.0
        ax1 = pw.Brick(figsize=pu.dashboard_brick_size(4.8, 4.0, layout_width))
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

        ax2 = pw.Brick(figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width))
        self.plot_rsd_bar(
            rsd_data=rsd_data,
            qc_label=qc_label,
            actual_label=actual_label,
            ax=ax2,
        )
        _bind_legends_to_axes(ax2)

        ax3 = pw.Brick(figsize=pu.dashboard_brick_size(5.2, 4.0, layout_width))
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
        ax4 = pw.Brick(figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width))
        self.plot_sd_od_scatter(
            metrics_df=pca_res["metrics_df"],
            sd_limit=pca_res["sd_limit"],
            od_limit=pca_res["od_limit"],
            ax=ax4,
            show_legend=False,
            annotate_thresholds=True,
            is_flags=is_flags,
            orf_flags=orf_flags,
        )

        ax5 = pw.Brick(figsize=pu.dashboard_brick_size(8.8, 4.0, layout_width))
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

        ax6 = pw.Brick(figsize=pu.dashboard_brick_size(1.2, 4.0, layout_width))
        self.plot_outlier_standalone_legend(
            metrics_df=pca_res["metrics_df"],
            sd_limit=pca_res["sd_limit"],
            od_limit=pca_res["od_limit"],
            ax=ax6,
            is_flags=is_flags,
            orf_flags=orf_flags,
            include_thresholds=False,
        )
        _bind_legends_to_axes(ax6)

        return (ax1 | ax2 | ax3) / (ax4 | ax5 | ax6)
