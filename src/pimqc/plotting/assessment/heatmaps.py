"""Correlation heatmaps and colorbar legends for quality assessment.

The module renders pooled-QC and inter-batch correlation matrices using the
shared heatmap typography and visible-cell-edge utilities.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional

from .. import plot_utils as pu
from ..heatmap import (
    draw_visible_heatmap_cell_edges,
    heatmap_annotation_fontsize,
)


class AssessmentHeatmapMixin:
    """Render assessment correlation heatmaps and their legends."""

    # Matrix heatmaps and system trends are implemented below.
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

        # Clustering Strategy Routing
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

        # Batch Colors Preparation
        custom_cmap = pu.custom_linear_cmap(
            ["white", pu.PRIMARY_ACCENT_COLOR], 100
        )
        color_map = mcolors.ListedColormap(
            pu.extract_linear_cmap(custom_cmap, cmin=0.2, cmax=1.0)
        ).with_extremes(bad="white")

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

        # Canvas & Layout Integration (Matplotlib GridSpec)
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

        annot_size = heatmap_annotation_fontsize(
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

        # Dendrogram Renderer Engine
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

        # Main Lower-Triangle Heatmap
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
                pu.format_colorbar_axes(cbar_ax)

        draw_visible_heatmap_cell_edges(
            ax=ax_heatmap,
            visible_mask=np.ones_like(geom_mask, dtype=bool),
            linewidth=cell_edge_lw,
            edgecolor="white",
            zorder=3,
        )
        draw_visible_heatmap_cell_edges(
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

        # Standardized Formatting & Inset Legend Injection
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

        # Context-Aware Dynamic Sizing & Annotation Logic
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

        # Heatmap Rendering
        mask = np.triu(np.ones_like(batch_corr_matrix, dtype=bool), k=1)
        custom_cmap = pu.custom_linear_cmap(
            ["white", pu.PRIMARY_ACCENT_COLOR], 100
        )
        color_map = mcolors.ListedColormap(
            pu.extract_linear_cmap(custom_cmap, cmin=0.2, cmax=1.0)
        ).with_extremes(bad="white")

        cbar_ax = (
            current_ax.inset_axes([1.05, 0.1, 0.05, 0.8])
            if show_colorbar
            else None
        )

        # Dynamically adjust decimal format to save space for medium cohorts
        annot_fmt = ".3f" if n_batches <= 10 else ".2f"
        annot_size = heatmap_annotation_fontsize(
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
            pu.format_colorbar_axes(cbar_ax)

        draw_visible_heatmap_cell_edges(
            ax=current_ax,
            visible_mask=np.ones_like(mask, dtype=bool),
            linewidth=cell_edge_lw,
            edgecolor="white",
            zorder=3,
        )
        draw_visible_heatmap_cell_edges(
            ax=current_ax,
            visible_mask=~mask,
            linewidth=cell_edge_lw,
            edgecolor="k",
            zorder=4,
        )
        for spine in current_ax.spines.values():
            spine.set_visible(False)

        # Dense Tick Layout
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

        # Standard Formatting
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
        ).with_extremes(bad="white")
        scalar_map = plt.cm.ScalarMappable(
            norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=color_map
        )
        colorbar = fig.colorbar(
            scalar_map,
            cax=current_ax,
            format="%.2f",
            label=f"{method.title()} Correlation",
        )
        pu.format_colorbar_axes(colorbar.ax)
        return fig

    # Dimensionality Reduction and Outlier Plots
