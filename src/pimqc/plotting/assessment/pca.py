"""PCA diagnostics and legends for quality assessment.

The module renders pooled-QC/sample score plots and formats their diagnostic
annotation without computing PCA scores or assessment metrics.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.legend import Legend
from typing import Any

from .. import annotation_layout as al
from .. import plot_utils as pu


class AssessmentPcaMixin:
    """Render PCA panels, diagnostics, and their shared legend."""

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

    @staticmethod
    def _format_pca_diagnostics_annotation(
        pca_diagnostics: dict[str, Any], batches: pd.Series
    ) -> str:
        """Format PCA diagnostics for the combinations supported by the data.

        Batch silhouette is meaningful only when the plotted data contain at
        least two valid batches. Single-batch datasets therefore omit that
        diagnostic instead of displaying an uninformative ``N/A`` value.

        Args:
            pca_diagnostics: Calculated PCA quality-assessment metrics.
            batches: Batch labels represented in the PCA scatter data.

        Returns:
            Annotation text containing all applicable PCA diagnostics.
        """

        def format_metric(value: Any) -> str:
            return f"{value:.4f}" if pd.notna(value) else "N/A"

        annotation_lines = [
            "Relative Dispersion: "
            f"{format_metric(pca_diagnostics.get('relative_dispersion'))}"
        ]
        if batches.dropna().nunique() > 1:
            annotation_lines.append(
                "Batch Silhouette: "
                f"{format_metric(pca_diagnostics.get('batch_silhouette'))}"
            )
        annotation_lines.append(
            "Centrality Shift: "
            f"{format_metric(pca_diagnostics.get('centrality_shift'))}"
        )
        return "\n".join(annotation_lines)

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

    def plot_pca_diagnostics_legend(
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

    # PCA diagnostic panel

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

        annot_text = self._format_pca_diagnostics_annotation(
            pca_diagnostics=pca_diagnostics,
            batches=plot_df[batch],
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

        al.place_annotation_with_legend_awareness(
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
