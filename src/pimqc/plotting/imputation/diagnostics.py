"""Masked-value reconstruction diagnostics for imputation.

The module renders distribution-fidelity and reconstruction-error panels from
precomputed benchmark arrays and metrics.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ...constants import DEFAULT_RANDOM_SEED
from ...statistics import metrics as su
from .. import annotation_layout as al
from .. import plot_utils as pu


class ImputationDiagnosticsMixin:
    """Provide masked-value density and reconstruction scatter diagnostics."""

    def _plot_masked_distribution_fidelity(
        self,
        true_vals: np.ndarray,
        pred_vals: np.ndarray,
        metrics: dict[str, float] | None = None,
        ax: plt.Axes | None = None,
        compact_title: bool = False,
        article_compact: bool = False,
        show_legend: bool = False,
    ) -> plt.Figure | plt.Axes:
        """
        Plot score-aligned density fidelity for pooled masked nonblank values.
        """
        from scipy.stats import gaussian_kde

        return_fig = ax is None
        if ax is None:
            fig, ax = plt.subplots(
                figsize=pu.article_brick_size(5.0, 4.0)
                if article_compact
                else (5.0, 4.0)
            )

        truth = np.asarray(true_vals, dtype=float)
        reconstruction = np.asarray(pred_vals, dtype=float)
        truth = truth[np.isfinite(truth)]
        reconstruction = reconstruction[np.isfinite(reconstruction)]

        if truth.size < 2 or reconstruction.size < 2:
            ax.text(
                0.5,
                0.5,
                "Insufficient masked values for density estimation.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                bbox=pu.ai_ready_text_bbox(),
                zorder=10,
            )
            self._apply_standard_format(
                ax=ax,
                title="Masked-Value Distribution Fidelity"
                if compact_title
                else "Masked-Value Distribution Fidelity",
                xlabel="log2 Intensity",
                ylabel="Relative Density",
                append_stage=False,
            )
            return fig if return_fig else ax

        x_min = min(float(np.min(truth)), float(np.min(reconstruction)))
        x_max = max(float(np.max(truth)), float(np.max(reconstruction)))
        margin = (x_max - x_min) * 0.10 if x_max > x_min else 1.0
        x_grid = np.linspace(x_min - margin, x_max + margin, 500)

        def _evaluate_kde(
            values: np.ndarray,
            seed_sequence: np.random.SeedSequence,
        ) -> np.ndarray:
            kde_values = values.copy()
            if np.nanstd(kde_values) < 1e-6:
                rng = np.random.default_rng(seed_sequence)
                kde_values += rng.normal(0.0, 1e-4, size=kde_values.size)
            return gaussian_kde(kde_values)(x_grid)

        base_seed = int(
            self.imp_obj.attrs.get("global_seed", DEFAULT_RANDOM_SEED)
        )
        truth_seed, reconstruction_seed = np.random.SeedSequence(
            base_seed
        ).spawn(2)
        truth_density = _evaluate_kde(truth, truth_seed)
        reconstruction_density = _evaluate_kde(
            reconstruction,
            reconstruction_seed,
        )
        y_max = max(
            float(np.nanmax(truth_density)),
            float(np.nanmax(reconstruction_density)),
        )

        pu.mark_preserve_alpha(ax)
        ax.fill_between(
            x_grid,
            truth_density,
            color="tab:gray",
            alpha=0.20,
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(
            x_grid,
            truth_density,
            color="black",
            linestyle="--",
            linewidth=0.75 if article_compact else 1.5,
            zorder=2,
        )
        ax.plot(
            x_grid,
            reconstruction_density,
            color=pu.PRIMARY_ACCENT_COLOR,
            linewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
            zorder=3,
        )
        ax.set_ylim(0.0, y_max * 1.35)

        metric_annotation = None
        if metrics:
            jsd = su.finite_or_nan(metrics.get("JSD_Total"))
            wasserstein = su.finite_or_nan(
                metrics.get("Wasserstein_Normalized")
            )
            annotation_lines = []
            if np.isfinite(jsd):
                annotation_lines.append(
                    f"JSD: {jsd:.3f}"
                    if article_compact
                    else f"Jensen-Shannon distance: {jsd:.3f}"
                )
            if np.isfinite(wasserstein):
                annotation_lines.append(
                    f"W: {wasserstein:.3f}"
                    if article_compact
                    else f"Normalized Wasserstein distance: {wasserstein:.3f}"
                )
            if annotation_lines:
                metric_annotation = ax.text(
                    0.5,
                    0.5,
                    "\n".join(annotation_lines),
                    transform=ax.transAxes,
                    fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                    horizontalalignment="right",
                    verticalalignment="top",
                    bbox=pu.ai_ready_text_bbox(
                        pad=0.25 if article_compact else 0.4
                    ),
                    zorder=10,
                )

        if show_legend:
            import matplotlib.lines as mlines

            ax.legend(
                handles=[
                    mlines.Line2D(
                        [],
                        [],
                        color="black",
                        linestyle="--",
                        linewidth=0.75 if article_compact else 1.5,
                        label="Known masked values",
                    ),
                    mlines.Line2D(
                        [],
                        [],
                        color=pu.PRIMARY_ACCENT_COLOR,
                        linewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
                        label="Reconstructed values",
                    ),
                ]
            )
            self._format_single_legend(
                ax=ax,
                group_title="Masked-value density reference",
                loc="lower right",
                bbox_to_anchor=None,
                max_item_rows=6,
            )

        if metric_annotation is not None:
            al.place_annotation_with_legend_awareness(
                ax=ax,
                text_artist=metric_annotation,
                occupancy_arrays=[
                    np.column_stack((x_grid, truth_density)),
                    np.column_stack((x_grid, reconstruction_density)),
                ],
            )

        self._apply_standard_format(
            ax=ax,
            title="Masked-Value Distribution Fidelity"
            if compact_title
            else "Masked-Value Distribution Fidelity",
            xlabel="log2 Intensity",
            ylabel="Relative Density",
            append_stage=False,
        )
        return fig if return_fig else ax

    def _plot_kde_standalone_legend(
        self,
        ax: plt.Axes,
        legend_cols: int = 3,
        loc: str = "upper left",
        bbox_to_anchor: tuple[float, float] | None = (0.0, 1.0),
    ) -> plt.Axes:
        """Draw a standalone legend for masked-density fidelity overlays."""
        import matplotlib.lines as mlines

        ax.axis("off")
        handles = [
            mlines.Line2D(
                [],
                [],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Known masked values",
            ),
            mlines.Line2D(
                [],
                [],
                color=pu.PRIMARY_ACCENT_COLOR,
                linestyle="-",
                linewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
                label="Reconstructed values",
            ),
        ]
        ax.legend(
            handles=handles,
            title="Masked-value density",
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            ncol=legend_cols,
            frameon=True,
            edgecolor="k",
            borderaxespad=0.0,
        )
        self._format_single_legend(
            ax=ax,
            group_title="Masked-value density reference",
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            legend_cols=legend_cols,
            borderaxespad=0.0,
        )
        return ax

    def _plot_nrmse_scatter(
        self,
        true_vals: np.ndarray,
        pred_vals: np.ndarray,
        metrics: dict[str, float],
        method_name: str = "",
        axis_lims: tuple[float, float] | None = None,
        compact_title: bool = False,
        show_method_in_title: bool = True,
        show_colorbar: bool = True,
        article_compact: bool = False,
        ax: plt.Axes | None = None,
    ) -> plt.Figure | plt.Axes:
        """Plot hexbin scatter of true vs imputed values from mask test."""
        if ax is None:
            fig, current_ax = plt.subplots(
                figsize=pu.article_brick_size(5.0, 4.0)
                if article_compact
                else (5.0, 4.0)
            )
        else:
            current_ax = ax
            fig = current_ax.figure

        if axis_lims is not None:
            ax_min, ax_max = axis_lims
            extent = (ax_min, ax_max, ax_min, ax_max)
            lim_min, lim_max = ax_min, ax_max
        else:
            d_min = min(true_vals.min(), pred_vals.min())
            d_max = max(true_vals.max(), pred_vals.max())
            margin = (d_max - d_min) * 0.05

            lim_min, lim_max = d_min - margin, d_max + margin
            extent = (lim_min, lim_max, lim_min, lim_max)

        color_map = pu.custom_linear_cmap(
            color_list=["white", pu.PRIMARY_ACCENT_COLOR],
            n_colors=256,
            cmin=0.1,
            cmax=1.0,
        )

        hb = current_ax.hexbin(
            x=true_vals,
            y=pred_vals,
            gridsize=40,
            extent=extent,
            cmap=color_map,
            mincnt=1,
        )

        current_ax.plot(
            [lim_min, lim_max],
            [lim_min, lim_max],
            color="tab:gray",
            linestyle="--",
            linewidth=0.6 if article_compact else 1.0,
            zorder=3,
        )

        nrmse_total = float(metrics.get("NRMSE_Total", np.nan))
        nrmse_low = float(metrics.get("NRMSE_Low", np.nan))
        annot_lines = [f"NRMSE (total): {nrmse_total:.4f}"]
        if np.isfinite(nrmse_low):
            annot_lines.append(f"NRMSE (low): {nrmse_low:.4f}")
        annot_text = "\n".join(annot_lines)

        title_str = "MAR Masked Simulation"
        if method_name:
            is_selected = method_name.strip().startswith("*")
            clean_name = method_name.replace("*", "").strip()
            clean_upper = clean_name.upper()
            if clean_upper in ("KNN", "LLS", "BPCA"):
                display = clean_upper
            elif clean_upper in ("QRILC", "QRLIC"):
                display = "QRILC"
            elif clean_upper in ("MINPROB", "PROB"):
                display = "MinProb"
            elif clean_upper == "MEDIAN":
                display = "Median"
            else:
                display = clean_name.title()

            if is_selected:
                display = f"* {display}"

            if show_method_in_title:
                title_str = (
                    display if compact_title else f"{title_str} ({display})"
                )

        self._apply_standard_format(
            ax=current_ax,
            title=title_str,
            xlabel="Known Masked Intensity (log2)",
            ylabel="Reconstructed Intensity (log2)",
            append_stage=False,
        )

        if axis_lims is not None:
            current_ax.set_xlim(ax_min, ax_max)
            current_ax.set_ylim(ax_min, ax_max)

        if show_colorbar:
            cb = fig.colorbar(hb, ax=current_ax)
            cb.set_label("Log10(Count)")
            pu.format_colorbar_axes(cb.ax)
        al.add_auto_annotation(
            ax=current_ax,
            text=annot_text,
            occupancy_arrays=[np.column_stack((true_vals, pred_vals))],
            fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
            bbox=pu.ai_ready_text_bbox(pad=0.25 if article_compact else 0.4),
        )

        if ax is None:
            return fig
        return current_ax
