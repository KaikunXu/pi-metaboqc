"""Diagnostic visualizations for missing-value imputation.

MetaboVisualizerImputer presents masked-value reconstruction, distribution
fidelity, candidate scores, sample-structure preservation, and fixed-method
outcomes. It converts metrics stored by the imputation stage into compact
figures and report panels without altering imputed values or selections.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from loguru import logger

from ...statistics import metrics as su
from ...visualization import plot_utils as pu
from ...core import model
from ...visualization import base as visualizer_classes


class MetaboVisualizerImputer(visualizer_classes.BaseMetaboVisualizer):
    """Visualization suite for evaluating imputation accuracy."""

    def __init__(
        self,
        raw_obj: model.MetaboInt,
        imp_obj: model.MetaboInt,
    ) -> None:
        """Initialize the imputation visualizer."""
        super().__init__(metabo_obj=imp_obj)
        self.raw_obj = raw_obj.astype(float).replace({0: np.nan})
        self.imp_obj = imp_obj.astype(float)

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

        def _evaluate_kde(values: np.ndarray, seed: int) -> np.ndarray:
            kde_values = values.copy()
            if np.nanstd(kde_values) < 1e-6:
                rng = np.random.default_rng(seed)
                kde_values += rng.normal(0.0, 1e-4, size=kde_values.size)
            return gaussian_kde(kde_values)(x_grid)

        truth_density = _evaluate_kde(truth, seed=123)
        reconstruction_density = _evaluate_kde(reconstruction, seed=456)
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
                ax.text(
                    0.96,
                    0.96,
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

        threshold = metrics.get("Threshold")
        if threshold is not None:
            current_ax.axvline(
                x=threshold,
                color="tab:gray",
                linestyle="--",
                linewidth=0.6 if article_compact else 1.0,
            )
            current_ax.axhline(
                y=threshold,
                color="tab:gray",
                linestyle="--",
                linewidth=0.6 if article_compact else 1.0,
            )

        nrmse_total = float(metrics.get("NRMSE_Total", np.nan))
        nrmse_low = float(metrics.get("NRMSE_Low", np.nan))
        annot_text = (
            f"NRMSE Total: {nrmse_total:.4f}\nNRMSE Low: {nrmse_low:.4f}"
        )

        current_ax.text(
            0.96,
            0.02,
            annot_text,
            transform=current_ax.transAxes,
            fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
            verticalalignment="bottom",
            horizontalalignment="right",
            clip_on=False,
            bbox=pu.ai_ready_text_bbox(pad=0.25 if article_compact else 0.4),
            zorder=10,
        )

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

        if ax is None:
            return fig
        return current_ax

    @staticmethod
    def _format_imputation_method_label(method_name: str) -> str:
        """Return a compact display label for an imputation method."""
        method_map = {
            "KNN": "KNN",
            "LLS": "LLS",
            "BPCA": "BPCA",
            "QRILC": "QRILC",
            "MINPROB": "MinProb",
            "PROB": "MinProb",
            "MEDIAN": "Median",
        }
        return method_map.get(str(method_name).upper(), str(method_name))

    @staticmethod
    def _method_key(method_name: str) -> str:
        """Normalize imputation method labels for robust matching."""
        return str(method_name).replace(" ", "").replace("-", "").upper()

    def plot_imputation_score_summary(
        self,
        results_dict: dict[
            str, tuple[dict[str, float], np.ndarray, np.ndarray]
        ],
        best_method: str,
        ax: plt.Axes | None = None,
        show_legend: bool = False,
    ) -> plt.Axes:
        """Plot MAR imputation AUTO score components."""
        try:
            import patchworklib as pw
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError("patchworklib is required for this plot.")

        if ax is None:
            current_ax = pw.Brick(
                figsize=(4, 4), label="imputation_nrmse_summary"
            )
        else:
            current_ax = ax

        summary_rows = []
        best_key = self._method_key(best_method)
        for method_name, (metrics, _, _) in results_dict.items():
            nrmse_total = float(metrics.get("NRMSE_Total", np.nan))
            summary_rows.append(
                {
                    "method": method_name,
                    "label": self._format_imputation_method_label(method_name),
                    "nrmse_total": nrmse_total,
                    "reconstruction_score": metrics.get("Reconstruction_Score"),
                    "distribution_preservation_score": metrics.get(
                        "Distribution_Preservation_Score"
                    ),
                    "sample_structure_score": metrics.get(
                        "Sample_Structure_Score"
                    ),
                    "auto_score": metrics.get("Auto_Score"),
                    "selected": self._method_key(method_name) == best_key,
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.replace([np.inf, -np.inf], np.nan)
        has_auto_score = summary_df["auto_score"].notna().any()
        if has_auto_score:
            summary_df = summary_df.dropna(subset=["auto_score"])
            summary_df = summary_df.sort_values(
                by=["auto_score", "nrmse_total", "label"],
                ascending=[False, True, True],
            ).reset_index(drop=True)
        else:
            summary_df = summary_df.dropna(subset=["nrmse_total"])
            summary_df = summary_df.sort_values(
                by=["nrmse_total", "label"], ascending=[False, True]
            ).reset_index(drop=True)

        if summary_df.empty:
            current_ax.axis("off")
            return current_ax

        y_pos = np.arange(len(summary_df))
        if has_auto_score:
            score_cols = [
                "reconstruction_score",
                "distribution_preservation_score",
                "sample_structure_score",
            ]
            weights = {
                "reconstruction_score": 0.65,
                "distribution_preservation_score": 0.20,
                "sample_structure_score": 0.15,
            }
            label_map = {
                "reconstruction_score": "Masked reconstruction",
                "distribution_preservation_score": "Distribution fidelity",
                "sample_structure_score": "Sample structure preservation",
            }
            color_map = {
                "reconstruction_score": pu.get_equivalent_hex(
                    pu.PRIMARY_ACCENT_COLOR, alpha=1.0
                ),
                "distribution_preservation_score": pu.get_equivalent_hex(
                    "tab:gray", alpha=0.75
                ),
                "sample_structure_score": pu.get_equivalent_hex(
                    "tab:gray", alpha=0.45
                ),
            }
            left = np.zeros(len(summary_df), dtype=float)
            for score_col in score_cols:
                left_start = left.copy()
                values = []
                for _, row in summary_df.iterrows():
                    available_weight = sum(
                        weights[col]
                        for col in score_cols
                        if np.isfinite(su.finite_or_nan(row.get(col)))
                    )
                    if available_weight <= 0:
                        values.append(0.0)
                        continue
                    score_value = np.clip(
                        su.finite_or_nan(row.get(score_col)), 0.0, 1.0
                    )
                    values.append(
                        score_value * weights[score_col] / available_weight
                    )
                values_arr = np.asarray(values, dtype=float)
                current_ax.barh(
                    y_pos,
                    values_arr,
                    left=left,
                    color=color_map[score_col],
                    edgecolor="k",
                    linewidth=0.5,
                    height=0.58,
                    label=label_map[score_col],
                )
                for y_idx, row in enumerate(summary_df.itertuples()):
                    score_value = su.finite_or_nan(getattr(row, score_col))
                    if values_arr[y_idx] < 0.10 or not np.isfinite(score_value):
                        continue
                    face_color = color_map[score_col]
                    current_ax.text(
                        left_start[y_idx] + values_arr[y_idx] / 2.0,
                        y_idx,
                        f"{score_value:.2f}",
                        va="center",
                        ha="center",
                        fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                        color=pu.get_contrast_color(face_color),
                        clip_on=True,
                    )
                left += values_arr

        else:
            selected_color = pu.get_equivalent_hex(
                pu.PRIMARY_ACCENT_COLOR, alpha=1.0
            )
            background_color = pu.get_equivalent_hex("tab:gray", alpha=0.75)
            bar_colors = [
                selected_color if bool(row.selected) else background_color
                for row in summary_df.itertuples()
            ]
            current_ax.barh(
                y_pos,
                summary_df["nrmse_total"],
                color=bar_colors,
                edgecolor="k",
                linewidth=0.5,
                height=0.58,
            )
            left = summary_df["nrmse_total"].to_numpy(dtype=float)

        y_labels = [
            f"* {row.label}" if bool(row.selected) else str(row.label)
            for row in summary_df.itertuples()
        ]
        current_ax.set_yticks(y_pos)
        current_ax.set_yticklabels(y_labels)
        current_ax.invert_yaxis()

        if has_auto_score:
            x_upper = float(np.nanmax(left)) if left.size else 1.0
            x_upper = min(1.08, max(x_upper + 0.08, x_upper * 1.10, 0.20))
            current_ax.set_xlim(0, x_upper)
            label_values = summary_df["auto_score"].to_numpy(dtype=float)
        else:
            xmax = float(summary_df["nrmse_total"].max())
            current_ax.set_xlim(0, xmax * 1.2 if xmax > 0 else 1)
            label_values = summary_df["nrmse_total"].to_numpy(dtype=float)

        for y_idx, row in enumerate(summary_df.itertuples()):
            value = float(label_values[y_idx])
            label_x = float(left[y_idx])
            current_ax.text(
                min(label_x + 0.015, current_ax.get_xlim()[1] * 0.97),
                y_idx,
                f"{value:.3f}" if has_auto_score else f"{value:.4f}",
                va="center",
                ha="left",
                fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
            )

        title = (
            "Auto Imputation Method Selection"
            if has_auto_score
            else "MAR Imputer Ranking"
        )
        if show_legend and has_auto_score:
            legend_handles = [
                mpatches.Patch(
                    facecolor=color_map[score_col],
                    edgecolor="k",
                    linewidth=0.5,
                    label=label_map[score_col],
                )
                for score_col in score_cols
            ]
            current_ax.legend(handles=legend_handles)
            self._format_single_legend(
                ax=current_ax,
                group_title="Imputation score components",
                loc="lower right",
                bbox_to_anchor=None,
                max_item_rows=6,
            )
        self._apply_standard_format(
            ax=current_ax,
            title=title,
            xlabel=(
                "Weighted contribution to overall score"
                if has_auto_score
                else "NRMSE Total"
            ),
            append_stage=False,
        )
        current_ax.tick_params(axis="y", length=0)
        return current_ax

    def plot_imputation_score_legend(
        self,
        ax: plt.Axes,
        legend_cols: int | None = None,
        fontsize: float = pu.DEFAULT_LEGEND_FONTSIZE,
        title_fontsize: float = pu.DEFAULT_LEGEND_TITLE_FONTSIZE,
        article_compact: bool = False,
    ) -> plt.Axes:
        """Draw a standalone legend for MAR imputation score components."""
        import matplotlib.patches as mpatches

        ax.axis("off")
        legend_linewidth = pu.DEFAULT_AXIS_LINEWIDTH
        handles = [
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex(
                    pu.PRIMARY_ACCENT_COLOR, alpha=1.0
                ),
                edgecolor="k",
                linewidth=legend_linewidth,
                label="Masked reconstruction",
            ),
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=0.75),
                edgecolor="k",
                linewidth=legend_linewidth,
                label="Distribution fidelity",
            ),
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=0.45),
                edgecolor="k",
                linewidth=legend_linewidth,
                label="Sample structure preservation",
            ),
        ]
        ax.legend(handles=handles)
        self._format_single_legend(
            ax=ax,
            group_title="Imputation score components",
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            legend_cols=legend_cols,
            max_item_rows=6,
            borderaxespad=0.0,
            handlelength=1.0 if article_compact else 1.8,
            handletextpad=0.3 if article_compact else 0.8,
            labelspacing=0.25 if article_compact else 0.5,
            borderpad=0.3 if article_compact else 0.4,
            fontsize=fontsize,
            title_fontsize=title_fontsize,
        )
        if article_compact:
            self._apply_article_legend_style(
                ax=ax,
                fontsize=fontsize,
                title_fontsize=title_fontsize,
            )
        return ax

    def plot_imputation_dashboard_legend(
        self,
        ax: plt.Axes,
        fontsize: float = pu.DEFAULT_LEGEND_FONTSIZE,
        title_fontsize: float = pu.DEFAULT_LEGEND_TITLE_FONTSIZE,
    ) -> plt.Axes:
        """Draw score-component and masked-density legends in one panel."""
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches

        score_handles = [
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex(
                    pu.PRIMARY_ACCENT_COLOR, alpha=1.0
                ),
                edgecolor="k",
                linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
                label="Masked reconstruction",
            ),
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=0.75),
                edgecolor="k",
                linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
                label="Distribution fidelity",
            ),
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=0.45),
                edgecolor="k",
                linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
                label="Sample structure preservation",
            ),
        ]
        density_handles = [
            mlines.Line2D(
                [],
                [],
                color="black",
                linestyle="--",
                linewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
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

        self._plot_grouped_standalone_legends(
            ax=ax,
            legend_groups=[
                ("Imputation score components", score_handles),
                ("Masked-value density reference", density_handles),
            ],
            loc="upper left",
            start_bbox=(0.0, 1.0),
            row_gap=0.04,
            max_item_rows=6,
            borderaxespad=0.0,
            fontsize=fontsize,
            title_fontsize=title_fontsize,
        )
        return ax

    def plot_imputation_article_score_legend(self, ax: plt.Axes) -> plt.Axes:
        """Draw a right-side score legend for the imputation article panel."""
        return self.plot_imputation_score_legend(
            ax=ax,
            fontsize=pu.ARTICLE_LEGEND_FONTSIZE,
            title_fontsize=pu.ARTICLE_LEGEND_TITLE_FONTSIZE,
            article_compact=True,
        )

    def plot_imputation_article_density_legend(self, ax: plt.Axes) -> plt.Axes:
        """Draw a right-side density legend for the imputation article panel."""
        import matplotlib.lines as mlines

        ax.axis("off")
        density_handles = [
            mlines.Line2D(
                [],
                [],
                color="black",
                linestyle="--",
                linewidth=0.75,
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
        ax.legend(handles=density_handles)
        self._format_single_legend(
            ax=ax,
            group_title="Masked-value density reference",
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            max_item_rows=6,
            borderaxespad=0.0,
            fontsize=pu.ARTICLE_LEGEND_FONTSIZE,
            title_fontsize=pu.ARTICLE_LEGEND_TITLE_FONTSIZE,
            handlelength=1.0,
            handletextpad=0.3,
            labelspacing=0.25,
            borderpad=0.3,
        )
        self._apply_article_legend_style(
            ax=ax,
            fontsize=pu.ARTICLE_LEGEND_FONTSIZE,
            title_fontsize=pu.ARTICLE_LEGEND_TITLE_FONTSIZE,
        )
        return ax

    def _resolve_article_benchmark_item(
        self,
        results_dict: dict[
            str, tuple[dict[str, float], np.ndarray, np.ndarray]
        ],
        best_method: str,
    ) -> tuple[str, tuple[dict[str, float], np.ndarray, np.ndarray]] | None:
        """
        Return the selected AUTO benchmark tuple without changing candidate
        order.
        """
        selected_key = self._method_key(best_method)
        for method_name, item in results_dict.items():
            if self._method_key(method_name) == selected_key:
                return method_name, item
        return next(iter(results_dict.items()), None)

    def plot_imputation_reconstruction_article_dashboard(
        self,
        results_dict: dict[
            str, tuple[dict[str, float], np.ndarray, np.ndarray]
        ],
        best_method: str,
    ) -> object | None:
        """
        Create a compact AUTO selection and masked-reconstruction manuscript
        panel.
        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning(
                "patchworklib not found. Skipping imputation article panel."
            )
            return None

        best_item = self._resolve_article_benchmark_item(
            results_dict, best_method
        )
        if best_item is None:
            return None

        method_name, (metrics, true_vals, pred_vals) = best_item
        pw.clear()
        panel_height = pu.ARTICLE_PANEL_HEIGHT_IN

        summary_ax = pw.Brick(
            figsize=pu.article_brick_size(1.85, panel_height),
            label="article_imputation_summary",
        )
        self.plot_imputation_score_summary(
            results_dict=results_dict,
            best_method=best_method,
            ax=summary_ax,
            show_legend=False,
        )
        self._apply_article_panel_format(
            summary_ax,
            title="Auto Imputation Method Selection",
        )

        scatter_ax = pw.Brick(
            figsize=pu.article_brick_size(1.85, panel_height),
            label="article_imputation_masked_nrmse",
        )
        self._plot_nrmse_scatter(
            true_vals=true_vals,
            pred_vals=pred_vals,
            metrics=metrics,
            method_name=self._format_imputation_method_label(method_name),
            compact_title=False,
            show_method_in_title=False,
            show_colorbar=False,
            article_compact=True,
            ax=scatter_ax,
        )
        self._apply_article_panel_format(
            scatter_ax,
            title="MAR Masked Simulation",
        )
        scatter_ax.set_xlabel("Known Masked Intensity (log2)")
        scatter_ax.set_ylabel("Reconstructed Intensity (log2)")

        legend_ax = pw.Brick(
            figsize=pu.article_brick_size(1.30, panel_height),
            label="article_imputation_score_legend",
        )
        self.plot_imputation_article_score_legend(ax=legend_ax)
        return summary_ax | scatter_ax | legend_ax

    def plot_imputation_preservation_article_dashboard(
        self,
        results_dict: dict[
            str, tuple[dict[str, float], np.ndarray, np.ndarray]
        ],
        best_method: str,
    ) -> object | None:
        """Create a compact fidelity and sample-structure manuscript panel."""
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning(
                "patchworklib not found. Skipping imputation article panel."
            )
            return None

        best_item = self._resolve_article_benchmark_item(
            results_dict, best_method
        )
        if best_item is None:
            return None

        _, (metrics, true_vals, pred_vals) = best_item
        pw.clear()
        panel_height = pu.ARTICLE_PANEL_HEIGHT_IN

        density_ax = pw.Brick(
            figsize=pu.article_brick_size(1.85, panel_height),
            label="article_imputation_density",
        )
        self._plot_masked_distribution_fidelity(
            true_vals=true_vals,
            pred_vals=pred_vals,
            metrics=metrics,
            ax=density_ax,
            compact_title=False,
            article_compact=True,
            show_legend=False,
        )
        self._apply_article_panel_format(
            density_ax,
            title="Masked-Value Distribution Fidelity",
        )

        sample_ax = pw.Brick(
            figsize=pu.article_brick_size(1.85, panel_height),
            label="article_imputation_sample_structure",
        )
        pu.plot_sample_structure_change_map(
            ax=sample_ax,
            raw_obj=self.raw_obj,
            transformed_obj=self.imp_obj,
            structure_metrics=metrics,
            title="Sample Structure Change Map",
            compact_style=True,
        )
        self._apply_article_panel_format(
            sample_ax,
            title="Sample Structure Change Map",
        )

        legend_ax = pw.Brick(
            figsize=pu.article_brick_size(1.30, panel_height),
            label="article_imputation_density_legend",
        )
        self.plot_imputation_article_density_legend(ax=legend_ax)
        return density_ax | sample_ax | legend_ax

    def plot_imputation_structure_metrics(
        self,
        results_dict: dict[
            str, tuple[dict[str, float], np.ndarray, np.ndarray]
        ],
        best_method: str,
        metric_group: str = "structure",
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot preservation metrics for MAR imputation candidates."""
        try:
            import patchworklib as pw
        except ImportError:
            raise ImportError("patchworklib is required for this plot.")

        if ax is None:
            current_ax = pw.Brick(
                figsize=(4.0, 4.0), label="imputation_structure"
            )
        else:
            current_ax = ax

        rows = []
        best_key = self._method_key(best_method)
        for method_name, (metrics, _, _) in results_dict.items():
            rows.append(
                {
                    "method": method_name,
                    "label": self._format_imputation_method_label(method_name),
                    "Jensen-Shannon preservation": metrics.get("JSD_Score"),
                    "Wasserstein preservation": metrics.get(
                        "Wasserstein_Score"
                    ),
                    "Trustworthiness": metrics.get("Trustworthiness"),
                    "Distance rank preservation": metrics.get(
                        "Distance_Rank_Preservation"
                    ),
                    "Distance scale preservation": metrics.get(
                        "Distance_Scale_Preservation"
                    ),
                    "auto_score": metrics.get("Auto_Score"),
                    "selected": self._method_key(method_name) == best_key,
                }
            )

        group_key = str(metric_group).lower().strip()
        if group_key in {"distribution", "dist"}:
            metric_cols = [
                "Jensen-Shannon preservation",
                "Wasserstein preservation",
            ]
            metric_labels = [
                "Jensen-Shannon\npreservation",
                "Wasserstein\npreservation",
            ]
            title = "Distribution Preservation"
        elif group_key in {"structure", "sample", "sample_structure"}:
            metric_cols = [
                "Trustworthiness",
                "Distance rank preservation",
                "Distance scale preservation",
            ]
            metric_labels = [
                "Trustworthiness",
                "Distance-rank\npreservation",
                "Distance-scale\npreservation",
            ]
            title = "Sample Structure Preservation"
        else:
            raise ValueError(
                "metric_group must be 'distribution' or 'structure'."
            )

        metric_df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
        for col in ["auto_score", *metric_cols]:
            metric_df[col] = pd.to_numeric(metric_df[col], errors="coerce")
        metric_df = metric_df.dropna(subset=metric_cols, how="all")
        metric_df = metric_df.sort_values(
            by=["auto_score", "label"], ascending=[False, True]
        ).reset_index(drop=True)

        if metric_df.empty:
            current_ax.axis("off")
            return current_ax

        matrix = metric_df[metric_cols].to_numpy(dtype=float)
        cmap = pu.score_heatmap_cmap()
        annot_size = pu.heatmap_annotation_fontsize(
            current_ax,
            n_rows=matrix.shape[0],
            n_cols=matrix.shape[1],
            default_size=pu.DEFAULT_SCORE_HEATMAP_ANNOTATION_FONTSIZE,
            max_size=pu.DEFAULT_SCORE_HEATMAP_ANNOTATION_FONTSIZE,
            min_size=4.0,
        )
        current_ax.imshow(
            np.ma.masked_invalid(matrix),
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
        )
        current_ax.set_xticks(np.arange(len(metric_cols)))
        current_ax.set_xticklabels(metric_labels)
        current_ax.set_yticks(np.arange(len(metric_df)))
        current_ax.set_yticklabels(
            [
                f"* {row.label}" if bool(row.selected) else str(row.label)
                for row in metric_df.itertuples()
            ]
        )
        current_ax.set_xticks(np.arange(-0.5, len(metric_cols), 1), minor=True)
        current_ax.set_yticks(np.arange(-0.5, len(metric_df), 1), minor=True)
        grid_lw = pu.DEFAULT_HEATMAP_CELL_LINEWIDTH
        current_ax.grid(
            which="minor", color="k", linestyle="-", linewidth=grid_lw
        )
        current_ax.tick_params(which="minor", bottom=False, left=False)

        for y_idx in range(matrix.shape[0]):
            for x_idx in range(matrix.shape[1]):
                value = matrix[y_idx, x_idx]
                if not np.isfinite(value):
                    label = "NA"
                    color = "0.35"
                else:
                    label = f"{value:.2f}"
                    color = pu.get_contrast_color(cmap(value))
                current_ax.text(
                    x_idx,
                    y_idx,
                    label,
                    ha="center",
                    va="center",
                    fontsize=annot_size,
                    color=color,
                )

        self._apply_standard_format(
            ax=current_ax,
            title=title,
            xlabel="",
            ylabel="",
            append_stage=False,
        )
        pu.rotate_xticks_if_overlapping(current_ax)
        for spine in current_ax.spines.values():
            spine.set_visible(False)
        current_ax.tick_params(axis="both", length=0)
        return current_ax

    def plot_imputation_preservation_scorecard(
        self,
        results_dict: dict[
            str, tuple[dict[str, float], np.ndarray, np.ndarray]
        ],
        best_method: str,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """
        Plot distribution and sample-structure preservation scores together.
        """
        try:
            import patchworklib as pw
        except ImportError:
            raise ImportError("patchworklib is required for this plot.")

        if ax is None:
            current_ax = pw.Brick(
                figsize=(4.8, 4.0), label="imputation_scorecard"
            )
        else:
            current_ax = ax

        best_key = self._method_key(best_method)
        rows = []
        for method_name, (metrics, _, _) in results_dict.items():
            rows.append(
                {
                    "method": method_name,
                    "label": self._format_imputation_method_label(method_name),
                    "Jensen-Shannon preservation": metrics.get("JSD_Score"),
                    "Wasserstein preservation": metrics.get(
                        "Wasserstein_Score"
                    ),
                    "Trustworthiness": metrics.get("Trustworthiness"),
                    "Distance rank preservation": metrics.get(
                        "Distance_Rank_Preservation"
                    ),
                    "Distance scale preservation": metrics.get(
                        "Distance_Scale_Preservation"
                    ),
                    "auto_score": metrics.get("Auto_Score"),
                    "selected": self._method_key(method_name) == best_key,
                }
            )

        metric_cols = [
            "Jensen-Shannon preservation",
            "Wasserstein preservation",
            "Trustworthiness",
            "Distance rank preservation",
            "Distance scale preservation",
        ]
        metric_labels = [
            "Jensen-Shannon\npreservation",
            "Wasserstein\npreservation",
            "Trustworthiness",
            "Distance-rank\npreservation",
            "Distance-scale\npreservation",
        ]
        metric_df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
        for col in ["auto_score", *metric_cols]:
            metric_df[col] = pd.to_numeric(metric_df[col], errors="coerce")
        metric_df = metric_df.dropna(subset=metric_cols, how="all")
        metric_df = metric_df.sort_values(
            by=["auto_score", "label"], ascending=[False, True]
        ).reset_index(drop=True)

        if metric_df.empty:
            current_ax.axis("off")
            return current_ax

        matrix = metric_df[metric_cols].to_numpy(dtype=float)
        cmap = pu.score_heatmap_cmap()
        annot_size = pu.heatmap_annotation_fontsize(
            current_ax,
            n_rows=matrix.shape[0],
            n_cols=matrix.shape[1],
            default_size=pu.DEFAULT_SCORE_HEATMAP_ANNOTATION_FONTSIZE,
            max_size=pu.DEFAULT_SCORE_HEATMAP_ANNOTATION_FONTSIZE,
            min_size=4.0,
        )
        current_ax.imshow(
            np.ma.masked_invalid(matrix),
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
        )
        current_ax.set_xticks(np.arange(len(metric_cols)))
        current_ax.set_xticklabels(metric_labels)
        current_ax.set_yticks(np.arange(len(metric_df)))
        current_ax.set_yticklabels(
            [
                f"* {row.label}" if bool(row.selected) else str(row.label)
                for row in metric_df.itertuples()
            ]
        )
        n_rows, n_cols = matrix.shape
        for x_pos in np.arange(-0.5, n_cols, 1.0):
            current_ax.plot(
                [x_pos, x_pos],
                [-0.5, n_rows - 0.5],
                color="k",
                linewidth=pu.DEFAULT_HEATMAP_CELL_LINEWIDTH,
                zorder=3,
            )
        for y_pos in np.arange(-0.5, n_rows, 1.0):
            current_ax.plot(
                [-0.5, n_cols - 0.5],
                [y_pos, y_pos],
                color="k",
                linewidth=pu.DEFAULT_HEATMAP_CELL_LINEWIDTH,
                zorder=3,
            )

        for y_idx in range(matrix.shape[0]):
            for x_idx in range(matrix.shape[1]):
                value = matrix[y_idx, x_idx]
                if not np.isfinite(value):
                    label = "NA"
                    color = "0.35"
                else:
                    label = f"{value:.2f}"
                    color = pu.get_contrast_color(cmap(value))
                current_ax.text(
                    x_idx,
                    y_idx,
                    label,
                    ha="center",
                    va="center",
                    fontsize=annot_size,
                    color=color,
                )

        dist_color = pu.get_equivalent_hex("tab:gray", alpha=0.75)
        struct_color = pu.get_equivalent_hex("tab:gray", alpha=0.45)
        group_specs = [
            (-0.5, 2.0, "Distribution fidelity", dist_color),
            (1.5, 3.0, "Sample structure preservation", struct_color),
        ]
        for x_start, width, label, face_color in group_specs:
            current_ax.add_patch(
                plt.Rectangle(
                    (x_start, -1.05),
                    width,
                    0.38,
                    facecolor=face_color,
                    edgecolor="k",
                    linewidth=pu.DEFAULT_HEATMAP_CELL_LINEWIDTH,
                    zorder=5,
                    clip_on=False,
                )
            )
            current_ax.text(
                x_start + width / 2.0,
                -0.86,
                label,
                ha="center",
                va="center",
                fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                color=pu.get_contrast_color(face_color),
                zorder=6,
                clip_on=False,
            )

        current_ax.set_ylim(len(metric_df) - 0.5, -1.18)
        self._apply_standard_format(
            ax=current_ax,
            title="Candidate Preservation Scorecard",
            xlabel="",
            ylabel="",
            append_stage=False,
            tick_fontsize=pu.DEFAULT_AXIS_TICK_FONTSIZE,
        )
        pu.rotate_xticks_if_overlapping(current_ax)
        for spine in current_ax.spines.values():
            spine.set_visible(False)
        current_ax.tick_params(axis="both", length=0)
        return current_ax

    def plot_imputation_auto_dashboard(
        self,
        results_dict: dict[
            str, tuple[dict[str, float], np.ndarray, np.ndarray]
        ],
        best_method: str,
    ) -> object | None:
        """Create the final MAR imputation Auto-selection dashboard."""
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("Module 'patchworklib' not found. Skipping grid.")
            return None

        if not results_dict:
            return None

        g_min, g_max = float("inf"), float("-inf")
        sorted_items = sorted(
            results_dict.items(),
            key=lambda item: (
                float(item[1][0].get("Auto_Score", np.nan)),
                -float(item[1][0].get("NRMSE_Total", np.nan)),
                self._format_imputation_method_label(item[0]),
            ),
            reverse=True,
        )

        for _, (_, true_vals, pred_vals) in sorted_items:
            g_min = min(g_min, true_vals.min(), pred_vals.min())
            g_max = max(g_max, true_vals.max(), pred_vals.max())

        margin = (g_max - g_min) * 0.05
        shared_lims = (g_min - margin, g_max + margin)
        best_key = self._method_key(best_method)
        best_item = next(
            (
                item
                for item in sorted_items
                if self._method_key(item[0]) == best_key
            ),
            sorted_items[0],
        )

        pw.clear()
        layout_width = 12.5
        ax_summary = pw.Brick(
            figsize=pu.dashboard_brick_size(4.3, 4.0, layout_width),
            label="imputation_score_summary",
        )
        self.plot_imputation_score_summary(
            results_dict=results_dict,
            best_method=best_method,
            ax=ax_summary,
        )
        ax_scorecard = pw.Brick(
            figsize=pu.dashboard_brick_size(6.0, 4.0, layout_width),
            label="imputation_scorecard",
        )
        self.plot_imputation_preservation_scorecard(
            results_dict=results_dict,
            best_method=best_method,
            ax=ax_scorecard,
        )
        ax_legend = pw.Brick(
            figsize=pu.dashboard_brick_size(2.2, 4.0, layout_width),
            label="imputation_dashboard_legend",
        )
        self.plot_imputation_dashboard_legend(ax=ax_legend)

        method_name, (metrics, true_vals, pred_vals) = best_item
        ax_best_scatter = pw.Brick(
            figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
            label="best_nrmse_scatter",
        )
        self._plot_nrmse_scatter(
            true_vals,
            pred_vals,
            metrics,
            method_name=self._format_imputation_method_label(method_name),
            axis_lims=shared_lims,
            compact_title=False,
            show_method_in_title=False,
            article_compact=True,
            ax=ax_best_scatter,
        )

        ax_density = pw.Brick(
            figsize=pu.dashboard_brick_size(4.25, 4.0, layout_width),
            label="imputation_masked_density",
        )
        self._plot_masked_distribution_fidelity(
            true_vals=true_vals,
            pred_vals=pred_vals,
            metrics=metrics,
            ax=ax_density,
            compact_title=False,
            article_compact=True,
        )
        ax_sample_structure = pw.Brick(
            figsize=pu.dashboard_brick_size(4.25, 4.0, layout_width),
            label="imputation_sample_structure_preservation",
        )
        pu.plot_sample_structure_change_map(
            ax=ax_sample_structure,
            raw_obj=self.raw_obj,
            transformed_obj=self.imp_obj,
            structure_metrics=metrics,
            title="Sample Structure Change Map",
            compact_style=True,
        )

        top_row = ax_summary | ax_scorecard | ax_legend
        diagnostic_row = ax_best_scatter | ax_density | ax_sample_structure

        return top_row / diagnostic_row

    def plot_imputation_method_dashboard(
        self,
        metrics: dict[str, float],
        true_vals: np.ndarray,
        pred_vals: np.ndarray,
        method_name: str,
    ) -> object | None:
        """
        Create a fixed-method imputation dashboard without Auto score panels.
        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("Module 'patchworklib' not found. Skipping grid.")
            return None

        if true_vals is None or pred_vals is None or len(true_vals) == 0:
            return None

        d_min = min(float(np.nanmin(true_vals)), float(np.nanmin(pred_vals)))
        d_max = max(float(np.nanmax(true_vals)), float(np.nanmax(pred_vals)))
        margin = (d_max - d_min) * 0.05 if d_max > d_min else 1.0
        shared_lims = (d_min - margin, d_max + margin)

        pw.clear()
        layout_width = 13.2
        ax_scatter = pw.Brick(
            figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
            label="method_nrmse_scatter",
        )
        self._plot_nrmse_scatter(
            true_vals,
            pred_vals,
            metrics,
            method_name=self._format_imputation_method_label(method_name),
            axis_lims=shared_lims,
            compact_title=False,
            show_method_in_title=False,
            article_compact=True,
            ax=ax_scatter,
        )

        ax_density = pw.Brick(
            figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
            label="method_masked_density",
        )
        self._plot_masked_distribution_fidelity(
            true_vals=true_vals,
            pred_vals=pred_vals,
            metrics=metrics,
            ax=ax_density,
            compact_title=False,
            article_compact=True,
        )

        ax_sample_structure = pw.Brick(
            figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
            label="method_sample_structure_preservation",
        )
        pu.plot_sample_structure_change_map(
            ax=ax_sample_structure,
            raw_obj=self.raw_obj,
            transformed_obj=self.imp_obj,
            structure_metrics=metrics,
            title="Sample Structure Change Map",
            compact_style=True,
        )

        ax_legend = pw.Brick(
            figsize=pu.dashboard_brick_size(1.2, 8.0, layout_width),
            label="method_kde_legend",
        )
        self._plot_kde_standalone_legend(
            ax=ax_legend,
            legend_cols=1,
            loc="center left",
            bbox_to_anchor=(0.0, 0.5),
        )

        return ax_scatter | ax_density | ax_sample_structure | ax_legend

    def plot_imputation_nrmse_appendix_grid(
        self,
        results_dict: dict[
            str, tuple[dict[str, float], np.ndarray, np.ndarray]
        ],
    ) -> object | None:
        """
        Create a 2 x 3 appendix grid of candidate masked-reconstruction plots.
        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("Module 'patchworklib' not found. Skipping grid.")
            return None

        if not results_dict:
            return None

        sorted_items = sorted(
            results_dict.items(),
            key=lambda item: (
                float(item[1][0].get("Auto_Score", np.nan)),
                -float(item[1][0].get("NRMSE_Total", np.nan)),
                self._format_imputation_method_label(item[0]),
            ),
            reverse=True,
        )
        g_min, g_max = float("inf"), float("-inf")
        for _, (_, true_vals, pred_vals) in sorted_items:
            g_min = min(g_min, true_vals.min(), pred_vals.min())
            g_max = max(g_max, true_vals.max(), pred_vals.max())

        margin = (g_max - g_min) * 0.05
        shared_lims = (g_min - margin, g_max + margin)
        best_key = self._method_key(sorted_items[0][0])

        pw.clear()
        scatter_bricks: list[object] = []
        for idx, (method_name, (metrics, true_vals, pred_vals)) in enumerate(
            sorted_items[:6]
        ):
            ax_scatter = pw.Brick(
                figsize=pu.dashboard_brick_size(3.6, 3.6, 10.8),
                label=f"nrmse_appendix_scatter_{idx + 1}",
            )
            display_method = self._format_imputation_method_label(method_name)
            if self._method_key(method_name) == best_key:
                display_method = f"* {display_method}"
            self._plot_nrmse_scatter(
                true_vals,
                pred_vals,
                metrics,
                method_name=display_method,
                axis_lims=shared_lims,
                compact_title=False,
                ax=ax_scatter,
            )
            scatter_bricks.append(ax_scatter)

        while len(scatter_bricks) < 6:
            ax_blank = pw.Brick(
                figsize=pu.dashboard_brick_size(3.6, 3.6, 10.8),
                label=f"nrmse_appendix_scatter_blank_{len(scatter_bricks)}",
            )
            ax_blank.axis("off")
            scatter_bricks.append(ax_blank)

        return (scatter_bricks[0] | scatter_bricks[1] | scatter_bricks[2]) / (
            scatter_bricks[3] | scatter_bricks[4] | scatter_bricks[5]
        )

    def plot_imputation_density_overlay(
        self,
        true_vals: np.ndarray,
        pred_vals: np.ndarray,
        metrics: dict[str, float] | None = None,
    ) -> object | None:
        """Create a standalone score-aligned masked-density fidelity panel."""
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping density overlay.")
            return None

        pw.clear()

        layout_width = 7.2
        ax_density = pw.Brick(
            figsize=pu.dashboard_brick_size(5.0, 4.0, layout_width),
            label="masked_density_fidelity",
        )
        self._plot_masked_distribution_fidelity(
            true_vals=true_vals,
            pred_vals=pred_vals,
            metrics=metrics,
            ax=ax_density,
            compact_title=False,
        )
        ax_legend = pw.Brick(
            figsize=pu.dashboard_brick_size(2.2, 4.0, layout_width),
            label="masked_density_legend",
        )
        self._plot_kde_standalone_legend(ax=ax_legend, legend_cols=1)

        return ax_density | ax_legend
