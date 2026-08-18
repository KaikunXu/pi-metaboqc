"""Dashboard composition and legends for normalization.

Standard dashboards, standalone legends, and retained experimental manuscript
layouts are assembled from the diagnostic and scorecard sibling modules.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from loguru import logger

from .. import plot_utils as pu


class NormalizationDashboardMixin:
    """Assemble standard and experimental normalization dashboards."""

    def plot_normalization_dashboard_legend(
        self,
        ax: plt.Axes,
        fontsize: float = pu.DEFAULT_LEGEND_FONTSIZE,
        title_fontsize: float = pu.DEFAULT_LEGEND_TITLE_FONTSIZE,
        article_compact: bool = False,
        layout_cols: int = 1,
    ) -> plt.Axes:
        """Draw grouped score-component and stage legends for the dashboard."""
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches

        legend_linewidth = pu.DEFAULT_AXIS_LINEWIDTH
        line_width = pu.DEFAULT_GUIDE_LINEWIDTH
        marker_size = pu.DEFAULT_LEGEND_MARKER_SIZE
        score_cols, label_map, color_map = (
            self._normalization_score_component_style()
        )
        score_handles = [
            mpatches.Patch(
                facecolor=color_map[col],
                edgecolor="k",
                linewidth=legend_linewidth,
                label=label_map[col],
            )
            for col in score_cols
        ]
        rle_handles = [
            mpatches.Patch(
                facecolor=self.pal["Before Norm"],
                edgecolor="k",
                linewidth=legend_linewidth,
                label="Before Norm",
            ),
            mpatches.Patch(
                facecolor=self.pal["After Norm"],
                edgecolor="k",
                linewidth=legend_linewidth,
                label="After Norm",
            ),
        ]
        variance_handles = [
            mlines.Line2D(
                [],
                [],
                color=self.pal["Before Norm"],
                linestyle="--",
                marker="o",
                linewidth=line_width,
                markersize=marker_size,
                label="Before Norm",
            ),
            mlines.Line2D(
                [],
                [],
                color=self.pal["After Norm"],
                linestyle="-",
                marker="o",
                linewidth=line_width,
                markersize=marker_size,
                label="After Norm",
            ),
        ]
        distance_handles = [
            mlines.Line2D(
                [0],
                [0],
                color=self.pal["Before Norm"],
                marker="o",
                linestyle="",
                markeredgecolor="k",
                markeredgewidth=0.5 if article_compact else 0.25,
                markersize=marker_size,
                label="Before Norm",
            ),
            mlines.Line2D(
                [0],
                [0],
                color=self.pal["After Norm"],
                marker="o",
                linestyle="",
                markeredgecolor="k",
                markeredgewidth=0.5 if article_compact else 0.25,
                markersize=marker_size,
                label="After Norm",
            ),
        ]

        self._plot_grouped_standalone_legends(
            ax=ax,
            legend_groups=[
                ("Normalization score components", score_handles),
                ("QC RLE alignment stage", rle_handles),
                ("QC variance stabilization stage", variance_handles),
                ("QC structure distance stage", distance_handles),
            ],
            loc="upper left",
            start_bbox=(0.0, 1.0),
            row_gap=0.035,
            layout_cols=layout_cols,
            column_gap=0.12,
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

    def plot_normalization_article_legend(self, ax: plt.Axes) -> plt.Axes:
        """
        Draw an experimental manuscript legend for normalization. This
        revision-oriented interface is excluded from the default pipeline.
        """
        return self.plot_normalization_dashboard_legend(
            ax=ax,
            fontsize=pu.ARTICLE_LEGEND_FONTSIZE,
            title_fontsize=pu.ARTICLE_LEGEND_TITLE_FONTSIZE,
            article_compact=True,
            layout_cols=1,
        )

    def plot_normalization_dashboard(self) -> object | None:
        """
        Combine score-aligned normalization diagnostics into a PW dashboard.
        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping dashboard.")
            return None

        pw.clear()

        auto_summary = self.norm.attrs.get("selection", {}).get(
            "candidate_summary"
        )
        is_auto = bool(auto_summary)

        if is_auto:
            layout_width = 13.7
            ax_auto = self.plot_normalization_score_summary(
                auto_summary=auto_summary,
                figsize=pu.dashboard_brick_size(6.2, 4.0, layout_width),
                show_legend=False,
            )
            if ax_auto is None:
                ax_auto = pw.Brick(
                    figsize=pu.dashboard_brick_size(4.5, 4.0, layout_width),
                    label="Auto_Score_Spacer",
                )
                ax_auto.axis("off")

            ax_scorecard = pw.Brick(
                figsize=pu.dashboard_brick_size(4.9, 4.0, layout_width),
                label="Norm_Preservation_Scorecard",
            )
            self.plot_normalization_preservation_scorecard(
                auto_summary=auto_summary,
                ax=ax_scorecard,
            )
            ax_legend = pw.Brick(
                figsize=pu.dashboard_brick_size(2.6, 4.0, layout_width),
                label="normalization_dashboard_legend",
            )
            self.plot_normalization_dashboard_legend(ax=ax_legend)
            row1 = ax_auto | ax_scorecard | ax_legend

            ax_qc_variance = pw.Brick(
                figsize=pu.dashboard_brick_size(3.0, 4.0, layout_width),
                label="QC_Variance",
            )
            self._plot_qc_variance_stabilization(
                ax=ax_qc_variance,
                show_legend=False,
                article_compact=True,
            )
            ax_qc_structure = pw.Brick(
                figsize=pu.dashboard_brick_size(3.0, 4.0, layout_width),
                label="QC_Structure",
            )
            self._plot_qc_structure_improvement(
                ax=ax_qc_structure,
                show_legend=False,
                article_compact=True,
            )
            ax_sample_structure = pw.Brick(
                figsize=pu.dashboard_brick_size(3.0, 4.0, layout_width),
                label="Sample_Structure",
            )
            self._plot_sample_structure_preservation(
                ax_geom=ax_sample_structure, compact_style=True
            )
            ax_qc_alignment = pw.Brick(
                figsize=pu.dashboard_brick_size(3.0, 4.0, layout_width),
                label="QC_Alignment",
            )
            self._plot_qc_rle_boxplot(
                ax=ax_qc_alignment,
                show_legend=False,
                article_compact=True,
            )
            row2 = (
                ax_qc_alignment
                | ax_qc_variance
                | ax_qc_structure
                | ax_sample_structure
            )

            return row1 / row2

        layout_width = 8.0
        ax_qc_alignment = pw.Brick(
            figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
            label="QC_Alignment",
        )
        self._plot_qc_rle_boxplot(ax=ax_qc_alignment, article_compact=True)
        ax_qc_variance = pw.Brick(
            figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
            label="QC_Variance",
        )
        self._plot_qc_variance_stabilization(
            ax=ax_qc_variance, article_compact=True
        )
        row1 = ax_qc_alignment | ax_qc_variance

        ax_qc_structure = pw.Brick(
            figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
            label="QC_Structure",
        )
        self._plot_qc_structure_improvement(
            ax=ax_qc_structure, article_compact=True
        )
        ax_sample_structure = pw.Brick(
            figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
            label="Sample_Structure",
        )
        self._plot_sample_structure_preservation(
            ax_geom=ax_sample_structure, compact_style=True
        )
        row2 = ax_qc_structure | ax_sample_structure

        return row1 / row2

    def plot_normalization_article_dashboard(self) -> object | None:
        """Create an experimental AUTO normalization manuscript panel.

        This revision-oriented interface is excluded from the default pipeline.
        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning(
                "patchworklib not found. Skipping normalization article panel."
            )
            return None

        auto_summary = self.norm.attrs.get("selection", {}).get(
            "candidate_summary"
        )
        if not auto_summary:
            return None

        pw.clear()
        panel_height = pu.ARTICLE_PANEL_HEIGHT_IN

        summary_ax = self.plot_normalization_score_summary(
            auto_summary=auto_summary,
            figsize=pu.article_brick_size(1.42, panel_height),
            show_legend=False,
        )
        if summary_ax is None:
            summary_ax = pw.Brick(
                figsize=pu.article_brick_size(1.42, panel_height),
                label="article_normalization_summary",
            )
            summary_ax.axis("off")
        self._apply_article_panel_format(
            summary_ax,
            title="Auto Normalization Method Selection",
        )

        rle_ax = pw.Brick(
            figsize=pu.article_brick_size(1.20, panel_height),
            label="article_normalization_rle",
        )
        self._plot_qc_rle_boxplot(
            ax=rle_ax,
            show_legend=False,
            article_compact=True,
        )
        self._apply_article_panel_format(
            rle_ax,
            title="QC RLE\nAlignment Change",
        )

        variance_ax = pw.Brick(
            figsize=pu.article_brick_size(1.20, panel_height),
            label="article_normalization_variance",
        )
        self._plot_qc_variance_stabilization(
            ax=variance_ax,
            show_legend=False,
            article_compact=True,
        )
        self._apply_article_panel_format(
            variance_ax,
            title="QC Variance Stabilization",
        )
        variance_ax.set_xlabel("Mean QC log2 Intensity")
        variance_ax.set_ylabel("QC dispersion")

        structure_ax = pw.Brick(
            figsize=pu.article_brick_size(1.20, panel_height),
            label="article_normalization_structure",
        )
        self._plot_qc_structure_improvement(
            ax=structure_ax,
            show_legend=False,
            article_compact=True,
        )
        self._apply_article_panel_format(
            structure_ax,
            title="QC Structure Distance Change",
        )
        structure_ax.set_ylabel("QC distance (log scale)")

        legend_ax = pw.Brick(
            figsize=pu.article_brick_size(1.30, panel_height),
            label="article_normalization_legend",
        )
        self.plot_normalization_article_legend(ax=legend_ax)
        return summary_ax | rle_ax | variance_ax | structure_ax | legend_ax
