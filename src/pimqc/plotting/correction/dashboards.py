"""Dashboard composition and legends for signal correction.

Standard dashboards, standalone legends, candidate appendices, and retained
experimental manuscript layouts are assembled from sibling panel modules.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from ...constants import DEFAULT_RANDOM_SEED
from ...processing.correction.algorithms import _format_correction_method_label
from .. import plot_utils as pu
from ..sample_structure import plot_sample_structure_change_map

FitPredictCallable = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
CorrectionModel = (
    RandomForestRegressor
    | TransformedTargetRegressor
    | Pipeline
    | FitPredictCallable
)


class CorrectionDashboardMixin:
    """Assemble standard and experimental correction dashboards."""

    def plot_correction_dashboard_legend(
        self,
        ax: plt.Axes,
        show_cv: bool = True,
        fontsize: float = pu.DEFAULT_LEGEND_FONTSIZE,
        title_fontsize: float = pu.DEFAULT_LEGEND_TITLE_FONTSIZE,
        article_compact: bool = False,
    ) -> plt.Axes:
        """Draw grouped score-component and correction-mode legends."""
        import matplotlib.patches as mpatches

        legend_linewidth = pu.DEFAULT_AXIS_LINEWIDTH

        score_handles = [
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex(
                    pu.PRIMARY_ACCENT_COLOR, alpha=1.0
                ),
                edgecolor="k",
                linewidth=legend_linewidth,
                label="Median QC-RSD improvement",
            ),
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex(
                    pu.PRIMARY_ACCENT_COLOR, alpha=0.67
                ),
                edgecolor="k",
                linewidth=legend_linewidth,
                label="Feature-wise QC-RSD improvement",
            ),
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=0.6),
                edgecolor="k",
                linewidth=legend_linewidth,
                label="Sample structure preservation",
            ),
        ]

        mode_handles = [
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=1.0),
                edgecolor="k",
                linewidth=legend_linewidth,
                label="Baseline",
            )
        ]
        if show_cv:
            mode_handles.append(
                mpatches.Patch(
                    facecolor=pu.get_equivalent_hex(
                        pu.PRIMARY_ACCENT_COLOR, alpha=0.33
                    ),
                    edgecolor="k",
                    linewidth=legend_linewidth,
                    linestyle="--",
                    label="OOF model",
                )
            )
        mode_handles.append(
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex(
                    pu.PRIMARY_ACCENT_COLOR, alpha=1.0
                ),
                edgecolor="k",
                linewidth=legend_linewidth,
                label="Full model",
            )
        )

        self._plot_grouped_standalone_legends(
            ax=ax,
            legend_groups=[
                ("Correction score components", score_handles),
                ("QC-RSD evaluation stage", mode_handles),
            ],
            loc="upper left",
            start_bbox=(0.0, 1.0),
            row_gap=0.04,
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

    def plot_correction_article_legend(
        self,
        ax: plt.Axes,
        show_oof: bool,
    ) -> plt.Axes:
        """Draw an experimental manuscript legend for correction.

        This revision-oriented interface is excluded from the default pipeline.
        """
        return self.plot_correction_dashboard_legend(
            ax=ax,
            show_cv=show_oof,
            fontsize=pu.ARTICLE_LEGEND_FONTSIZE,
            title_fontsize=pu.ARTICLE_LEGEND_TITLE_FONTSIZE,
            article_compact=True,
        )

    def plot_correction_article_dashboard(
        self,
        results_store: dict[str, dict[str, Any]],
        selected_method: str,
    ) -> object | None:
        """
        Create an experimental score-aligned correction manuscript panel.

        This revision-oriented interface is excluded from the default pipeline.
        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning(
                "patchworklib not found. Skipping correction article panel."
            )
            return None

        if selected_method not in results_store:
            return None

        pw.clear()
        selected_result = results_store[selected_method]
        panel_height = pu.ARTICLE_PANEL_HEIGHT_IN

        summary_ax = pw.Brick(
            figsize=pu.article_brick_size(1.85, panel_height),
            label="article_correction_summary",
        )
        self.plot_correction_score_summary(
            results_store=results_store,
            selected_method=selected_method,
            ax=summary_ax,
            show_legend=False,
        )
        self._apply_article_panel_format(
            summary_ax,
            title="Auto Correction Method Selection",
        )

        rsd_ax = pw.Brick(
            figsize=pu.article_brick_size(1.70, panel_height),
            label="article_correction_qc_rsd",
        )
        self.plot_corr_rsd(
            stage_dfs=selected_result["stage_dfs"],
            stage_oof_dfs=selected_result.get("stage_oof_dfs", {}),
            ax=rsd_ax,
            show_legend=False,
            article_compact=True,
        )
        self._apply_article_panel_format(
            rsd_ax,
            title="QC-RSD Distribution",
        )

        ecdf_ax = pw.Brick(
            figsize=pu.article_brick_size(1.70, panel_height),
            label="article_correction_featurewise",
        )
        self.plot_featurewise_qc_rsd_improvement_ecdf(
            result=selected_result,
            ax=ecdf_ax,
            article_compact=True,
        )
        self._apply_article_panel_format(
            ecdf_ax,
            title="Feature-wise QC-RSD Improvement",
        )
        ecdf_ax.set_xlabel("QC-RSD relative improvement")
        ecdf_ax.set_ylabel("Cumulative fraction")

        legend_ax = pw.Brick(
            figsize=pu.article_brick_size(1.30, panel_height),
            label="article_correction_legend",
        )
        self.plot_correction_article_legend(
            ax=legend_ax,
            show_oof=bool(selected_result.get("stage_oof_dfs")),
        )
        return summary_ax | rsd_ax | ecdf_ax | legend_ax

    def plot_correction_dashboard(
        self,
        results_store: dict[str, dict[str, Any]],
        selected_method: str,
        include_auto_summary: bool = True,
    ) -> object | None:
        """Combine correction selection and selected-method diagnostics."""
        try:
            import patchworklib as pw
        except ImportError:
            raise ImportError("patchworklib is required for this plot.")

        pw.clear()
        if not results_store:
            return None

        row1 = None
        if include_auto_summary:
            layout_width = 12.3
            summary_brick = pw.Brick(
                figsize=pu.dashboard_brick_size(5.0, 4.0, layout_width),
                label="correction_eval_summary",
            )
            self.plot_correction_score_summary(
                results_store=results_store,
                selected_method=selected_method,
                ax=summary_brick,
                show_legend=False,
            )
            structure_brick = pw.Brick(
                figsize=pu.dashboard_brick_size(4.8, 4.0, layout_width),
                label="correction_preservation_scorecard",
            )
            self.plot_correction_preservation_scorecard(
                results_store=results_store,
                selected_method=selected_method,
                ax=structure_brick,
            )
            legend_brick = pw.Brick(
                figsize=pu.dashboard_brick_size(2.5, 4.0, layout_width),
                label="correction_dashboard_legend",
            )
            self.plot_correction_dashboard_legend(ax=legend_brick)
            row1 = summary_brick | structure_brick | legend_brick

        if selected_method not in results_store:
            return row1

        selected_result = results_store[selected_method]
        layout_width = 12.0
        selected_rsd = pw.Brick(
            figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
            label="selected_correction_qc_rsd",
        )
        self.plot_corr_rsd(
            stage_dfs=selected_result["stage_dfs"],
            stage_oof_dfs=selected_result.get("stage_oof_dfs", {}),
            ax=selected_rsd,
            show_legend=not include_auto_summary,
            article_compact=True,
        )
        selected_rsd.set_title(
            "QC-RSD Distribution",
            fontsize=pu.DEFAULT_TITLE_FONTSIZE,
            fontweight="bold",
        )

        featurewise_ecdf = pw.Brick(
            figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
            label="selected_featurewise_qc_rsd_ecdf",
        )
        self.plot_featurewise_qc_rsd_improvement_ecdf(
            result=selected_result,
            ax=featurewise_ecdf,
            article_compact=True,
        )

        sample_structure = pw.Brick(
            figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
            label="selected_correction_sample_structure",
        )
        final_stage_df = list(selected_result["stage_dfs"].values())[-1]
        plot_sample_structure_change_map(
            ax=sample_structure,
            raw_obj=self.corr,
            transformed_obj=final_stage_df,
            structure_metrics=selected_result.get(
                "sample_structure_metrics", {}
            ),
            seed=int(
                self.corr.attrs.get("global_seed", DEFAULT_RANDOM_SEED)
            ),
            title="Sample Structure Change Map",
            compact_style=True,
        )

        row2 = selected_rsd | featurewise_ecdf | sample_structure
        return row1 / row2 if row1 is not None else row2

    def plot_correction_candidate_dashboard(
        self, results_store: dict[str, dict[str, Any]], selected_method: str
    ) -> object | None:
        """Assemble the QC-RSD appendix dashboard for AUTO candidates."""
        try:
            import patchworklib as pw
        except ImportError:
            raise ImportError("patchworklib is required for this plot.")

        pw.clear()
        if not results_store:
            return None

        panel_width = 3.7
        panel_height = 4.0
        layout_width = panel_width * 3.0
        bricks: dict[str, object] = {}
        method_rows = [
            ["QC-RLSC", "robust QC-RLSC", "QC-SVR"],
            ["SERRF", "RUV-III", "WaveICA 2.0"],
        ]
        detail_methods = [method for row in method_rows for method in row]
        shared_y_limits = self._resolve_dashboard_corr_rsd_ylim(results_store)

        for method in detail_methods:
            if method not in results_store:
                continue
            res = results_store[method]
            stage_dfs = res["stage_dfs"]
            stage_oof_dfs = res.get("stage_oof_dfs", {})
            safe_label = re.sub(r"[^A-Za-z0-9_]+", "_", f"rsd_box_{method}")

            b = pw.Brick(
                figsize=pu.dashboard_brick_size(
                    panel_width, panel_height, layout_width
                ),
                label=safe_label,
            )

            self.plot_corr_rsd(
                stage_dfs=stage_dfs,
                stage_oof_dfs=stage_oof_dfs,
                ax=b,
                show_legend=False,
                y_limits=shared_y_limits,
                article_compact=True,
            )

            method_label = _format_correction_method_label(method)
            title = (
                f"* {method_label}"
                if method == selected_method
                else method_label
            )
            b.set_title(
                title,
                fontsize=pu.DEFAULT_TITLE_FONTSIZE,
                fontweight="bold",
            )
            bricks[method] = b

        plot_rows = []
        for row_methods in method_rows:
            row_bricks = [
                bricks[method] for method in row_methods if method in bricks
            ]
            if not row_bricks:
                continue
            row = row_bricks[0]
            for brick in row_bricks[1:]:
                row = row | brick
            plot_rows.append(row)

        if not plot_rows:
            return None

        legend_brick = pw.Brick(
            figsize=pu.dashboard_brick_size(
                panel_width * 3.0, 0.55, layout_width
            ),
            label="correction_mode_legend",
        )
        self.plot_rsd_standalone_legend(
            ax=legend_brick,
            show_cv=True,
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            legend_cols=3,
        )

        grid_pw = plot_rows[0]
        for row in plot_rows[1:]:
            grid_pw = grid_pw / row

        # Patchworklib's default 0.5-inch operator margin is appropriate
        # between full plot rows but excessive for a shallow legend strip.
        # Tighten only this final composition and restore the global setting.
        previous_margin = pw.param.get("margin", 0.5)
        try:
            pw.param["margin"] = 0.10
            candidate_dashboard = grid_pw / legend_brick
        finally:
            pw.param["margin"] = previous_margin
        return candidate_dashboard
