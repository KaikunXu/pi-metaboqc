"""Candidate ranking and preservation scorecards for normalization.

The module maps normalization-specific metrics and labels onto the shared
score renderers; dashboard composition remains in ``dashboards.py``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from typing import Any

from .. import plot_utils as pu
from ..components import (
    CandidateScoreRenderer,
    MetricScorecardRenderer,
    ScoreComponentSpec,
    ScorecardMetricSpec,
)
from ...processing.normalization.methods import NORMALIZATION_METHODS


class NormalizationScorecardMixin:
    """Render normalization candidate scores and preservation metrics."""

    @staticmethod
    def _normalization_score_component_style() -> tuple[
        list[str],
        dict[str, str],
        dict[str, str],
    ]:
        """Return the score-component ordering, labels, and colors."""
        score_cols = [
            "rle_alignment_change_score",
            "variance_stabilization_score",
            "qc_structure_change_score",
            "sample_structure_score",
        ]
        label_map = {
            "rle_alignment_change_score": "QC RLE alignment change",
            "variance_stabilization_score": "QC variance stabilization",
            "qc_structure_change_score": "QC structure distance change",
            "sample_structure_score": "Sample structure preservation",
        }
        color_map = {
            "rle_alignment_change_score": pu.get_equivalent_hex(
                pu.PRIMARY_ACCENT_COLOR, alpha=1.0
            ),
            "variance_stabilization_score": pu.get_equivalent_hex(
                pu.PRIMARY_ACCENT_COLOR, alpha=0.67
            ),
            "qc_structure_change_score": pu.get_equivalent_hex(
                pu.PRIMARY_ACCENT_COLOR, alpha=0.33
            ),
            "sample_structure_score": pu.get_equivalent_hex(
                "tab:gray", alpha=0.5
            ),
        }
        return score_cols, label_map, color_map

    def plot_normalization_score_summary(
        self,
        auto_summary: list[dict[str, Any]] | pd.DataFrame | None = None,
        figsize: tuple[float, float] = (4.0, 4.0),
        show_legend: bool = True,
    ) -> object | None:
        """Plot Auto normalization weighted score components as stacked bars."""
        if auto_summary is None:
            auto_summary = self.norm.attrs.get("selection", {}).get(
                "candidate_summary"
            )
        if not auto_summary:
            return None

        try:
            import patchworklib as pw
        except ImportError as e:
            logger.warning(f"Skipping Auto normalization stacked bar plot: {e}")
            return None

        summary_df = pd.DataFrame(auto_summary).copy()
        if summary_df.empty:
            return None

        score_cols, label_map, color_map = (
            self._normalization_score_component_style()
        )
        contribution_weights = self.norm._AUTO_SCORE_COMPONENT_WEIGHTS
        for col in ["overall_score", *score_cols]:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")

        plot_df = summary_df.copy()
        plot_df["status"] = plot_df["status"].fillna("failed")
        scoreable_mask = (
            plot_df["status"].eq("ok") & plot_df["overall_score"].notna()
        )
        if not scoreable_mask.any():
            return None

        plot_df["_sort_score"] = plot_df["overall_score"].fillna(-1.0)
        plot_df = plot_df.sort_values(
            by=["_sort_score", "method"], ascending=[False, True]
        ).reset_index(drop=True)

        plot_df["label"] = plot_df["method"].map(
            NORMALIZATION_METHODS.display_name
        )
        renderer = CandidateScoreRenderer(
            [
                ScoreComponentSpec(
                    column=column,
                    label=label_map[column],
                    color=color_map[column],
                    weight=contribution_weights[column],
                    annotation_threshold=0.09,
                )
                for column in score_cols
            ]
        )
        ax = pw.Brick(figsize=figsize, label="auto_norm_stacked_bar")
        renderer.render(
            ax,
            plot_df,
            total_column="overall_score",
            status_column="status",
            scale_to_total=True,
            bar_height=0.62,
        )

        if show_legend:
            ax.legend(
                handles=renderer.legend_handles(),
                loc="lower right",
                bbox_to_anchor=None,
            )
            self._format_single_legend(
                ax,
                loc="lower right",
                bbox_to_anchor=None,
                group_title="Normalization score components",
                max_item_rows=6,
            )

        self._apply_standard_format(
            ax=ax,
            title="Auto Normalization Method Selection",
            xlabel="Weighted contribution to overall score",
            append_stage=False,
        )
        return ax

    def plot_normalization_preservation_scorecard(
        self,
        auto_summary: list[dict[str, Any]] | pd.DataFrame | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Axes | None:
        """Plot candidate-level sample-structure preservation scores."""
        if auto_summary is None:
            auto_summary = self.norm.attrs.get("selection", {}).get(
                "candidate_summary"
            )
        if not auto_summary:
            return None

        try:
            import patchworklib as pw
        except ImportError as e:
            logger.warning(f"Skipping Auto normalization scorecard plot: {e}")
            return None

        current_ax = (
            pw.Brick(
                figsize=pu.dashboard_brick_size(4.8, 4.0, 8.0),
                label="normalization_preservation_scorecard",
            )
            if ax is None
            else ax
        )
        score_cols = [
            "sample_structure_trustworthiness",
            "sample_structure_rank_preservation",
            "sample_structure_scale_preservation",
        ]
        metric_labels = [
            "Trustworthiness",
            "Distance-rank\npreservation",
            "Distance-scale\npreservation",
        ]

        score_df = pd.DataFrame(auto_summary).copy()
        if score_df.empty:
            current_ax.axis("off")
            return current_ax

        for col in ["overall_score", *score_cols]:
            score_df[col] = pd.to_numeric(score_df[col], errors="coerce")
        score_df = score_df.loc[score_df["status"].eq("ok")].copy()
        score_df = score_df.dropna(subset=score_cols, how="all")
        score_df = score_df.sort_values(
            by=["overall_score", "method"], ascending=[False, True]
        ).reset_index(drop=True)
        if score_df.empty:
            current_ax.axis("off")
            return current_ax

        score_df["label"] = score_df["method"].map(
            NORMALIZATION_METHODS.display_name
        )
        MetricScorecardRenderer(
            [
                ScorecardMetricSpec(column, label)
                for column, label in zip(score_cols, metric_labels)
            ]
        ).render(current_ax, score_df)

        self._apply_standard_format(
            current_ax,
            title="Candidate Preservation Scorecard",
            xlabel="",
            ylabel="",
            append_stage=False,
        )
        pu.rotate_xticks_if_overlapping(current_ax)
        current_ax.tick_params(axis="both", length=0)
        return current_ax
