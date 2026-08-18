"""Correction candidate ranking and preservation scorecards.

This module renders the two matrix-like summaries used to compare AUTO
correction candidates. It contains no dashboard assembly or correction
algorithms; those responsibilities remain in sibling plotting and processing
modules respectively.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...processing.correction.algorithms import (
    _format_correction_method_label,
)
from .. import plot_utils as pu
from ..components import (
    CandidateScoreRenderer,
    MetricScorecardRenderer,
    ScoreComponentSpec,
    ScorecardMetricSpec,
)


class CorrectionScorecardMixin:
    """Render candidate score contributions and preservation matrices."""

    def plot_correction_score_summary(
        self,
        results_store: dict[str, dict[str, Any]],
        selected_method: str,
        ax: plt.Axes | None = None,
        show_legend: bool = True,
    ) -> plt.Axes:
        """Plot weighted AUTO correction score components."""
        try:
            import patchworklib as pw
        except ImportError:
            raise ImportError("patchworklib is required for this plot.")

        if ax is None:
            current_ax = pw.Brick(
                figsize=(9.0, 3.0), label="correction_eval_summary"
            )
        else:
            current_ax = ax

        summary_rows = []
        for method, result in results_store.items():
            method_label = _format_correction_method_label(method)
            summary_rows.append(
                {
                    "method": method,
                    "label": method_label,
                    "selected": method == selected_method,
                    "eval_rsd": result.get("eval_rsd"),
                    "median_qc_rsd_improvement_score": result.get(
                        "median_qc_rsd_improvement_score"
                    ),
                    "featurewise_qc_rsd_improvement_score": result.get(
                        "featurewise_qc_rsd_improvement_score"
                    ),
                    "sample_structure_score": result.get(
                        "sample_structure_score"
                    ),
                    "auto_score": result.get("auto_score"),
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.replace([np.inf, -np.inf], np.nan)
        summary_df = summary_df.dropna(subset=["auto_score"])
        summary_df = summary_df.sort_values(
            by=["auto_score", "label"], ascending=[False, True]
        ).reset_index(drop=True)

        if summary_df.empty:
            current_ax.axis("off")
            return current_ax

        renderer = CandidateScoreRenderer(
            [
                ScoreComponentSpec(
                    "median_qc_rsd_improvement_score",
                    "Median QC-RSD improvement",
                    pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=1.0),
                    0.35,
                    0.11,
                ),
                ScoreComponentSpec(
                    "featurewise_qc_rsd_improvement_score",
                    "Feature-wise QC-RSD improvement",
                    pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=0.67),
                    0.35,
                    0.11,
                ),
                ScoreComponentSpec(
                    "sample_structure_score",
                    "Sample structure preservation",
                    pu.get_equivalent_hex("tab:gray", alpha=0.6),
                    0.30,
                    0.11,
                ),
            ]
        )
        renderer.render(
            current_ax,
            summary_df,
            total_column="auto_score",
        )

        self._apply_standard_format(
            current_ax,
            title="Auto Correction Method Selection",
            xlabel="Weighted contribution to overall score",
            append_stage=False,
        )
        if show_legend:
            current_ax.legend(handles=renderer.legend_handles())
            self._format_single_legend(
                ax=current_ax,
                group_title="Correction score components",
                loc="lower right",
                bbox_to_anchor=None,
                max_item_rows=6,
            )
        current_ax.tick_params(axis="y", length=0)

        return current_ax

    def plot_correction_preservation_scorecard(
        self,
        results_store: dict[str, dict[str, Any]],
        selected_method: str,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot actual-sample structure metrics used by AUTO correction."""
        try:
            import patchworklib as pw
        except ImportError:
            raise ImportError("patchworklib is required for this plot.")

        if ax is None:
            current_ax = pw.Brick(
                figsize=(3.6, 4.0), label="correction_preservation_scorecard"
            )
        else:
            current_ax = ax

        rows = []
        for method, result in results_store.items():
            method_label = _format_correction_method_label(method)
            rows.append(
                {
                    "method": method,
                    "label": method_label,
                    "selected": method == selected_method,
                    "sample_structure_score": result.get(
                        "sample_structure_score"
                    ),
                    "Trustworthiness": result.get(
                        "sample_structure_metrics", {}
                    ).get("sample_structure_trustworthiness"),
                    "Distance rank preservation": result.get(
                        "sample_structure_metrics", {}
                    ).get("sample_structure_rank_preservation"),
                    "Distance scale preservation": result.get(
                        "sample_structure_metrics", {}
                    ).get("sample_structure_scale_preservation"),
                    "auto_score": result.get("auto_score"),
                }
            )

        summary_df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
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
        for col in ["auto_score", "sample_structure_score", *metric_cols]:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")
        summary_df = summary_df.dropna(subset=metric_cols, how="all")
        summary_df = summary_df.sort_values(
            by=["auto_score", "label"], ascending=[False, True]
        ).reset_index(drop=True)

        if summary_df.empty:
            current_ax.axis("off")
            return current_ax

        MetricScorecardRenderer(
            [
                ScorecardMetricSpec(column, label)
                for column, label in zip(metric_cols, metric_labels)
            ]
        ).render(current_ax, summary_df)

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
