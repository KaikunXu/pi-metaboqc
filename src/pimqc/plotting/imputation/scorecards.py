"""Candidate ranking and preservation scorecards for imputation.

The module adapts imputation-specific metrics to the shared score renderers;
dashboard composition remains in ``dashboards.py``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import plot_utils as pu
from ..components import (
    CandidateScoreRenderer,
    MetricScorecardRenderer,
    ScoreComponentSpec,
    ScorecardMetricSpec,
)
from ...processing.imputation.methods import IMPUTATION_METHODS


class ImputationScorecardMixin:
    """Render imputation candidate scores and preservation metrics."""

    def plot_imputation_score_summary(
        self,
        results_dict: dict[
            str, tuple[dict[str, float], np.ndarray, np.ndarray]
        ],
        selected_method: str,
        ax: plt.Axes | None = None,
        show_legend: bool = False,
    ) -> plt.Axes:
        """Plot MAR imputation AUTO score components or fallback NRMSE."""
        try:
            import patchworklib as pw
        except ImportError as exc:
            raise ImportError(
                "patchworklib is required for this plot."
            ) from exc

        current_ax = (
            pw.Brick(figsize=(4, 4), label="imputation_nrmse_summary")
            if ax is None
            else ax
        )
        best_key = IMPUTATION_METHODS.canonicalize(
            selected_method, strict=False
        )
        rows = []
        for method_name, (metrics, _, _) in results_dict.items():
            rows.append(
                {
                    "method": method_name,
                    "label": IMPUTATION_METHODS.display_name(method_name),
                    "nrmse_total": metrics.get("NRMSE_Total"),
                    "reconstruction_score": metrics.get("Reconstruction_Score"),
                    "distribution_preservation_score": metrics.get(
                        "Distribution_Preservation_Score"
                    ),
                    "sample_structure_score": metrics.get(
                        "Sample_Structure_Score"
                    ),
                    "auto_score": metrics.get("Auto_Score"),
                    "selected": (
                        IMPUTATION_METHODS.canonicalize(
                            method_name, strict=False
                        )
                        == best_key
                    ),
                }
            )

        summary_df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
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

        if has_auto_score:
            renderer = CandidateScoreRenderer(
                [
                    ScoreComponentSpec(
                        "reconstruction_score",
                        "Masked reconstruction",
                        pu.get_equivalent_hex(
                            pu.PRIMARY_ACCENT_COLOR, alpha=1.0
                        ),
                        0.65,
                    ),
                    ScoreComponentSpec(
                        "distribution_preservation_score",
                        "Distribution fidelity",
                        pu.get_equivalent_hex("tab:gray", alpha=0.75),
                        0.20,
                    ),
                    ScoreComponentSpec(
                        "sample_structure_score",
                        "Sample structure preservation",
                        pu.get_equivalent_hex("tab:gray", alpha=0.45),
                        0.15,
                    ),
                ]
            )
            renderer.render(
                current_ax,
                summary_df,
                total_column="auto_score",
            )
            if show_legend:
                current_ax.legend(handles=renderer.legend_handles())
                self._format_single_legend(
                    ax=current_ax,
                    group_title="Imputation score components",
                    loc="lower right",
                    bbox_to_anchor=None,
                    max_item_rows=6,
                )
            title = "Auto Imputation Method Selection"
            xlabel = "Weighted contribution to overall score"
        else:
            y_positions = np.arange(len(summary_df))
            selected_color = pu.get_equivalent_hex(
                pu.PRIMARY_ACCENT_COLOR, alpha=1.0
            )
            background_color = pu.get_equivalent_hex("tab:gray", alpha=0.75)
            values = summary_df["nrmse_total"].to_numpy(dtype=float)
            current_ax.barh(
                y_positions,
                values,
                color=[
                    selected_color if row.selected else background_color
                    for row in summary_df.itertuples()
                ],
                edgecolor="k",
                linewidth=0.5,
                height=0.58,
            )
            current_ax.set_yticks(y_positions)
            current_ax.set_yticklabels(
                [
                    f"* {row.label}" if row.selected else str(row.label)
                    for row in summary_df.itertuples()
                ]
            )
            current_ax.invert_yaxis()
            x_max = float(np.nanmax(values)) if values.size else 1.0
            current_ax.set_xlim(0, x_max * 1.2 if x_max > 0 else 1.0)
            for y_index, value in enumerate(values):
                current_ax.text(
                    min(
                        value + 0.015,
                        current_ax.get_xlim()[1] * 0.97,
                    ),
                    y_index,
                    f"{value:.4f}",
                    va="center",
                    ha="left",
                    fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                )
            current_ax.tick_params(axis="y", length=0)
            title = "MAR Imputer Ranking"
            xlabel = "NRMSE Total"

        self._apply_standard_format(
            ax=current_ax,
            title=title,
            xlabel=xlabel,
            append_stage=False,
        )
        return current_ax

    def plot_imputation_preservation_scorecard(
        self,
        results_dict: dict[
            str, tuple[dict[str, float], np.ndarray, np.ndarray]
        ],
        selected_method: str,
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

        best_key = IMPUTATION_METHODS.canonicalize(
            selected_method, strict=False
        )
        rows = []
        for method_name, (metrics, _, _) in results_dict.items():
            rows.append(
                {
                    "method": method_name,
                    "label": IMPUTATION_METHODS.display_name(method_name),
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
                    "selected": (
                        IMPUTATION_METHODS.canonicalize(
                            method_name, strict=False
                        )
                        == best_key
                    ),
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

        MetricScorecardRenderer(
            [
                ScorecardMetricSpec(column, label)
                for column, label in zip(metric_cols, metric_labels)
            ]
        ).render(current_ax, metric_df)

        dist_color = pu.get_equivalent_hex("tab:gray", alpha=0.75)
        struct_color = pu.get_equivalent_hex("tab:gray", alpha=0.45)
        group_specs = [
            (-0.5, 2.0, "Distribution fidelity", dist_color),
            (1.5, 3.0, "Sample structure preservation", struct_color),
        ]
        for x_start, width, label, face_color in group_specs:
            # The reusable scorecard bounds cell borders to the data matrix,
            # so each header needs only its own semantic group outline.
            current_ax.add_patch(
                plt.Rectangle(
                    (x_start, -1.05),
                    width,
                    0.38,
                    facecolor=face_color,
                    edgecolor="k",
                    linewidth=pu.DEFAULT_HEATMAP_CELL_LINEWIDTH,
                    zorder=4,
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
                zorder=5,
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
