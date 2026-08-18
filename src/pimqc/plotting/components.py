"""Render reusable candidate summaries and metric scorecards.

The renderers implement only stable drawing mechanics. Stage plotters
supply their own columns, labels, weights, colors, and metric specifications,
preserving domain-specific rules and additional dashboard decoration.
"""

from dataclasses import dataclass
from typing import Callable, Iterable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plot_utils as pu
from .heatmap import heatmap_annotation_fontsize, score_heatmap_cmap


@dataclass(frozen=True)
class ScoreComponentSpec:
    """Describe one weighted component in a candidate score plot."""

    column: str
    label: str
    color: str
    weight: float
    annotation_threshold: float = 0.10


class CandidateScoreRenderer:
    """Render weighted candidate scores while preserving stage-specific data."""

    def __init__(self, components: Iterable[ScoreComponentSpec]) -> None:
        """Initialize the stacked-score renderer.

        Args:
            components: Ordered component definitions and visual styles.

        Raises:
            ValueError: If no score components are supplied.
        """
        self.components = tuple(components)
        if not self.components:
            raise ValueError("At least one score component is required.")

    def render(
        self,
        ax: plt.Axes,
        frame: pd.DataFrame,
        *,
        total_column: str,
        label_column: str = "label",
        selected_column: str = "selected",
        status_column: str | None = None,
        valid_status: str = "ok",
        scale_to_total: bool = False,
        bar_height: float = 0.58,
    ) -> plt.Axes:
        """Render a stacked horizontal score chart from a prepared table."""
        data = frame.copy()
        component_columns = [spec.column for spec in self.components]
        for column in [total_column, *component_columns]:
            data[column] = pd.to_numeric(data[column], errors="coerce")

        contributions = np.zeros((len(data), len(self.components)), dtype=float)
        # Renormalize weights over finite components so a missing optional
        # metric does not automatically penalize an otherwise valid candidate.
        for row_index, (_, row) in enumerate(data.iterrows()):
            if status_column and row.get(status_column) != valid_status:
                continue
            available = [
                spec
                for spec in self.components
                if np.isfinite(row.get(spec.column, np.nan))
            ]
            available_weight = sum(spec.weight for spec in available)
            if available_weight <= 0:
                continue
            for component_index, spec in enumerate(self.components):
                value = row.get(spec.column, np.nan)
                if np.isfinite(value):
                    contributions[row_index, component_index] = (
                        np.clip(float(value), 0.0, 1.0)
                        * spec.weight
                        / available_weight
                    )
            if scale_to_total:
                # Some stages supply a separately guarded total score; retain
                # its magnitude while using component weights for composition.
                total = row.get(total_column, np.nan)
                raw_total = contributions[row_index].sum()
                if np.isfinite(total) and raw_total > 0:
                    contributions[row_index] *= (
                        np.clip(float(total), 0.0, 1.0) / raw_total
                    )

        y_positions = np.arange(len(data))
        left = np.zeros(len(data), dtype=float)
        for component_index, spec in enumerate(self.components):
            values = contributions[:, component_index]
            starts = left.copy()
            ax.barh(
                y_positions,
                values,
                left=left,
                height=bar_height,
                color=spec.color,
                edgecolor="k",
                linewidth=0.5,
                label=spec.label,
                zorder=3,
            )
            for y_index, (_, row) in enumerate(data.iterrows()):
                score = row.get(spec.column, np.nan)
                if values[y_index] < spec.annotation_threshold:
                    continue
                if not np.isfinite(score):
                    continue
                ax.text(
                    starts[y_index] + values[y_index] / 2.0,
                    y_index,
                    f"{float(score):.2f}",
                    va="center",
                    ha="center",
                    fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                    color=pu.get_contrast_color(spec.color),
                    clip_on=True,
                    zorder=4,
                )
            left += values

        x_upper = float(np.nanmax(left)) if left.size else 1.0
        x_upper = min(1.08, max(x_upper + 0.08, x_upper * 1.10, 0.20))
        for y_index, (_, row) in enumerate(data.iterrows()):
            total = row.get(total_column, np.nan)
            is_valid = (
                not status_column or row.get(status_column) == valid_status
            )
            text = (
                f"{float(total):.3f}"
                if is_valid and np.isfinite(total)
                else "failed"
            )
            ax.text(
                min(float(left[y_index]) + 0.015, x_upper * 0.97),
                y_index,
                text,
                va="center",
                ha="left",
                fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                color="0.15" if text != "failed" else "0.45",
                style="normal" if text != "failed" else "italic",
            )

        labels = []
        for _, row in data.iterrows():
            label = str(row[label_column])
            labels.append(
                f"* {label}" if bool(row.get(selected_column, False)) else label
            )
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, x_upper)
        ax.set_ylim(-0.5, len(data) - 0.5)
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)
        return ax

    def legend_handles(self) -> list[mpatches.Patch]:
        """Return standard legend handles for the configured components."""
        return [
            mpatches.Patch(
                facecolor=spec.color,
                edgecolor="k",
                linewidth=0.5,
                label=spec.label,
            )
            for spec in self.components
        ]


@dataclass(frozen=True)
class ScorecardMetricSpec:
    """Describe one scorecard column."""

    column: str
    label: str
    formatter: Callable[[float], str] = lambda value: f"{value:.2f}"


class MetricScorecardRenderer:
    """Render a candidate-by-metric scorecard from stage-prepared values."""

    def __init__(self, metrics: Iterable[ScorecardMetricSpec]) -> None:
        """Initialize the metric-scorecard renderer.

        Args:
            metrics: Ordered scorecard columns and value formatters.

        Raises:
            ValueError: If no metric specifications are supplied.
        """
        self.metrics = tuple(metrics)
        if not self.metrics:
            raise ValueError("At least one scorecard metric is required.")

    def render(
        self,
        ax: plt.Axes,
        frame: pd.DataFrame,
        *,
        label_column: str = "label",
        selected_column: str = "selected",
    ) -> plt.Axes:
        """Render a bounded 0-1 heatmap while retaining missing cells."""
        columns = [metric.column for metric in self.metrics]
        matrix = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy()
        cmap = score_heatmap_cmap()
        annotation_size = heatmap_annotation_fontsize(
            ax,
            n_rows=matrix.shape[0],
            n_cols=matrix.shape[1],
            default_size=pu.DEFAULT_SCORE_HEATMAP_ANNOTATION_FONTSIZE,
            max_size=pu.DEFAULT_SCORE_HEATMAP_ANNOTATION_FONTSIZE,
            min_size=4.0,
        )
        # Mask invalid cells instead of mapping NA to the lowest score color.
        ax.imshow(
            np.ma.masked_invalid(matrix),
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
        )
        ax.set_xticks(np.arange(len(columns)))
        ax.set_xticklabels([metric.label for metric in self.metrics])
        ax.set_yticks(np.arange(len(frame)))
        ax.set_yticklabels(
            [
                f"* {row[label_column]}"
                if bool(row.get(selected_column, False))
                else str(row[label_column])
                for _, row in frame.iterrows()
            ]
        )
        # Draw borders only across the matrix extent. The individual lines let
        # stage scorecards extend their y limits for grouped headers without
        # leaking a border into that semantic header row.
        n_rows, n_cols = matrix.shape
        for x_position in np.arange(-0.5, n_cols, 1.0):
            ax.plot(
                [x_position, x_position],
                [-0.5, n_rows - 0.5],
                color="k",
                linewidth=pu.DEFAULT_HEATMAP_CELL_LINEWIDTH,
                zorder=3,
            )
        for y_position in np.arange(-0.5, n_rows, 1.0):
            ax.plot(
                [-0.5, n_cols - 0.5],
                [y_position, y_position],
                color="k",
                linewidth=pu.DEFAULT_HEATMAP_CELL_LINEWIDTH,
                zorder=3,
            )

        for y_index in range(matrix.shape[0]):
            for x_index, metric in enumerate(self.metrics):
                value = matrix[y_index, x_index]
                label = (
                    "NA" if not np.isfinite(value) else metric.formatter(value)
                )
                color = (
                    "0.35"
                    if not np.isfinite(value)
                    else pu.get_contrast_color(cmap(value))
                )
                ax.text(
                    x_index,
                    y_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=annotation_size,
                    color=color,
                )
        ax.tick_params(axis="both", length=0)
        return ax
