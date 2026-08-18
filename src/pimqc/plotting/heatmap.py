"""Shared heatmap primitives with focused, reusable responsibilities.

These helpers centralize heatmap rendering concerns for multiple stage
plotters without embedding stage-specific metrics, workflow decisions, or
dashboard layouts.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from . import plot_utils as pu


def score_heatmap_cmap(
    base_color: str = pu.PRIMARY_ACCENT_COLOR,
    n_colors: int = 256,
    cmin: float = 0.2,
    cmax: float = 1.0,
) -> mpl.colors.Colormap:
    """Return the standard white-to-accent score heatmap colormap."""
    custom_cmap = pu.custom_linear_cmap(["white", base_color], n_colors)
    return mpl.colors.ListedColormap(
        pu.extract_linear_cmap(custom_cmap, cmin, cmax, n_colors)
    )


def heatmap_annotation_fontsize(
    ax: plt.Axes,
    n_rows: int,
    n_cols: int,
    default_size: float = pu.DEFAULT_ANNOTATION_FONTSIZE,
    max_size: float = pu.DEFAULT_ANNOTATION_FONTSIZE,
    min_size: float = 4.0,
    fill_ratio: float = 0.62,
) -> float:
    """Estimate a readable annotation size for a heatmap."""
    ax_w, ax_h = pu.axis_size_inches(ax)
    row_font = pu.dense_tick_fontsize(
        n_items=max(1, n_rows),
        axis_inches=ax_h,
        default_size=default_size,
        max_size=max_size,
        min_size=min_size,
        fill_ratio=fill_ratio,
        force_dense=True,
    )
    col_font = pu.dense_tick_fontsize(
        n_items=max(1, n_cols),
        axis_inches=ax_w,
        default_size=default_size,
        max_size=max_size,
        min_size=min_size,
        fill_ratio=fill_ratio,
        force_dense=True,
    )
    return min(row_font, col_font)


def draw_visible_heatmap_cell_edges(
    ax: plt.Axes,
    visible_mask: np.ndarray,
    linewidth: float,
    edgecolor: str = "k",
    zorder: float = 4,
) -> None:
    """Draw each visible heatmap cell edge once for vector output."""
    edge_segments: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for row_idx, col_idx in np.argwhere(visible_mask):
        x0, x1 = float(col_idx), float(col_idx + 1)
        y0, y1 = float(row_idx), float(row_idx + 1)
        edge_segments.update(
            {
                ((x0, y0), (x1, y0)),
                ((x0, y1), (x1, y1)),
                ((x0, y0), (x0, y1)),
                ((x1, y0), (x1, y1)),
            }
        )
    if edge_segments:
        ax.add_collection(
            LineCollection(
                list(edge_segments),
                colors=edgecolor,
                linewidths=linewidth,
                zorder=zorder,
                clip_on=False,
            )
        )
