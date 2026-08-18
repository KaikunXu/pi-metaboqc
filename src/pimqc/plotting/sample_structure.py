"""Cross-stage diagnostic panel renderers.

These functions contain project-level diagnostic semantics shared by several
processing stages. Low-level styling and layout primitives remain in
``plot_utils``; numerical geometry is supplied by ``statistics.metrics``.
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..constants import DEFAULT_RANDOM_SEED
from ..statistics import metrics as su
from ..statistics import sample_structure as structure_stats
from . import annotation_layout as al
from . import plot_utils as pu


def _finite_metric(metrics: Mapping[str, object], *keys: str) -> float:
    """Return the first finite metric value among aliases."""
    for key in keys:
        value = su.finite_or_nan(metrics.get(key))
        if np.isfinite(value):
            return float(np.clip(value, 0.0, 1.0))
    return float("nan")


def plot_sample_structure_change_map(
    ax: plt.Axes,
    raw_obj: pd.DataFrame,
    transformed_obj: pd.DataFrame,
    structure_metrics: Mapping[str, float] | None = None,
    sample_cols: pd.Index | None = None,
    seed: int = DEFAULT_RANDOM_SEED,
    max_features: int | None = 5000,
    scale_log_ratio_tol: float = 0.25,
    scale_rel_delta_tol: float = 0.35,
    title: str = "Sample Structure Change Map",
    compact_style: bool = False,
) -> plt.Axes:
    """Plot sample-level scale, rank, and local-trustworthiness changes.

    Args:
        ax: Target axes.
        raw_obj: Matrix before processing.
        transformed_obj: Matrix after processing.
        structure_metrics: Optional precomputed global preservation metrics.
        sample_cols: Optional sample columns to include in geometry.
        seed: Deterministic feature-subsampling seed.
        max_features: Maximum features used for geometry calculation.
        scale_log_ratio_tol: Practical scale-change tolerance.
        scale_rel_delta_tol: Practical relative-distance tolerance.
        title: Panel title.
        compact_style: Use article-sized markers and text.

    Returns:
        The supplied axes.
    """
    metrics = dict(structure_metrics or {})
    if not metrics:
        metrics = structure_stats.calc_sample_structure_preservation(
            raw_obj=raw_obj,
            transformed_obj=transformed_obj,
            sample_cols=sample_cols,
            max_features=max_features,
            seed=seed,
            scale_log_ratio_tol=scale_log_ratio_tol,
            scale_rel_delta_tol=scale_rel_delta_tol,
        )

    geometry = structure_stats.calc_sample_structure_arrays(
        raw_obj=raw_obj,
        transformed_obj=transformed_obj,
        sample_cols=sample_cols,
        max_features=max_features,
        seed=seed,
    )["geometry"]
    trust_score = _finite_metric(
        metrics,
        "sample_structure_trustworthiness",
        "Trustworthiness",
        "trustworthiness",
    )
    rank_score = _finite_metric(
        metrics,
        "sample_structure_rank_preservation",
        "Distance_Rank_Preservation",
        "distance_rank_preservation",
    )
    scale_score = _finite_metric(
        metrics,
        "sample_structure_scale_preservation",
        "Distance_Scale_Preservation",
        "distance_scale_preservation",
    )

    plot_df = pd.concat(
        [
            pd.to_numeric(
                geometry.get("sample_log2_distance_ratio", pd.Series()),
                errors="coerce",
            ).rename("scale_shift"),
            pd.to_numeric(
                geometry.get("sample_distance_rank_rho", pd.Series()),
                errors="coerce",
            ).rename("rank_rho"),
            pd.to_numeric(
                geometry.get(
                    "sample_neighborhood_trustworthiness", pd.Series()
                ),
                errors="coerce",
            ).rename("local_trust"),
        ],
        axis=1,
    ).dropna(subset=["scale_shift", "rank_rho"])

    if plot_df.empty:
        ax.text(
            0.5,
            0.5,
            "Insufficient sample data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
            bbox=pu.ai_ready_text_bbox(),
        )
        ax.set_title(title)
        return ax

    edges = np.array([0.0, 0.90, 0.95, 0.98, 1.000001])
    cmap = mpl.colors.ListedColormap(
        [
            pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=0.20),
            pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=0.42),
            pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=0.68),
            pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=1.00),
        ]
    )
    norm = mpl.colors.BoundaryNorm(edges, cmap.N, clip=True)
    local_trust = plot_df["local_trust"].fillna(trust_score)
    scatter = ax.scatter(
        plot_df["scale_shift"],
        plot_df["rank_rho"],
        c=local_trust,
        cmap=cmap,
        norm=norm,
        s=14.0 if compact_style else 28.0,
        edgecolor="k",
        linewidth=0.15 if compact_style else 0.25,
        zorder=3,
    )
    ax.axvline(0.0, color="0.35", linestyle="--", linewidth=0.6, zorder=1)
    ax.axhline(1.0, color="0.55", linestyle=":", linewidth=0.6, zorder=1)
    x_extent = max(float(np.nanmax(np.abs(plot_df["scale_shift"]))), 0.15)
    ax.set_xlim(-1.15 * x_extent, 1.15 * x_extent)
    y_low = min(float(np.nanmin(plot_df["rank_rho"])), 0.85)
    ax.set_ylim(max(-1.0, y_low - 0.05), 1.03)

    colorbar_regions = [
        ((0.94, 0.13, 0.025, 0.28), (0.70, 0.08, 1.01, 0.46), "right"),
        ((0.94, 0.59, 0.025, 0.28), (0.70, 0.54, 1.01, 0.92), "right"),
        ((0.04, 0.13, 0.025, 0.28), (-0.01, 0.08, 0.30, 0.46), "left"),
        ((0.04, 0.59, 0.025, 0.28), (-0.01, 0.54, 0.30, 0.92), "left"),
    ]
    axes_xy = ax.transAxes.inverted().transform(
        ax.transData.transform(plot_df[["scale_shift", "rank_rho"]])
    )
    colorbar_box, blocked, side = min(
        colorbar_regions,
        key=lambda item: int(
            np.sum(
                (axes_xy[:, 0] >= item[1][0])
                & (axes_xy[:, 0] <= item[1][2])
                & (axes_xy[:, 1] >= item[1][1])
                & (axes_xy[:, 1] <= item[1][3])
            )
        ),
    )
    colorbar_ax = ax.inset_axes(colorbar_box)
    colorbar = ax.figure.colorbar(
        scatter,
        cax=colorbar_ax,
        boundaries=edges,
        ticks=[0.90, 0.95, 0.98],
        spacing="uniform",
        drawedges=True,
    )
    colorbar.ax.set_yticklabels(["0.90", "0.95", "0.98"])
    colorbar.ax.yaxis.set_ticks_position("left" if side == "right" else "right")
    colorbar.ax.yaxis.set_label_position("left" if side == "right" else "right")
    colorbar.set_label(
        "Local trustworthiness",
        labelpad=2,
    )

    # Finalize every fixed decoration before measuring collision obstacles.
    # In particular, the inset colorbar's tight bounding box includes its
    # ticks and inward-facing label only after typography has been applied.
    ax.set_title(title)
    ax.set_xlabel("Median log2 distance ratio\n(after / before)")
    ax.set_ylabel("Distance-rank correlation")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    pu.change_fontsize(ax)
    pu.change_weight(ax)
    pu.format_colorbar_axes(colorbar.ax)

    note_lines = []
    if np.isfinite(trust_score):
        note_lines.append(f"Global T(k): {trust_score:.3f}")
    if np.isfinite(rank_score):
        note_lines.append(f"Distance-rank preservation: {rank_score:.3f}")
    if np.isfinite(scale_score):
        note_lines.append(f"Distance-scale preservation: {scale_score:.3f}")
    if note_lines:
        al.add_auto_annotation(
            ax=ax,
            text="\n".join(note_lines),
            occupancy_arrays=[plot_df[["scale_shift", "rank_rho"]].to_numpy()],
            blocked_regions=[blocked],
            fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
            bbox=pu.ai_ready_text_bbox(pad=0.20 if compact_style else 0.30),
        )
    return ax
