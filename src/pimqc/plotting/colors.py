"""Shared categorical palette construction for scientific visualizations.

The helpers in this module centralize deterministic color assignment without
embedding stage-specific metrics, workflow decisions, or dashboard layouts.
"""

from __future__ import annotations

from collections.abc import Iterable

from . import plot_utils as pu


def build_categorical_palette(
    values: Iterable[object],
    *,
    colors: tuple[str, str] = ("white", "tab:red"),
    cmin: float = 0.5,
    cmax: float = 1.0,
) -> dict[object, str]:
    """Build a deterministic color mapping for categorical values."""
    unique_values = sorted(set(values), key=str)
    if not unique_values:
        return {}
    cmap = pu.custom_linear_cmap(list(colors), n_colors=100)
    palette = pu.extract_linear_cmap(
        cmap=cmap,
        cmin=cmin,
        cmax=cmax,
        n_colors=len(unique_values),
    )
    return dict(zip(unique_values, palette))
