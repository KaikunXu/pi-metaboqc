# src/pimqc/plot_utils.py
"""
Script purpose: Provide reusable plotting helpers for visual consistency.

This module contains small matplotlib utilities for color blending, contrast
selection, colormap extraction, axis formatting, tick rotation, font/weight
updates, bar-value annotation, and confidence ellipse drawing. These functions
are intentionally generic so assessment, filtering, correction, imputation,
normalization, and builder visualizers can share the same figure behavior.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.text import Text
import pandas as pd
import re
from typing import List, Mapping, Optional, Union
import warnings

DEFAULT_AXIS_FORMAT = "normal"
DEFAULT_FORMAT_AXIS = "xy"
DEFAULT_AXIS_TICK_FONTSIZE = 14
DEFAULT_AXIS_LABEL_FONTSIZE = 14
DEFAULT_TITLE_FONTSIZE = 16
DEFAULT_ANNOTATION_FONTSIZE = 10
DEFAULT_AXIS_TICK_WEIGHT = "normal"
DEFAULT_AXIS_LABEL_WEIGHT = "normal"
DEFAULT_TITLE_WEIGHT = "bold"
DEFAULT_TICK_ROTATION = 45
DEFAULT_ROTATION_AXIS = "x"
DEFAULT_SCI_LOWER_THRESHOLD = 1e-3
DEFAULT_SCI_UPPER_THRESHOLD = 1e4
DEFAULT_SCI_POWER_LIMITS = (0, 0)
DEFAULT_VECTOR_FONT_FAMILY = "Helvetica"
DEFAULT_VECTOR_LINEBREAK_MODE = "keep"
DEFAULT_VECTOR_HYPHEN_REPLACEMENT = "\u2013"
PRESERVE_ALPHA_ATTR = "_pi_metaboqc_preserve_alpha"

# Compact manuscript-composite typography.  These values are intentionally
# separate from the standard dashboard defaults above.
ARTICLE_AXIS_TICK_FONTSIZE = 6.0
ARTICLE_AXIS_LABEL_FONTSIZE = 7.0
ARTICLE_TITLE_FONTSIZE = 8.0
ARTICLE_ANNOTATION_FONTSIZE = 5.5
ARTICLE_LEGEND_FONTSIZE = 5.0
ARTICLE_LEGEND_TITLE_FONTSIZE = 5.5

# Shared visual theme for QC, preprocessing selection, and diagnostic emphasis.
# Dataset-builder sample-type plots intentionally retain their own red/blue/gray palette.
PRIMARY_ACCENT_COLOR = "tab:blue"
NEUTRAL_COLOR = "tab:gray"


def ai_ready_text_bbox(pad: float = 0.25) -> dict[str, object]:
    """Return an opaque white text box suitable for editable SVG/PDF output."""
    return {
        "boxstyle": f"round,pad={pad}",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 1.0,
    }


def format_percentile_label(percentile: float) -> str:
    """Format a 0-1 percentile fraction as a compact percentile label."""
    pct_value = float(percentile) * 100
    if np.isclose(pct_value, round(pct_value)):
        pct_text = f"{pct_value:.0f}"
    else:
        pct_text = f"{pct_value:.1f}".rstrip("0").rstrip(".")
    return f"P{pct_text}"


def normalize_vector_text(
    text: object,
    replace_hyphen: bool = True,
    replace_spaces: bool = True,
    linebreak_mode: str = DEFAULT_VECTOR_LINEBREAK_MODE,
    hyphen_replacement: str = DEFAULT_VECTOR_HYPHEN_REPLACEMENT,
) -> str:
    """Normalize plot text for editable PDF/SVG output in vector editors.

    The function targets common Adobe Illustrator import issues: text splitting
    around hyphenated terms and multi-line labels. MathText strings are left
    unchanged because Matplotlib handles them as separate glyph layout objects.
    """
    if text is None:
        return ""

    normalized = str(text)
    if not normalized or "$" in normalized or "\\math" in normalized:
        return normalized

    linebreak_key = str(linebreak_mode).lower().strip()
    if linebreak_key in {"space", "single_line", "single-line"}:
        normalized = re.sub(r"\s*[\r\n]+\s*", " ", normalized)
    elif linebreak_key in {"slash", "solidus"}:
        normalized = re.sub(r"\s*[\r\n]+\s*", " / ", normalized)
    elif linebreak_key in {"semicolon", "semi"}:
        normalized = re.sub(r"\s*[\r\n]+\s*", "; ", normalized)
    elif linebreak_key not in {"keep", "preserve"}:
        raise ValueError(
            "linebreak_mode must be one of: keep, preserve, space, slash, semicolon."
        )

    normalized = re.sub(r"[ \t]{2,}", " ", normalized)

    if replace_hyphen:
        # U+2011 and word-joiner hyphen can trigger Arial fallback in Matplotlib.
        # En dash is supported by Arial and avoids the common Illustrator split
        # behavior seen around regular ASCII hyphens.
        normalized = re.sub(
            r"(?<=[A-Za-z0-9])-(?=[A-Za-z0-9])",
            hyphen_replacement,
            normalized,
        )

    if replace_spaces:
        # Keep compact method/version labels such as "WaveICA 2.0" together.
        normalized = re.sub(
            r"\b([A-Za-z][A-Za-z0-9]*)[^\S\r\n]+(\d+(?:\.\d+)?)\b",
            lambda match: f"{match.group(1)}\u00a0{match.group(2)}",
            normalized,
        )
        # Keep numeric values attached to short units or percent signs. Use
        # regular NBSP instead of narrow NBSP for broader PDF/SVG compatibility.
        normalized = re.sub(
            r"\b(\d+(?:\.\d+)?)[^\S\r\n]+(%|[A-Za-z]{1,5}\b)",
            lambda match: f"{match.group(1)}\u00a0{match.group(2)}",
            normalized,
        )

    return normalized


def normalize_text_artist_for_vector_export(
    text_artist: Text,
    target_font: str | list[str] | tuple[str, ...] = DEFAULT_VECTOR_FONT_FAMILY,
    replace_hyphen: bool = True,
    replace_spaces: bool = True,
    linebreak_mode: str = DEFAULT_VECTOR_LINEBREAK_MODE,
    hyphen_replacement: str = DEFAULT_VECTOR_HYPHEN_REPLACEMENT,
) -> None:
    """Apply vector-export text normalization to one Matplotlib Text artist."""
    current_text = text_artist.get_text()
    if current_text:
        text_artist.set_text(
            normalize_vector_text(
                current_text,
                replace_hyphen=replace_hyphen,
                replace_spaces=replace_spaces,
                linebreak_mode=linebreak_mode,
                hyphen_replacement=hyphen_replacement,
            )
        )

    text_artist.set_fontfamily(target_font)
    text_artist.set_fontstretch("normal")
    text_artist.set_fontstyle("normal")
    text_artist.set_clip_on(False)


def normalize_figure_text_for_vector_export(
    fig: plt.Figure,
    target_font: str | list[str] | tuple[str, ...] = DEFAULT_VECTOR_FONT_FAMILY,
    replace_hyphen: bool = True,
    replace_spaces: bool = True,
    linebreak_mode: str = DEFAULT_VECTOR_LINEBREAK_MODE,
    hyphen_replacement: str = DEFAULT_VECTOR_HYPHEN_REPLACEMENT,
) -> None:
    """Normalize all Matplotlib text artists before PDF/SVG export."""
    for text_artist in fig.findobj(match=Text):
        normalize_text_artist_for_vector_export(
            text_artist=text_artist,
            target_font=target_font,
            replace_hyphen=replace_hyphen,
            replace_spaces=replace_spaces,
            linebreak_mode=linebreak_mode,
            hyphen_replacement=hyphen_replacement,
        )


def get_equivalent_hex(
    color: Union[str, tuple, list],
    alpha: Optional[float] = 1.0,
    bg_color: Union[str, tuple] = "white",
) -> str:
    """
    Convert a transparent color to its visually equivalent solid Hex color.

    This blends the target color with a background color (default white)
    using the specified alpha value. This is highly useful for rendering
    engines or export formats that drop or poorly support alpha channels.

    Args:
        + color: Color name ("tab:blue"), hex ("#cccccc"), or RGB/RGBA tuple.
        Tuples can be 0.0-1.0 scale or 0-255 scale (e.g., (123, 234, 12)).
        + alpha (float): Transparency level (0.0 to 1.0).
        + bg_color (str): Background color to blend against.

    Returns:
        str: Solid 6-digit Hex color code (e.g., "#8fbbda").
    """
    # 1. Normalize 0-255 scale RGB/RGBA tuples to 0.0-1.0 scale
    if isinstance(color, (tuple, list)):
        if any(c > 1.0 for c in color):
            color = tuple(c / 255.0 for c in color)

    # 2. Extract base RGBA (mcolors automatically handles names and hex)
    try:
        r, g, b, original_a = mcolors.to_rgba(color)
    except ValueError:
        raise ValueError(f"Invalid color format provided: {color}")

    bg_r, bg_g, bg_b, _ = mcolors.to_rgba(bg_color)

    # Determine final alpha (override original alpha if parameter is explicitly passed)
    final_alpha = alpha if alpha is not None else original_a

    # 3. Perform standard alpha blending against the background
    # Formula: Result = Foreground * alpha + Background * (1 - alpha)
    blend_r = r * final_alpha + bg_r * (1.0 - final_alpha)
    blend_g = g * final_alpha + bg_g * (1.0 - final_alpha)
    blend_b = b * final_alpha + bg_b * (1.0 - final_alpha)

    # 4. Convert blended RGB back to a solid 6-digit Hex string
    return mcolors.to_hex((blend_r, blend_g, blend_b))


def solidify_rgba_array(
    rgba_values: np.ndarray,
    bg_color: Union[str, tuple] = "white",
) -> np.ndarray:
    """Blend visible RGBA colors onto a background and return export-safe colors.

    Fully transparent entries are kept transparent. Turning them into opaque
    white would make hidden helper patches, spines, or collection rows visible
    in SVG/PDF exports and may cover downstream artists in vector editors.
    """
    rgba_array = np.asarray(rgba_values, dtype=float)
    if rgba_array.size == 0:
        return rgba_array

    original_shape = rgba_array.shape
    if rgba_array.ndim == 1:
        rgba_array = rgba_array.reshape(1, -1)
    if rgba_array.shape[1] == 3:
        rgba_array = np.column_stack([rgba_array, np.ones(len(rgba_array))])
    if rgba_array.shape[1] != 4:
        return rgba_values

    bg_rgba = np.asarray(mcolors.to_rgba(bg_color), dtype=float)
    alpha = rgba_array[:, 3:4]
    solid_rgb = rgba_array[:, :3] * alpha + bg_rgba[:3] * (1.0 - alpha)
    solid_rgba = np.column_stack([solid_rgb, np.ones(len(rgba_array))])
    transparent_mask = rgba_array[:, 3] <= 0
    if np.any(transparent_mask):
        solid_rgba[transparent_mask] = rgba_array[transparent_mask]
    return solid_rgba.reshape(original_shape)


def mark_preserve_alpha(artist: object) -> object:
    """Mark a Matplotlib artist or Axes to keep real alpha during vector export."""
    try:
        setattr(artist, PRESERVE_ALPHA_ATTR, True)
    except Exception:
        pass
    return artist


def artist_preserves_alpha(artist: object) -> bool:
    """Return whether an artist was marked to keep true alpha transparency."""
    return bool(getattr(artist, PRESERVE_ALPHA_ATTR, False))


def solidify_artist_alpha(
    artist: object,
    bg_color: Union[str, tuple] = "white",
) -> None:
    """Convert one Matplotlib artist's transparent colors to opaque colors.

    This is intended for SVG/PDF export where Adobe Illustrator can mishandle
    transparent patches, collections, legend handles, or text bounding boxes.
    """
    if artist_preserves_alpha(artist):
        return

    def _solid_color(color: object, artist_alpha: float | None = None) -> object:
        """Return an equivalent opaque color while preserving invalid sentinels."""
        if color is None or (isinstance(color, str) and color == "none"):
            return color
        try:
            rgba = mcolors.to_rgba(color)
        except (TypeError, ValueError):
            return color
        effective_alpha = rgba[3] if artist_alpha is None else rgba[3] * artist_alpha
        if effective_alpha <= 0:
            return "none"
        if effective_alpha >= 1.0:
            return mcolors.to_hex(rgba[:3])
        return get_equivalent_hex(rgba[:3], alpha=effective_alpha, bg_color=bg_color)

    try:
        artist_alpha = artist.get_alpha() if hasattr(artist, "get_alpha") else None
    except (TypeError, ValueError):
        artist_alpha = None

    if hasattr(artist, "get_facecolor") and hasattr(artist, "set_facecolor"):
        try:
            facecolor = artist.get_facecolor()
            if np.asarray(facecolor).ndim <= 1:
                artist.set_facecolor(_solid_color(facecolor, artist_alpha))
            else:
                artist.set_facecolor(solidify_rgba_array(facecolor, bg_color))
        except (AttributeError, TypeError, ValueError):
            pass

    if hasattr(artist, "get_edgecolor") and hasattr(artist, "set_edgecolor"):
        try:
            edgecolor = artist.get_edgecolor()
            if np.asarray(edgecolor).ndim <= 1:
                artist.set_edgecolor(_solid_color(edgecolor, artist_alpha))
            else:
                artist.set_edgecolor(solidify_rgba_array(edgecolor, bg_color))
        except (AttributeError, TypeError, ValueError):
            pass

    if hasattr(artist, "get_facecolors") and hasattr(artist, "set_facecolors"):
        try:
            facecolors = artist.get_facecolors()
            if len(facecolors):
                artist.set_facecolors(solidify_rgba_array(facecolors, bg_color))
        except (AttributeError, TypeError, ValueError):
            pass

    if hasattr(artist, "get_edgecolors") and hasattr(artist, "set_edgecolors"):
        try:
            edgecolors = artist.get_edgecolors()
            if len(edgecolors):
                artist.set_edgecolors(solidify_rgba_array(edgecolors, bg_color))
        except (AttributeError, TypeError, ValueError):
            pass

    if hasattr(artist, "get_color") and hasattr(artist, "set_color"):
        try:
            artist.set_color(_solid_color(artist.get_color(), artist_alpha))
        except (AttributeError, TypeError, ValueError):
            pass

    for getter_name, setter_name in [
        ("get_markerfacecolor", "set_markerfacecolor"),
        ("get_markeredgecolor", "set_markeredgecolor"),
    ]:
        if hasattr(artist, getter_name) and hasattr(artist, setter_name):
            try:
                color_value = getattr(artist, getter_name)()
                getattr(artist, setter_name)(_solid_color(color_value, artist_alpha))
            except (AttributeError, TypeError, ValueError):
                pass

    if hasattr(artist, "get_bbox_patch"):
        try:
            bbox_patch = artist.get_bbox_patch()
            if bbox_patch is not None:
                solidify_artist_alpha(bbox_patch, bg_color=bg_color)
        except (AttributeError, TypeError, ValueError):
            pass

    if (
        artist_alpha is not None
        and 0 < artist_alpha < 1.0
        and hasattr(artist, "set_alpha")
    ):
        try:
            artist.set_alpha(1.0)
        except (AttributeError, TypeError, ValueError):
            pass


def solidify_figure_alpha(
    fig: plt.Figure,
    bg_color: Union[str, tuple] = "white",
) -> None:
    """Convert transparent Matplotlib artists in a figure to opaque colors."""
    for ax in fig.axes:
        if not getattr(ax, "axison", True):
            try:
                ax.patch.set_facecolor("none")
                ax.patch.set_edgecolor("none")
                mark_preserve_alpha(ax.patch)
            except (AttributeError, TypeError, ValueError):
                pass

    preserved_artist_ids: set[int] = set()
    for ax in fig.axes:
        if artist_preserves_alpha(ax):
            preserved_artist_ids.update(id(item) for item in ax.findobj())
            preserved_artist_ids.add(id(ax))

    for artist in fig.findobj():
        if id(artist) in preserved_artist_ids:
            continue
        solidify_artist_alpha(artist=artist, bg_color=bg_color)


def get_contrast_color(bg_color: Union[str, tuple]) -> str:
    """Calculate text color (black or white) to maximize contrast.

    Computes the perceived luminance of the background color, accounting
    for alpha transparency (blended with a white canvas).

    Args:
        bg_color: A matplotlib-compatible color string or RGBA tuple.

    Returns:
        "white" if the background is dark, "k" (black) if it is light.
    """
    import matplotlib.colors as mcolors

    r, g, b, a = mcolors.to_rgba(bg_color)

    # Blend with a white background assuming transparency shows white
    r = r * a + 1.0 * (1 - a)
    g = g * a + 1.0 * (1 - a)
    b = b * a + 1.0 * (1 - a)

    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.65 else "k"


def get_cmap(palette: str = "Set1") -> mpl.colors.Colormap:
    """Get a matplotlib colormap by name."""
    return mpl.colormaps[palette]


def custom_linear_cmap(
    color_list: List[str] = ["#FFFFFF", "#1F77B4"],
    n_colors: int = 100,
    cmin: float = 0.0,
    cmax: float = 1.0,
) -> mpl.colors.LinearSegmentedColormap:
    """Create a truncated custom linear segmented colormap."""
    base_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "Base_Cmap", colors=color_list, N=256
    )
    sampled_colors = base_cmap(np.linspace(cmin, cmax, n_colors))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "Truncated_Cmap", colors=sampled_colors, N=n_colors
    )
    cmap.set_bad(color="tab:gray")
    return cmap


def extract_qual_cmap(
    cmap: mpl.colors.Colormap, n_colors: Optional[int] = None
) -> List[str]:
    """Extract hexadecimal colors from a qualitative colormap."""
    if n_colors is not None and n_colors >= cmap.N:
        warnings.warn(
            "The resampled number is greater than the total number.",
            category=UserWarning,
        )
    n = n_colors if n_colors is not None else cmap.N
    colors = [mpl.colors.to_hex(cmap(i)).upper() for i in np.arange(0, n)]
    return colors


def extract_linear_cmap(
    cmap: mpl.colors.Colormap,
    cmin: float = 0.0,
    cmax: float = 1.0,
    n_colors: Optional[int] = None,
) -> List[str]:
    """Extract hexadecimal colors from a linear colormap given a range."""
    if n_colors is None:
        n_colors = cmap.N
    colors = [
        mpl.colors.to_hex(i).upper() for i in cmap(np.linspace(cmin, cmax, n_colors))
    ]
    return colors


def change_axis_format(
    ax: plt.Axes,
    axis_format: str = DEFAULT_AXIS_FORMAT,
    axis: str = DEFAULT_FORMAT_AXIS,
    sci_lower_threshold: float = DEFAULT_SCI_LOWER_THRESHOLD,
    sci_upper_threshold: float = DEFAULT_SCI_UPPER_THRESHOLD,
    sci_power_limits: tuple[int, int] = DEFAULT_SCI_POWER_LIMITS,
) -> None:
    """Change the tick format of specified axes.

    Supported formats:
        - "percentage", "percent", "pct": display tick values as percentages.
        - "scientific notation", "sci": force scientific notation.
        - "auto_sci", "smart_sci", "auto scientific": switch to scientific
          notation only when finite tick magnitudes are very large or very small.
    """

    def _finite_tick_magnitudes(axis_obj: object) -> np.ndarray:
        """Return finite non-zero major tick magnitudes for one axis."""
        ticks = np.asarray(axis_obj.get_majorticklocs(), dtype=float)
        ticks = ticks[np.isfinite(ticks)]
        ticks = np.abs(ticks[ticks != 0])
        return ticks

    def _axis_needs_scientific(axis_obj: object) -> bool:
        """Return whether tick magnitudes exceed the auto-scientific thresholds."""
        tick_magnitudes = _finite_tick_magnitudes(axis_obj)
        if tick_magnitudes.size == 0:
            return False
        return bool(
            np.nanmax(tick_magnitudes) >= sci_upper_threshold
            or np.nanmin(tick_magnitudes) <= sci_lower_threshold
        )

    def _apply_scientific(axis_obj: object, force: bool = True) -> None:
        """Apply a ScalarFormatter configured for scientific notation."""
        formatter = mticker.ScalarFormatter(useMathText=False)
        formatter.set_useOffset(False)
        formatter.set_scientific(True)
        formatter.set_powerlimits(sci_power_limits if force else (-3, 4))
        axis_obj.set_major_formatter(formatter)

    def _apply_plain(axis_obj: object) -> None:
        """Apply a plain ScalarFormatter without offset text."""
        formatter = mticker.ScalarFormatter(useMathText=False)
        formatter.set_useOffset(False)
        formatter.set_scientific(False)
        axis_obj.set_major_formatter(formatter)

    format_key = str(axis_format).lower().replace("-", "_").strip()
    auto_sci_formats = {
        "auto",
        "auto_sci",
        "smart_sci",
        "auto_scientific",
        "smart_scientific",
        "scientific_auto",
        "auto scientific",
        "smart scientific",
    }
    sci_formats = {"scientific notation", "scientific_notation", "sci"}
    pct_formats = {"percentage", "percent", "pct"}

    if axis in ("x", "xy"):
        if format_key in pct_formats:
            ax.xaxis.set_major_locator(mticker.FixedLocator(ax.get_xticks()))
            ax.set_xticklabels(["{:,.0f}".format(100 * x) for x in ax.get_xticks()])
        elif format_key in sci_formats and ax.get_xscale() == "linear":
            _apply_scientific(ax.xaxis, force=True)
        elif format_key in auto_sci_formats and ax.get_xscale() == "linear":
            if _axis_needs_scientific(ax.xaxis):
                _apply_scientific(ax.xaxis, force=True)
            else:
                _apply_plain(ax.xaxis)
    if axis in ("y", "xy"):
        if format_key in pct_formats:
            ax.yaxis.set_major_locator(mticker.FixedLocator(ax.get_yticks()))
            ax.set_yticklabels(["{:,.0f}".format(100 * x) for x in ax.get_yticks()])
        elif format_key in sci_formats and ax.get_yscale() == "linear":
            _apply_scientific(ax.yaxis, force=True)
        elif format_key in auto_sci_formats and ax.get_yscale() == "linear":
            if _axis_needs_scientific(ax.yaxis):
                _apply_scientific(ax.yaxis, force=True)
            else:
                _apply_plain(ax.yaxis)


def apply_smart_axis_notation(
    ax: plt.Axes,
    axis: str = DEFAULT_FORMAT_AXIS,
    sci_lower_threshold: float = DEFAULT_SCI_LOWER_THRESHOLD,
    sci_upper_threshold: float = DEFAULT_SCI_UPPER_THRESHOLD,
    sci_power_limits: tuple[int, int] = DEFAULT_SCI_POWER_LIMITS,
) -> None:
    """Apply automatic scientific notation only to plain numeric axes."""

    def _can_reformat(axis_obj: object, axis_scale: str) -> bool:
        """Return whether an axis uses a plain numeric formatter."""
        if axis_scale != "linear":
            return False
        formatter = axis_obj.get_major_formatter()
        return isinstance(formatter, mticker.ScalarFormatter)

    if axis in ("x", "xy") and _can_reformat(ax.xaxis, ax.get_xscale()):
        change_axis_format(
            ax=ax,
            axis_format="auto_sci",
            axis="x",
            sci_lower_threshold=sci_lower_threshold,
            sci_upper_threshold=sci_upper_threshold,
            sci_power_limits=sci_power_limits,
        )
    if axis in ("y", "xy") and _can_reformat(ax.yaxis, ax.get_yscale()):
        change_axis_format(
            ax=ax,
            axis_format="auto_sci",
            axis="y",
            sci_lower_threshold=sci_lower_threshold,
            sci_upper_threshold=sci_upper_threshold,
            sci_power_limits=sci_power_limits,
        )


def change_fontsize(
    ax: plt.Axes,
    axis_ticks_fontsize: float = DEFAULT_AXIS_TICK_FONTSIZE,
    axis_label_fontsize: float = DEFAULT_AXIS_LABEL_FONTSIZE,
    title_fontsize: float = DEFAULT_TITLE_FONTSIZE,
    axis: str = DEFAULT_FORMAT_AXIS,
) -> None:
    """Change the fontsize of axis ticks, labels, and title."""
    if axis in ("x", "xy"):
        ax.xaxis.label.set_fontsize(axis_label_fontsize)
        for tick in ax.get_xticklabels():
            tick.set_fontsize(axis_ticks_fontsize)
    if axis in ("y", "xy"):
        ax.yaxis.label.set_fontsize(axis_label_fontsize)
        for tick in ax.get_yticklabels():
            tick.set_fontsize(axis_ticks_fontsize)
    ax.title.set_fontsize(title_fontsize)


def change_weight(
    ax: plt.Axes,
    axis_ticks_weight: str = DEFAULT_AXIS_TICK_WEIGHT,
    axis_label_weight: str = DEFAULT_AXIS_LABEL_WEIGHT,
    title_weight: str = DEFAULT_TITLE_WEIGHT,
    axis: str = DEFAULT_FORMAT_AXIS,
) -> None:
    """Change the font weight of axis ticks, labels, and title."""
    if axis in ("x", "xy"):
        ax.xaxis.label.set_weight(axis_label_weight)
        for tick in ax.get_xticklabels():
            tick.set_weight(axis_ticks_weight)
    if axis in ("y", "xy"):
        ax.yaxis.label.set_weight(axis_label_weight)
        for tick in ax.get_yticklabels():
            tick.set_weight(axis_ticks_weight)
    ax.title.set_weight(title_weight)


def change_axis_rotation(
    ax: plt.Axes,
    rotation: float = DEFAULT_TICK_ROTATION,
    axis: str = DEFAULT_ROTATION_AXIS,
) -> None:
    """Rotate the major tick labels of the specified axes."""
    if axis in ("x", "xy"):
        plt.setp(
            ax.xaxis.get_majorticklabels(),
            rotation=rotation,
            ha={0: "center", 90: "center"}.get(rotation, "right"),
            va="top",
        )
    if axis in ("y", "xy"):
        plt.setp(
            ax.yaxis.get_majorticklabels(),
            rotation=rotation,
            ha={0: "right", 90: "center"}.get(rotation, "right"),
            va={0: "center", 90: "center"}.get(rotation, "top"),
        )


def rotate_xticks_if_overlapping(
    ax: plt.Axes,
    rotation: float = 45.0,
    pad: float = 4.0,
    min_gap_px: float = 2.0,
) -> bool:
    """Rotate x tick labels only when their rendered boxes overlap."""
    labels = [label for label in ax.get_xticklabels() if label.get_visible()]
    if len(labels) < 2:
        return False

    try:
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        boxes = [
            label.get_window_extent(renderer=renderer)
            for label in labels
            if label.get_text()
        ]
    except Exception:
        boxes = []

    has_overlap = False
    if len(boxes) >= 2:
        boxes = sorted(boxes, key=lambda box: box.x0)
        has_overlap = any(
            boxes[idx].x1 + min_gap_px > boxes[idx + 1].x0
            for idx in range(len(boxes) - 1)
        )

    if has_overlap:
        plt.setp(labels, rotation=rotation, ha="right", va="top")
        ax.tick_params(axis="x", pad=pad)
    else:
        plt.setp(labels, rotation=0, ha="center", va="top")
        ax.tick_params(axis="x", pad=pad)

    return has_overlap


def axis_size_inches(ax: plt.Axes) -> tuple[float, float]:
    """Return the rendered Axes size in inches."""
    fig_w, fig_h = ax.figure.get_size_inches()
    bbox = ax.get_position()
    return max(bbox.width * fig_w, 1e-6), max(bbox.height * fig_h, 1e-6)


def index_to_tick_labels(index: pd.Index) -> list[str]:
    """Convert regular or MultiIndex labels into display strings."""
    if isinstance(index, pd.MultiIndex):
        return ["-".join(map(str, item)) for item in index.to_list()]
    return [str(item) for item in index]


def compact_tick_label(label_text: str, max_chars: int | None = None) -> str:
    """Shorten dense tick labels while preserving batch/sample cues."""
    parts = re.split("-", str(label_text))
    if len(parts) > 4:
        label_text = "-".join([parts[0]] + parts[4:])
    else:
        label_text = str(label_text)

    if max_chars is None or len(label_text) <= max_chars:
        return label_text

    max_chars = max(6, int(max_chars))
    head_chars = max(3, int(max_chars * 0.55))
    tail_chars = max(2, max_chars - head_chars - 3)
    return f"{label_text[:head_chars]}...{label_text[-tail_chars:]}"


def tick_labels_need_compaction(
    labels: list[str],
    n_items: int,
    axis_inches: float,
    default_size: float = DEFAULT_AXIS_TICK_FONTSIZE,
    chars_per_inch: float = 13.0,
) -> bool:
    """Return whether dense labels require size reduction or compaction."""
    if n_items <= 0:
        return False
    cell_inches = axis_inches / max(1, n_items)
    max_label_len = max([len(str(label)) for label in labels] or [1])
    label_inches = max_label_len / chars_per_inch
    default_cell_points = axis_inches * 72.0 / max(1, n_items)
    return label_inches > cell_inches * 1.25 or default_cell_points < default_size


def dense_tick_fontsize(
    n_items: int,
    axis_inches: float,
    default_size: float = DEFAULT_AXIS_TICK_FONTSIZE,
    max_size: float = DEFAULT_AXIS_TICK_FONTSIZE,
    min_size: float = 1.4,
    fill_ratio: float = 0.70,
    force_dense: bool = False,
) -> float:
    """Estimate a readable font size for one label per plotted item."""
    cell_points = axis_inches * 72.0 / max(1, n_items)
    if not force_dense:
        return min(max_size, default_size)
    return max(min_size, min(max_size, cell_points * fill_ratio))


def heatmap_annotation_fontsize(
    ax: plt.Axes,
    n_rows: int,
    n_cols: int,
    default_size: float = 11.0,
    max_size: float = 12.0,
    min_size: float = 6.0,
    fill_ratio: float = 0.62,
) -> float:
    """Estimate a readable annotation fontsize for heatmap cells."""
    ax_w, ax_h = axis_size_inches(ax)
    row_font = dense_tick_fontsize(
        n_items=max(1, n_rows),
        axis_inches=ax_h,
        default_size=default_size,
        max_size=max_size,
        min_size=min_size,
        fill_ratio=fill_ratio,
        force_dense=True,
    )
    col_font = dense_tick_fontsize(
        n_items=max(1, n_cols),
        axis_inches=ax_w,
        default_size=default_size,
        max_size=max_size,
        min_size=min_size,
        fill_ratio=fill_ratio,
        force_dense=True,
    )
    return min(row_font, col_font)


def score_heatmap_cmap(
    base_color: str = PRIMARY_ACCENT_COLOR,
    n_colors: int = 256,
    cmin: float = 0.2,
    cmax: float = 1.0,
) -> mpl.colors.Colormap:
    """Return the standard white-to-accent score heatmap colormap."""
    custom_cmap = custom_linear_cmap(["white", base_color], n_colors=n_colors)
    return mpl.colors.ListedColormap(
        extract_linear_cmap(
            cmap=custom_cmap,
            cmin=cmin,
            cmax=cmax,
            n_colors=n_colors,
        )
    )


def plot_sample_structure_change_map(
    ax: plt.Axes,
    raw_obj: pd.DataFrame,
    transformed_obj: pd.DataFrame,
    structure_metrics: Mapping[str, float] | None = None,
    sample_cols: pd.Index | None = None,
    seed: int = 123,
    max_features: int | None = 5000,
    scale_log_ratio_tol: float = 0.25,
    scale_rel_delta_tol: float = 0.35,
    title: str = "Sample Structure Change Map",
    compact_style: bool = False,
) -> plt.Axes:
    """Map sample-wise structure changes and report score-aligned global metrics."""
    from . import stat_utils as su

    metrics = dict(structure_metrics or {})
    if not metrics:
        metrics = su.calc_sample_structure_preservation(
            raw_obj=raw_obj,
            transformed_obj=transformed_obj,
            sample_cols=sample_cols,
            max_features=max_features,
            seed=seed,
            scale_log_ratio_tol=scale_log_ratio_tol,
            scale_rel_delta_tol=scale_rel_delta_tol,
        )

    geometry = su.calc_sample_structure_arrays(
        raw_obj=raw_obj,
        transformed_obj=transformed_obj,
        sample_cols=sample_cols,
        max_features=max_features,
        seed=seed,
    )["geometry"]

    def _metric(*keys: str) -> float:
        for key in keys:
            value = su.finite_or_nan(metrics.get(key))
            if np.isfinite(value):
                return float(np.clip(value, 0.0, 1.0))
        return float("nan")

    trustworthiness_score = _metric(
        "sample_structure_trustworthiness",
        "Trustworthiness",
        "trustworthiness",
    )
    rank_score = _metric(
        "sample_structure_rank_preservation",
        "Distance_Rank_Preservation",
        "distance_rank_preservation",
    )
    scale_score = _metric(
        "sample_structure_scale_preservation",
        "Distance_Scale_Preservation",
        "distance_scale_preservation",
    )
    sample_shift = geometry.get("sample_log2_distance_ratio", pd.Series(dtype=float))
    sample_rank = geometry.get("sample_distance_rank_rho", pd.Series(dtype=float))
    sample_trust = geometry.get(
        "sample_neighborhood_trustworthiness",
        pd.Series(dtype=float),
    )
    if not isinstance(sample_shift, pd.Series):
        sample_shift = pd.Series(sample_shift, dtype=float)
    if not isinstance(sample_rank, pd.Series):
        sample_rank = pd.Series(sample_rank, dtype=float)
    if not isinstance(sample_trust, pd.Series):
        sample_trust = pd.Series(sample_trust, dtype=float)
    plot_df = pd.concat(
        [
            pd.to_numeric(sample_shift, errors="coerce").rename("scale_shift"),
            pd.to_numeric(sample_rank, errors="coerce").rename("rank_rho"),
            pd.to_numeric(sample_trust, errors="coerce").rename("local_trust"),
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
            fontsize=5.0 if compact_style else DEFAULT_ANNOTATION_FONTSIZE,
            bbox=ai_ready_text_bbox(),
        )
        ax.set_title(title)
        ax.set_xlabel("Median log2 distance ratio")
        ax.set_ylabel("Distance-rank correlation")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        change_fontsize(ax)
        change_weight(ax)
        return ax

    annotation_size = 5.0 if compact_style else DEFAULT_ANNOTATION_FONTSIZE
    point_size = 14.0 if compact_style else 28.0
    point_lw = 0.15 if compact_style else 0.25
    trust_bin_edges = np.array([0.0, 0.90, 0.95, 0.98, 1.000001])
    trust_bin_labels = ["<0.90", "0.90-0.95", "0.95-0.98", "\u22650.98"]
    trust_bin_colors = [
        get_equivalent_hex(PRIMARY_ACCENT_COLOR, alpha=0.20),
        get_equivalent_hex(PRIMARY_ACCENT_COLOR, alpha=0.42),
        get_equivalent_hex(PRIMARY_ACCENT_COLOR, alpha=0.68),
        get_equivalent_hex(PRIMARY_ACCENT_COLOR, alpha=1.00),
    ]
    cmap = mpl.colors.ListedColormap(trust_bin_colors)
    trust_norm = mpl.colors.BoundaryNorm(
        boundaries=trust_bin_edges,
        ncolors=cmap.N,
        clip=True,
    )
    local_trust = plot_df["local_trust"].fillna(trustworthiness_score)
    scatter = ax.scatter(
        plot_df["scale_shift"],
        plot_df["rank_rho"],
        c=local_trust,
        cmap=cmap,
        norm=trust_norm,
        s=point_size,
        edgecolor="k",
        linewidth=point_lw,
        zorder=3,
    )
    ax.axvline(0.0, color="0.35", linestyle="--", linewidth=0.6, zorder=1)
    ax.axhline(1.0, color="0.55", linestyle=":", linewidth=0.6, zorder=1)
    colorbar_ax = ax.inset_axes([0.92, 0.16, 0.025, 0.30])
    trust_bin_centers = (trust_bin_edges[:-1] + trust_bin_edges[1:]) / 2.0
    colorbar = ax.figure.colorbar(
        scatter,
        cax=colorbar_ax,
        boundaries=trust_bin_edges,
        ticks=trust_bin_centers,
        spacing="uniform",
    )
    colorbar.ax.set_yticklabels(trust_bin_labels)
    colorbar.ax.tick_params(labelsize=annotation_size, length=1.5, width=0.4)
    colorbar.set_label(
        "Local neighborhood preservation",
        fontsize=annotation_size,
        labelpad=2,
    )

    score_note = []
    if np.isfinite(trustworthiness_score):
        score_note.append(f"Global T(k): {trustworthiness_score:.3f}")
    if np.isfinite(rank_score):
        score_note.append(f"Distance-rank preservation: {rank_score:.3f}")
    if np.isfinite(scale_score):
        score_note.append(f"Distance-scale preservation: {scale_score:.3f}")
    if score_note:
        ax.text(
            0.98,
            0.03,
            "\n".join(score_note),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=annotation_size,
            bbox=ai_ready_text_bbox(pad=0.20 if compact_style else 0.30),
            zorder=4,
        )
    ax.set_title(title)
    ax.set_xlabel("Median log2 distance ratio\n(after / before)")
    ax.set_ylabel("Distance-rank correlation")
    x_extent = max(
        float(np.nanmax(np.abs(plot_df["scale_shift"].to_numpy()))),
        0.15,
    )
    ax.set_xlim(-1.15 * x_extent, 1.15 * x_extent)
    y_low = min(float(np.nanmin(plot_df["rank_rho"])), 0.85)
    ax.set_ylim(max(-1.0, y_low - 0.05), 1.03)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    change_fontsize(ax)
    change_weight(ax)
    return ax


def plot_sample_distance_scale_shift_panel(
    ax: plt.Axes,
    raw_obj: pd.DataFrame,
    transformed_obj: pd.DataFrame,
    sample_cols: pd.Index | None = None,
    seed: int = 123,
    max_features: int | None = 5000,
    max_points: int = 250,
    title: str = "Sample Distance-Scale Shift",
    compact_annotations: bool = False,
    compact_style: bool = False,
) -> plt.Axes:
    """Plot actual-sample robust-distance scale shifts after processing.

    The panel is intentionally label-free: it uses actual samples only by default
    and visualizes the per-sample median log2 ratio of transformed/raw robust
    distances. The annotations report raw geometric diagnostics, so this panel
    complements rather than reproduces the Auto score components.
    """
    from . import stat_utils as su

    structure = su.calc_sample_structure_arrays(
        raw_obj=raw_obj,
        transformed_obj=transformed_obj,
        sample_cols=sample_cols,
        max_features=max_features,
        seed=seed,
    )
    geom_metrics = structure["geometry"]
    sample_shift = geom_metrics.get("sample_log2_distance_ratio", pd.Series(dtype=float))
    if not isinstance(sample_shift, pd.Series):
        sample_shift = pd.Series(sample_shift, dtype=float)
    sample_shift = pd.to_numeric(sample_shift, errors="coerce").dropna()

    rank_loss = su.finite_or_nan(geom_metrics.get("rank_loss"))
    rank_rho = 1.0 - rank_loss if np.isfinite(rank_loss) else float("nan")
    median_rel_delta = su.finite_or_nan(geom_metrics.get("median_relative_delta"))
    median_log2_ratio = su.finite_or_nan(
        geom_metrics.get("median_sample_log2_distance_ratio")
    )
    n_neighbors = su.finite_or_nan(geom_metrics.get("n_neighbors"))

    metric_note: list[str] = []
    if compact_annotations:
        trustworthiness = su.finite_or_nan(
            geom_metrics.get("neighborhood_trustworthiness")
        )
        if np.isfinite(trustworthiness):
            metric_note.append(f"T(k): {trustworthiness:.3f}")
        if np.isfinite(rank_rho):
            metric_note.append(f"rho: {rank_rho:.3f}")
        if np.isfinite(median_rel_delta):
            metric_note.append(f"Median |delta d|: {median_rel_delta:.3f}")
    else:
        if np.isfinite(rank_rho):
            metric_note.append(f"Distance rank rho: {rank_rho:.3f}")
        if np.isfinite(median_log2_ratio):
            metric_note.append(
                f"Median log2 distance ratio: {median_log2_ratio:.3f}"
            )
        if np.isfinite(median_rel_delta):
            metric_note.append(f"Median relative |delta d|: {median_rel_delta:.3f}")
        if np.isfinite(n_neighbors):
            metric_note.append(f"kNN reference: k={int(n_neighbors)}")

    if sample_shift.empty:
        ax.text(
            0.5,
            0.5,
            "Insufficient sample data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=DEFAULT_ANNOTATION_FONTSIZE,
            bbox=ai_ready_text_bbox(),
            zorder=10,
        )
        ax.set_title(title)
        ax.set_xlabel("Sample-level log2 distance ratio")
        ax.set_ylabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        change_fontsize(ax)
        change_weight(ax)
        return ax

    x_values = sample_shift.to_numpy(dtype=float)
    box_linewidth = 0.5 if compact_style else 1.2
    median_linewidth = 0.65 if compact_style else 1.2
    whisker_linewidth = 0.5 if compact_style else 1.0
    marker_size = 10 if compact_style else 18
    marker_linewidth = 0.15 if compact_style else 0.25
    ref_linewidth = 0.6 if compact_style else 1.1
    arrow_linewidth = 0.6 if compact_style else 0.9
    compact_note_fontsize = 4.0 if compact_style else 5.5

    ax.boxplot(
        x_values,
        vert=False,
        positions=[0],
        widths=0.36,
        patch_artist=True,
        showfliers=False,
        boxprops={
            "facecolor": "white",
            "edgecolor": get_equivalent_hex(PRIMARY_ACCENT_COLOR, alpha=1.0),
            "linewidth": box_linewidth,
        },
        medianprops={"color": "0.15", "linewidth": median_linewidth},
        whiskerprops={"color": "0.35", "linewidth": whisker_linewidth},
        capprops={"color": "0.35", "linewidth": whisker_linewidth},
    )

    rng = np.random.default_rng(seed)
    plot_values = x_values
    if plot_values.size > max_points:
        keep = rng.choice(plot_values.size, size=max_points, replace=False)
        plot_values = plot_values[keep]
    y_jitter = rng.normal(loc=0.0, scale=0.045, size=plot_values.size)
    ax.scatter(
        plot_values,
        y_jitter,
        color=get_equivalent_hex(PRIMARY_ACCENT_COLOR, alpha=1.0),
        edgecolor="k",
        linewidth=marker_linewidth,
        s=marker_size,
        zorder=3,
    )
    ax.axvline(0, color="0.20", linestyle="--", linewidth=ref_linewidth, zorder=2)

    finite_vals = sample_shift.replace([np.inf, -np.inf], np.nan).dropna()
    x_low, x_high = np.nanpercentile(finite_vals.to_numpy(dtype=float), [2, 98])
    x_abs = max(abs(float(x_low)), abs(float(x_high)), 0.25)
    ax.set_xlim(-x_abs * 1.25, x_abs * 1.25)
    ax.set_ylim(-0.45, 0.45)
    ax.set_yticks([0])
    ax.set_yticklabels(["Actual\nsamples"])

    ax.text(
        0.02,
        0.96,
        "Compressed",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=compact_note_fontsize if compact_annotations else DEFAULT_ANNOTATION_FONTSIZE,
        color="0.35",
        zorder=6,
    )
    ax.annotate(
        "",
        xy=(0.05, 0.90),
        xytext=(0.30, 0.90),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "-|>",
            "color": "0.45",
            "linewidth": arrow_linewidth,
            "mutation_scale": 5 if compact_style else (6 if compact_annotations else 9),
            "shrinkA": 0,
            "shrinkB": 0,
        },
        annotation_clip=False,
    )
    ax.text(
        0.98,
        0.96,
        "Expanded",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=compact_note_fontsize if compact_annotations else DEFAULT_ANNOTATION_FONTSIZE,
        color="0.35",
        zorder=6,
    )
    ax.annotate(
        "",
        xy=(0.95, 0.90),
        xytext=(0.70, 0.90),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "-|>",
            "color": "0.45",
            "linewidth": arrow_linewidth,
            "mutation_scale": 5 if compact_style else (6 if compact_annotations else 9),
            "shrinkA": 0,
            "shrinkB": 0,
        },
        annotation_clip=False,
    )

    if metric_note:
        ax.text(
            0.98,
            0.03,
            "\n".join(metric_note),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=compact_note_fontsize if compact_annotations else DEFAULT_ANNOTATION_FONTSIZE,
            color="0.25",
            clip_on=False,
            bbox=ai_ready_text_bbox(pad=0.25 if compact_annotations else 0.4),
            zorder=10,
        )

    ax.set_title(title)
    ax.set_xlabel("Sample-level log2 distance ratio")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    change_fontsize(ax)
    change_weight(ax)
    return ax


def dense_label_char_limit(n_items: int) -> int:
    """Resolve a conservative character cap for very dense tick labels."""
    if n_items <= 20:
        return 28
    if n_items <= 40:
        return 22
    if n_items <= 70:
        return 16
    return 12


def apply_batch_tick_colors(
    labels: list[Text], tick_color_dict: dict[object, str]
) -> None:
    """Color dense tick labels by batch prefix when possible."""
    for label in labels:
        text = label.get_text()
        for batch, color in tick_color_dict.items():
            if text.startswith(str(batch)):
                label.set_color(color)
                break


def show_values_on_bars(
    axs: Union[plt.Axes, np.ndarray],
    value_format: str = "{:.2f}",
    fontsize: float = 11,
    show_percentage: bool = False,
    pct_type: str = "total",
    stacked: bool = False,
    skip_zero: bool = True,
    threshold_pct: float = 0.10,
) -> None:
    """Annotate bar plots with their values automatically.

    Args:
        axs: A single matplotlib Axes or a numpy array of Axes.
        value_format: Format string for the numerical value.
        fontsize: Font size of the text annotation.
        show_percentage: Whether to calculate and append percentage.
        pct_type: 'total' (plot-level) or 'group' (hue-container-level).
        stacked: Enable intelligent parsing for stacked bar charts.
        skip_zero: Do not annotate patches with a height of exactly 0.
        threshold_pct: Hide annotations if height is less than this
            percentage of the stack's total height (stacked=True only).
    """

    def _draw_label(
        ax: plt.Axes,
        p: mpl.patches.Patch,
        total: float,
        stack_total: Optional[float] = None,
    ) -> None:
        height = p.get_height()
        if skip_zero and height == 0:
            return

        # Check threshold for stacked patches to avoid clutter
        if stacked and stack_total is not None and stack_total > 0:
            if (height / stack_total) < threshold_pct:
                return

        value = value_format.format(height)
        if show_percentage and total > 0:
            value += "\n({:.1f}%)".format(100 * height / total)

        _x = p.get_x() + p.get_width() / 2

        # Split color logic based on placement position
        if stacked:
            _y = p.get_y() + height / 2
            va = "center"
            # Inner patches use auto-adaptive color based on background
            c = get_contrast_color(p.get_facecolor())
        else:
            _y = p.get_y() + height
            va = "bottom" if height >= 0 else "top"
            # Outside annotations must be black against the white canvas
            c = "k"

        ax.text(
            _x, _y, value, ha="center", va=va, rotation=0, fontsize=fontsize, color=c
        )

    def _show_on_single_plot(ax: plt.Axes) -> None:
        if stacked and ax.containers:
            stack_totals = {}
            for container in ax.containers:
                for p in container:
                    if skip_zero and p.get_height() == 0:
                        continue
                    x_center = p.get_x() + p.get_width() / 2
                    cur_top = p.get_y() + p.get_height()
                    if x_center in stack_totals:
                        stack_totals[x_center] = max(stack_totals[x_center], cur_top)
                    else:
                        stack_totals[x_center] = cur_top

            for container in ax.containers:
                for p in container:
                    x_center = p.get_x() + p.get_width() / 2
                    st_tot = stack_totals.get(x_center, 0)
                    _draw_label(ax, p, 0, stack_total=st_tot)

            # Total value on top of the stacked bars (Always Black)
            max_h = max(stack_totals.values()) if stack_totals else 1
            for x_c, total_h in stack_totals.items():
                if skip_zero and total_h == 0:
                    continue
                ax.text(
                    x_c,
                    total_h + (max_h * 0.02),
                    value_format.format(total_h),
                    ha="center",
                    va="bottom",
                    color="k",
                    fontsize=fontsize + 1,
                    fontweight="bold",
                )
        else:
            if show_percentage and pct_type == "group" and ax.containers:
                for container in ax.containers:
                    valid_p = [p for p in container if p in ax.patches]
                    if not valid_p:
                        continue
                    grp_tot = float(np.sum([p.get_height() for p in valid_p]))
                    for p in valid_p:
                        _draw_label(ax, p, grp_tot)
            else:
                v_p = ax.patches
                tot_h = float(np.sum([p.get_height() for p in v_p]))
                for p in v_p:
                    _draw_label(ax, p, tot_h)

    if isinstance(axs, np.ndarray):
        for _, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


def confidence_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes,
    n_std: float = 3.0,
    facecolor: str = "none",
    **kwargs: object,
) -> mpl.patches.Ellipse:
    """Create a plot of the covariance confidence ellipse of `x` and `y`."""
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    if x.size != y.size:
        raise ValueError("x and y must be the same size.")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(np.mean(x), np.mean(y))
    )
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
