"""Collision-aware placement for scientific plot annotations.

This module owns geometry-based placement shared by all pi-metaboqc plotters.
It separates three layout problems that require different constraints:

* free annotation blocks start from eight conventional edge regions and are
  refined continuously in axes coordinates;
* threshold labels move along their reference line and choose one of the two
  perpendicular sides;
* curve labels search positions sampled uniformly over the complete arc.

Placement uses rendered marker extents, patch and text bounding boxes,
legends, colorbars, line paths, and user-supplied blocked regions. Multiple
threshold labels are allocated jointly with a bounded beam search. If no
collision-free internal solution exists, the allocator increases the
perpendicular padding, reserves additional axis space, and only then tries an
external label with a short connector.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.legend import Legend
from matplotlib.transforms import Affine2D

from .plot_utils import DEFAULT_ANNOTATION_FONTSIZE

BBoxTuple = tuple[float, float, float, float]

_FREE_ANCHORS = (
    ((0.96, 0.02), "right", "bottom"),
    ((0.96, 0.98), "right", "top"),
    ((0.04, 0.02), "left", "bottom"),
    ((0.04, 0.98), "left", "top"),
    ((0.96, 0.50), "right", "center"),
    ((0.04, 0.50), "left", "center"),
    ((0.50, 0.98), "center", "top"),
    ((0.50, 0.02), "center", "bottom"),
)


@dataclass(frozen=True)
class _Obstacles:
    """Rendered obstacles expressed in the target axes coordinate system."""

    boxes: np.ndarray
    line_paths: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class _ReferenceCandidate:
    """One constrained placement candidate for a reference-line label."""

    center: tuple[float, float]
    anchor: tuple[float, float]
    bbox: BBoxTuple
    orientation: str
    side: int
    outside: bool
    base_cost: float
    hard_collisions: int


def _overlap_area(first: BBoxTuple, second: BBoxTuple) -> float:
    """Return the intersection area of two axes-coordinate rectangles."""
    x0, y0, x1, y1 = first
    bx0, by0, bx1, by1 = second
    return max(0.0, min(x1, bx1) - max(x0, bx0)) * max(
        0.0, min(y1, by1) - max(y0, by0)
    )


def _expand_bbox(
    bbox: BBoxTuple,
    x_padding: float,
    y_padding: float,
) -> BBoxTuple:
    """Expand an axes-coordinate rectangle by independent margins."""
    x0, y0, x1, y1 = bbox
    return (
        x0 - x_padding,
        y0 - y_padding,
        x1 + x_padding,
        y1 + y_padding,
    )


def _out_of_bounds(bbox: BBoxTuple) -> float:
    """Return total axes-coordinate overflow outside the plotting rectangle."""
    x0, y0, x1, y1 = bbox
    return (
        max(0.0, -x0) + max(0.0, x1 - 1.0) + max(0.0, -y0) + max(0.0, y1 - 1.0)
    )


def _bbox_from_anchor(
    xy: tuple[float, float],
    width: float,
    height: float,
    ha: str,
    va: str,
) -> BBoxTuple:
    """Construct a text rectangle from its anchor and alignment."""
    x, y = xy
    if ha == "left":
        x0, x1 = x, x + width
    elif ha == "right":
        x0, x1 = x - width, x
    else:
        x0, x1 = x - width / 2.0, x + width / 2.0

    if va == "bottom":
        y0, y1 = y, y + height
    elif va == "top":
        y0, y1 = y - height, y
    else:
        y0, y1 = y - height / 2.0, y + height / 2.0
    return (x0, y0, x1, y1)


def _clamp_anchor(
    xy: tuple[float, float],
    width: float,
    height: float,
    ha: str,
    va: str,
    margin: float = 0.015,
) -> tuple[float, float]:
    """Clamp an aligned text anchor so its rectangle remains inside axes."""
    x, y = xy
    if ha == "left":
        x = np.clip(x, margin, 1.0 - width - margin)
    elif ha == "right":
        x = np.clip(x, width + margin, 1.0 - margin)
    else:
        x = np.clip(x, width / 2.0 + margin, 1.0 - width / 2.0 - margin)

    if va == "bottom":
        y = np.clip(y, margin, 1.0 - height - margin)
    elif va == "top":
        y = np.clip(y, height + margin, 1.0 - margin)
    else:
        y = np.clip(
            y,
            height / 2.0 + margin,
            1.0 - height / 2.0 - margin,
        )
    return (float(x), float(y))


def _line_intersects_bbox(path: np.ndarray, bbox: BBoxTuple) -> bool:
    """Return whether a polyline intersects or enters a rectangle."""
    if path.ndim != 2 or path.shape[0] == 0 or path.shape[1] < 2:
        return False
    finite = path[np.all(np.isfinite(path[:, :2]), axis=1), :2]
    if finite.size == 0:
        return False

    x0, y0, x1, y1 = bbox
    inside = (
        (finite[:, 0] >= x0)
        & (finite[:, 0] <= x1)
        & (finite[:, 1] >= y0)
        & (finite[:, 1] <= y1)
    )
    if np.any(inside):
        return True

    for start, end in pairwise(finite):
        dx, dy = end - start
        p = (-dx, dx, -dy, dy)
        q = (
            start[0] - x0,
            x1 - start[0],
            start[1] - y0,
            y1 - start[1],
        )
        lower, upper = 0.0, 1.0
        for p_value, q_value in zip(p, q):
            if np.isclose(p_value, 0.0):
                if q_value < 0.0:
                    break
                continue
            ratio = q_value / p_value
            if p_value < 0.0:
                lower = max(lower, ratio)
            else:
                upper = min(upper, ratio)
            if lower > upper:
                break
        else:
            return True
    return False


def _figure_axes_bboxes(ax: plt.Axes) -> list[BBoxTuple]:
    """Return visible sibling and child axes extents in local axes units."""
    inverse = ax.transAxes.inverted()
    boxes: list[BBoxTuple] = []
    for other in ax.figure.axes:
        if other is ax or not other.get_visible():
            continue
        try:
            display_bbox = other.get_window_extent()
            bbox = display_bbox.transformed(inverse)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue
        values = (bbox.x0, bbox.y0, bbox.x1, bbox.y1)
        if not np.all(np.isfinite(values)):
            continue
        # Patchworklib and twinned axes can add a wrapper axes over the same
        # plotting rectangle. It is not a neighboring panel and must not make
        # every internal annotation candidate look occupied.
        if _overlap_area(values, (0.0, 0.0, 1.0, 1.0)) > 0.5:
            continue
        boxes.append(values)
    return boxes


def _artist_bbox(
    ax: plt.Axes,
    artist: object,
    renderer: object,
) -> BBoxTuple | None:
    """Return one visible artist extent in axes coordinates when available.

    Child axes such as inset colorbars need their tight bounding box because
    the plain axes extent excludes tick labels, axis labels, and offset text.
    Ordinary artists continue to use their window extent.
    """
    try:
        if isinstance(artist, plt.Axes):
            bbox = artist.get_tightbbox(renderer)
            if bbox is None:
                bbox = artist.get_window_extent(renderer)
        else:
            bbox = artist.get_window_extent(renderer)
        bbox = bbox.transformed(ax.transAxes.inverted())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    values = (bbox.x0, bbox.y0, bbox.x1, bbox.y1)
    return values if np.all(np.isfinite(values)) else None


def _path_collection_bboxes(
    ax: plt.Axes,
    collection: PathCollection,
) -> list[BBoxTuple]:
    """Return per-marker extents, including rendered marker size and shape.

    Matplotlib stores marker paths and size transforms separately from their
    offsets. Transform each unique path/size combination once, then translate
    the resulting display-space rectangle to every matching offset. This is
    equivalent to measuring markers individually but avoids hundreds of
    expensive path-extent calls on dense diagnostic scatters.
    """
    try:
        paths = collection.get_paths()
        transforms = np.asarray(collection.get_transforms(), dtype=float)
        offsets = np.ma.asarray(collection.get_offsets(), dtype=float)
        if not paths or offsets.ndim != 2 or offsets.shape[1] < 2:
            return []
        valid = ~np.ma.getmaskarray(offsets[:, :2]).any(axis=1)
        offset_values = np.asarray(offsets[:, :2].filled(np.nan), dtype=float)
        valid &= np.all(np.isfinite(offset_values), axis=1)
        if not np.any(valid):
            return []
        display_centers = collection.get_offset_transform().transform(
            offset_values
        )
    except (AttributeError, TypeError, ValueError):
        return []

    inverse = ax.transAxes.inverted()
    linewidths = np.asarray(collection.get_linewidths(), dtype=float)
    linewidth_pixels = linewidths * ax.figure.dpi / 144.0
    boxes: list[BBoxTuple] = []
    n_paths = len(paths)
    n_transforms = max(1, len(transforms))
    n_linewidths = max(1, len(linewidth_pixels))
    group_keys = {
        (index % n_paths, index % n_transforms, index % n_linewidths)
        for index in np.flatnonzero(valid)
    }
    for path_index, transform_index, linewidth_index in group_keys:
        marker_indices = np.asarray(
            [
                index
                for index in np.flatnonzero(valid)
                if index % n_paths == path_index
                and index % n_transforms == transform_index
                and index % n_linewidths == linewidth_index
            ],
            dtype=int,
        )
        if marker_indices.size == 0:
            continue
        try:
            size_transform = (
                Affine2D(transforms[transform_index])
                if len(transforms)
                else Affine2D()
            )
            marker_path = size_transform.transform_path(paths[path_index])
            marker_path = collection.get_transform().transform_path(marker_path)
            local_bbox = marker_path.get_extents()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue
        edge_pad = (
            float(linewidth_pixels[linewidth_index])
            if len(linewidth_pixels)
            else 0.0
        )
        centers = display_centers[marker_indices]
        lower = centers + np.array(
            [local_bbox.x0 - edge_pad, local_bbox.y0 - edge_pad]
        )
        upper = centers + np.array(
            [local_bbox.x1 + edge_pad, local_bbox.y1 + edge_pad]
        )
        lower_axes = inverse.transform(lower)
        upper_axes = inverse.transform(upper)
        boxes.extend(
            (
                float(x0),
                float(y0),
                float(x1),
                float(y1),
            )
            for (x0, y0), (x1, y1) in zip(lower_axes, upper_axes)
            if np.all(np.isfinite([x0, y0, x1, y1]))
        )
    return boxes


def _line_path_in_axes(ax: plt.Axes, line: object) -> np.ndarray | None:
    """Transform a Line2D path, including blended transforms, into axes units."""
    try:
        display_path = line.get_transform().transform_path(line.get_path())
        vertices = ax.transAxes.inverted().transform(display_path.vertices)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    finite = vertices[np.all(np.isfinite(vertices), axis=1)]
    return finite if finite.size else None


def _explicit_point_boxes(
    ax: plt.Axes,
    occupancy_arrays: Sequence[np.ndarray] | None,
    point_padding: float,
) -> list[BBoxTuple]:
    """Convert explicit data-coordinate point arrays into small safe boxes."""
    if occupancy_arrays is None:
        return []
    boxes: list[BBoxTuple] = []
    for values in occupancy_arrays:
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[1] < 2 or not array.size:
            continue
        array = array[np.all(np.isfinite(array[:, :2]), axis=1), :2]
        if not array.size:
            continue
        axes_points = ax.transAxes.inverted().transform(
            ax.transData.transform(array)
        )
        for x_value, y_value in axes_points:
            boxes.append(
                (
                    x_value - point_padding,
                    y_value - point_padding,
                    x_value + point_padding,
                    y_value + point_padding,
                )
            )
    return boxes


def _collect_obstacles(
    ax: plt.Axes,
    *,
    occupancy_arrays: Sequence[np.ndarray] | None = None,
    occupancy_artists: Sequence[object] | None = None,
    blocked_regions: Sequence[BBoxTuple] | None = None,
    exclude: Sequence[object] | None = None,
    point_padding: float = 0.008,
    include_sibling_axes: bool = False,
) -> _Obstacles:
    """Collect all rendered geometry used by annotation collision scoring."""
    excluded = {id(item) for item in (exclude or [])}
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    boxes = list(blocked_regions or [])
    line_paths: list[np.ndarray] = []
    has_marker_collections = False

    # A PathCollection needs per-marker extents; its aggregate extent is not
    # reliable and would either miss points or reserve the entire data cloud.
    for collection in ax.collections:
        if id(collection) in excluded or not collection.get_visible():
            continue
        if isinstance(collection, PathCollection):
            has_marker_collections = True
            boxes.extend(_path_collection_bboxes(ax, collection))
        else:
            bbox = _artist_bbox(ax, collection, renderer)
            if bbox is not None:
                boxes.append(bbox)

    for patch in ax.patches:
        if (
            patch is ax.patch
            or id(patch) in excluded
            or not patch.get_visible()
        ):
            continue
        # Stage diagnostics use low-zorder rectangles to shade decision
        # regions. They are contextual backgrounds rather than marks that a
        # label can hide, so reserving their complete rectangle would reject
        # every candidate on that side of a threshold. Data bars and other
        # foreground patches retain their normal positive z-order and remain
        # collision obstacles.
        if float(patch.get_zorder()) <= 0.0:
            continue
        bbox = _artist_bbox(ax, patch, renderer)
        if bbox is not None:
            boxes.append(bbox)

    for text_artist in ax.texts:
        if id(text_artist) in excluded or not text_artist.get_visible():
            continue
        if not str(text_artist.get_text()).strip():
            continue
        bbox = _artist_bbox(ax, text_artist, renderer)
        if bbox is not None:
            boxes.append(bbox)

    # Axes titles are separate Text attributes rather than members of
    # ``ax.texts``. Reserving their real extents prevents curve labels near the
    # top edge from colliding with panel titles.
    for title_artist in (
        ax.title,
        getattr(ax, "_left_title", None),
        getattr(ax, "_right_title", None),
    ):
        if title_artist is None or id(title_artist) in excluded:
            continue
        if (
            not title_artist.get_visible()
            or not title_artist.get_text().strip()
        ):
            continue
        bbox = _artist_bbox(ax, title_artist, renderer)
        if bbox is not None:
            boxes.append(bbox)

    selected_artists = list(occupancy_artists or [])
    legend = ax.get_legend()
    if legend is not None:
        selected_artists.append(legend)
    selected_artists.extend(
        artist
        for artist in getattr(ax, "artists", [])
        if isinstance(artist, Legend)
    )
    selected_artists.extend(getattr(ax, "child_axes", []))
    for artist in selected_artists:
        if id(artist) in excluded:
            continue
        if not getattr(artist, "get_visible", lambda: True)():
            continue
        bbox = _artist_bbox(ax, artist, renderer)
        if bbox is not None:
            boxes.append(bbox)

    if include_sibling_axes:
        boxes.extend(_figure_axes_bboxes(ax))
    # Explicit arrays are a fallback for data not already represented by a
    # rendered scatter. Avoid duplicating every marker when callers provide
    # the same data for backward-compatible occupancy hints.
    if not has_marker_collections:
        boxes.extend(_explicit_point_boxes(ax, occupancy_arrays, point_padding))

    for line in ax.lines:
        if id(line) in excluded or not line.get_visible():
            continue
        path = _line_path_in_axes(ax, line)
        if path is not None:
            line_paths.append(path)
    box_array = np.asarray(boxes, dtype=float).reshape((-1, 4))
    return _Obstacles(box_array, tuple(line_paths))


def _score_bbox(
    bbox: BBoxTuple,
    obstacles: _Obstacles,
    *,
    safety_x: float,
    safety_y: float,
    require_inside: bool,
) -> tuple[float, int]:
    """Score one label rectangle against rendered obstacle geometry."""
    padded = _expand_bbox(bbox, safety_x, safety_y)
    hard_collisions = 0
    overlap_area = 0.0
    clearance_cost = 0.0

    # Vectorized rectangle scoring is the main performance path for dense
    # scatters and barplots, where hundreds of rendered marker/patch boxes are
    # compared with dozens of candidate label positions.
    boxes = obstacles.boxes
    if boxes.size:
        padded_width = np.maximum(
            0.0,
            np.minimum(padded[2], boxes[:, 2])
            - np.maximum(padded[0], boxes[:, 0]),
        )
        padded_height = np.maximum(
            0.0,
            np.minimum(padded[3], boxes[:, 3])
            - np.maximum(padded[1], boxes[:, 1]),
        )
        padded_overlap = padded_width * padded_height
        collision_mask = padded_overlap > 0.0
        hard_collisions += int(np.count_nonzero(collision_mask))
        clearance_cost += float(np.sum(padded_overlap[collision_mask]))

        overlap_width = np.maximum(
            0.0,
            np.minimum(bbox[2], boxes[:, 2]) - np.maximum(bbox[0], boxes[:, 0]),
        )
        overlap_height = np.maximum(
            0.0,
            np.minimum(bbox[3], boxes[:, 3]) - np.maximum(bbox[1], boxes[:, 1]),
        )
        overlap_area += float(np.sum(overlap_width * overlap_height))

        clear_mask = ~collision_mask
        if np.any(clear_mask):
            clear_boxes = boxes[clear_mask]
            dx = np.maximum.reduce(
                (
                    clear_boxes[:, 0] - bbox[2],
                    bbox[0] - clear_boxes[:, 2],
                    np.zeros(len(clear_boxes)),
                )
            )
            dy = np.maximum.reduce(
                (
                    clear_boxes[:, 1] - bbox[3],
                    bbox[1] - clear_boxes[:, 3],
                    np.zeros(len(clear_boxes)),
                )
            )
            gaps = np.hypot(dx, dy)
            safety = max(safety_x, safety_y)
            near = gaps < safety
            clearance_cost += float(np.sum((safety - gaps[near]) ** 2))

    for path in obstacles.line_paths:
        if _line_intersects_bbox(path, padded):
            hard_collisions += 1
            clearance_cost += 0.01
        if _line_intersects_bbox(path, bbox):
            overlap_area += 0.01

    # Keep the requested safety distance from the axes boundary as well as
    # from artists; this also leaves a small visual gutter below panel titles.
    overflow = _out_of_bounds(padded) if require_inside else 0.0
    if require_inside and overflow > 0.0:
        hard_collisions += 1
    score = (
        hard_collisions * 1_000_000.0
        + overlap_area * 100_000.0
        + clearance_cost * 10_000.0
        + overflow * 1_000_000.0
    )
    return score, hard_collisions


def _measure_text(
    ax: plt.Axes,
    text: str,
    *,
    fontsize: float,
    rotation: float = 0.0,
    **kwargs: object,
) -> tuple[float, float]:
    """Measure rendered text width and height in axes coordinates."""
    probe = ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        rotation=rotation,
        visible=True,
        **kwargs,
    )
    try:
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        bbox = probe.get_window_extent(renderer).transformed(
            ax.transAxes.inverted()
        )
        return (max(0.01, float(bbox.width)), max(0.01, float(bbox.height)))
    finally:
        probe.remove()


def _annotation_background_padding(
    ax: plt.Axes,
    text_artist: object,
    renderer: object,
) -> float:
    """Return the largest rendered bbox overhang around a text artist.

    ``Text.get_window_extent`` measures glyphs but excludes the ``bbox``
    patch attached to a text artist. Opaque annotation backgrounds therefore
    need an additional layout margin or their rounded padding can touch an
    axis spine even when the glyph rectangle itself is fully inside the axes.

    Args:
        ax: Axes owning the annotation.
        text_artist: Matplotlib text artist being positioned.
        renderer: Active figure renderer.

    Returns:
        Largest non-negative bbox overhang in axes coordinates.
    """
    try:
        bbox_patch = text_artist.get_bbox_patch()
        if bbox_patch is None:
            return 0.0
        inverse = ax.transAxes.inverted()
        text_bbox = text_artist.get_window_extent(renderer).transformed(inverse)
        patch_bbox = bbox_patch.get_window_extent(renderer).transformed(inverse)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return 0.0
    return max(
        0.0,
        float(text_bbox.x0 - patch_bbox.x0),
        float(patch_bbox.x1 - text_bbox.x1),
        float(text_bbox.y0 - patch_bbox.y0),
        float(patch_bbox.y1 - text_bbox.y1),
    )


def artist_bboxes_in_axes(
    ax: plt.Axes,
    artists: Sequence[object] | None = None,
    include_legend: bool = True,
    include_child_axes: bool = True,
) -> list[BBoxTuple]:
    """Return visible artist extents in the target axes coordinate system.

    Path collections are returned as one rectangle per rendered marker. This
    makes the helper suitable for layout scoring, unlike aggregate collection
    extents that discard marker size information.

    Args:
        ax: Target Matplotlib axes.
        artists: Explicit artists to measure.
        include_legend: Include legends owned by ``ax``.
        include_child_axes: Include inset axes such as colorbars.

    Returns:
        Axes-coordinate ``(x0, y0, x1, y1)`` rectangles.
    """
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    selected = list(artists or [])
    if include_legend:
        legend = ax.get_legend()
        if legend is not None:
            selected.append(legend)
        selected.extend(
            artist
            for artist in getattr(ax, "artists", [])
            if isinstance(artist, Legend)
        )
    if include_child_axes:
        selected.extend(getattr(ax, "child_axes", []))

    boxes: list[BBoxTuple] = []
    for artist in selected:
        if not getattr(artist, "get_visible", lambda: True)():
            continue
        if isinstance(artist, PathCollection):
            boxes.extend(_path_collection_bboxes(ax, artist))
            continue
        bbox = _artist_bbox(ax, artist, renderer)
        if bbox is not None:
            boxes.append(bbox)
    return boxes


def legend_bboxes_in_axes(
    ax: plt.Axes,
    legend: object | None = None,
) -> list[BBoxTuple]:
    """Return local legend extents as axes-coordinate rectangles."""
    return artist_bboxes_in_axes(
        ax,
        artists=[legend] if legend is not None else None,
        include_legend=legend is None,
        include_child_axes=False,
    )


def legend_is_inside_axes(ax: plt.Axes, legend: object | None = None) -> bool:
    """Return whether a local legend overlaps the axes plotting rectangle."""
    for x0, y0, x1, y1 in legend_bboxes_in_axes(ax, legend=legend):
        if min(1.0, x1) > max(0.0, x0) and min(1.0, y1) > max(0.0, y0):
            return True
    return False


def _free_candidate_score(
    xy: tuple[float, float],
    *,
    width: float,
    height: float,
    ha: str,
    va: str,
    obstacles: _Obstacles,
    origin: tuple[float, float],
    safety: float,
) -> tuple[float, int, BBoxTuple]:
    """Score a free annotation position including a small drift penalty."""
    bbox = _bbox_from_anchor(xy, width, height, ha, va)
    score, hard = _score_bbox(
        bbox,
        obstacles,
        safety_x=safety,
        safety_y=safety,
        require_inside=True,
    )
    drift = float(np.hypot(xy[0] - origin[0], xy[1] - origin[1]))
    return score + drift * 0.1, hard, bbox


def _refine_free_candidate(
    start: tuple[float, float],
    *,
    width: float,
    height: float,
    ha: str,
    va: str,
    obstacles: _Obstacles,
    safety: float,
) -> tuple[tuple[float, float], float, int, BBoxTuple]:
    """Continuously refine one of the eight conventional annotation regions."""
    origin = _clamp_anchor(start, width, height, ha, va)
    current = origin
    score, hard, bbox = _free_candidate_score(
        current,
        width=width,
        height=height,
        ha=ha,
        va=va,
        obstacles=obstacles,
        origin=origin,
        safety=safety,
    )
    directions = (
        (-1.0, -1.0),
        (-1.0, 0.0),
        (-1.0, 1.0),
        (0.0, -1.0),
        (0.0, 1.0),
        (1.0, -1.0),
        (1.0, 0.0),
        (1.0, 1.0),
    )
    for step in (0.08, 0.04, 0.02, 0.01, 0.005):
        improved = True
        while improved:
            improved = False
            best = (current, score, hard, bbox)
            for dx, dy in directions:
                proposed = _clamp_anchor(
                    (current[0] + dx * step, current[1] + dy * step),
                    width,
                    height,
                    ha,
                    va,
                )
                if (
                    np.hypot(proposed[0] - origin[0], proposed[1] - origin[1])
                    > 0.24
                ):
                    continue
                candidate_score, candidate_hard, candidate_bbox = (
                    _free_candidate_score(
                        proposed,
                        width=width,
                        height=height,
                        ha=ha,
                        va=va,
                        obstacles=obstacles,
                        origin=origin,
                        safety=safety,
                    )
                )
                if candidate_score + 1e-9 < best[1]:
                    best = (
                        proposed,
                        candidate_score,
                        candidate_hard,
                        candidate_bbox,
                    )
            if best[0] != current:
                current, score, hard, bbox = best
                improved = True
    return current, score, hard, bbox


def _resolve_data_arrays(
    ax: plt.Axes,
    occupancy_arrays: Sequence[np.ndarray] | None,
) -> list[np.ndarray]:
    """Return finite data arrays for axis-expansion calculations."""
    arrays: list[np.ndarray] = []
    if occupancy_arrays is not None:
        arrays.extend(
            np.asarray(values, dtype=float) for values in occupancy_arrays
        )
    else:
        for collection in ax.collections:
            try:
                arrays.append(np.asarray(collection.get_offsets(), dtype=float))
            except (AttributeError, TypeError, ValueError):
                continue
        for line in ax.lines:
            try:
                x_values = np.asarray(line.get_xdata(), dtype=float)
                y_values = np.asarray(line.get_ydata(), dtype=float)
            except (AttributeError, TypeError, ValueError):
                continue
            if x_values.size == y_values.size and x_values.size:
                arrays.append(np.column_stack((x_values, y_values)))
    return [
        values[np.all(np.isfinite(values[:, :2]), axis=1), :2]
        for values in arrays
        if values.ndim == 2 and values.shape[1] >= 2 and values.size
    ]


def _expand_axis_for_annotation(
    ax: plt.Axes,
    *,
    side: str,
    occupancy_arrays: Sequence[np.ndarray] | None,
    reserve_fraction: float,
    axis: str,
) -> bool:
    """Reserve a data-space band on one edge and report whether it changed."""
    arrays = _resolve_data_arrays(ax, occupancy_arrays)
    if arrays:
        points = np.vstack(arrays)
        column = 1 if axis == "y" else 0
        content_low = float(np.nanmin(points[:, column]))
        content_high = float(np.nanmax(points[:, column]))
    else:
        data_bbox = ax.dataLim
        content_low = data_bbox.y0 if axis == "y" else data_bbox.x0
        content_high = data_bbox.y1 if axis == "y" else data_bbox.x1
    if not np.all(np.isfinite([content_low, content_high])):
        return False

    low, high = ax.get_ylim() if axis == "y" else ax.get_xlim()
    scale = ax.get_yscale() if axis == "y" else ax.get_xscale()
    use_log = scale == "log" and min(content_low, content_high, low, high) > 0
    if use_log:
        content_low, content_high, low, high = np.log10(
            [content_low, content_high, low, high]
        )
    fraction = min(0.42, max(0.08, float(reserve_fraction)))
    if side in {"bottom", "left"}:
        required = high - (high - content_low) / (1.0 - fraction)
        limits = (min(low, required), high)
    else:
        required = low + (content_high - low) / (1.0 - fraction)
        limits = (low, max(high, required))
    if use_log:
        limits = tuple(float(10**value) for value in limits)

    previous = ax.get_ylim() if axis == "y" else ax.get_xlim()
    if axis == "y":
        ax.set_ylim(*limits)
    else:
        ax.set_xlim(*limits)
    return not np.allclose(previous, limits)


def place_annotation_in_least_occupied_corner(
    ax: plt.Axes,
    text_artist: object,
    occupancy_arrays: Sequence[np.ndarray] | None = None,
    blocked_regions: Sequence[BBoxTuple] | None = None,
    point_padding: float = 0.008,
) -> dict[str, object]:
    """Place a free annotation using eight regions plus continuous refinement.

    Args:
        ax: Target axes.
        text_artist: Existing Matplotlib text artist to reposition.
        occupancy_arrays: Optional data-coordinate point arrays.
        blocked_regions: Additional axes-coordinate rectangles.
        point_padding: Minimum axes-coordinate clearance around point centers
            supplied through ``occupancy_arrays``.

    Returns:
        Placement metadata, including the final rectangle and collision count.
    """
    try:
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        measured = text_artist.get_window_extent(renderer).transformed(
            ax.transAxes.inverted()
        )
        width = min(0.92, max(0.02, float(measured.width)))
        height = min(0.75, max(0.02, float(measured.height)))
        background_padding = _annotation_background_padding(
            ax,
            text_artist,
            renderer,
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        width, height = 0.46, 0.18
        background_padding = 0.0

    obstacles = _collect_obstacles(
        ax,
        occupancy_arrays=occupancy_arrays,
        blocked_regions=blocked_regions,
        exclude=[text_artist],
        point_padding=point_padding,
    )
    visual_gutter = max(_axes_padding(ax, 1.0))
    safety = max(
        0.008,
        point_padding,
        background_padding + visual_gutter,
    )
    placements: list[dict[str, object]] = []
    for rank, (start, ha, va) in enumerate(_FREE_ANCHORS):
        xy, score, hard, bbox = _refine_free_candidate(
            start,
            width=width,
            height=height,
            ha=ha,
            va=va,
            obstacles=obstacles,
            safety=safety,
        )
        placements.append(
            {
                "xy": xy,
                "ha": ha,
                "va": va,
                "bbox": bbox,
                "score": score + rank * 1e-5,
                "hard_collisions": hard,
                "point_overlap": hard,
                "blocked_overlap": float(hard > 0),
            }
        )
    best = min(placements, key=lambda item: float(item["score"]))
    text_artist.set_position(best["xy"])
    text_artist.set_ha(str(best["ha"]))
    text_artist.set_va(str(best["va"]))
    if background_padding > 0.0:
        best = {
            **best,
            "bbox": _expand_bbox(
                best["bbox"],
                background_padding,
                background_padding,
            ),
        }
    return best


def place_annotation_with_legend_awareness(
    ax: plt.Axes,
    text_artist: object,
    occupancy_arrays: Sequence[np.ndarray] | None = None,
    blocked_regions: Sequence[BBoxTuple] | None = None,
    legend: object | None = None,
    expand_axes: bool = False,
) -> dict[str, object]:
    """Place a free annotation after legend and colorbar layout is finalized."""
    regions = list(blocked_regions or [])
    if legend is not None and legend_is_inside_axes(ax, legend=legend):
        regions.extend(legend_bboxes_in_axes(ax, legend=legend))
    placement = place_annotation_in_least_occupied_corner(
        ax=ax,
        text_artist=text_artist,
        occupancy_arrays=occupancy_arrays,
        blocked_regions=regions,
    )
    if expand_axes and int(placement["hard_collisions"]) > 0:
        bbox = placement["bbox"]
        side = "bottom" if placement["va"] == "bottom" else "top"
        changed = _expand_axis_for_annotation(
            ax,
            side=side,
            occupancy_arrays=occupancy_arrays,
            reserve_fraction=float(bbox[3]) - float(bbox[1]) + 0.04,
            axis="y",
        )
        if changed:
            placement = place_annotation_with_legend_awareness(
                ax=ax,
                text_artist=text_artist,
                occupancy_arrays=occupancy_arrays,
                blocked_regions=blocked_regions,
                legend=legend,
                expand_axes=False,
            )
    return placement


def add_auto_annotation(
    ax: plt.Axes,
    text: str,
    occupancy_arrays: Sequence[np.ndarray] | None = None,
    blocked_regions: Sequence[BBoxTuple] | None = None,
    expand_axes: bool = True,
    **kwargs: object,
) -> object:
    """Create and place a free annotation after fixed guides are rendered."""
    artist = ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        clip_on=False,
        zorder=10,
        **kwargs,
    )
    place_annotation_with_legend_awareness(
        ax=ax,
        text_artist=artist,
        occupancy_arrays=occupancy_arrays,
        blocked_regions=blocked_regions,
        expand_axes=expand_axes,
    )
    return artist


def _axes_padding(ax: plt.Axes, points: float) -> tuple[float, float]:
    """Convert a physical point distance to x/y axes fractions."""
    display_bbox = ax.get_window_extent()
    pixels = points * ax.figure.dpi / 72.0
    return (
        pixels / max(float(display_bbox.width), 1.0),
        pixels / max(float(display_bbox.height), 1.0),
    )


def _reference_coordinate(
    ax: plt.Axes,
    value: float,
    orientation: str,
) -> float:
    """Convert a data-space reference value to one axes coordinate."""
    if orientation == "horizontal":
        display = ax.get_yaxis_transform().transform((0.0, value))
        return float(ax.transAxes.inverted().transform(display)[1])
    display = ax.get_xaxis_transform().transform((value, 0.0))
    return float(ax.transAxes.inverted().transform(display)[0])


def _reference_candidates(
    ax: plt.Axes,
    reference: Mapping[str, object],
    obstacles: _Obstacles,
    *,
    pad_multiplier: float,
    outside: bool,
    safety_points: float,
    text_size: tuple[float, float] | None = None,
    line_coordinate: float | None = None,
    outside_obstacles: _Obstacles | None = None,
) -> list[_ReferenceCandidate]:
    """Generate continuously spaced candidates for one constrained label."""
    orientation = str(reference.get("orientation", "horizontal"))
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError("orientation must be 'horizontal' or 'vertical'")
    value = float(reference["value"])
    text = str(reference["text"])
    fontsize = float(reference.get("fontsize", DEFAULT_ANNOTATION_FONTSIZE))
    rotation = 90.0 if orientation == "vertical" else 0.0
    width, height = text_size or _measure_text(
        ax, text, fontsize=fontsize, rotation=rotation
    )
    x_pad, y_pad = _axes_padding(ax, safety_points)
    base_pad = float(reference.get("pad_points", 3.0)) * pad_multiplier
    pad_x, pad_y = _axes_padding(ax, base_pad)
    if line_coordinate is None:
        line_coordinate = _reference_coordinate(ax, value, orientation)
    external_geometry = outside_obstacles or obstacles
    candidates: list[_ReferenceCandidate] = []

    if orientation == "horizontal":
        feasible = np.linspace(
            width / 2.0 + 0.015, 1.0 - width / 2.0 - 0.015, 61
        )
        # A small preference for the right side matches conventional threshold
        # labeling while the collision score remains dominant.
        ordered = sorted(feasible, key=lambda item: abs(item - 0.88))
        for side in (1, -1):
            center_y = line_coordinate + side * (height / 2.0 + pad_y)
            for rank, along in enumerate(ordered):
                center = (float(along), float(center_y))
                bbox = (
                    center[0] - width / 2.0,
                    center[1] - height / 2.0,
                    center[0] + width / 2.0,
                    center[1] + height / 2.0,
                )
                score, hard = _score_bbox(
                    bbox,
                    obstacles,
                    safety_x=x_pad,
                    safety_y=y_pad,
                    require_inside=True,
                )
                candidates.append(
                    _ReferenceCandidate(
                        center=center,
                        anchor=(center[0], line_coordinate),
                        bbox=bbox,
                        orientation=orientation,
                        side=side,
                        outside=False,
                        base_cost=score
                        + rank * 0.002
                        + (0.01 if side < 0 else 0.0),
                        hard_collisions=hard,
                    )
                )
        if outside:
            for side_x in (1, -1):
                x_value = (
                    1.0 + width / 2.0 + pad_x
                    if side_x > 0
                    else -width / 2.0 - pad_x
                )
                for side in (1, -1):
                    y_value = line_coordinate + side * (height / 2.0 + pad_y)
                    center = (x_value, y_value)
                    bbox = (
                        center[0] - width / 2.0,
                        center[1] - height / 2.0,
                        center[0] + width / 2.0,
                        center[1] + height / 2.0,
                    )
                    score, hard = _score_bbox(
                        bbox,
                        external_geometry,
                        safety_x=x_pad,
                        safety_y=y_pad,
                        require_inside=False,
                    )
                    edge = 1.0 if side_x > 0 else 0.0
                    candidates.append(
                        _ReferenceCandidate(
                            center=center,
                            anchor=(edge, line_coordinate),
                            bbox=bbox,
                            orientation=orientation,
                            side=side,
                            outside=True,
                            base_cost=score + 20.0,
                            hard_collisions=hard,
                        )
                    )
    else:
        feasible = np.linspace(
            height / 2.0 + 0.015, 1.0 - height / 2.0 - 0.015, 61
        )
        # Vertical limits conventionally read from the top downward.
        ordered = sorted(feasible, key=lambda item: abs(item - 0.88))
        for side in (-1, 1):
            center_x = line_coordinate + side * (width / 2.0 + pad_x)
            for rank, along in enumerate(ordered):
                center = (float(center_x), float(along))
                bbox = (
                    center[0] - width / 2.0,
                    center[1] - height / 2.0,
                    center[0] + width / 2.0,
                    center[1] + height / 2.0,
                )
                score, hard = _score_bbox(
                    bbox,
                    obstacles,
                    safety_x=x_pad,
                    safety_y=y_pad,
                    require_inside=True,
                )
                candidates.append(
                    _ReferenceCandidate(
                        center=center,
                        anchor=(line_coordinate, center[1]),
                        bbox=bbox,
                        orientation=orientation,
                        side=side,
                        outside=False,
                        base_cost=score
                        + rank * 0.002
                        + (0.01 if side > 0 else 0.0),
                        hard_collisions=hard,
                    )
                )
        if outside:
            for side_y in (1, -1):
                y_value = (
                    1.0 + height / 2.0 + pad_y
                    if side_y > 0
                    else -height / 2.0 - pad_y
                )
                for side in (-1, 1):
                    x_value = line_coordinate + side * (width / 2.0 + pad_x)
                    center = (x_value, y_value)
                    bbox = (
                        center[0] - width / 2.0,
                        center[1] - height / 2.0,
                        center[0] + width / 2.0,
                        center[1] + height / 2.0,
                    )
                    score, hard = _score_bbox(
                        bbox,
                        external_geometry,
                        safety_x=x_pad,
                        safety_y=y_pad,
                        require_inside=False,
                    )
                    edge = 1.0 if side_y > 0 else 0.0
                    candidates.append(
                        _ReferenceCandidate(
                            center=center,
                            anchor=(line_coordinate, edge),
                            bbox=bbox,
                            orientation=orientation,
                            side=side,
                            outside=True,
                            base_cost=score + 20.0,
                            hard_collisions=hard,
                        )
                    )
    return candidates


def _joint_reference_solution(
    candidate_groups: Sequence[Sequence[_ReferenceCandidate]],
    *,
    safety_x: float,
    safety_y: float,
    beam_width: int = 256,
) -> tuple[list[_ReferenceCandidate], float, int]:
    """Allocate all reference labels jointly with bounded beam search."""
    states: list[tuple[float, int, list[_ReferenceCandidate]]] = [(0.0, 0, [])]
    for group in candidate_groups:
        ranked = sorted(group, key=lambda item: item.base_cost)[:48]
        expanded_states: list[tuple[float, int, list[_ReferenceCandidate]]] = []
        for state_score, state_hard, selected in states:
            for candidate in ranked:
                pair_score = 0.0
                pair_hard = 0
                padded = _expand_bbox(candidate.bbox, safety_x, safety_y)
                for prior in selected:
                    prior_padded = _expand_bbox(prior.bbox, safety_x, safety_y)
                    if _overlap_area(padded, prior_padded) > 0.0:
                        pair_hard += 1
                        pair_score += 1_000_000.0
                        pair_score += (
                            _overlap_area(
                                candidate.bbox,
                                prior.bbox,
                            )
                            * 100_000.0
                        )
                expanded_states.append(
                    (
                        state_score + candidate.base_cost + pair_score,
                        state_hard + candidate.hard_collisions + pair_hard,
                        [*selected, candidate],
                    )
                )
        states = sorted(expanded_states, key=lambda item: (item[1], item[0]))[
            :beam_width
        ]
    best_score, best_hard, best_candidates = min(
        states,
        key=lambda item: (item[1], item[0]),
    )
    return best_candidates, best_score, best_hard


def _render_reference_annotations(
    ax: plt.Axes,
    references: Sequence[Mapping[str, object]],
    placements: Sequence[_ReferenceCandidate],
) -> list[object]:
    """Render a jointly selected set of threshold-label placements."""
    annotations: list[object] = []
    for reference, placement in zip(references, placements):
        arrowprops = None
        if placement.outside:
            arrowprops = {
                "arrowstyle": "-",
                "color": str(reference.get("color", "k")),
                "linewidth": 0.5,
                "shrinkA": 0,
                "shrinkB": 0,
            }
        annotation = ax.annotate(
            str(reference["text"]),
            xy=placement.anchor,
            xycoords=ax.transAxes,
            xytext=placement.center,
            textcoords=ax.transAxes,
            ha="center",
            va="center",
            rotation=(
                90
                if str(reference.get("orientation", "horizontal")) == "vertical"
                else 0
            ),
            fontsize=float(
                reference.get("fontsize", DEFAULT_ANNOTATION_FONTSIZE)
            ),
            color=str(reference.get("color", "k")),
            arrowprops=arrowprops,
            annotation_clip=False,
            clip_on=False,
            zorder=5,
        )
        annotations.append(annotation)
    return annotations


def _expand_for_reference_solution(
    ax: plt.Axes,
    references: Sequence[Mapping[str, object]],
    occupancy_arrays: Sequence[np.ndarray] | None,
) -> bool:
    """Reserve the conventional trailing edge for constrained line labels."""
    orientations = [
        str(reference.get("orientation", "horizontal"))
        for reference in references
    ]
    if orientations.count("vertical") > orientations.count("horizontal"):
        return _expand_axis_for_annotation(
            ax,
            side="top",
            occupancy_arrays=occupancy_arrays,
            reserve_fraction=0.16,
            axis="y",
        )
    return _expand_axis_for_annotation(
        ax,
        side="right",
        occupancy_arrays=occupancy_arrays,
        reserve_fraction=0.18,
        axis="x",
    )


def annotate_reference_lines(
    ax: plt.Axes,
    references: Sequence[Mapping[str, object]],
    occupancy_arrays: Sequence[np.ndarray] | None = None,
    occupancy_artists: Sequence[object] | None = None,
    blocked_regions: Sequence[BBoxTuple] | None = None,
    allow_outside: bool = True,
    expand_axis: bool = True,
    safety_points: float = 2.0,
    point_padding: float = 0.008,
) -> list[object]:
    """Jointly place labels for horizontal and vertical reference lines.

    Horizontal labels move continuously along x and choose above or below the
    line. Vertical labels move continuously along y and choose left or right.
    Candidate sets for all lines are optimized together, so a locally optimal
    first label cannot strand a later label.

    The fallback order is deterministic: normal internal padding, enlarged
    internal padding, axis expansion, and finally external labels with short
    connectors. Existing markers, patches, legends, colorbars, text, and lines
    all contribute rendered geometry to the collision score.

    Args:
        ax: Target Matplotlib axes.
        references: Mappings containing at least ``value`` and ``text``.
        occupancy_arrays: Optional data-coordinate point arrays.
        occupancy_artists: Additional artists to reserve.
        blocked_regions: Additional axes-coordinate rectangles.
        allow_outside: Allow the external-label fallback stage.
        expand_axis: Allow axis expansion before the external-label fallback.
        safety_points: Required physical clearance from obstacles.
        point_padding: Axes-coordinate radius for explicit occupancy points.

    Returns:
        Rendered annotations in the same order as ``references``.
    """
    reference_list = list(references)
    if not reference_list:
        return []
    for reference in reference_list:
        if "value" not in reference or "text" not in reference:
            raise ValueError("Each reference requires 'value' and 'text'.")

    obstacles = _collect_obstacles(
        ax,
        occupancy_arrays=occupancy_arrays,
        occupancy_artists=occupancy_artists,
        blocked_regions=blocked_regions,
        point_padding=point_padding,
    )
    sibling_boxes = np.asarray(_figure_axes_bboxes(ax), dtype=float).reshape(
        (-1, 4)
    )
    outside_boxes = (
        np.vstack((obstacles.boxes, sibling_boxes))
        if sibling_boxes.size
        else obstacles.boxes
    )
    outside_obstacles = _Obstacles(outside_boxes, obstacles.line_paths)
    safety_x, safety_y = _axes_padding(ax, safety_points)
    best_solution: tuple[list[_ReferenceCandidate], float, int] | None = None
    prepared_geometry: list[tuple[tuple[float, float], float]] = []
    for reference in reference_list:
        orientation = str(reference.get("orientation", "horizontal"))
        fontsize = float(reference.get("fontsize", DEFAULT_ANNOTATION_FONTSIZE))
        prepared_geometry.append(
            (
                _measure_text(
                    ax,
                    str(reference["text"]),
                    fontsize=fontsize,
                    rotation=90.0 if orientation == "vertical" else 0.0,
                ),
                _reference_coordinate(
                    ax,
                    float(reference["value"]),
                    orientation,
                ),
            )
        )

    # Each stage is evaluated jointly. We stop at the first collision-free
    # solution to preserve the documented fallback priority.
    stages = [(1.0, False), (1.8, False), (2.8, False)]
    for pad_multiplier, outside in stages:
        groups = [
            _reference_candidates(
                ax,
                reference,
                obstacles,
                pad_multiplier=pad_multiplier,
                outside=outside,
                safety_points=safety_points,
                text_size=prepared_geometry[index][0],
                line_coordinate=prepared_geometry[index][1],
                outside_obstacles=outside_obstacles,
            )
            for index, reference in enumerate(reference_list)
        ]
        solution = _joint_reference_solution(
            groups,
            safety_x=safety_x,
            safety_y=safety_y,
        )
        if best_solution is None or (solution[2], solution[1]) < (
            best_solution[2],
            best_solution[1],
        ):
            best_solution = solution
        if solution[2] == 0:
            return _render_reference_annotations(
                ax, reference_list, solution[0]
            )

    if expand_axis and _expand_for_reference_solution(
        ax,
        reference_list,
        occupancy_arrays,
    ):
        return annotate_reference_lines(
            ax=ax,
            references=reference_list,
            occupancy_arrays=occupancy_arrays,
            occupancy_artists=occupancy_artists,
            blocked_regions=blocked_regions,
            allow_outside=allow_outside,
            expand_axis=False,
            safety_points=safety_points,
            point_padding=point_padding,
        )

    # External labels are a true last resort. They are intentionally tested
    # only after internal candidates and optional axis expansion so compact
    # dashboard panels do not prefer connectors merely because their rendered
    # text occupies a larger fraction of the axes.
    if allow_outside:
        outside_groups = [
            _reference_candidates(
                ax,
                reference,
                obstacles,
                pad_multiplier=2.8,
                outside=True,
                safety_points=safety_points,
                text_size=prepared_geometry[index][0],
                line_coordinate=prepared_geometry[index][1],
                outside_obstacles=outside_obstacles,
            )
            for index, reference in enumerate(reference_list)
        ]
        outside_solution = _joint_reference_solution(
            outside_groups,
            safety_x=safety_x,
            safety_y=safety_y,
        )
        if outside_solution[2] == 0:
            return _render_reference_annotations(
                ax,
                reference_list,
                outside_solution[0],
            )
        if best_solution is None or (
            outside_solution[2],
            outside_solution[1],
        ) < (best_solution[2], best_solution[1]):
            best_solution = outside_solution
    assert best_solution is not None
    return _render_reference_annotations(ax, reference_list, best_solution[0])


def annotate_reference_line(
    ax: plt.Axes,
    value: float,
    text: str,
    orientation: str = "horizontal",
    color: str = "k",
    fontsize: float = DEFAULT_ANNOTATION_FONTSIZE,
    pad_points: float = 3.0,
    occupancy_arrays: Sequence[np.ndarray] | None = None,
    occupancy_artists: Sequence[object] | None = None,
    blocked_regions: Sequence[BBoxTuple] | None = None,
    allow_outside: bool = True,
    expand_axis: bool = True,
    point_padding: float = 0.015,
) -> object:
    """Place one reference label through the shared constrained allocator."""
    return annotate_reference_lines(
        ax=ax,
        references=[
            {
                "value": value,
                "text": text,
                "orientation": orientation,
                "color": color,
                "fontsize": fontsize,
                "pad_points": pad_points,
            }
        ],
        occupancy_arrays=occupancy_arrays,
        occupancy_artists=occupancy_artists,
        blocked_regions=blocked_regions,
        allow_outside=allow_outside,
        expand_axis=expand_axis,
        point_padding=point_padding,
    )[0]


def _arc_length_samples(
    axes_path: np.ndarray,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate path positions and unit tangents at uniform arc lengths."""
    segments = np.diff(axes_path, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    keep = np.concatenate(([True], np.diff(cumulative) > 1e-12))
    cumulative = cumulative[keep]
    path = axes_path[keep]
    if len(path) == 1 or cumulative[-1] <= 0.0:
        return path, np.array([[1.0, 0.0]])

    targets = np.linspace(0.0, cumulative[-1], n_samples)
    x_values = np.interp(targets, cumulative, path[:, 0])
    y_values = np.interp(targets, cumulative, path[:, 1])
    positions = np.column_stack((x_values, y_values))
    tangents = np.gradient(positions, axis=0)
    norms = np.linalg.norm(tangents, axis=1)
    norms[norms <= 1e-12] = 1.0
    tangents /= norms[:, None]
    return positions, tangents


def annotate_reference_curve(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    text: str,
    color: str = "k",
    fontsize: float = DEFAULT_ANNOTATION_FONTSIZE,
    pad_points: float = 3.0,
    occupancy_arrays: Sequence[np.ndarray] | None = None,
    blocked_regions: Sequence[BBoxTuple] | None = None,
) -> object:
    """Place a curve label using dense, arc-length-uniform candidates."""
    x_data = np.asarray(x_values, dtype=float)
    y_data = np.asarray(y_values, dtype=float)
    finite = np.isfinite(x_data) & np.isfinite(y_data)
    x_data, y_data = x_data[finite], y_data[finite]
    if x_data.size == 0:
        raise ValueError("x_values and y_values must contain finite pairs")

    data_path = np.column_stack((x_data, y_data))
    axes_path = ax.transAxes.inverted().transform(
        ax.transData.transform(data_path)
    )
    n_samples = int(np.clip(len(axes_path) * 2, 48, 160))
    positions, tangents = _arc_length_samples(axes_path, n_samples)
    width, height = _measure_text(ax, text, fontsize=fontsize)
    pad_x, pad_y = _axes_padding(ax, pad_points)
    safety_x, safety_y = _axes_padding(ax, 2.0)

    obstacles = _collect_obstacles(
        ax,
        occupancy_arrays=occupancy_arrays,
        blocked_regions=blocked_regions,
        point_padding=0.008,
    )
    obstacles = _Obstacles(
        obstacles.boxes,
        (*obstacles.line_paths, axes_path),
    )
    candidates: list[
        tuple[float, int, tuple[float, float], tuple[float, float]]
    ] = []
    for rank, (position, tangent) in enumerate(zip(positions, tangents)):
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        support = abs(normal[0]) * width / 2.0 + abs(normal[1]) * height / 2.0
        pad = max(pad_x, pad_y)
        for side in (1, -1):
            center = position + side * normal * (support + pad)
            bbox = (
                center[0] - width / 2.0,
                center[1] - height / 2.0,
                center[0] + width / 2.0,
                center[1] + height / 2.0,
            )
            score, hard = _score_bbox(
                bbox,
                obstacles,
                safety_x=safety_x,
                safety_y=safety_y,
                require_inside=True,
            )
            # Prefer the central arc only when collision scores are equal.
            center_preference = abs(rank / max(1, n_samples - 1) - 0.5)
            candidates.append(
                (
                    score
                    + center_preference * 0.01
                    + (0.001 if side < 0 else 0.0),
                    hard,
                    (float(center[0]), float(center[1])),
                    (float(position[0]), float(position[1])),
                )
            )
    _, _, center, anchor = min(candidates, key=lambda item: (item[1], item[0]))
    return ax.annotate(
        text,
        xy=anchor,
        xycoords=ax.transAxes,
        xytext=center,
        textcoords=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        annotation_clip=False,
        clip_on=False,
        zorder=5,
    )
