"""Tests for shared annotation and reference-line placement helpers."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from pimqc.plotting import annotation_layout as al
from pimqc.plotting import plot_utils as pu


def test_auto_annotation_accounts_for_in_axes_legend() -> None:
    """An in-axes legend is reported and does not cover the note."""
    fig, ax = plt.subplots()
    points = np.array([[0.9, 0.9], [0.95, 0.95]])
    ax.scatter(points[:, 0], points[:, 1], label="Observed")
    ax.legend(loc="upper right")

    note = al.add_auto_annotation(
        ax=ax,
        text="Diagnostics",
        occupancy_arrays=[points],
    )
    fig.canvas.draw()

    assert al.legend_is_inside_axes(ax)
    assert note.get_position() != (0.96, 0.98)
    plt.close(fig)


def test_reference_line_orientation_matches_line_direction() -> None:
    """Horizontal and vertical reference labels use matching orientation."""
    fig, ax = plt.subplots()
    horizontal = al.annotate_reference_line(
        ax=ax,
        value=0.5,
        text="Horizontal",
        orientation="horizontal",
    )
    vertical = al.annotate_reference_line(
        ax=ax,
        value=0.5,
        text="Vertical",
        orientation="vertical",
    )

    assert horizontal.get_rotation() == 0
    assert vertical.get_rotation() == 90
    assert horizontal.arrow_patch is None
    assert vertical.arrow_patch is None
    plt.close(fig)


def test_reference_line_tie_breaking_follows_reading_conventions() -> None:
    """Prefer the upper and right portions of otherwise empty lines."""
    fig, ax = plt.subplots()
    vertical, horizontal = al.annotate_reference_lines(
        ax=ax,
        references=[
            {"value": 0.5, "text": "Vertical", "orientation": "vertical"},
            {
                "value": 0.5,
                "text": "Horizontal",
                "orientation": "horizontal",
            },
        ],
        expand_axis=False,
    )

    assert vertical.xy[1] > 0.75
    assert horizontal.xy[0] > 0.75
    plt.close(fig)


def test_auto_annotation_reserves_inset_axes() -> None:
    """A note avoids the axes rectangle occupied by an inset colorbar."""
    fig, ax = plt.subplots()
    inset = ax.inset_axes([0.72, 0.72, 0.24, 0.22])
    inset.set_axis_off()
    note = al.add_auto_annotation(
        ax=ax,
        text="Diagnostics",
        occupancy_arrays=[np.empty((0, 2))],
    )
    fig.canvas.draw()

    note_bbox = note.get_window_extent(fig.canvas.get_renderer()).transformed(
        ax.transAxes.inverted()
    )
    inset_bbox = al.artist_bboxes_in_axes(
        ax,
        artists=[inset],
        include_legend=False,
        include_child_axes=False,
    )[0]
    overlap_width = max(
        0.0, min(note_bbox.x1, inset_bbox[2]) - max(note_bbox.x0, inset_bbox[0])
    )
    overlap_height = max(
        0.0, min(note_bbox.y1, inset_bbox[3]) - max(note_bbox.y0, inset_bbox[1])
    )

    assert overlap_width * overlap_height == 0.0
    plt.close(fig)


def test_auto_annotation_reserves_colorbar_decorations() -> None:
    """Avoid colorbar ticks and labels outside the inset axes rectangle."""
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    scatter = ax.scatter([0.4, 0.6], [0.4, 0.6], c=[0.9, 1.0])
    colorbar_ax = ax.inset_axes([0.90, 0.02, 0.025, 0.35])
    colorbar = fig.colorbar(scatter, cax=colorbar_ax)
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.set_label("Local trustworthiness", labelpad=2)
    pu.format_colorbar_axes(colorbar.ax)
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    axes_only = colorbar_ax.get_window_extent(renderer).transformed(
        ax.transAxes.inverted()
    )
    decorated = al.artist_bboxes_in_axes(
        ax,
        artists=[colorbar_ax],
        include_legend=False,
        include_child_axes=False,
    )[0]
    assert decorated[0] < axes_only.x0

    note = al.add_auto_annotation(
        ax=ax,
        text="Global score: 0.99\nRank preservation: 0.95",
        expand_axes=False,
        bbox=pu.ai_ready_text_bbox(pad=0.3),
    )
    fig.canvas.draw()
    note_bbox = note.get_bbox_patch().get_window_extent(renderer).transformed(
        ax.transAxes.inverted()
    )
    overlap_width = max(
        0.0,
        min(note_bbox.x1, decorated[2]) - max(note_bbox.x0, decorated[0]),
    )
    overlap_height = max(
        0.0,
        min(note_bbox.y1, decorated[3]) - max(note_bbox.y0, decorated[1]),
    )

    assert overlap_width * overlap_height == 0.0
    plt.close(fig)


def test_auto_annotation_collects_scatter_points_when_unspecified() -> None:
    """Infer scatter occupancy instead of falling back to the first corner."""
    fig, ax = plt.subplots()
    points = np.array([[0.88, 0.05], [0.92, 0.08], [0.96, 0.04]])
    ax.scatter(points[:, 0], points[:, 1])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    note = al.add_auto_annotation(ax=ax, text="Diagnostics", expand_axes=False)

    assert note.get_position() != (0.96, 0.02)
    plt.close(fig)


def test_auto_annotation_keeps_opaque_background_clear_of_axes() -> None:
    """Reserve rounded-box padding in addition to the text glyph extent."""
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    note = al.add_auto_annotation(
        ax=ax,
        text="Relative Dispersion: 0.1234\nCentrality Shift: 0.5678",
        expand_axes=False,
        bbox=pu.ai_ready_text_bbox(pad=0.4),
    )
    fig.canvas.draw()
    patch_bbox = note.get_bbox_patch().get_window_extent(
        fig.canvas.get_renderer()
    )
    axes_bbox = ax.get_window_extent(fig.canvas.get_renderer())

    assert patch_bbox.x0 > axes_bbox.x0
    assert patch_bbox.y0 > axes_bbox.y0
    assert patch_bbox.x1 < axes_bbox.x1
    assert patch_bbox.y1 < axes_bbox.y1
    plt.close(fig)


def test_auto_annotation_refines_the_selected_region_continuously() -> None:
    """Move within an edge region instead of choosing only fixed anchors."""
    fig, ax = plt.subplots()
    fixed_anchors = [
        (0.96, 0.02),
        (0.96, 0.98),
        (0.04, 0.02),
        (0.04, 0.98),
        (0.96, 0.50),
        (0.04, 0.50),
        (0.50, 0.98),
        (0.50, 0.02),
    ]
    blocked = [
        (x_value - 0.09, y_value - 0.09, x_value + 0.09, y_value + 0.09)
        for x_value, y_value in fixed_anchors
    ]
    note = ax.text(0.5, 0.5, "Diagnostics", transform=ax.transAxes)

    placement = al.place_annotation_in_least_occupied_corner(
        ax,
        note,
        blocked_regions=blocked,
    )

    assert placement["hard_collisions"] == 0
    assert not any(
        np.allclose(note.get_position(), anchor) for anchor in fixed_anchors
    )
    plt.close(fig)


def test_auto_annotation_can_reserve_data_space_when_all_regions_are_busy() -> (
    None
):
    """Expand an axis only when every internal note candidate is occupied."""
    fig, ax = plt.subplots()
    grid = np.linspace(0.0, 1.0, 21)
    x_values, y_values = np.meshgrid(grid, grid)
    ax.scatter(x_values.ravel(), y_values.ravel())
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    al.add_auto_annotation(ax=ax, text="Dense diagnostics", expand_axes=True)

    assert ax.get_ylim() != (0.0, 1.0)
    plt.close(fig)


def test_reference_line_can_use_an_outside_candidate() -> None:
    """Permit a threshold label outside the axes when all line bands are busy."""
    fig, ax = plt.subplots()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    note = al.annotate_reference_line(
        ax=ax,
        value=0.5,
        text="Dense threshold",
        orientation="horizontal",
        blocked_regions=[(0.0, 0.0, 1.0, 1.0)],
        allow_outside=True,
        expand_axis=False,
    )
    fig.canvas.draw()
    bbox = note.get_window_extent(fig.canvas.get_renderer()).transformed(
        ax.transAxes.inverted()
    )

    assert bbox.x0 >= 1.0 - 1e-9 or bbox.x1 <= 1e-9
    assert note.arrow_patch is not None
    plt.close(fig)


def test_marker_obstacles_include_rendered_marker_size() -> None:
    """Measure marker extents instead of treating scatters as point centers."""
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    marker = ax.scatter([0.5], [0.5], s=2500)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    bbox = al.artist_bboxes_in_axes(
        ax,
        artists=[marker],
        include_legend=False,
        include_child_axes=False,
    )[0]

    assert bbox[2] - bbox[0] > 0.1
    assert bbox[3] - bbox[1] > 0.1
    plt.close(fig)


def test_reference_labels_are_allocated_jointly() -> None:
    """Two nearby limits receive non-overlapping labels in one allocation."""
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.axvline(0.48, color="gray")
    ax.axvline(0.52, color="gray")
    notes = al.annotate_reference_lines(
        ax,
        references=[
            {"value": 0.48, "text": "First limit", "orientation": "vertical"},
            {"value": 0.52, "text": "Second limit", "orientation": "vertical"},
        ],
        expand_axis=False,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    first = notes[0].get_window_extent(renderer)
    second = notes[1].get_window_extent(renderer)

    assert not first.overlaps(second)
    plt.close(fig)


def test_curve_annotation_searches_beyond_five_legacy_samples() -> None:
    """Search the full curve by arc length rather than five fixed indices."""
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    x_values = np.linspace(0.0, 1.0, 101)
    y_values = 0.2 + 0.6 * x_values**2
    ax.plot(x_values, y_values, color="gray")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    legacy_indices = np.rint(np.linspace(0, 100, 5)).astype(int)
    legacy_data = np.column_stack(
        (x_values[legacy_indices], y_values[legacy_indices])
    )
    legacy_axes = ax.transAxes.inverted().transform(
        ax.transData.transform(legacy_data)
    )
    blocked = [
        (x_value - 0.06, y_value - 0.06, x_value + 0.06, y_value + 0.06)
        for x_value, y_value in legacy_axes
    ]
    note = al.annotate_reference_curve(
        ax,
        x_values=x_values,
        y_values=y_values,
        text="Blank/QC cutoff",
        blocked_regions=blocked,
    )

    anchor = np.asarray(note.xy, dtype=float)
    distances = np.linalg.norm(legacy_axes - anchor, axis=1)
    assert float(np.min(distances)) > 0.06
    plt.close(fig)


def test_reference_layout_ignores_overlapping_wrapper_axes() -> None:
    """Patchwork-style wrapper axes do not force labels outside the panel."""
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    wrapper = fig.add_axes(ax.get_position(), frameon=False)
    wrapper.set_axis_off()
    ax.axhline(0.5, color="gray")

    note = al.annotate_reference_line(
        ax,
        value=0.5,
        text="Internal limit",
        orientation="horizontal",
        expand_axis=False,
    )
    fig.canvas.draw()
    bbox = note.get_window_extent(fig.canvas.get_renderer()).transformed(
        ax.transAxes.inverted()
    )

    assert bbox.x0 >= 0.0 and bbox.x1 <= 1.0
    assert bbox.y0 >= 0.0 and bbox.y1 <= 1.0
    assert note.arrow_patch is None
    plt.close(fig)


def test_reference_layout_does_not_treat_shaded_regions_as_data_marks() -> None:
    """Decision-region shading does not force a threshold label outside."""
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    ax.add_patch(
        Rectangle(
            (0.5, 0.0),
            0.5,
            0.6,
            transform=ax.transAxes,
            facecolor="tab:blue",
            alpha=0.1,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.axvline(0.5, color="gray")

    note = al.annotate_reference_line(
        ax,
        value=0.5,
        text="Vertical limit",
        orientation="vertical",
        expand_axis=False,
    )
    fig.canvas.draw()
    bbox = note.get_window_extent(fig.canvas.get_renderer()).transformed(
        ax.transAxes.inverted()
    )

    assert bbox.x0 >= 0.0 and bbox.x1 <= 1.0
    assert bbox.y0 >= 0.0 and bbox.y1 <= 1.0
    assert note.arrow_patch is None
    plt.close(fig)


def test_curve_annotation_avoids_the_panel_title() -> None:
    """A high curve label reserves the rendered title rectangle."""
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.set_title("Blank/QC Check")
    x_values = np.linspace(0.0, 1.0, 101)
    y_values = 0.82 + 0.12 * x_values
    ax.plot(x_values, y_values, color="gray")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    note = al.annotate_reference_curve(
        ax,
        x_values=x_values,
        y_values=y_values,
        text="Blank/QC cutoff = 0.20",
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    assert not note.get_window_extent(renderer).overlaps(
        ax.title.get_window_extent(renderer)
    )
    plt.close(fig)
