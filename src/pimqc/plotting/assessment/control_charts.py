"""Reference-feature control charts for quality assessment.

The module renders Shewhart-style traces for reference sample groups while
leaving metric calculation and stage orchestration to ``pimqc.processing``.
"""

from __future__ import annotations

import matplotlib.lines as mlines
import pandas as pd
import seaborn as sns

from ...core import model
from .. import plot_utils as pu


class AssessmentControlChartMixin:
    """Render reference-feature control charts."""

    def plot_ref_shewhart_chart(
        self,
        ref_data: pd.DataFrame,
        valid_feats: list[str],
        sample_type: str,
        batch: str,
        inject_order: str,
        qc_label: str,
        actual_label: str,
        bound_type: str,
        ref_type: str = "IS",
    ) -> object | None:
        """
        Plot Shewhart control charts with adaptive PW panels and one legend.
        """
        try:
            import patchworklib as pw
        except ImportError:
            return None

        if not valid_feats:
            return None

        pw.clear()
        ref_type_upper = ref_type.upper()
        plot_df = ref_data.reset_index().copy()
        plot_df[sample_type] = pd.Categorical(
            plot_df[sample_type],
            categories=[actual_label, qc_label],
            ordered=True,
        )
        plot_df[batch] = plot_df[batch].astype("category")
        plot_df = plot_df.sort_values(by=sample_type, ascending=True)

        # Use symmetrical marker mappings for IS and ORF flags.
        v_color = (
            pu.PRIMARY_ACCENT_COLOR if ref_type_upper == "IS" else "tab:orange"
        )
        v_ls = "--" if ref_type_upper == "IS" else "-."

        # Generate analytical control chart bricks sequentially
        plot_bricks = []
        panel_cols = 1 if len(valid_feats) == 1 else 2
        shewhart_source_width = 13.0
        shewhart_legend_source_width = 2.5
        shewhart_layout_width = (
            shewhart_source_width + shewhart_legend_source_width
        )
        panel_size = pu.dashboard_brick_size(6.5, 2.0, shewhart_layout_width)
        for feat_idx, feat in enumerate(valid_feats):
            brick = pw.Brick(
                figsize=panel_size,
                label=f"{ref_type_upper}_shewhart_{feat_idx}",
            )

            sns.scatterplot(
                ax=brick,
                data=plot_df,
                x=inject_order,
                y=feat,
                s=pu.DEFAULT_SCATTER_MARKER_AREA,
                edgecolor="k",
                linewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
                style=batch,
                palette={
                    qc_label: pu.PRIMARY_ACCENT_COLOR,
                    actual_label: pu.NEUTRAL_COLOR,
                },
                hue=sample_type,
                hue_order=[qc_label, actual_label],
                markers=self.style_map,
            )

            solid, lower, upper = model.MetaboInt.calculate_boundaries(
                x=ref_data[feat].values, boundary_type=bound_type
            )

            is_out = (plot_df[feat] < lower) | (plot_df[feat] > upper)
            outliers_data = plot_df[is_out]
            if not outliers_data.empty:
                brick.scatter(
                    outliers_data[inject_order],
                    outliers_data[feat],
                    facecolors="none",
                    edgecolors=v_color,
                    s=60,
                    linewidths=pu.DEFAULT_GUIDE_LINEWIDTH,
                    linestyle=v_ls,
                    zorder=0,
                )

            brick.axhline(y=solid, color="k", linestyle="-", linewidth=1.5)
            brick.axhline(y=lower, color="k", linestyle="--", linewidth=1.5)
            brick.axhline(y=upper, color="k", linestyle="--", linewidth=1.5)

            self._apply_standard_format(
                ax=brick,
                title=feat,
                xlabel=inject_order,
                ylabel="Intensity",
                append_stage=True,
            )
            pu.change_axis_format(ax=brick, axis_format="sci", axis="y")
            pu.change_fontsize(ax=brick, axis="y")
            pu.change_weight(ax=brick, axis="y")
            offset_text = brick.yaxis.get_offset_text()
            offset_text.set_fontsize(pu.DEFAULT_AXIS_TICK_FONTSIZE)
            offset_text.set_weight(pu.DEFAULT_AXIS_TICK_WEIGHT)

            if brick.get_legend():
                brick.get_legend().remove()

            plot_bricks.append(brick)

        # Construct the standalone comprehensive master legend brick
        row_bricks = []
        for row_start in range(0, len(plot_bricks), panel_cols):
            row_items = plot_bricks[row_start : row_start + panel_cols]
            if panel_cols == 2 and len(row_items) == 1:
                spacer = pw.Brick(
                    figsize=panel_size,
                    label=f"{ref_type_upper}_shewhart_spacer_{row_start}",
                )
                spacer.axis("off")
                row_items.append(spacer)

            row = row_items[0]
            for item in row_items[1:]:
                row = row | item
            row_bricks.append(row)

        plot_grid = row_bricks[0]
        for row in row_bricks[1:]:
            plot_grid = plot_grid / row

        legend_height_source = max(2.0, len(row_bricks) * 2.0)
        leg_brick = pw.Brick(
            figsize=pu.dashboard_brick_size(
                shewhart_legend_source_width,
                legend_height_source,
                shewhart_layout_width,
            ),
            label=f"{ref_type_upper}_shewhart_legend",
        )
        leg_brick.axis("off")

        legend_handles = []
        legend_labels = []

        # Consolidate groups by merging Outlier Status directly into Sample Type
        group_titles = [sample_type, batch, "Control Limits"]

        # Group A: Sample Type & Outlier Status (Unified Dimension)
        legend_handles.append(
            mlines.Line2D([], [], color="none", label=sample_type)
        )
        legend_labels.append(sample_type)
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=pu.PRIMARY_ACCENT_COLOR,
                marker="o",
                linestyle="none",
                markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                markeredgecolor="k",
                markeredgewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
                label=qc_label,
            )
        )
        legend_labels.append(qc_label)
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color="tab:gray",
                marker="o",
                linestyle="none",
                markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                markeredgecolor="k",
                markeredgewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
                label=actual_label,
            )
        )
        legend_labels.append(actual_label)

        # Append the hollow halo indicator directly inside the Sample Type group
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color="none",
                markeredgecolor=v_color,
                marker="o",
                markersize=10,
                markeredgewidth=2.0,
                linestyle=v_ls,
                label=f"{ref_type_upper} Outlier",
            )
        )
        legend_labels.append(f"{ref_type_upper} Outlier")

        # Group B: Chronological Batch Configurations (Aligned linewidth)
        legend_handles.append(mlines.Line2D([], [], color="none", label=batch))
        legend_labels.append(batch)
        for b_val in self.all_batches:
            m_style = self.style_map.get(b_val, "o")
            legend_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color="tab:gray",
                    marker=m_style,
                    linestyle="none",
                    markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                    markeredgecolor="k",
                    markeredgewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
                    label=str(b_val),
                )
            )
            legend_labels.append(str(b_val))

        # Group C: Statistical Boundary Thresholds
        legend_handles.append(
            mlines.Line2D([], [], color="none", label="Control Limits")
        )
        legend_labels.append("Control Limits")

        if str(bound_type).upper() == "IQR":
            solid_label, low_label, up_label = (
                "Median",
                "Q1 - 1.5 IQR",
                "Q3 + 1.5 IQR",
            )
        else:
            solid_label, low_label, up_label = (
                "Mean",
                "Mean - 3 Std",
                "Mean + 3 Std",
            )

        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color="k",
                ls="-",
                lw=pu.DEFAULT_GUIDE_LINEWIDTH,
                label=solid_label,
            )
        )
        legend_labels.append(solid_label)
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color="k",
                ls="--",
                lw=pu.DEFAULT_GUIDE_LINEWIDTH,
                label=low_label,
            )
        )
        legend_labels.append(low_label)
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color="k",
                ls="--",
                lw=pu.DEFAULT_GUIDE_LINEWIDTH,
                label=up_label,
            )
        )
        legend_labels.append(up_label)

        leg_brick.legend(legend_handles, legend_labels)

        # Format layout into 2 parallel columns for optimized space distribution
        self._format_multi_legends(
            ax=leg_brick,
            group_titles=group_titles,
            loc="upper left",
            start_bbox=(0.05, 0.95),
            row_gap=0.04,
            layout_cols=1,
            column_gap=0.1,
            sublegend_cols=1,
        )

        if hasattr(leg_brick.figure, "legends"):
            for leg in list(leg_brick.figure.legends):
                leg_brick.add_artist(leg)
            leg_brick.figure.legends.clear()

        return plot_grid | leg_brick
