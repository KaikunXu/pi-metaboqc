"""Missing-value filtering decision flowchart.

The module draws the stage decision topology and adapts it to the presence or
absence of biological-group metadata. It does not assemble dashboards.
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import pandas as pd

from .. import plot_utils as pu


class FilteringFlowchartMixin:
    """Render the decision flowchart used by filtering dashboards."""

    def _plot_mv_filtering_flowchart(
        self,
        df: pd.DataFrame,
        ax: plt.Axes,
        mnar_group_mv_tol: float,
        mnar_qc_mv_tol: float,
        active_base_tol: float,
        has_group_info: bool,
        mnar_intensity_pct: float = 0.1,
        margin_left: float = 0.0,
        margin_right: float = 0.0,
        margin_top: float = 0.0,
        margin_bottom: float = 0.0,
        compact: bool = False,
    ) -> None:
        """
        Horizontal flowchart with strictly QC-anchored logic.
        Dynamically adapts topology (removes Group Rescue nodes completely
        if no bio-group info exists) and re-balances X-axis coordinates.
        """
        total = len(df)
        count_group = sum(df["Stage1_Status"].str.contains("Group"))
        df_s2 = df[~df["Stage1_Status"].str.contains("Group")]
        count_qc = sum(df_s2["Stage1_Status"].str.contains("QC"))
        df_s3 = df_s2[~df_s2["Stage1_Status"].str.contains("QC")]
        count_mar = sum(df_s3["Stage1_Status"] == "MAR")
        count_inv = sum(df_s3["Stage1_Status"] == "INVALID")

        ax.axis("off")

        # Keep only a small safety margin around the outermost nodes and arrows
        # so the compact flowchart aligns with neighboring dashboard panels.
        ax.set_xlim(0.2 - margin_left, 32.9 + margin_right)
        ax.set_ylim(0.5 - margin_bottom, 9.35 + margin_top)

        color_mar = pu.PRIMARY_ACCENT_COLOR
        color_mnar = pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=0.5)
        color_inv = "tab:gray"
        color_pass = "white"
        box_style = "round,pad=0.12,rounding_size=0.18"
        node_fontsize = 7.0 if compact else (12 if has_group_info else 14)
        node_body_fontsize = 5.5 if compact else 10.0
        flow_linewidth = pu.DEFAULT_AXIS_LINEWIDTH if compact else 1.2
        arrow_linewidth = pu.DEFAULT_GUIDE_LINEWIDTH if compact else 2.0
        intensity_label = (
            f"QC intensity <= {pu.format_percentile_label(mnar_intensity_pct)}"
        )

        def _node(
            x: float,
            y: float,
            text: str,
            bg: str,
            width: float = 5.2,
            height: float = 1.5,
            fontsize: float | None = None,
            body_fontsize: float | None = None,
            line_step: float | None = None,
        ) -> dict[str, float]:
            """Draw a fixed-size flowchart node and return its data bounds."""
            text_color = pu.get_contrast_color(bg)
            text_fontsize = node_fontsize if fontsize is None else fontsize
            text_body_fontsize = (
                node_body_fontsize if body_fontsize is None else body_fontsize
            )
            text_line_step = (
                (0.49 if has_group_info else 0.55)
                if line_step is None
                else line_step
            )
            patch = mpatches.FancyBboxPatch(
                (x - width / 2, y - height / 2),
                width,
                height,
                boxstyle=box_style,
                facecolor=bg,
                edgecolor="k",
                linewidth=flow_linewidth,
                zorder=3,
                clip_on=False,
            )
            ax.add_patch(patch)
            text_lines = text.splitlines()
            if len(text_lines) == 1:
                ax.text(
                    x,
                    y,
                    text,
                    ha="center",
                    va="center",
                    multialignment="center",
                    fontsize=text_fontsize,
                    fontweight="semibold",
                    color=text_color,
                    zorder=4,
                )
            else:
                total_text_height = (len(text_lines) - 1) * text_line_step
                start_y = y + total_text_height / 2
                for line_idx, line_text in enumerate(text_lines):
                    is_title_line = line_idx == 0
                    ax.text(
                        x,
                        start_y - line_idx * text_line_step,
                        line_text,
                        ha="center",
                        va="center",
                        multialignment="center",
                        fontsize=text_fontsize
                        if is_title_line
                        else text_body_fontsize,
                        fontweight="semibold" if is_title_line else "normal",
                        color=text_color,
                        zorder=4,
                    )
            return {"x": x, "y": y, "width": width, "height": height}

        def _anchor(node: dict[str, float], side: str) -> tuple[float, float]:
            """Return one boundary midpoint for a node."""
            x = float(node["x"])
            y = float(node["y"])
            half_w = float(node["width"]) / 2
            half_h = float(node["height"]) / 2
            if side == "left":
                return (x - half_w, y)
            if side == "right":
                return (x + half_w, y)
            if side == "top":
                return (x, y + half_h)
            if side == "bottom":
                return (x, y - half_h)
            return (x, y)

        def _arrow(
            node_a: dict[str, float],
            node_b: dict[str, float],
            style: str = "horizontal",
        ) -> None:
            kwargs = dict(
                arrowstyle="-|>",
                color="gray",
                lw=arrow_linewidth,
                mutation_scale=8 if compact else 15,
                zorder=2,
                shrinkA=0,
                shrinkB=0,
                clip_on=False,
            )

            if style == "horizontal":
                start = _anchor(node_a, "right")
                end = _anchor(node_b, "left")
                arrow = mpatches.FancyArrowPatch(posA=start, posB=end, **kwargs)
            elif style == "vertical":
                if float(node_b["y"]) >= float(node_a["y"]):
                    start = _anchor(node_a, "top")
                    end = _anchor(node_b, "bottom")
                else:
                    start = _anchor(node_a, "bottom")
                    end = _anchor(node_b, "top")
                arrow = mpatches.FancyArrowPatch(posA=start, posB=end, **kwargs)
            elif style == "step_h":
                start = _anchor(node_a, "right")
                end = _anchor(node_b, "left")
                mid_x = (start[0] + end[0]) / 2
                path = mpath.Path(
                    [start, (mid_x, start[1]), (mid_x, end[1]), end],
                    [
                        mpath.Path.MOVETO,
                        mpath.Path.LINETO,
                        mpath.Path.LINETO,
                        mpath.Path.LINETO,
                    ],
                )
                arrow = mpatches.FancyArrowPatch(path=path, **kwargs)
            else:
                start = _anchor(node_a, "right")
                end = _anchor(node_b, "left")
                arrow = mpatches.FancyArrowPatch(posA=start, posB=end, **kwargs)
            ax.add_patch(arrow)

        # Full pipeline with four logical columns when BioGroup is available.
        if has_group_info:
            str_group = (
                f"Max MV >= {mnar_group_mv_tol * 100:.0f}%\n"
                f"Min MV <= {active_base_tol * 100:.0f}%"
            )
            qc_cond = (
                f"QC MV > {mnar_qc_mv_tol * 100:.0f}%\n"
                f"{intensity_label}\n"
                f"Min group MV <= {active_base_tol * 100:.0f}%"
            )

            node_root = _node(3.0, 5, f"Raw Features\n(n={total})", color_pass)
            node_c1 = _node(
                9.8,
                5,
                f"Group Rescue\n{str_group}",
                color_pass,
                width=5.7,
                height=1.95,
            )
            node_g = _node(
                9.8,
                8.5,
                f"MNAR Group\n(n={count_group})",
                color_mnar,
                width=5.1,
                height=1.35,
            )
            node_c2 = _node(
                16.6,
                5,
                f"QC Rescue\n{qc_cond}",
                color_pass,
                width=5.9,
                height=2.45,
                body_fontsize=(
                    pu.DEFAULT_ANNOTATION_FONTSIZE if compact else 10.0
                ),
                line_step=0.41,
            )
            node_q = _node(
                16.6,
                8.5,
                f"MNAR QC\n(n={count_qc})",
                color_mnar,
                width=5.1,
                height=1.35,
            )
            node_c3 = _node(
                23.4,
                5,
                "MAR Eligibility\nMin group MV "
                f"<= {active_base_tol * 100:.0f}%",
                color_pass,
                width=5.6,
                height=1.75,
                body_fontsize=(
                    pu.DEFAULT_ANNOTATION_FONTSIZE if compact else 10.0
                ),
                line_step=0.42,
            )
            node_mar = _node(
                30.5,
                7.5,
                f"MAR\n(n={count_mar})",
                color_mar,
                width=4.4,
                height=1.25,
            )
            node_inv = _node(
                30.5,
                2.5,
                f"INVALID\n(n={count_inv})",
                color_inv,
                width=4.4,
                height=1.25,
            )

            _arrow(node_root, node_c1, "horizontal")
            _arrow(node_c1, node_c2, "horizontal")
            _arrow(node_c1, node_g, "vertical")
            _arrow(node_c2, node_c3, "horizontal")
            _arrow(node_c2, node_q, "vertical")
            _arrow(node_c3, node_mar, "step_h")
            _arrow(node_c3, node_inv, "step_h")

        # Simplified three-column pipeline when BioGroup is unavailable.
        else:
            qc_cond = f"QC MV > {mnar_qc_mv_tol * 100:.0f}%\n{intensity_label}"

            node_root = _node(3.2, 5, f"Raw Features\n(n={total})", color_pass)
            node_c2 = _node(
                12.0,
                5,
                f"QC Rescue\n{qc_cond}",
                color_pass,
                width=5.3,
                height=1.75,
            )
            node_q = _node(
                12.0,
                8.5,
                f"MNAR QC\n(n={count_qc})",
                color_mnar,
                width=4.6,
                height=1.35,
            )
            node_c3 = _node(
                21.0,
                5,
                f"QC MV Check\nQC MV >= {active_base_tol * 100:.0f}%",
                color_pass,
                width=5.0,
                height=1.45,
            )
            node_mar = _node(
                30.0,
                7.5,
                f"MAR\n(n={count_mar})",
                color_mar,
                width=4.4,
                height=1.25,
            )
            node_inv = _node(
                30.0,
                2.5,
                f"INVALID\n(n={count_inv})",
                color_inv,
                width=4.4,
                height=1.25,
            )

            _arrow(node_root, node_c2, "horizontal")
            _arrow(node_c2, node_c3, "horizontal")
            _arrow(node_c2, node_q, "vertical")
            _arrow(node_c3, node_mar, "step_h")
            _arrow(node_c3, node_inv, "step_h")
