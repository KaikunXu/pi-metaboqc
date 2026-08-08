"""Diagnostic visualizations for sample and feature filtering.

MetaboVisualizerFilter explains high-missing sample exclusion, MAR/MNAR rescue,
blank-to-QC checks, QC-RSD filtering, and feature retention across both filter
stages. Its flowcharts, scatter plots, distributions, and dashboards are
presentation layers over filtering results rather than filtering logic.
"""

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.lines as mlines

from loguru import logger

from ...visualization import plot_utils as pu
from ...visualization import base as visualizer_classes


from .analysis import MetaboIntFilter


class MetaboVisualizerFilter(visualizer_classes.BaseMetaboVisualizer):
    """Visualization suite for metabolomics filtering results."""

    def __init__(self, engine: MetaboIntFilter) -> None:
        """Initialize with the filtering engine."""
        super().__init__(metabo_obj=engine)
        self.engine = engine

    # =========================================================================
    # High-Misssing Values Samples Filtering
    # =========================================================================
    def _plot_sample_mv_stripplot(
        self,
        track_df: pd.DataFrame,
        tol: float,
        ax: plt.Axes | None = None,
        article_compact: bool = False,
    ) -> plt.Figure | plt.Axes | None:
        """Plot sample missing rates using a stripplot, annotating outliers."""
        if track_df.empty:
            return None if ax is None else ax

        if ax is None:
            fig, current_ax = plt.subplots(
                figsize=pu.article_brick_size(4.0, 4.0)
                if article_compact
                else (4.0, 4.0)
            )
        else:
            current_ax = ax
            fig = current_ax.figure

        df_plot = track_df.copy()

        color_dict = {
            t: pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=1.0)
            if "QC" in str(t).upper()
            else pu.get_equivalent_hex("tab:gray", alpha=1.0)
            for t in df_plot["Sample_Type"].unique()
        }
        cat_order = df_plot["Sample_Type"].unique().tolist()
        x_lookup = {cat: idx for idx, cat in enumerate(cat_order)}
        rng = np.random.default_rng(123)

        for sample_type, sub_df in df_plot.groupby("Sample_Type", sort=False):
            x_base = x_lookup[sample_type]
            jitter = rng.uniform(-0.18, 0.18, size=len(sub_df))
            current_ax.scatter(
                np.full(len(sub_df), x_base) + jitter,
                sub_df["MV_Rate_Pct"].to_numpy(dtype=float),
                s=(
                    pu.DEFAULT_COMPACT_SCATTER_MARKER_AREA
                    if article_compact
                    else pu.DEFAULT_SCATTER_MARKER_AREA
                ),
                marker="o",
                facecolor=color_dict[sample_type],
                edgecolor="k",
                linewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
                zorder=3,
                label=str(sample_type),
            )
        current_ax.set_xticks(range(len(cat_order)))
        current_ax.set_xticklabels([str(cat) for cat in cat_order])

        # Threshold line
        tol_pct = tol * 100
        current_ax.axhline(
            tol_pct,
            color="k",
            linestyle="--",
            linewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
            label=f"Sample-exclusion threshold ({tol_pct:.0f}%)",
        )

        # Smart Annotation for Outliers
        outliers = df_plot[df_plot["MV_Rate_Pct"] > tol_pct]
        for idx, row in outliers.iterrows():
            x_pos = cat_order.index(row["Sample_Type"])
            y_pos = row["MV_Rate_Pct"]
            current_ax.annotate(
                str(idx),
                (x_pos, y_pos),
                xytext=(12, 0),
                textcoords="offset points",
                fontsize=pu.ARTICLE_ANNOTATION_FONTSIZE
                if article_compact
                else 8,
                color="darkred",
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2", fc="white", ec="k", alpha=1.0
                ),
                arrowprops=dict(arrowstyle="-", color="k", lw=1.0),
            )

        self._apply_standard_format(
            ax=current_ax,
            title="Sample-Level Missing-Value Rates",
            xlabel="Sample type",
            ylabel="Missing-value rate (%)",
            append_stage=False,
        )

        sample_handles = [
            mlines.Line2D([], [], color="none", label="Sample type"),
            *[
                mlines.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    color=color_dict[sample_type],
                    markeredgecolor="k",
                    markeredgewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
                    markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                    label=str(sample_type),
                )
                for sample_type in cat_order
            ],
            mlines.Line2D([], [], color="none", label="Thresholds"),
            mlines.Line2D(
                [],
                [],
                color="k",
                linestyle="--",
                linewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
                label=f"Sample-exclusion threshold ({tol_pct:.0f}%)",
            ),
        ]
        current_ax.legend(handles=sample_handles)
        self._format_multi_legends(
            ax=current_ax,
            group_titles=["Sample type", "Thresholds"],
            loc="upper right",
            start_bbox=(0.98, 0.98),
            layout_cols=1,
            sublegend_cols=1,
        )

        # Adjust Y-limit slightly to prevent top annotations from clipping
        actual_y_max = max(df_plot["MV_Rate_Pct"].max(), tol_pct)
        padding = actual_y_max * 0.15 if actual_y_max > 0 else 10
        current_ax.set_ylim(-5, actual_y_max + padding)
        return fig if ax is None else current_ax

    # =========================================================================
    # High-MV Features Filtering Unified Summary Dashboard (1+N Layout)
    # =========================================================================
    def _plot_group_rescue_scatter(
        self,
        df: pd.DataFrame,
        max_col: str,
        min_col: str,
        mnar_group_mv_tol: float,
        active_base_tol: float,
        ax: plt.Axes,
        title: str,
        article_compact: bool = False,
    ) -> None:
        """Scatter plot visualizing the 2D logic of Group MNAR rescue."""

        if df.empty:
            ax.axis("off")
            return

        color_mnar = pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=0.5)
        color_pending = pu.get_equivalent_hex("tab:gray", alpha=1.0)

        df_plot = df.copy()

        df_plot["Step_Status"] = np.where(
            df_plot["Stage1_Status"].str.contains("Group"),
            "MNAR (Group)",
            "Pending",
        )

        df_plot = df_plot.sort_values(by="Step_Status", ascending=False)

        marker_map = {"MNAR (Group)": "X", "Pending": "o"}
        for status, sub_df in df_plot.groupby("Step_Status", sort=False):
            ax.scatter(
                sub_df[max_col].to_numpy(dtype=float),
                sub_df[min_col].to_numpy(dtype=float),
                s=(
                    pu.DEFAULT_COMPACT_SCATTER_MARKER_AREA
                    if article_compact
                    else pu.DEFAULT_SCATTER_MARKER_AREA
                ),
                marker=marker_map[status],
                facecolor={
                    "MNAR (Group)": color_mnar,
                    "Pending": color_pending,
                }[status],
                edgecolor="k",
                linewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
                zorder=3,
            )

        tol_max_pct = mnar_group_mv_tol * 100
        tol_min_pct = active_base_tol * 100

        ax.plot(
            [0, 100],
            [0, 100],
            color=pu.get_equivalent_hex("tab:gray", alpha=0.5),
            linestyle="-.",
            zorder=1,
        )

        ax.axvline(tol_max_pct, color="k", linestyle="--")
        ax.axhline(tol_min_pct, color="k", linestyle=":")

        ax.add_patch(
            mpatches.Rectangle(
                (tol_max_pct, -5),
                105 - tol_max_pct,
                tol_min_pct + 5,
                facecolor=pu.get_equivalent_hex(
                    pu.PRIMARY_ACCENT_COLOR, alpha=0.10
                ),
                edgecolor="none",
                zorder=0,
                clip_on=True,
            )
        )

        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)

        handles = [
            mlines.Line2D([], [], color="none", label="Status"),
            mlines.Line2D(
                [],
                [],
                color=color_mnar,
                marker="X",
                linestyle="",
                label="MNAR (Group)",
                markeredgecolor="k",
                markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                markeredgewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
            ),
            mlines.Line2D(
                [],
                [],
                color=color_pending,
                marker="o",
                linestyle="",
                label="Pending",
                markeredgecolor="k",
                markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                markeredgewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
            ),
            mlines.Line2D([], [], color="none", label="Thresholds"),
            mlines.Line2D(
                [], [], color="gray", linestyle="-.", label="y=x Limit"
            ),
            mlines.Line2D(
                [],
                [],
                color="k",
                linestyle="--",
                label=f"Max Group MV Cutoff >= {tol_max_pct:.0f}%",
            ),
            mlines.Line2D(
                [],
                [],
                color="k",
                linestyle=":",
                label=f"Min Group MV Cutoff <= {tol_min_pct:.0f}%",
            ),
        ]

        ax.legend(handles=handles)
        self._format_multi_legends(
            ax=ax,
            group_titles=["Status", "Thresholds"],
            loc="upper left",
            start_bbox=(0.05, 1.0),
            layout_cols=1,
            sublegend_cols=1,
            **(
                {
                    "fontsize": pu.DEFAULT_LEGEND_FONTSIZE,
                    "title_fontsize": pu.DEFAULT_LEGEND_TITLE_FONTSIZE,
                    "borderaxespad": 0.0,
                    "handlelength": 1.0,
                    "handletextpad": 0.3,
                    "labelspacing": 0.25,
                    "borderpad": 0.3,
                }
                if article_compact
                else {}
            ),
        )
        self._apply_standard_format(
            ax=ax,
            title=title,
            xlabel="Max Group MV (%)",
            ylabel="Min Group MV (%)",
            append_stage=False,
        )
        ax.title.set_weight("bold")

    def _plot_qc_rescue_scatter(
        self,
        df: pd.DataFrame,
        mnar_qc_mv_tol: float,
        mnar_int_threshold: float | None,
        ax: plt.Axes,
        title: str,
        mnar_intensity_pct: float = 0.1,
        article_compact: bool = False,
    ) -> None:
        """
        Diagnostic scatter for Step 2 with dual-threshold L-shape.
        Uses advanced 2.5D Bubble mapping:
        Color/Shape -> Rescue Status
        Bubble Size -> Min Group MV Pct (The 3rd continuous metric)
        """

        if df.empty:
            ax.axis("off")
            return

        color_mnar = pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=0.5)
        color_blocked = pu.get_equivalent_hex("tab:gray", alpha=1.0)
        color_pending = pu.get_equivalent_hex("tab:gray", alpha=1.0)
        df_plot = df.copy()
        intensity_label = (
            f"QC intensity <= {pu.format_percentile_label(mnar_intensity_pct)}"
        )

        def _determine_status(row: pd.Series) -> str:
            if "QC" in row["Stage1_Status"]:
                return "MNAR (QC)"
            elif (
                (row["QC_MV_Pct"] > mnar_qc_mv_tol * 100)
                and (mnar_int_threshold is not None)
                and (row["Log2_Intensity"] <= mnar_int_threshold)
            ):
                return "Blocked by Group Valid"
            else:
                return "Pending"

        df_plot["Step_Status"] = df_plot.apply(_determine_status, axis=1)
        df_plot = df_plot.sort_values(by="Step_Status", ascending=False)

        has_group_info = "Min_Group_MV_Pct" in df_plot.columns
        if has_group_info:
            raw_sizes = (
                df_plot["Min_Group_MV_Pct"].fillna(0).to_numpy(dtype=float)
            )
            size_min, size_max = np.nanmin(raw_sizes), np.nanmax(raw_sizes)
            if (
                np.isfinite(size_min)
                and np.isfinite(size_max)
                and size_max > size_min
            ):
                marker_min = 9 if article_compact else 18
                marker_span = 27 if article_compact else 54
                df_plot["_Marker_Size"] = (
                    marker_min
                    + (raw_sizes - size_min)
                    / (size_max - size_min)
                    * marker_span
                )
            else:
                df_plot["_Marker_Size"] = (
                    pu.DEFAULT_COMPACT_SCATTER_MARKER_AREA
                    if article_compact
                    else pu.DEFAULT_SCATTER_MARKER_AREA
                )
        else:
            df_plot["_Marker_Size"] = (
                pu.DEFAULT_COMPACT_SCATTER_MARKER_AREA
                if article_compact
                else pu.DEFAULT_SCATTER_MARKER_AREA
            )

        color_map = {
            "MNAR (QC)": color_mnar,
            "Blocked by Group Valid": color_blocked,
            "Pending": color_pending,
        }
        marker_map = {
            "MNAR (QC)": "X",
            "Blocked by Group Valid": "v",
            "Pending": "o",
        }
        for status, sub_df in df_plot.groupby("Step_Status", sort=False):
            ax.scatter(
                sub_df["Log2_Intensity"].to_numpy(dtype=float),
                sub_df["QC_MV_Pct"].to_numpy(dtype=float),
                s=sub_df["_Marker_Size"].to_numpy(dtype=float),
                marker=marker_map[status],
                facecolor=color_map[status],
                edgecolor="k",
                linewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
                zorder=3,
            )

        qc_mv_cutoff_pct = mnar_qc_mv_tol * 100
        ax.axhline(
            qc_mv_cutoff_pct, color="k", linestyle=":", label="MV Cutoff"
        )

        if mnar_int_threshold is not None:
            finite_x = pd.to_numeric(df_plot["Log2_Intensity"], errors="coerce")
            finite_x = finite_x[np.isfinite(finite_x)]
            if finite_x.empty:
                x_min, x_max = (
                    mnar_int_threshold - 1.0,
                    mnar_int_threshold + 1.0,
                )
            else:
                x_min = min(float(finite_x.min()), float(mnar_int_threshold))
                x_max = max(float(finite_x.max()), float(mnar_int_threshold))
                x_padding = max((x_max - x_min) * 0.05, 0.5)
                x_min -= x_padding
                x_max += x_padding

            y_min, y_max = -5.0, 105.0
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            rescue_width = max(float(mnar_int_threshold) - x_min, 0.0)
            rescue_height = y_max - qc_mv_cutoff_pct
            if rescue_width > 0 and rescue_height > 0:
                ax.add_patch(
                    mpatches.Rectangle(
                        (x_min, qc_mv_cutoff_pct),
                        rescue_width,
                        rescue_height,
                        facecolor=pu.get_equivalent_hex(
                            pu.PRIMARY_ACCENT_COLOR, alpha=0.10
                        ),
                        edgecolor="none",
                        zorder=0,
                        clip_on=True,
                    )
                )
            ax.axvline(
                mnar_int_threshold,
                color="k",
                linestyle="--",
                label=intensity_label,
            )
        else:
            ax.set_ylim(-5.0, 105.0)

        handles = [
            mlines.Line2D([], [], color="none", label="Status"),
            mlines.Line2D(
                [],
                [],
                color=color_mnar,
                marker="X",
                linestyle="",
                label="MNAR (QC)",
                markeredgecolor="k",
                markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                markeredgewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
            ),
            mlines.Line2D(
                [],
                [],
                color=color_blocked,
                marker="v",
                linestyle="",
                label="Blocked",
                markeredgecolor="k",
                markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                markeredgewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
            ),
            mlines.Line2D(
                [],
                [],
                color=color_pending,
                marker="o",
                linestyle="",
                label="Pending",
                markeredgecolor="k",
                markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                markeredgewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
            ),
            mlines.Line2D([], [], color="none", label="Thresholds"),
            mlines.Line2D(
                [],
                [],
                color="k",
                linestyle=":",
                label=f"MV > {qc_mv_cutoff_pct:.0f}%",
            ),
            mlines.Line2D(
                [], [], color="k", linestyle="--", label=intensity_label
            ),
        ]

        group_titles = ["Status", "Thresholds"]

        if has_group_info:
            handles.extend(
                [
                    mlines.Line2D([], [], color="none", label="Size reference"),
                    mlines.Line2D(
                        [],
                        [],
                        color="white",
                        marker="o",
                        markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                        label="Larger =\nHigher Min Group MV",
                        markerfacecolor="white",
                        markeredgecolor="gray",
                        linestyle="",
                    ),
                ]
            )
            group_titles.append("Size reference")

        ax.legend(handles=handles)
        self._format_multi_legends(
            ax=ax,
            group_titles=group_titles,
            loc="upper right",
            start_bbox=(0.98, 1.0),
            layout_cols=1,
            sublegend_cols=1,
            **(
                {
                    "fontsize": pu.DEFAULT_LEGEND_FONTSIZE,
                    "title_fontsize": pu.DEFAULT_LEGEND_TITLE_FONTSIZE,
                    "borderaxespad": 0.0,
                    "handlelength": 1.0,
                    "handletextpad": 0.3,
                    "labelspacing": 0.25,
                    "borderpad": 0.3,
                }
                if article_compact
                else {}
            ),
        )

        self._apply_standard_format(
            ax=ax,
            title=title,
            xlabel="log2(Median QC Intensity)",
            ylabel="QC Missing Rate (%)",
            append_stage=False,
        )
        ax.title.set_weight("bold")

    def _plot_cutoff_histogram(
        self,
        df: pd.DataFrame,
        x_col: str,
        hue_col: str,
        tol: float,
        palette: dict[str, str],
        hue_order: list[str],
        ax: plt.Axes,
        title: str,
        x_label: str,
        article_compact: bool = False,
    ) -> None:
        """Generic histogram using explicit bar patches for vector editing."""
        if df.empty:
            ax.axis("off")
            return

        bin_edges = np.arange(0, 105, 5)
        bin_width = float(np.diff(bin_edges).min())
        plot_df = df[[x_col, hue_col]].copy()
        plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
        plot_df = plot_df.dropna(subset=[x_col])

        for z_idx, category in enumerate(hue_order, start=2):
            if category not in palette:
                continue
            values = plot_df.loc[plot_df[hue_col] == category, x_col].to_numpy(
                dtype=float
            )
            if values.size == 0:
                continue
            counts, _ = np.histogram(values, bins=bin_edges)
            ax.bar(
                bin_edges[:-1],
                counts,
                width=bin_width,
                align="edge",
                color=palette[category],
                edgecolor="k",
                linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
                label=category,
                zorder=z_idx,
            )

        ax.axvline(
            tol * 100,
            color="k",
            linestyle=":",
            lw=pu.DEFAULT_GUIDE_LINEWIDTH,
        )

        handles = [mlines.Line2D([], [], color="none", label="Status")]
        handles.extend(
            [
                mpatches.Patch(facecolor=palette[cat], edgecolor="k", label=cat)
                for cat in hue_order
                if cat in palette
            ]
        )

        handles.append(mlines.Line2D([], [], color="none", label="Thresholds"))
        handles.append(
            mlines.Line2D(
                [],
                [],
                color="k",
                linestyle=":",
                label=f"Min Group MV <= {tol * 100:.0f}%",
            )
        )

        ax.legend(handles=handles)
        self._format_multi_legends(
            ax=ax,
            group_titles=["Status", "Thresholds"],
            loc="upper right",
            start_bbox=(0.98, 1.0),
            layout_cols=1,
            sublegend_cols=1,
            **(
                {
                    "fontsize": pu.DEFAULT_LEGEND_FONTSIZE,
                    "title_fontsize": pu.DEFAULT_LEGEND_TITLE_FONTSIZE,
                    "borderaxespad": 0.0,
                    "handlelength": 1.0,
                    "handletextpad": 0.3,
                    "labelspacing": 0.25,
                    "borderpad": 0.3,
                }
                if article_compact
                else {}
            ),
        )

        self._apply_standard_format(
            ax=ax,
            title=title,
            xlabel=x_label,
            ylabel="Feature Count",
            append_stage=False,
        )
        ax.title.set_weight("bold")

    def _plot_pipeline_flowchart_atom(
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

        ax.set_xlim(0 - margin_left, 33 + margin_right)
        ax.set_ylim(0 - margin_bottom, 10 + margin_top)

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

        # =====================================================================
        # Topology A: Full Pipeline (With BioGroups)
        # 4 Logical Columns distributed across X=[2.0, 9.5, 17.0, 24.5, 31.0]
        # =====================================================================
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

        # =====================================================================
        # Topology B: Simplified Pipeline (No BioGroups)
        # 3 Logical Columns distributed dynamically across X=[3.0, 12.0, 21.0,
        # 30.0]
        # =====================================================================
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

    def plot_mv_filtering_summary_grid(
        self,
        tracking_df: pd.DataFrame,
        active_base_tol: float,
        mnar_group_mv_tol: float | None = None,
        mnar_qc_mv_tol: float = 0.2,
        mnar_int_threshold: float | None = None,
        mnar_intensity_pct: float = 0.1,
    ) -> object | None:
        """
        Orchestrates a unified diagnostic dashboard for Stage-1 filtering.
        Dynamically adapts layout based on biological grouping and utilizes
        patchworklib to align subplots precisely via topological rules.
        """
        try:
            import patchworklib as pw
        except ImportError:
            return None

        pw.clear()

        # 1. Initialize data copy and evaluate biological grouping status
        df_curr = tracking_df.copy()
        has_group_info = ("Max_Group_MV_Pct" in df_curr.columns) and (
            df_curr["Max_Group_MV_Pct"].notna().any()
        )

        # 2. Build the universal Sample MV Stripplot Brick
        layout_width = 12.0
        ax_sample = pw.Brick(
            figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
            label="sample_mv",
        )
        sample_track = self.engine.stats.get("sample_tracking", pd.DataFrame())
        sample_mv_tol = self.engine.attrs.get("sample_mv_tol", 0.5)
        self._plot_sample_mv_stripplot(
            sample_track, sample_mv_tol, ax=ax_sample, article_compact=True
        )

        # 3. Dynamic layout assembly based on biological grouping
        if has_group_info:
            # --- Layout A: With Groups (1+2 Top, 1+1+1 Bottom) ---

            # Flowchart ratio is 2 units wide to match the bottom 2 plots
            ax_flow = pw.Brick(
                figsize=pu.dashboard_brick_size(8.0, 4.0, layout_width),
                label="flowchart",
            )
            self._plot_pipeline_flowchart_atom(
                df=df_curr,
                ax=ax_flow,
                mnar_group_mv_tol=mnar_group_mv_tol,
                mnar_qc_mv_tol=mnar_qc_mv_tol,
                active_base_tol=active_base_tol,
                has_group_info=True,
                mnar_intensity_pct=mnar_intensity_pct,
                compact=True,
                margin_right=0.0,
            )

            # Subplot S1: Group Rescue Scatter
            ax_group_rescue = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="s1",
            )
            self._plot_group_rescue_scatter(
                df_curr,
                "Max_Group_MV_Pct",
                "Min_Group_MV_Pct",
                mnar_group_mv_tol,
                active_base_tol,
                ax_group_rescue,
                "Group-level MNAR Rescue",
                article_compact=True,
            )
            # Cascade remaining features downward
            mask_group = df_curr["Stage1_Status"].str.contains("Group")
            df_curr = df_curr[~mask_group]

            # Subplot S2: QC Rescue Scatter
            ax_qc_rescue = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="s2",
            )
            self._plot_qc_rescue_scatter(
                df_curr,
                mnar_qc_mv_tol,
                mnar_int_threshold,
                ax_qc_rescue,
                "QC-level MNAR Rescue",
                mnar_intensity_pct=mnar_intensity_pct,
                article_compact=True,
            )
            # Cascade remaining features downward
            mask_qc = df_curr["Stage1_Status"].str.contains("QC")
            df_curr = df_curr[~mask_qc]

            # Subplot S3: Base Threshold Check Histogram
            ax_base_check = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="s3",
            )
            self._plot_cutoff_histogram(
                df_curr,
                "Min_Group_MV_Pct",
                "Stage1_Status",
                active_base_tol,
                {"MAR": pu.PRIMARY_ACCENT_COLOR, "INVALID": pu.NEUTRAL_COLOR},
                ["MAR", "INVALID"],
                ax_base_check,
                ("MAR Eligibility Check"),
                "Min Group-level MV (%)",
                article_compact=True,
            )

            # Column-first topology binding to enforce strict vertical alignment
            # Prevents width stretching caused by the axis-off flowchart
            col_left = ax_sample / ax_group_rescue
            col_right = ax_flow / (ax_qc_rescue | ax_base_check)
            return col_left | col_right

        else:
            # --- Layout B: No Groups (1 Full-width Top, 1+1+1 Bottom) ---

            # Flowchart ratio is 3 units wide to span the entire top row
            ax_flow = pw.Brick(
                figsize=pu.dashboard_brick_size(12.0, 4.0, layout_width),
                label="flowchart",
            )
            self._plot_pipeline_flowchart_atom(
                df=df_curr,
                ax=ax_flow,
                mnar_group_mv_tol=None,
                mnar_qc_mv_tol=mnar_qc_mv_tol,
                active_base_tol=active_base_tol,
                has_group_info=False,
                mnar_intensity_pct=mnar_intensity_pct,
                compact=True,
            )

            # Subplot S2: QC Rescue Scatter (Acts as Step 1 here)
            ax_qc_rescue = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="s2",
            )
            self._plot_qc_rescue_scatter(
                df_curr,
                mnar_qc_mv_tol,
                mnar_int_threshold,
                ax_qc_rescue,
                "QC-level MNAR Rescue",
                mnar_intensity_pct=mnar_intensity_pct,
                article_compact=True,
            )
            mask_qc = df_curr["Stage1_Status"].str.contains("QC")
            df_curr = df_curr[~mask_qc]

            # Subplot S3: Base Threshold Check Histogram (Acts as Step 2 here)
            ax_base_check = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="s3",
            )
            self._plot_cutoff_histogram(
                df_curr,
                "QC_MV_Pct",
                "Stage1_Status",
                active_base_tol,
                {"MAR": pu.PRIMARY_ACCENT_COLOR, "INVALID": pu.NEUTRAL_COLOR},
                ["MAR", "INVALID"],
                ax_base_check,
                "QC-level MV Check",
                "QC-level MV (%)",
                article_compact=True,
            )

            # Row-first topology binding: Full width top over 3 equal bottom
            row_bottom = ax_sample | ax_qc_rescue | ax_base_check
            return ax_flow / row_bottom

    # =========================================================================
    # Manuscript-only filtering article dashboards
    # =========================================================================
    def plot_high_mv_filter_article_dashboard(self) -> object | None:
        """Create a compact three-panel summary of high-MV feature screening.

        The manuscript-only layout retains the three decision diagnostics used
        to classify group-rescued MNAR, QC-rescued MNAR, and MAR features. It
        is deliberately independent of the full Stage 1 dashboard so the
        standard report layout and its typography remain unchanged.
        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning(
                "patchworklib not found. Skipping article dashboard."
            )
            return None

        tracking_df = self.engine.stats.get("stage1_tracking", pd.DataFrame())
        if tracking_df.empty:
            logger.warning(
                "Stage 1 tracking data are unavailable for article export."
            )
            return None

        has_group_info = (
            "Max_Group_MV_Pct" in tracking_df.columns
            and tracking_df["Max_Group_MV_Pct"].notna().any()
        )
        if not has_group_info:
            logger.warning(
                "Group-level MNAR rescue is unavailable; skipping high-MV "
                "article dashboard."
            )
            return None

        sample_type = self.engine.attrs.get("sample_type", "Sample Type")
        sample_dict = self.engine.attrs.get("sample_dict", {})
        qc_label = sample_dict.get("QC sample", "QC")
        qc_mask = (
            self.engine.columns.get_level_values(sample_type) == qc_label
            if sample_type in self.engine.columns.names
            else np.zeros(self.engine.shape[1], dtype=bool)
        )
        mnar_int_threshold = None
        if qc_mask.any():
            mnar_intensity_pct = self.engine.attrs.get(
                "mnar_intensity_pct", 0.1
            )
            raw_threshold = (
                self.engine.loc[:, qc_mask]
                .median(axis=1)
                .quantile(mnar_intensity_pct)
            )
            mnar_int_threshold = np.log2(raw_threshold + 1)

        active_base_tol = self.engine.attrs.get("mv_group_tol", 0.5)
        mnar_group_mv_tol = self.engine.attrs.get("mnar_group_mv_tol", 0.8)
        mnar_qc_mv_tol = self.engine.attrs.get("mnar_qc_mv_tol", 0.2)
        mnar_intensity_pct = self.engine.attrs.get("mnar_intensity_pct", 0.1)

        pw.clear()
        # Patchworklib adds fixed label/legend padding. This width yields an
        # approximately 17.7 cm export, within the ACS double-column limit.
        panel_size = pu.article_brick_size(1.72, 1.72)
        ax_group = pw.Brick(figsize=panel_size, label="article_group_rescue")
        ax_qc = pw.Brick(figsize=panel_size, label="article_qc_rescue")
        ax_mar = pw.Brick(figsize=panel_size, label="article_mar_eligibility")

        self._plot_group_rescue_scatter(
            tracking_df,
            "Max_Group_MV_Pct",
            "Min_Group_MV_Pct",
            mnar_group_mv_tol,
            active_base_tol,
            ax_group,
            "Group-level MNAR Rescue",
            article_compact=True,
        )
        self._apply_article_panel_format(ax_group, "Group-level MNAR Rescue")

        after_group = tracking_df[
            ~tracking_df["Stage1_Status"].str.contains("Group", na=False)
        ]
        self._plot_qc_rescue_scatter(
            after_group,
            mnar_qc_mv_tol,
            mnar_int_threshold,
            ax_qc,
            "QC-level MNAR Rescue",
            mnar_intensity_pct=mnar_intensity_pct,
            article_compact=True,
        )
        self._apply_article_panel_format(ax_qc, "QC-level MNAR Rescue")

        after_qc = after_group[
            ~after_group["Stage1_Status"].str.contains("QC", na=False)
        ]
        self._plot_cutoff_histogram(
            after_qc,
            "Min_Group_MV_Pct",
            "Stage1_Status",
            active_base_tol,
            {"MAR": pu.PRIMARY_ACCENT_COLOR, "INVALID": pu.NEUTRAL_COLOR},
            ["MAR", "INVALID"],
            ax_mar,
            "MAR Eligibility Check",
            "Min group MV (%)",
            article_compact=True,
        )
        self._apply_article_panel_format(ax_mar, "MAR Eligibility Check")

        return ax_group | ax_qc | ax_mar

    # =========================================================================
    # Low-quality Features Filtering Unified Summary Dashboard (1+N Layout)
    # =========================================================================
    def _plot_retained_count_steps(
        self,
        ax: plt.Axes | None = None,
        article_compact: bool = False,
    ) -> plt.Figure | plt.Axes:
        """Plot feature attrition cascade stacked bar chart by MAR/MNAR."""

        if ax is None:
            fig, current_ax = plt.subplots(
                figsize=pu.article_brick_size(4.5, 4.0)
                if article_compact
                else (4.5, 4.0)
            )
        else:
            current_ax = ax
            fig = current_ax.figure

        feature_counts = self.engine.stats.get("feature_counts", {})
        stats = self.engine.stats

        # Check if blank samples exist to dynamically adapt the X-axis steps
        blank_mean = stats.get("blank_mean")
        has_blanks = blank_mean is not None and not blank_mean.empty

        step_keys = [
            "raw",
            "post_stage1",
            "post_stage2_blank",
            "post_stage2_rsd",
        ]
        step_labels = [
            "Raw\nData",
            "High-MV\nCheck",
            "QC/Blank\nCheck",
            "QC RSD\nCheck",
        ]

        # Build valid indices, skipping Blank Check entirely if no blanks exist
        valid_idx = []
        for i, k in enumerate(step_keys):
            if k in feature_counts:
                if k == "post_stage2_blank" and not has_blanks:
                    continue
                valid_idx.append(i)

        if not valid_idx:
            return fig if ax is None else current_ax

        labels = [step_labels[i] for i in valid_idx]

        idx_mar = stats.get("idx_mar", pd.Index([]))
        idx_mnar = stats.get("idx_mnar", pd.Index([]))
        idx_dropped_blank = stats.get("idx_dropped_blank", pd.Index([]))
        idx_dropped_rsd = stats.get("idx_dropped_rsd", pd.Index([]))

        mar_base = len(idx_mar)
        mnar_base = len(idx_mnar)

        blank_drop_mar = len(idx_dropped_blank.intersection(idx_mar))
        blank_drop_mnar = len(idx_dropped_blank.intersection(idx_mnar))
        rsd_drop_mar = len(idx_dropped_rsd.intersection(idx_mar))
        rsd_drop_mnar = len(idx_dropped_rsd.intersection(idx_mnar))

        mar_all = np.array(
            [
                mar_base,
                mar_base,
                mar_base - blank_drop_mar,
                mar_base - blank_drop_mar - rsd_drop_mar,
            ]
        )

        mnar_all = np.array(
            [
                mnar_base,
                mnar_base,
                mnar_base - blank_drop_mnar,
                mnar_base - blank_drop_mnar - rsd_drop_mnar,
            ]
        )

        inv_base = max(0, feature_counts.get("raw", 0) - (mar_base + mnar_base))
        inv_all = np.array([inv_base, 0, 0, 0])

        mar_counts = mar_all[valid_idx]
        mnar_counts = mnar_all[valid_idx]
        inv_counts = inv_all[valid_idx]

        color_mar = pu.PRIMARY_ACCENT_COLOR
        color_mnar = pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=0.5)
        color_inv = "tab:gray"

        x = np.arange(len(labels))
        width = 0.55

        # Track dynamic bottoms for stacked bars to prevent empty legend items
        current_bottom = np.zeros(len(labels))

        if mar_base > 0:
            current_ax.bar(
                x,
                mar_counts,
                bottom=current_bottom,
                label="MAR",
                color=color_mar,
                edgecolor="k",
                width=width,
                linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
            )
            current_bottom += mar_counts

        if mnar_base > 0:
            current_ax.bar(
                x,
                mnar_counts,
                bottom=current_bottom,
                label="MNAR",
                color=color_mnar,
                edgecolor="k",
                width=width,
                linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
            )
            current_bottom += mnar_counts

        if inv_base > 0:
            current_ax.bar(
                x,
                inv_counts,
                bottom=current_bottom,
                label="Invalid",
                color=color_inv,
                edgecolor="k",
                width=width,
                linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
            )
            current_bottom += inv_counts

        totals = current_bottom

        current_ax.set_xticks(x)
        current_ax.set_xticklabels(labels)

        pu.show_values_on_bars(
            axs=current_ax,
            value_format="{:.0f}",
            fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
            stacked=True,
            skip_zero=True,
            threshold_pct=0.05,
        )

        self._apply_standard_format(
            ax=current_ax,
            title="Feature Retention Across Filtering Steps",
            xlabel="Filtering Steps",
            ylabel="Feature Count",
            append_stage=False,
        )

        self._format_single_legend(
            ax=current_ax,
            group_title="Feature type",
            loc="upper right",
            bbox_to_anchor=None,
            **(
                {
                    "fontsize": pu.DEFAULT_LEGEND_FONTSIZE,
                    "title_fontsize": pu.DEFAULT_LEGEND_TITLE_FONTSIZE,
                    "borderaxespad": 0.0,
                    "handlelength": 1.0,
                    "handletextpad": 0.3,
                    "labelspacing": 0.25,
                    "borderpad": 0.3,
                }
                if article_compact
                else {}
            ),
        )

        max_height = totals.max() if len(totals) > 0 else 1
        current_ax.set_ylim(0, max_height * 1.25)

        if ax is None:
            return fig
        return current_ax

    def _plot_qc_blank_scatter(
        self,
        ax: plt.Axes | None = None,
        article_compact: bool = False,
        legend_inside: bool = False,
    ) -> plt.Figure | plt.Axes | None:
        """Plots Log2 scatter of QC vs Blank intensities."""
        blank_mean = self.engine.stats.get("blank_mean")
        qc_mean = self.engine.stats.get("qc_mean")

        idx_mnar = pd.Index(self.engine.stats.get("idx_mnar", []))

        if blank_mean is None or qc_mean is None or blank_mean.empty:
            return None if ax is None else ax

        # Prepare data frame for plotting (treat missing blanks as 0)
        blank_safe = blank_mean.fillna(0).astype(float)

        df_plot = pd.DataFrame(
            {
                "QC": np.log2(qc_mean.astype(float) + 1),
                "Blank": np.log2(blank_safe + 1),
            }
        )

        df_plot["Feature Type"] = "MAR"
        valid_mnar = idx_mnar.intersection(df_plot.index)
        if not valid_mnar.empty:
            df_plot.loc[valid_mnar, "Feature Type"] = "MNAR"

        # Use blank_safe for ratios to match the filtering engine.
        # NaN <= 0.2 evaluates to False, falsely flagging them as Filtered.
        blank_qc_ratio_tol = self.engine.attrs.get("blank_qc_ratio_tol", 0.2)
        qc_safe = qc_mean.replace(0, np.finfo(float).eps).astype(float)

        df_plot["Status"] = np.where(
            blank_safe / qc_safe <= blank_qc_ratio_tol, "Retained", "Filtered"
        )

        # Sort DataFrame so MNAR points remain visible on top.
        # Alphabetical sorting ("MAR" < "MNAR") pushes MNAR to the bottom of
        # the DataFrame, causing seaborn to render them last and on top.
        df_plot = df_plot.sort_values(by="Feature Type", ascending=True)

        if ax is None:
            fig, current_ax = plt.subplots(
                figsize=pu.article_brick_size(5.0, 4.0)
                if article_compact
                else (5.0, 4.0)
            )
        else:
            current_ax = ax
            fig = current_ax.figure

        sns.scatterplot(
            data=df_plot,
            x="QC",
            y="Blank",
            ax=current_ax,
            hue="Status",
            palette={
                "Retained": pu.NEUTRAL_COLOR,
                "Filtered": pu.PRIMARY_ACCENT_COLOR,
            },
            style="Feature Type",
            markers={"MAR": "o", "MNAR": "X"},
            s=(
                pu.DEFAULT_COMPACT_SCATTER_MARKER_AREA
                if article_compact
                else pu.DEFAULT_SCATTER_MARKER_AREA
            ),
            edgecolor="k",
            linewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
        )

        lims = [
            np.min([current_ax.get_xlim(), current_ax.get_ylim()]),
            np.max([current_ax.get_xlim(), current_ax.get_ylim()]),
        ]
        x_line = np.linspace(max(0, lims[0]), lims[1], 200)
        current_ax.plot(
            x_line,
            np.log2(((2**x_line - 1) * blank_qc_ratio_tol) + 1),
            color="k",
            linestyle="--",
            linewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
            label=f"Mean Blank / Mean QC <= {blank_qc_ratio_tol:.2f}",
        )

        self._apply_standard_format(
            ax=current_ax,
            title="Blank/QC Check",
            xlabel="log2(Mean QC + 1)",
            ylabel="log2(Mean Blank + 1)",
            append_stage=False,
        )

        legend_handles = [
            mlines.Line2D([], [], color="none", label="Status"),
            mlines.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=pu.NEUTRAL_COLOR,
                markeredgecolor="k",
                markeredgewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
                markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                label="Retained",
            ),
            mlines.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=pu.PRIMARY_ACCENT_COLOR,
                markeredgecolor="k",
                markeredgewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
                markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                label="Filtered",
            ),
            mlines.Line2D([], [], color="none", label="Feature type"),
            mlines.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color="0.35",
                markeredgecolor="k",
                markeredgewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
                markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                label="MAR",
            ),
            mlines.Line2D(
                [],
                [],
                marker="X",
                linestyle="",
                color="0.35",
                markeredgecolor="k",
                markeredgewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
                markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                label="MNAR",
            ),
            mlines.Line2D([], [], color="none", label="Thresholds"),
            mlines.Line2D(
                [],
                [],
                color="k",
                linestyle="--",
                linewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
                label=f"Mean Blank / Mean QC <= {blank_qc_ratio_tol:.2f}",
            ),
        ]
        current_ax.legend(handles=legend_handles)
        self._format_multi_legends(
            ax=current_ax,
            group_titles=["Status", "Feature type", "Thresholds"],
            loc=(
                "upper left"
                if article_compact
                else ("upper right" if legend_inside else "upper left")
            ),
            start_bbox=(0.02, 0.98)
            if article_compact
            else ((0.98, 0.98) if legend_inside else (1.05, 1.0)),
            layout_cols=1,
            sublegend_cols=1,
            **(
                {
                    "fontsize": pu.DEFAULT_LEGEND_FONTSIZE,
                    "title_fontsize": pu.DEFAULT_LEGEND_TITLE_FONTSIZE,
                    "borderaxespad": 0.0,
                    "handlelength": 1.0,
                    "handletextpad": 0.3,
                    "labelspacing": 0.25,
                    "borderpad": 0.3,
                }
                if article_compact
                else {}
            ),
        )

        if ax is None:
            return fig
        return current_ax

    def _plot_rsd_dist(
        self,
        idx_mar: pd.Index | list[object],
        ax: plt.Axes | None = None,
        article_compact: bool = False,
    ) -> plt.Figure | plt.Axes | None:
        """
        Plot the MAR-only QC-RSD distribution used for reproducibility
        filtering.
        """
        qc_rsd_all = self.engine.stats.get("qc_rsd_all")
        if qc_rsd_all is None or qc_rsd_all.empty:
            return None if ax is None else ax

        if not isinstance(idx_mar, pd.Index):
            idx_mar = pd.Index(idx_mar)

        if ax is None:
            fig, current_ax = plt.subplots(
                figsize=pu.article_brick_size(4.0, 4.0)
                if article_compact
                else (4.0, 4.0)
            )
        else:
            current_ax = ax
            fig = current_ax.figure
        pu.mark_preserve_alpha(current_ax)

        qc_rsd_tol = self.engine.attrs.get("qc_rsd_tol", 0.3)
        mar_rsd = qc_rsd_all.loc[
            qc_rsd_all.index.intersection(idx_mar)
        ].dropna()
        if mar_rsd.empty:
            current_ax.axis("off")
            return fig if ax is None else current_ax

        max_rsd = float(mar_rsd.max())
        bin_edges = np.linspace(0, max_rsd, 50)
        mar_color = pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=1.0)

        sns.histplot(
            x=mar_rsd,
            color=mar_color,
            bins=bin_edges,
            kde=True,
            ax=current_ax,
            legend=False,
            edgecolor="k",
            linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
        )

        current_ax.axvline(
            x=qc_rsd_tol,
            color="k",
            linestyle="--",
            linewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
        )

        handles = [
            mpatches.Patch(
                facecolor=mar_color,
                edgecolor="k",
                linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
                label="MAR features",
            ),
            mlines.Line2D(
                [],
                [],
                color="k",
                linestyle="--",
                label=f"MAR Threshold ({qc_rsd_tol})",
                linewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
            ),
        ]

        legend_kwargs = getattr(self, "LEGEND_KWARGS", {}).copy()
        if article_compact:
            legend_kwargs.update(
                {
                    "fontsize": pu.DEFAULT_LEGEND_FONTSIZE,
                    "title_fontsize": pu.DEFAULT_LEGEND_TITLE_FONTSIZE,
                    "borderaxespad": 0.0,
                    "handlelength": 1.0,
                    "handletextpad": 0.3,
                    "labelspacing": 0.25,
                    "borderpad": 0.3,
                }
            )

        current_ax.legend(
            handles=handles,
            title="QC-RSD filter",
            loc="upper right",
            **legend_kwargs,
        )

        self._apply_standard_format(
            ax=current_ax,
            title="QC-RSD Check",
            xlabel="RSD",
            ylabel="Feature Count",
            append_stage=False,
        )

        if ax is None:
            return fig
        return current_ax

    def plot_quality_filtering_summary_grid(self) -> object | None:
        """Combine Stage 2 plots into a single figure using patchworklib.

        Dynamically adapts the grid layout: renders a 1x3 grid if Blank
        samples are present, or a 1x2 grid if Blank samples are missing.
        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping summary grid.")
            return None

        pw.clear()

        # 1. Detect if Blank data exists to determine the topology
        blank_mean = self.engine.stats.get("blank_mean")
        has_blanks = blank_mean is not None and not blank_mean.empty

        idx_mar = self.engine.stats.get("idx_mar", pd.Index([]))

        # 2. Topology A: 1x3 Grid (Blank samples exist)
        if has_blanks:
            layout_width = 12.0
            ax1 = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="qc_blank",
            )
            ax2 = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="qc_rsd",
            )
            ax3 = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="retention",
            )

            self._plot_qc_blank_scatter(ax=ax1, article_compact=True)
            self._plot_rsd_dist(idx_mar=idx_mar, ax=ax2, article_compact=True)
            self._plot_retained_count_steps(ax=ax3, article_compact=True)

            return ax1 | ax2 | ax3

        # 3. Topology B: 1x2 Grid (No Blank samples)
        else:
            layout_width = 8.0
            ax2 = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="qc_rsd",
            )
            ax3 = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="retention",
            )

            self._plot_rsd_dist(idx_mar=idx_mar, ax=ax2, article_compact=True)
            self._plot_retained_count_steps(ax=ax3, article_compact=True)

            return ax2 | ax3

    def plot_low_quality_filter_article_dashboard(self) -> object | None:
        """
        Create a compact three-panel summary of low-quality feature filtering.

        The QC-RSD panel deliberately reuses the MAR-only distribution used by
        the filtering engine. MNAR features remain absent from this diagnostic
        because they are exempt from the QC-RSD reproducibility filter.

        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning(
                "patchworklib not found. Skipping article dashboard."
            )
            return None

        blank_mean = self.engine.stats.get("blank_mean")
        idx_mar = self.engine.stats.get("idx_mar", pd.Index([]))
        if blank_mean is None or blank_mean.empty or len(idx_mar) == 0:
            logger.warning(
                "Blank/QC and MAR QC-RSD inputs are required for the "
                "article dashboard."
            )
            return None

        pw.clear()
        # Compensate for the smaller low-quality layout margin so the exported
        # dashboard matches the high-MV article dashboard at approximately 17.7
        # cm.
        panel_size = pu.article_brick_size(1.72, 1.72)
        ax_blank = pw.Brick(figsize=panel_size, label="article_blank_qc")
        ax_rsd = pw.Brick(figsize=panel_size, label="article_qc_rsd")
        ax_retention = pw.Brick(
            figsize=panel_size, label="article_feature_retention"
        )

        self._plot_qc_blank_scatter(
            ax=ax_blank,
            article_compact=True,
            legend_inside=True,
        )
        self._apply_article_panel_format(ax_blank, "Blank/QC Check")

        self._plot_rsd_dist(
            idx_mar=idx_mar,
            ax=ax_rsd,
            article_compact=True,
        )
        self._apply_article_panel_format(ax_rsd, "QC-RSD Check")

        self._plot_retained_count_steps(
            ax=ax_retention,
            article_compact=True,
        )
        self._apply_article_panel_format(
            ax_retention, "Feature Retention Across Filtering Steps"
        )

        return ax_blank | ax_rsd | ax_retention
