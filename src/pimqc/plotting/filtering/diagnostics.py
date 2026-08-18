"""Diagnostic panels for sample and feature filtering.

The module renders sample missingness, MNAR rescue, MAR eligibility, blank/QC,
QC-RSD, and feature-retention panels from precomputed filtering statistics.
"""

from __future__ import annotations

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ...constants import DEFAULT_RANDOM_SEED
from .. import annotation_layout as al
from .. import plot_utils as pu


class FilteringDiagnosticsMixin:
    """Render the individual panels consumed by filtering dashboards."""

    # High-missing-value filtering panels
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
        rng = np.random.default_rng(
            int(self.engine.attrs.get("global_seed", DEFAULT_RANDOM_SEED))
        )

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
        ]
        current_ax.legend(handles=sample_handles)
        self._format_multi_legends(
            ax=current_ax,
            group_titles=["Sample type"],
            loc="upper right",
            start_bbox=(0.98, 0.98),
            layout_cols=1,
            sublegend_cols=1,
        )

        # Adjust Y-limit slightly to prevent top annotations from clipping
        actual_y_max = max(df_plot["MV_Rate_Pct"].max(), tol_pct)
        padding = actual_y_max * 0.15 if actual_y_max > 0 else 10
        current_ax.set_ylim(-5, actual_y_max + padding)
        al.annotate_reference_line(
            ax=current_ax,
            value=tol_pct,
            text=f"Sample exclusion = {tol_pct:.0f}%",
            orientation="horizontal",
            occupancy_arrays=[
                np.column_stack(
                    (
                        df_plot["Sample_Type"].map(x_lookup).to_numpy(float),
                        df_plot["MV_Rate_Pct"].to_numpy(float),
                    )
                )
            ],
        )
        return fig if ax is None else current_ax

    # High-MV Features Filtering Unified Summary Dashboard (1+N Layout)
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
        ]

        ax.legend(handles=handles)
        self._format_multi_legends(
            ax=ax,
            group_titles=["Status"],
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
        al.annotate_reference_lines(
            ax=ax,
            references=[
                {
                    "value": tol_max_pct,
                    "text": f"Max group MV = {tol_max_pct:.0f}%",
                    "orientation": "vertical",
                },
                {
                    "value": tol_min_pct,
                    "text": f"Min group MV = {tol_min_pct:.0f}%",
                    "orientation": "horizontal",
                },
            ],
            occupancy_arrays=[
                df_plot[[max_col, min_col]].to_numpy(dtype=float)
            ],
        )

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
        ]

        group_titles = ["Status"]

        if has_group_info:
            marker_min = 9 if article_compact else 18
            marker_span = 27 if article_compact else 54
            handles.extend(
                [
                    mlines.Line2D([], [], color="none", label="Size reference"),
                    mlines.Line2D(
                        [],
                        [],
                        color="white",
                        marker="o",
                        markersize=np.sqrt(marker_min),
                        label="Lower Min Group MV",
                        markerfacecolor="white",
                        markeredgecolor="gray",
                        linestyle="",
                    ),
                    mlines.Line2D(
                        [],
                        [],
                        color="white",
                        marker="o",
                        markersize=np.sqrt(marker_min + marker_span),
                        label="Higher Min Group MV",
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
        references: list[dict[str, object]] = [
            {
                "value": qc_mv_cutoff_pct,
                "text": f"QC MV = {qc_mv_cutoff_pct:.0f}%",
                "orientation": "horizontal",
            }
        ]
        if mnar_int_threshold is not None:
            references.append(
                {
                    "value": mnar_int_threshold,
                    "text": f"QC intensity = {mnar_int_threshold:.2f}",
                    "orientation": "vertical",
                }
            )
        self._apply_standard_format(
            ax=ax,
            title=title,
            xlabel="log2(Median QC Intensity)",
            ylabel="QC Missing Rate (%)",
            append_stage=False,
        )
        ax.title.set_weight("bold")
        al.annotate_reference_lines(
            ax=ax,
            references=references,
            occupancy_arrays=[
                df_plot[["Log2_Intensity", "QC_MV_Pct"]].to_numpy(dtype=float)
            ],
        )

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

        ax.legend(handles=handles)
        self._format_multi_legends(
            ax=ax,
            group_titles=["Status"],
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
        al.annotate_reference_line(
            ax=ax,
            value=tol * 100,
            text=f"Min group MV = {tol * 100:.0f}%",
            orientation="vertical",
            occupancy_artists=list(ax.patches),
        )

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

        feature_counts = self.audit_tables.get("feature_counts", {})
        stats = self.audit_tables

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
        blank_mean = self.audit_tables.get("blank_mean")
        qc_mean = self.audit_tables.get("qc_mean")

        idx_mnar = pd.Index(self.audit_tables.get("idx_mnar", []))

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
        ]
        current_ax.legend(handles=legend_handles)
        self._format_multi_legends(
            ax=current_ax,
            group_titles=["Status", "Feature type"],
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
        al.annotate_reference_curve(
            ax=current_ax,
            x_values=x_line,
            y_values=np.log2(((2**x_line - 1) * blank_qc_ratio_tol) + 1),
            text=f"Blank/QC cutoff = {blank_qc_ratio_tol:.2f}",
            occupancy_arrays=[df_plot[["QC", "Blank"]].to_numpy(float)],
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
        qc_rsd_all = self.audit_tables.get("qc_rsd_all")
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
        al.annotate_reference_line(
            ax=current_ax,
            value=float(qc_rsd_tol),
            text=f"QC-RSD cutoff = {float(qc_rsd_tol):.2f}",
            orientation="vertical",
            occupancy_artists=list(current_ax.patches),
        )

        if ax is None:
            return fig
        return current_ax
