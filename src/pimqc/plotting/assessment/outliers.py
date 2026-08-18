"""RSD and outlier diagnostics for quality assessment.

The module renders feature-RSD distributions, score/orthogonal-distance
scatter plots, stacked outlier bars, and their standalone legends.
"""

from __future__ import annotations

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from .. import annotation_layout as al
from .. import plot_utils as pu


class AssessmentOutlierMixin:
    """Render RSD and multivariate outlier panels and legends."""

    def plot_rsd_standalone_legend(
        self,
        qc_label: str,
        actual_label: str,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Render the shared sample-type legend used by exported RSD panels."""
        standalone = ax is None
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(2.2, 1.6))
        else:
            current_ax = ax
            fig = current_ax.figure

        handles = [
            Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=1.0),
                edgecolor="black",
                linestyle="-",
                linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
                label=qc_label,
            ),
            Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=0.4),
                edgecolor="black",
                linestyle="--",
                linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
                label=actual_label,
            ),
        ]
        legend = current_ax.legend(
            handles=handles,
            title="Sample Type",
            loc="center left",
            bbox_to_anchor=(0.0, 0.5),
            **getattr(self, "LEGEND_KWARGS", {}),
        )
        # A figure-level legend lets ``bbox_inches='tight'`` crop to the
        # legend itself instead of retaining the otherwise empty axes canvas.
        if standalone and legend not in fig.legends:
            fig.legends.append(legend)
        if standalone:
            current_ax.set_visible(False)
        return fig

    # RSD and outlier diagnostic panels
    def plot_rsd_bar(
        self,
        rsd_data: dict[str, dict[str, int]],
        qc_label: str,
        actual_label: str,
        ax: plt.Axes | None = None,
        legend_mode: str = "local",
        title_mode: str = "full",
        **kwargs: object,
    ) -> plt.Figure:
        """Plots the RSD distribution using explicitly provided data.

        Converts the pre-calculated RSD dictionary into a format suitable
        for seaborn. Applies custom RGBA alpha blending, container styling,
        and removes zero-height patches to prevent annotation artifacts.

        Args:
            rsd_data (dict): Pre-calculated RSD distribution dictionary.
            qc_label (str): Label for QC samples.
            actual_label (str): Label for actual samples.
            ax (matplotlib.axes.Axes, optional): The target axes object.
            **kwargs: Additional formatting parameters.

        Returns:
            matplotlib.figure.Figure: The rendered figure object.
        """
        legend_mode = self._validate_legend_mode(legend_mode)

        # Initialize axes hierarchy
        if ax is None:
            fig, current_ax = plt.subplots(
                figsize=pu.dashboard_brick_size(4.0, 4.0, 14.0)
            )
        else:
            current_ax = ax
            fig = current_ax.figure

        labels = ["0-10%", "10-20%", "20-30%", ">30%"]
        x_pos = np.arange(len(labels), dtype=float)
        bar_width = 0.42
        qc_counts = np.asarray(
            [rsd_data.get("qc", {}).get(label, 0) for label in labels]
        )
        actual_counts = np.asarray(
            [rsd_data.get("actual", {}).get(label, 0) for label in labels]
        )
        qc_bar_colors = [
            pu.get_equivalent_hex(
                pu.PRIMARY_ACCENT_COLOR if label == ">30%" else pu.NEUTRAL_COLOR
            )
            for label in labels
        ]
        actual_bar_colors = [
            pu.get_equivalent_hex(
                pu.PRIMARY_ACCENT_COLOR
                if label == ">30%"
                else pu.NEUTRAL_COLOR,
                alpha=0.4,
            )
            for label in labels
        ]

        current_ax.bar(
            x_pos - bar_width / 2,
            qc_counts,
            width=bar_width,
            color=qc_bar_colors,
            edgecolor="black",
            linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
            linestyle="-",
            label=qc_label,
        )
        current_ax.bar(
            x_pos + bar_width / 2,
            actual_counts,
            width=bar_width,
            color=actual_bar_colors,
            edgecolor="black",
            linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
            linestyle="--",
            label=actual_label,
        )

        # Update axis limit and annotate after removing empty patches
        max_count = max(
            float(np.nanmax(qc_counts)), float(np.nanmax(actual_counts)), 1.0
        )
        current_ax.set_ylim(0, max_count * 1.3)
        pu.show_values_on_bars(
            axs=current_ax,
            show_percentage=True,
            value_format="{:.0f}",
            pct_type="group",
            fontsize=pu.DEFAULT_DENSE_BAR_ANNOTATION_FONTSIZE,
        )

        # Manually construct legend to ensure correct style mapping
        h_type = [
            Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=1.0),
                edgecolor="black",
                linestyle="-",
                linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
                label=qc_label,
            ),
            Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=0.4),
                edgecolor="black",
                linestyle="--",
                linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
                label=actual_label,
            ),
        ]

        # Bypass auto-formatters using global LEGEND_KWARGS
        legend_kwargs = getattr(self, "LEGEND_KWARGS", {}).copy()
        legend_kwargs.update({"title": "Sample Type", "loc": "best"})
        if legend_mode == "local":
            current_ax.legend(handles=h_type, **legend_kwargs)
        current_ax.set_xticks(x_pos)
        current_ax.set_xticklabels(labels)

        # Execute standardized axis formatting
        if hasattr(self, "_apply_standard_format"):
            rsd_title, append_stage = self._panel_title(
                "Feature RSD Distribution", title_mode
            )
            self._apply_standard_format(
                ax=current_ax,
                title=rsd_title,
                xlabel="RSD Bin",
                ylabel="Feature Count",
                append_stage=append_stage,
            )

        return fig

    # Outlier Detection
    def plot_sd_od_scatter(
        self,
        metrics_df: pd.DataFrame,
        sd_limit: float,
        od_limit: float,
        is_flags: pd.Series | None = None,
        orf_flags: pd.Series | None = None,
        ax: plt.Axes | None = None,
        show_legend: bool = True,
        annotate_thresholds: bool = False,
        legend_mode: str = "local",
        title_mode: str = "full",
    ) -> plt.Figure:
        """Plot SD-OD diagnostic scatter with multi-dimensional overlays."""
        legend_mode = self._validate_legend_mode(legend_mode)
        accent_solid = pu.PRIMARY_ACCENT_COLOR
        accent_alpha = pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=0.5)

        custom_pal = {
            "Normal": "tab:gray",
            "Strong Outlier": accent_alpha,
            "Orthogonal Outlier": accent_alpha,
            "Extreme Outlier": accent_solid,
        }
        custom_markers = {
            "Normal": "o",
            "Strong Outlier": "^",
            "Orthogonal Outlier": "s",
            "Extreme Outlier": "X",
        }

        if ax is None:
            fig, current_ax = plt.subplots(
                figsize=pu.dashboard_brick_size(4.0, 4.0, 14.0)
            )
        else:
            current_ax = ax
            fig = current_ax.figure

        sns.scatterplot(
            data=metrics_df,
            x="SD",
            y="OD",
            hue="Category",
            style="Category",
            palette=custom_pal,
            markers=custom_markers,
            s=pu.DEFAULT_SCATTER_MARKER_AREA,
            edgecolor="k",
            linewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
            ax=current_ax,
            zorder=2,
        )

        # Overlay halo effect for analytical IS outliers (accent dashed circle)
        if is_flags is not None and is_flags.any():
            outlier_idx = is_flags[is_flags].index.intersection(
                metrics_df.index
            )
            subset = metrics_df.loc[outlier_idx]
            if not subset.empty:
                current_ax.scatter(
                    subset["SD"],
                    subset["OD"],
                    s=60,
                    facecolors="none",
                    edgecolors=pu.PRIMARY_ACCENT_COLOR,
                    linewidths=pu.DEFAULT_GUIDE_LINEWIDTH,
                    linestyle="--",
                    zorder=3,
                    label="IS Outlier",
                )

        # Overlay halo effect for analytical ORF outliers (Orange dash-dot
        # circle)
        if orf_flags is not None and orf_flags.any():
            outlier_idx = orf_flags[orf_flags].index.intersection(
                metrics_df.index
            )
            subset = metrics_df.loc[outlier_idx]
            if not subset.empty:
                current_ax.scatter(
                    subset["SD"],
                    subset["OD"],
                    s=72,
                    facecolors="none",
                    edgecolors="tab:orange",
                    linewidths=pu.DEFAULT_GUIDE_LINEWIDTH,
                    linestyle="-.",
                    zorder=4,
                    label="ORF Outlier",
                )

        threshold_color = pu.get_equivalent_hex("k", alpha=0.6)
        current_ax.axvline(
            x=sd_limit,
            color=threshold_color,
            linestyle="--",
            linewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
        )
        current_ax.axhline(
            y=od_limit,
            color=threshold_color,
            linestyle="--",
            linewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
        )

        outlier_title, append_stage = self._panel_title(
            "Integrated Outlier Diagnostics", title_mode
        )
        self._apply_standard_format(
            ax=current_ax,
            xlabel="Score Distance (Hotelling's T2)",
            ylabel="Orthogonal Distance (SPE / DModX)",
            append_stage=append_stage,
            title=outlier_title,
        )

        if show_legend and legend_mode == "local":
            handles, labels = current_ax.get_legend_handles_labels()
            dummy_cat = mlines.Line2D([], [], color="none", label="Category")
            full_handles = [dummy_cat] + handles
            full_labels = ["Category"] + labels

            current_ax.legend(full_handles, full_labels)

            self._format_multi_legends(
                ax=current_ax,
                group_titles=["Category"],
                loc="upper left",
                start_bbox=(1.05, 1.0),
                row_gap=0.04,
                layout_cols=1,
                sublegend_cols=1,
            )
        else:
            self._remove_axis_legends(current_ax)

        current_ax.autoscale()
        # Place threshold labels after the final legend and axis limits settle.
        if annotate_thresholds:
            point_arrays = [
                current_ax.collections[0].get_offsets()
                if current_ax.collections
                else np.empty((0, 2))
            ]
            al.annotate_reference_lines(
                ax=current_ax,
                references=[
                    {
                        "value": sd_limit,
                        "text": f"SD limit = {sd_limit:.2f}",
                        "orientation": "vertical",
                        "color": threshold_color,
                    },
                    {
                        "value": od_limit,
                        "text": f"OD limit = {od_limit:.2f}",
                        "orientation": "horizontal",
                        "color": threshold_color,
                    },
                ],
                occupancy_arrays=point_arrays,
                expand_axis=True,
            )
        return fig

    def _plot_stat_outliers_bar(
        self,
        outliers_df: pd.DataFrame,
        sample_type: str,
        batch: str,
        sample_name: str,
        actual_label: str,
        target_param: str = "both",
        sd_limit: float | None = None,
        od_limit: float | None = None,
        show_normal: bool = False,
        is_flags: pd.Series | None = None,
        orf_flags: pd.Series | None = None,
        ax1: plt.Axes | None = None,
        ax2: plt.Axes | None = None,
        show_legend: bool = True,
    ) -> plt.Figure | None:
        """Plot outlier results with symmetrical reference flag encodings."""
        sample_types = outliers_df.index.get_level_values(sample_type)
        mask = sample_types == actual_label
        out_df = outliers_df[mask].copy()

        if out_df.empty:
            return None

        def _get_category(row: pd.Series) -> str:
            spe = row[("SPE-DModX", "Outliers (SPE-DModX)")]
            ht2 = row[("HT2", "Outliers (HT2)")]
            if spe and ht2:
                return "Extreme Outlier"
            elif spe:
                return "Orthogonal Outlier"
            elif ht2:
                return "Strong Outlier"
            return "Normal"

        cats = out_df.apply(_get_category, axis=1)

        if not show_normal:
            outlier_mask = cats != "Normal"

            if is_flags is not None:
                is_sub_mask = is_flags.loc[out_df.index].fillna(False).values
                outlier_mask = outlier_mask | is_sub_mask

            if orf_flags is not None:
                orf_sub_mask = orf_flags.loc[out_df.index].fillna(False).values
                outlier_mask = outlier_mask | orf_sub_mask

            out_df = out_df[outlier_mask].copy()
            cats = cats[outlier_mask].copy()

        if out_df.empty:
            return None

        io_col = getattr(self, "io_col", "Inject Order")
        if io_col in out_df.index.names:
            out_df = out_df.sort_index(level=io_col)
            cats = cats.loc[out_df.index]

        idx_df = out_df.index.to_frame()
        batch_str = idx_df[batch].astype(str)
        name_str = idx_df[sample_name].astype(str)
        new_idx = (batch_str + "-" + name_str).values

        # Symmetrically fetch reference flags with strict length checks
        if is_flags is not None:
            is_sub = is_flags.loc[out_df.index].fillna(False).values
        else:
            is_sub = np.zeros(len(out_df), dtype=bool)

        if orf_flags is not None:
            orf_sub = orf_flags.loc[out_df.index].fillna(False).values
        else:
            orf_sub = np.zeros(len(out_df), dtype=bool)

        labeled_idx = []
        for name, f_is, f_orf in zip(new_idx, is_sub, orf_sub):
            if f_is and f_orf:
                labeled_idx.append(f"{name} *#")
            elif f_is:
                labeled_idx.append(f"{name} *")
            elif f_orf:
                labeled_idx.append(f"{name} #")
            else:
                labeled_idx.append(name)
        new_idx = np.array(labeled_idx)

        out_df.index = new_idx
        cats.index = new_idx
        out_df = out_df.rename_axis(index=["Sample ID"])

        accent_solid = pu.PRIMARY_ACCENT_COLOR
        accent_alpha = pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=0.5)
        gray_col = pu.NEUTRAL_COLOR

        palette_spe = {
            "Extreme Outlier": accent_solid,
            "Orthogonal Outlier": accent_alpha,
            "Strong Outlier": gray_col,
            "Normal": gray_col,
        }
        palette_ht2 = {
            "Extreme Outlier": accent_solid,
            "Orthogonal Outlier": gray_col,
            "Strong Outlier": accent_alpha,
            "Normal": gray_col,
        }

        hatch_styles = {
            "Extreme Outlier": "",
            "Orthogonal Outlier": "///",
            "Strong Outlier": r"\\\\",
            "Normal": "",
        }
        cat_order = [
            "Extreme Outlier",
            "Orthogonal Outlier",
            "Strong Outlier",
            "Normal",
        ]

        if ax1 is None or ax2 is None:
            fig, (current_ax1, current_ax2) = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(out_df.shape[0] * 0.3 + 2, 7),
                sharex=True,
            )
        else:
            current_ax1, current_ax2 = ax1, ax2
            fig = current_ax1.figure

        axes_list = [current_ax1, current_ax2]
        metrics = ["SPE-DModX", "HT2"]
        cols = ["SPE-DModX", "Hotelling T2 Score"]
        palettes = [palette_spe, palette_ht2]

        n_samples = out_df.shape[0]
        ax_w, _ = pu.axis_size_inches(current_ax2)
        raw_tick_labels = new_idx.tolist()
        needs_dense_ticks = pu.tick_labels_need_compaction(
            labels=raw_tick_labels,
            n_items=n_samples,
            axis_inches=ax_w,
            default_size=pu.DEFAULT_AXIS_TICK_FONTSIZE,
        )
        max_tick_chars = pu.dense_label_char_limit(n_samples)
        display_idx = (
            [
                pu.compact_tick_label(label, max_tick_chars)
                for label in raw_tick_labels
            ]
            if needs_dense_ticks
            else raw_tick_labels
        )
        dynamic_tick_size = pu.dense_tick_fontsize(
            n_items=n_samples,
            axis_inches=ax_w,
            default_size=pu.DEFAULT_AXIS_TICK_FONTSIZE,
            max_size=pu.DEFAULT_AXIS_TICK_FONTSIZE,
            min_size=1.4,
            fill_ratio=0.78,
            force_dense=needs_dense_ticks,
        )
        max_full_ticks = max(40, int(ax_w * 14))
        show_all_ticks = (not needs_dense_ticks) or n_samples <= max_full_ticks
        max_sparse_ticks = max(45, int(ax_w * 10))
        step = max(1, int(np.ceil(n_samples / max_sparse_ticks)))

        for i, (ax, metric, col, pal) in enumerate(
            zip(axes_list, metrics, cols, palettes)
        ):
            threshold_spec = None
            df_plot = out_df[metric].reset_index()
            df_plot["Category"] = cats.values
            present_cats = [c for c in cat_order if c in cats.values]

            sns.barplot(
                ax=ax,
                data=df_plot,
                x="Sample ID",
                y=col,
                hue="Category",
                palette=pal,
                hue_order=present_cats,
                dodge=False,
            )

            for j, cat in enumerate(present_cats):
                if j < len(ax.containers):
                    for bar in ax.containers[j]:
                        bar.set_facecolor(pal[cat])
                        bar.set_edgecolor("black")
                        bar.set_linewidth(pu.DEFAULT_AXIS_LINEWIDTH)
                        bar.set_hatch(hatch_styles[cat])

            if i == 0 and od_limit is not None:
                ax.axhline(
                    y=od_limit,
                    color="k",
                    linestyle="--",
                    linewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
                    alpha=0.8,
                    zorder=2,
                )
                threshold_spec = (od_limit, f"OD limit = {od_limit:.2f}")
            elif i == 1 and sd_limit is not None:
                ax.axhline(
                    y=sd_limit,
                    color="k",
                    linestyle="--",
                    linewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
                    alpha=0.8,
                    zorder=2,
                )
                threshold_spec = (sd_limit, f"SD limit = {sd_limit:.2f}")

            if i == 0:
                self._apply_standard_format(
                    ax=ax,
                    title="Integrated Outlier Barplot",
                    title_fontsize=pu.DEFAULT_TITLE_FONTSIZE,
                    label_fontsize=pu.DEFAULT_AXIS_LABEL_FONTSIZE,
                    append_stage=True,
                    ylabel="Orthogonal Distance\n(SPE / DModX)",
                )
            else:
                self._apply_standard_format(
                    ax=ax,
                    title="",
                    xlabel="Sample ID",
                    title_fontsize=pu.DEFAULT_TITLE_FONTSIZE,
                    label_fontsize=pu.DEFAULT_AXIS_LABEL_FONTSIZE,
                    append_stage=False,
                    ylabel="Score Distance\n(Hotelling's T2)",
                )

            if ax.get_legend():
                ax.get_legend().remove()
            if threshold_spec is not None:
                al.annotate_reference_line(
                    ax=ax,
                    value=float(threshold_spec[0]),
                    text=threshold_spec[1],
                    orientation="horizontal",
                    color=pu.get_equivalent_hex("k", alpha=0.6),
                    occupancy_artists=list(ax.patches),
                    expand_axis=True,
                )

        current_ax1.set_xlabel("")
        current_ax1.tick_params(axis="x", bottom=False, labelbottom=False)
        current_ax2.tick_params(axis="x", bottom=True, labelbottom=True)
        current_ax2.set_xlabel("Sample ID")
        current_ax2.xaxis.label.set_fontsize(pu.DEFAULT_AXIS_LABEL_FONTSIZE)
        current_ax2.xaxis.label.set_fontweight(pu.DEFAULT_AXIS_LABEL_WEIGHT)

        current_ax2.set_xticks(np.arange(n_samples))
        current_ax2.set_xticklabels(
            display_idx,
            rotation=90,
            ha="right",
            va="center",
            fontsize=dynamic_tick_size,
            rotation_mode="anchor",
        )
        current_ax2.tick_params(axis="x", pad=1, length=2)

        cat_values = cats.values
        for idx, label in enumerate(current_ax2.xaxis.get_ticklabels()):
            full_text = (
                str(new_idx[idx]) if idx < len(new_idx) else label.get_text()
            )
            is_extreme = (
                idx < len(cat_values) and cat_values[idx] == "Extreme Outlier"
            )
            if "*" in full_text or "#" in full_text or is_extreme:
                label.set_color(pu.PRIMARY_ACCENT_COLOR)

        if not show_all_ticks:
            visible_indices = {0, n_samples - 1}
            for i in range(step, n_samples - 1, step):
                if (n_samples - 1 - i) > (step * 0.7):
                    visible_indices.add(i)

            for idx, label in enumerate(current_ax2.xaxis.get_ticklabels()):
                label.set_visible(idx in visible_indices)

        return fig

    def plot_outlier_standalone_legend(
        self,
        metrics_df: pd.DataFrame,
        sd_limit: float,
        od_limit: float,
        ax: plt.Axes | None = None,
        is_flags: pd.Series | None = None,
        orf_flags: pd.Series | None = None,
        complete_categories: bool = False,
        include_bar_diagnostics: bool = True,
        include_thresholds: bool = True,
    ) -> plt.Figure | plt.Axes:
        """Create a comprehensive unified legend for all outlier diagnostics."""
        standalone = ax is None
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(2.0, 4.0))
        else:
            current_ax = ax
            fig = current_ax.figure

        accent_solid = pu.PRIMARY_ACCENT_COLOR
        accent_alpha = pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=0.5)
        gray_col = pu.NEUTRAL_COLOR

        cat_styles = {
            "Extreme Outlier": {
                "color": accent_solid,
                "marker": "X",
                "hatch": "",
            },
            "Orthogonal Outlier": {
                "color": accent_alpha,
                "marker": "s",
                "hatch": "///",
            },
            "Strong Outlier": {
                "color": accent_alpha,
                "marker": "^",
                "hatch": r"\\\\",
            },
            "Normal": {"color": gray_col, "marker": "o", "hatch": ""},
        }

        def _reference_features_available(
            flags: pd.Series | None, feature_attr: str
        ) -> bool:
            """
            Return whether a reference-feature channel exists for this QA run.

            A boolean flag series can be present even when every sample is
            normal. The assessor's validated feature list is the authoritative
            availability check, so configured-but-invalid feature names do not
            create misleading legend entries.

            """
            source_obj = getattr(self, "obj", None)
            if source_obj is None:
                return flags is not None
            try:
                return bool(getattr(source_obj, feature_attr))
            except (AttributeError, KeyError, TypeError, ValueError):
                return False

        has_is_features = _reference_features_available(is_flags, "valid_is")
        has_orf_features = _reference_features_available(orf_flags, "valid_orf")

        present_categories = set(metrics_df["Category"].unique())
        legend_handles, legend_labels = [], []
        group_titles = ["Scatter Diagnostics"]
        if include_bar_diagnostics:
            group_titles.append("Bar Diagnostics")
        if include_thresholds:
            group_titles.append("Thresholds")

        # Group A: Scatter Diagnostics (Markers)
        legend_handles.append(
            mlines.Line2D([], [], color="none", label="Scatter Diagnostics")
        )
        legend_labels.append("Scatter Diagnostics")

        for label, style in cat_styles.items():
            if complete_categories or label in present_categories:
                h = mlines.Line2D(
                    [],
                    [],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle="none",
                    markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                    markeredgecolor="k",
                    markeredgewidth=pu.DEFAULT_MARKER_EDGEWIDTH,
                    label=label,
                )
                legend_handles.append(h)
                legend_labels.append(label)

        if has_is_features:
            halo_handle = mlines.Line2D(
                [],
                [],
                color="none",
                markeredgecolor=pu.PRIMARY_ACCENT_COLOR,
                marker="o",
                markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                markeredgewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
                linestyle="--",
                label="IS Outlier",
            )
            legend_handles.append(halo_handle)
            legend_labels.append("IS Outlier")

        if has_orf_features:
            orf_halo_handle = mlines.Line2D(
                [],
                [],
                color="none",
                markeredgecolor="tab:orange",
                marker="o",
                markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                markeredgewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
                linestyle="-.",
                label="ORF Outlier",
            )
            legend_handles.append(orf_halo_handle)
            legend_labels.append("ORF Outlier")

        if include_bar_diagnostics:
            # Group B: Bar Diagnostics (Hatch Styles)
            legend_handles.append(
                mlines.Line2D([], [], color="none", label="Bar Diagnostics")
            )
            legend_labels.append("Bar Diagnostics")

            for label, style in cat_styles.items():
                if (
                    complete_categories or label in present_categories
                ) and label != "Normal":
                    h = mpatches.Patch(
                        facecolor=style["color"],
                        edgecolor="black",
                        linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
                        hatch=style["hatch"],
                        label=label,
                    )
                    legend_handles.append(h)
                    legend_labels.append(label)

            if has_is_features:
                legend_handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        color="none",
                        markerfacecolor=pu.PRIMARY_ACCENT_COLOR,
                        markeredgecolor=pu.PRIMARY_ACCENT_COLOR,
                        marker=r"$\ast$",
                        markersize=pu.DEFAULT_LEGEND_MARKER_SIZE,
                        linestyle="none",
                        label="IS Outlier",
                    )
                )
                legend_labels.append("IS Outlier")

            if has_orf_features:
                legend_handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        color="none",
                        markerfacecolor="tab:orange",
                        markeredgecolor="tab:orange",
                        marker=r"$\#$",
                        markersize=10,
                        linestyle="none",
                        label="ORF Outlier",
                    )
                )
                legend_labels.append("ORF Outlier")

        if include_thresholds:
            # Group C: Thresholds (Lines)
            legend_handles.append(
                mlines.Line2D([], [], color="none", label="Thresholds")
            )
            legend_labels.append("Thresholds")

            if sd_limit is not None:
                legend_handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        color="k",
                        ls="--",
                        alpha=0.8,
                        lw=pu.DEFAULT_GUIDE_LINEWIDTH,
                        label=f"HT2 Limit ({sd_limit:.2f})",
                    )
                )
                legend_labels.append(f"HT2 Limit ({sd_limit:.2f})")

            if od_limit is not None:
                legend_handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        color="k",
                        ls="--",
                        alpha=0.8,
                        lw=pu.DEFAULT_GUIDE_LINEWIDTH,
                        label=f"SPE Limit ({od_limit:.2f})",
                    )
                )
                legend_labels.append(f"SPE Limit ({od_limit:.2f})")

        current_ax.legend(legend_handles, legend_labels)

        self._format_multi_legends(
            ax=current_ax,
            group_titles=group_titles,
            loc="upper left",
            start_bbox=(0.0, 1.0),
            row_gap=0.04,
            layout_cols=1,
            sublegend_cols=1,
        )
        # The grouped legends are figure artists after _format_multi_legends;
        # excluding the carrier axes keeps the standalone SVG compact.
        current_ax.axis("off")
        if standalone:
            current_ax.set_visible(False)

        return fig if ax is None else current_ax
