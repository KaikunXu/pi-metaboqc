"""Diagnostic panels for normalization results.

The module renders RLE, density, variance-stabilization, and sample/QC
structure diagnostics from precomputed normalization results.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

from ...constants import DEFAULT_RANDOM_SEED
from ...statistics import metrics as su
from .. import annotation_layout as al
from .. import plot_utils as pu
from ..sample_structure import plot_sample_structure_change_map


class NormalizationDiagnosticsMixin:
    """Render distribution, variance, and structure diagnostics."""

    def _plot_qc_rle_boxplot(
        self,
        ax: plt.Axes | None = None,
        max_points: int = 50000,
        show_legend: bool = True,
        article_compact: bool = False,
    ) -> plt.Figure | plt.Axes:
        """
        Plot QC-sample RLE center offset and spread before/after normalization.
        """
        if ax is None:
            fig, current_ax = plt.subplots(
                figsize=pu.article_brick_size(4.0, 4.0)
                if article_compact
                else (4.0, 4.0)
            )
        else:
            current_ax = ax
            fig = current_ax.figure

        plot_records = []
        sample_metric_values: dict[str, dict[str, pd.Series]] = {}
        for label, obj in self.stages:
            log_d = su._extract_log2_target(obj)
            if log_d is None or log_d.empty:
                continue

            qc_cols = obj._qc.columns.intersection(log_d.columns)
            if len(qc_cols) < 2:
                continue

            global_feature_median = log_d.median(axis=1)
            qc_rle = (
                log_d[qc_cols].astype(float).sub(global_feature_median, axis=0)
            )
            qc_rle = qc_rle.replace([np.inf, -np.inf], np.nan)

            sample_medians = qc_rle.median(axis=0).replace(
                [np.inf, -np.inf], np.nan
            )
            sample_q25 = qc_rle.quantile(0.25, axis=0)
            sample_q75 = qc_rle.quantile(0.75, axis=0)
            sample_iqrs = (sample_q75 - sample_q25).replace(
                [np.inf, -np.inf], np.nan
            )
            sample_iqrs = sample_iqrs.replace([np.inf, -np.inf], np.nan)

            sample_medians = sample_medians.dropna()
            sample_iqrs = sample_iqrs.dropna()
            if sample_medians.empty or sample_iqrs.empty:
                continue

            sample_metric_values[label] = {
                "RLE center offset": sample_medians.abs(),
                "RLE spread": sample_iqrs,
            }
            center_values = sample_metric_values[label]["RLE center offset"]
            spread_values = sample_metric_values[label]["RLE spread"]

            center_offset = su.finite_or_nan(center_values.median())
            rle_spread = su.finite_or_nan(spread_values.median())
            if not all(np.isfinite(v) for v in [center_offset, rle_spread]):
                continue

            plot_records.extend(
                [
                    {
                        "Metric": "RLE center offset",
                        "Stage": label,
                        "Value": center_offset,
                        "Q25": su.finite_or_nan(center_values.quantile(0.25)),
                        "Q75": su.finite_or_nan(center_values.quantile(0.75)),
                    },
                    {
                        "Metric": "RLE spread",
                        "Stage": label,
                        "Value": rle_spread,
                        "Q25": su.finite_or_nan(spread_values.quantile(0.25)),
                        "Q75": su.finite_or_nan(spread_values.quantile(0.75)),
                    },
                ]
            )

        annotation_note: str | None = None
        if plot_records:
            plot_df = pd.DataFrame(plot_records)
            metric_order = ["RLE center offset", "RLE spread"]
            stage_order = ["Before Norm", "After Norm"]
            note_lines = ["Relative change"]
            x_base = np.arange(len(metric_order), dtype=float)
            bar_width = 0.34
            offsets = {
                "Before Norm": -bar_width / 2,
                "After Norm": bar_width / 2,
            }
            bar_label_records = []
            bar_top_lookup: dict[tuple[str, str], float] = {}
            bar_label_top_lookup: dict[str, float] = {}

            for stage in stage_order:
                stage_df = plot_df[plot_df["Stage"].eq(stage)].set_index(
                    "Metric"
                )
                values = []
                lower_errors = []
                upper_errors = []
                for metric in metric_order:
                    value = su.finite_or_nan(
                        stage_df["Value"].get(metric, np.nan)
                    )
                    q25 = su.finite_or_nan(stage_df["Q25"].get(metric, np.nan))
                    q75 = su.finite_or_nan(stage_df["Q75"].get(metric, np.nan))
                    lower_error = (
                        max(value - q25, 0.0) if np.isfinite(q25) else 0.0
                    )
                    upper_error = (
                        max(q75 - value, 0.0) if np.isfinite(q75) else 0.0
                    )
                    values.append(value)
                    lower_errors.append(lower_error)
                    upper_errors.append(upper_error)
                    bar_top_lookup[(metric, stage)] = (
                        value + upper_error if np.isfinite(value) else 0.0
                    )

                bar_container = current_ax.bar(
                    x_base + offsets[stage],
                    values,
                    yerr=np.vstack([lower_errors, upper_errors]),
                    width=bar_width,
                    color=self.pal[stage],
                    edgecolor="k",
                    linewidth=0.5 if article_compact else 1.0,
                    label=stage,
                    zorder=3,
                    error_kw={
                        "ecolor": "0.20",
                        "elinewidth": 0.5 if article_compact else 1.0,
                        "capsize": 2.0 if article_compact else 3.0,
                        "capthick": 0.5 if article_compact else 1.0,
                        "zorder": 4,
                    },
                )
                for patch, value, upper_error in zip(
                    bar_container.patches,
                    values,
                    upper_errors,
                    strict=False,
                ):
                    bar_label_records.append(
                        {
                            "metric": metric,
                            "stage": stage,
                            "x": patch.get_x() + patch.get_width() / 2.0,
                            "value": value,
                            "y": value + upper_error,
                        }
                    )

            if show_legend:
                current_ax.legend(loc="best")
                self._format_single_legend(
                    ax=current_ax,
                    loc="best",
                    bbox_to_anchor=None,
                    group_title="QC RLE alignment stage",
                    max_item_rows=6,
                )

            y_max = float(
                np.nanmax(plot_df[["Value", "Q75"]].to_numpy(dtype=float))
            )
            y_upper = y_max * 1.18 if y_max > 0 else 1.0
            current_ax.set_ylim(0, y_upper)
            label_offset = y_upper * 0.018
            for label_record in bar_label_records:
                value = su.finite_or_nan(label_record["value"])
                if not np.isfinite(value):
                    continue
                metric = str(label_record["metric"])
                inside_bar = value > y_upper * 0.12
                if inside_bar:
                    text_y = max(value - y_upper * 0.035, value * 0.88)
                    va = "top"
                    text_color = "white"
                    label_top = value
                else:
                    text_y = label_record["y"] + label_offset
                    va = "bottom"
                    text_color = "0.15"
                    label_top = text_y + y_upper * 0.025
                current_ax.text(
                    label_record["x"],
                    text_y,
                    f"{value:.3f}",
                    ha="center",
                    va=va,
                    fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                    color=text_color,
                    zorder=4,
                )
                bar_label_top_lookup[metric] = max(
                    bar_label_top_lookup.get(metric, 0.0),
                    label_top,
                )
            current_ax.set_xlim(-0.5, len(metric_order) - 0.5)
            current_ax.set_xticks(x_base)
            current_ax.set_xticklabels(["RLE center\noffset", "RLE\nspread"])

            metric_note_labels = {
                "RLE center offset": "Center",
                "RLE spread": "Spread",
            }
            for metric in metric_order:
                before_subset = plot_df[
                    plot_df["Metric"].eq(metric)
                    & plot_df["Stage"].eq("Before Norm")
                ]["Value"]
                after_subset = plot_df[
                    plot_df["Metric"].eq(metric)
                    & plot_df["Stage"].eq("After Norm")
                ]["Value"]
                before_value = su.finite_or_nan(
                    before_subset.iloc[0] if not before_subset.empty else np.nan
                )
                after_value = su.finite_or_nan(
                    after_subset.iloc[0] if not after_subset.empty else np.nan
                )
                rel_reduction = 100.0 * su.relative_change_lower_better(
                    before_value, after_value
                )
                if np.isfinite(rel_reduction):
                    note_lines.append(
                        f"{metric_note_labels[metric]}: {rel_reduction:.1f}%"
                    )

            bracket_top = y_upper
            bracket_height = y_upper * 0.018
            bracket_gap = y_upper * 0.045
            for metric_idx, metric in enumerate(metric_order):
                p_val = self._paired_wilcoxon_pvalue(
                    sample_metric_values.get("Before Norm", {}).get(metric),
                    sample_metric_values.get("After Norm", {}).get(metric),
                )
                local_bar_top = max(
                    bar_top_lookup.get((metric, "Before Norm"), 0.0),
                    bar_top_lookup.get((metric, "After Norm"), 0.0),
                    bar_label_top_lookup.get(metric, 0.0),
                )
                y_level = local_bar_top + bracket_gap
                bracket_top = max(
                    bracket_top,
                    y_level + bracket_height + y_upper * 0.05,
                )
                self._add_pairwise_significance(
                    ax=current_ax,
                    x_left=x_base[metric_idx] + offsets["Before Norm"],
                    x_right=x_base[metric_idx] + offsets["After Norm"],
                    y=y_level,
                    text=self._pvalue_to_stars(p_val),
                    height=bracket_height,
                )
            current_ax.set_ylim(0, bracket_top)
            if not article_compact:
                clean_note_lines = [line for line in note_lines if line]
                if clean_note_lines:
                    annotation_note = "\n".join(clean_note_lines)
        else:
            current_ax.text(
                0.5,
                0.5,
                "Insufficient QC data",
                transform=current_ax.transAxes,
                ha="center",
                va="center",
                fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                bbox=pu.ai_ready_text_bbox(),
                zorder=10,
            )

        self._apply_standard_format(
            ax=current_ax,
            title="QC RLE Alignment Change",
            xlabel="QC RLE metric",
            ylabel="Median value (IQR)",
            append_stage=False,
        )
        if annotation_note is not None:
            al.add_auto_annotation(
                ax=current_ax,
                text=annotation_note,
                fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                bbox=pu.ai_ready_text_bbox(pad=0.35),
            )
        return fig if ax is None else current_ax

    @staticmethod
    def _pvalue_to_stars(p_value: float) -> str:
        """Convert a p-value into compact significance-star text."""
        if not np.isfinite(p_value):
            return "n/a"
        if p_value < 0.001:
            return "***"
        if p_value < 0.01:
            return "**"
        if p_value < 0.05:
            return "*"
        return "ns"

    @staticmethod
    def _paired_wilcoxon_pvalue(
        before_values: pd.Series | None,
        after_values: pd.Series | None,
    ) -> float:
        """
        Calculate a paired Wilcoxon signed-rank p-value for matched QC samples.
        """
        if before_values is None or after_values is None:
            return float("nan")

        common_index = before_values.index.intersection(after_values.index)
        if len(common_index) < 3:
            return float("nan")

        before_arr = before_values.loc[common_index].to_numpy(dtype=float)
        after_arr = after_values.loc[common_index].to_numpy(dtype=float)
        finite_mask = np.isfinite(before_arr) & np.isfinite(after_arr)
        before_arr = before_arr[finite_mask]
        after_arr = after_arr[finite_mask]
        if before_arr.size < 3 or np.allclose(before_arr, after_arr):
            return 1.0

        try:
            return float(
                stats.wilcoxon(
                    before_arr,
                    after_arr,
                    zero_method="wilcox",
                    alternative="two-sided",
                ).pvalue
            )
        except ValueError:
            return float("nan")

    @staticmethod
    def _add_pairwise_significance(
        ax: plt.Axes,
        x_left: float,
        x_right: float,
        y: float,
        text: str,
        height: float,
    ) -> None:
        """Draw a paired-comparison bracket with significance text."""
        ax.plot(
            [x_left, x_left, x_right, x_right],
            [y, y + height, y + height, y],
            color="0.20",
            linewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
            clip_on=False,
            zorder=5,
        )
        ax.text(
            (x_left + x_right) / 2.0,
            y + height * 1.20,
            text,
            ha="center",
            va="bottom",
            fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
            color="0.20",
            clip_on=False,
            zorder=6,
        )

    def _plot_density_kde(
        self,
        metrics: dict[str, Any] | None = None,
        ax_qc: plt.Axes | None = None,
        ax_sample: plt.Axes | None = None,
    ) -> plt.Figure | tuple[plt.Axes, plt.Axes]:
        """Plot Log2 intensity density overlay for QC and Samples."""
        return_fig = False
        if ax_qc is None or ax_sample is None:
            fig, (ax_qc, ax_sample) = plt.subplots(1, 2, figsize=(8, 4))
            return_fig = True

        for grp, current_ax in [("QC", ax_qc), ("Sample", ax_sample)]:
            pu.mark_preserve_alpha(current_ax)
            metric_note = None
            for label, obj in self.stages:
                log_d = su._extract_log2_target(obj)
                if log_d is None or log_d.empty:
                    continue

                if grp == "QC" and hasattr(obj, "_qc"):
                    cols = obj._qc.columns.intersection(log_d.columns)
                elif hasattr(obj, "_actual_sample"):
                    cols = obj._actual_sample.columns.intersection(
                        log_d.columns
                    )
                else:
                    cols = []

                if len(cols) > 0:
                    vals = log_d[cols].values.flatten()
                    vals = vals[~np.isnan(vals)]
                    if len(vals) > 0:
                        sns.kdeplot(
                            vals,
                            ax=current_ax,
                            label=label,
                            color=self.pal[label],
                            linewidth=2,
                            alpha=0.8,
                        )

            self._apply_standard_format(
                ax=current_ax,
                title=f"Density Overlay ({grp})",
                xlabel="Log2 Intensity",
                ylabel="Density",
                append_stage=False,
            )

            if metrics and "JSD" in metrics and grp in metrics["JSD"]:
                jsd_data = metrics["JSD"][grp].get("Before vs After", np.nan)

                if isinstance(jsd_data, dict):
                    jsd_val = jsd_data.get("JSD", jsd_data.get("jsd", np.nan))
                else:
                    jsd_val = jsd_data

                if not pd.isna(jsd_val):
                    annot_text = (
                        "Jensen-Shannon Divergence\n"
                        f"Before Norm vs After Norm: {float(jsd_val):.3f}"
                    )
                    metric_note = current_ax.text(
                        0.5,
                        0.5,
                        annot_text,
                        transform=current_ax.transAxes,
                        fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                        verticalalignment="bottom",
                        horizontalalignment="right",
                        clip_on=False,
                        bbox=pu.ai_ready_text_bbox(pad=0.4),
                        zorder=10,
                    )

            if current_ax.get_legend_handles_labels()[0]:
                current_ax.legend(loc="best")
                self._format_single_legend(
                    current_ax,
                    group_title="Distribution alignment stage",
                )
            if metric_note is not None:
                al.place_annotation_with_legend_awareness(
                    ax=current_ax,
                    text_artist=metric_note,
                    occupancy_arrays=None,
                    expand_axes=True,
                )

        if return_fig:
            plt.tight_layout()
            return fig
        return ax_qc, ax_sample

    def _plot_qc_variance_stabilization(
        self,
        ax: plt.Axes | None = None,
        show_legend: bool = True,
        article_compact: bool = False,
    ) -> plt.Figure | plt.Axes:
        """Plot QC mean-dispersion dependence before/after normalization."""
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

        stage_records = []
        for label, obj in self.stages:
            log_d = su._extract_log2_target(obj)
            if log_d is None or log_d.empty:
                continue

            qc_cols = obj._qc.columns.intersection(log_d.columns)
            if len(qc_cols) < 3:
                continue

            variance_metrics = self.norm._calc_qc_variance_stabilization_values(
                log_d,
                qc_cols=qc_cols,
            )
            feature_stats = variance_metrics["feature_stats"]
            trend_df = variance_metrics["trend"]
            if feature_stats.empty or trend_df.empty:
                continue

            stage_records.append(
                {
                    "label": label,
                    "trend": trend_df,
                    "metrics": variance_metrics,
                }
            )

        if not stage_records:
            current_ax.text(
                0.5,
                0.5,
                "Insufficient QC data",
                transform=current_ax.transAxes,
                ha="center",
                va="center",
                fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                bbox=pu.ai_ready_text_bbox(),
                zorder=10,
            )
            self._apply_standard_format(
                ax=current_ax,
                title="QC Variance Stabilization",
                xlabel="Mean QC log2 Intensity",
                ylabel="QC dispersion",
                append_stage=False,
            )
            return fig if ax is None else current_ax

        line_style_map = {
            "Before Norm": {"color": pu.NEUTRAL_COLOR, "linestyle": "--"},
            "After Norm": {"color": pu.PRIMARY_ACCENT_COLOR, "linestyle": "-"},
        }
        ribbon_color_map = {
            "Before Norm": (pu.NEUTRAL_COLOR, 0.25),
            "After Norm": (pu.PRIMARY_ACCENT_COLOR, 0.33),
        }
        for record in stage_records:
            label = record["label"]
            trend_df = record["trend"]
            style = line_style_map.get(label, line_style_map["After Norm"])
            ribbon_color, ribbon_alpha = ribbon_color_map.get(
                label, (pu.PRIMARY_ACCENT_COLOR, 0.33)
            )
            x_vals = trend_df["mean_intensity"].to_numpy(dtype=float)
            y_vals = trend_df["dispersion_median"].to_numpy(dtype=float)
            y_low = trend_df["dispersion_q25"].to_numpy(dtype=float)
            y_high = trend_df["dispersion_q75"].to_numpy(dtype=float)
            current_ax.fill_between(
                x_vals,
                y_low,
                y_high,
                color=ribbon_color,
                alpha=ribbon_alpha,
                linewidth=0,
                zorder=1 if label == "Before Norm" else 2,
            )
            current_ax.plot(
                x_vals,
                y_vals,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
                marker="o",
                markersize=2.0 if article_compact else 3.0,
                label=label,
                zorder=4,
            )

        before_metrics = next(
            (
                record["metrics"]
                for record in stage_records
                if record["label"] == "Before Norm"
            ),
            {},
        )
        after_metrics = next(
            (
                record["metrics"]
                for record in stage_records
                if record["label"] == "After Norm"
            ),
            {},
        )
        note_lines = []
        if before_metrics and after_metrics:
            if article_compact:
                note_lines.extend(
                    [
                        "|rho|: {:.3f} / {:.3f}".format(
                            before_metrics.get("mean_variance_abs_rho", np.nan),
                            after_metrics.get("mean_variance_abs_rho", np.nan),
                        ),
                        "|slope|: {:.3f} / {:.3f}".format(
                            before_metrics.get(
                                "mean_variance_abs_slope", np.nan
                            ),
                            after_metrics.get(
                                "mean_variance_abs_slope", np.nan
                            ),
                        ),
                    ]
                )
            else:
                note_lines.extend(
                    [
                        "Before / After",
                        "|rho|: {:.3f} / {:.3f}".format(
                            before_metrics.get("mean_variance_abs_rho", np.nan),
                            after_metrics.get("mean_variance_abs_rho", np.nan),
                        ),
                        "|slope|: {:.3f} / {:.3f}".format(
                            before_metrics.get(
                                "mean_variance_abs_slope", np.nan
                            ),
                            after_metrics.get(
                                "mean_variance_abs_slope", np.nan
                            ),
                        ),
                        "Median dispersion: {:.3f} / {:.3f}".format(
                            before_metrics.get("qc_dispersion_median", np.nan),
                            after_metrics.get("qc_dispersion_median", np.nan),
                        ),
                    ]
                )
        if show_legend:
            current_ax.legend(loc="center right")
            self._format_single_legend(
                ax=current_ax,
                loc="center right",
                bbox_to_anchor=None,
                group_title="QC variance stabilization stage",
                max_item_rows=6,
            )

        self._apply_standard_format(
            ax=current_ax,
            title="QC Variance Stabilization",
            xlabel="Mean QC log2 Intensity",
            ylabel="QC dispersion",
            append_stage=False,
        )
        clean_note_lines = [line for line in note_lines if line]
        if clean_note_lines:
            al.add_auto_annotation(
                ax=current_ax,
                text="\n".join(clean_note_lines),
                fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                bbox=pu.ai_ready_text_bbox(
                    pad=0.25 if article_compact else 0.35
                ),
            )
        return fig if ax is None else current_ax

    def _plot_qc_structure_improvement(
        self,
        ax: plt.Axes | None = None,
        max_features: int = 5000,
        max_pair_points: int = 300,
        show_legend: bool = True,
        article_compact: bool = False,
    ) -> plt.Figure | plt.Axes:
        """Plot before/after multivariate QC-distance distributions."""
        if ax is None:
            fig, current_ax = plt.subplots(
                figsize=pu.article_brick_size(4.0, 4.0)
                if article_compact
                else (4.0, 4.0)
            )
        else:
            current_ax = ax
            fig = current_ax.figure

        summary_by_stage = {}
        for label, obj in self.stages:
            log_d = su._extract_log2_target(obj)
            if log_d is None or log_d.empty:
                continue

            qc_cols = obj._qc.columns.intersection(log_d.columns)
            qc_structure = self.norm._calc_qc_structure_values(
                log_d,
                qc_cols=qc_cols,
                max_features=max_features,
                seed=int(
                    self.norm.attrs.get("global_seed", DEFAULT_RANDOM_SEED)
                ),
            )
            distances = qc_structure["qc_centroid_distance"]
            if distances.empty:
                continue

            summary_by_stage[label] = qc_structure

        before_summary = summary_by_stage.get("Before Norm", {})
        after_summary = summary_by_stage.get("After Norm", {})
        if not before_summary or not after_summary:
            current_ax.text(
                0.5,
                0.5,
                "Insufficient QC data",
                transform=current_ax.transAxes,
                ha="center",
                va="center",
                fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                bbox=pu.ai_ready_text_bbox(),
                zorder=10,
            )
            self._apply_standard_format(
                ax=current_ax,
                title="QC Structure Distance Change",
                xlabel="QC distance metric",
                ylabel="Robust QC distance",
                append_stage=False,
            )
            return fig if ax is None else current_ax

        def _clean_distance_series(values: object) -> pd.Series:
            """Convert a stored QC-distance vector to finite numeric values."""
            if not isinstance(values, pd.Series):
                values = pd.Series(values, dtype=float)
            return (
                pd.to_numeric(values, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

        distance_specs = [
            {
                "label": "Distance to\nQC centroid",
                "short_label": "Centroid",
                "before": _clean_distance_series(
                    before_summary.get(
                        "qc_centroid_distance", pd.Series(dtype=float)
                    )
                ),
                "after": _clean_distance_series(
                    after_summary.get(
                        "qc_centroid_distance", pd.Series(dtype=float)
                    )
                ),
                "point_limit": None,
            },
            {
                "label": "QC-QC\npairwise distance",
                "short_label": "Pairwise",
                "before": _clean_distance_series(
                    before_summary.get(
                        "qc_pairwise_distance", pd.Series(dtype=float)
                    )
                ),
                "after": _clean_distance_series(
                    after_summary.get(
                        "qc_pairwise_distance", pd.Series(dtype=float)
                    )
                ),
                "point_limit": max_pair_points,
            },
        ]
        distance_specs = [
            spec
            for spec in distance_specs
            if not spec["before"].empty and not spec["after"].empty
        ]
        if not distance_specs:
            current_ax.text(
                0.5,
                0.5,
                "Insufficient QC-distance data",
                transform=current_ax.transAxes,
                ha="center",
                va="center",
                fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
            )
            self._apply_standard_format(
                ax=current_ax,
                title="QC Structure Distance Change",
                xlabel="QC distance metric",
                ylabel="Robust QC distance",
                append_stage=False,
            )
            return fig if ax is None else current_ax

        rng = np.random.default_rng(
            int(self.norm.attrs.get("global_seed", DEFAULT_RANDOM_SEED))
        )
        stage_order = ["Before Norm", "After Norm"]
        stage_offsets = {"Before Norm": -0.17, "After Norm": 0.17}
        box_width = 0.22 if article_compact else 0.28
        box_linewidth = 0.5 if article_compact else 1.25
        median_linewidth = 0.65 if article_compact else 1.25
        whisker_linewidth = 0.5 if article_compact else 1.0
        all_values: list[np.ndarray] = []
        note_lines = (
            [] if article_compact else ["Median distance (Before / After)"]
        )

        for metric_idx, spec in enumerate(distance_specs):
            stage_medians: dict[str, float] = {}
            for stage in stage_order:
                values = (
                    spec["before"] if stage == "Before Norm" else spec["after"]
                )
                values = values.astype(float).replace([np.inf, -np.inf], np.nan)
                values = values.dropna()
                if values.empty:
                    continue

                value_array = values.to_numpy(dtype=float)
                all_values.append(value_array[np.isfinite(value_array)])
                x_pos = metric_idx + stage_offsets[stage]
                current_ax.boxplot(
                    value_array,
                    positions=[x_pos],
                    widths=box_width,
                    patch_artist=True,
                    showfliers=False,
                    boxprops={
                        "facecolor": "white",
                        "edgecolor": self.pal[stage],
                        "linewidth": box_linewidth,
                    },
                    medianprops={
                        "color": "0.15",
                        "linewidth": median_linewidth,
                    },
                    whiskerprops={
                        "color": self.pal[stage],
                        "linewidth": whisker_linewidth,
                    },
                    capprops={
                        "color": self.pal[stage],
                        "linewidth": whisker_linewidth,
                    },
                )

                point_values = value_array
                point_limit = spec["point_limit"]
                if point_limit is not None and point_values.size > point_limit:
                    keep = rng.choice(
                        point_values.size, size=int(point_limit), replace=False
                    )
                    point_values = point_values[keep]
                jitter = rng.normal(
                    loc=0.0, scale=0.025, size=point_values.size
                )
                current_ax.scatter(
                    np.full(point_values.size, x_pos, dtype=float) + jitter,
                    point_values,
                    color=self.pal[stage],
                    edgecolor="k",
                    linewidth=0.15 if article_compact else 0.20,
                    s=10 if article_compact else 16,
                    zorder=3,
                )
                stage_medians[stage] = su.finite_or_nan(values.median())

            if all(
                np.isfinite(stage_medians.get(stage, np.nan))
                for stage in stage_order
            ):
                current_ax.plot(
                    [
                        metric_idx + stage_offsets["Before Norm"],
                        metric_idx + stage_offsets["After Norm"],
                    ],
                    [
                        stage_medians["Before Norm"],
                        stage_medians["After Norm"],
                    ],
                    color="0.25",
                    linewidth=0.6 if article_compact else 1.0,
                    zorder=4,
                )

            before_median = stage_medians.get("Before Norm", np.nan)
            after_median = stage_medians.get("After Norm", np.nan)
            if np.isfinite(before_median) and np.isfinite(after_median):
                improvement = su.relative_change_lower_better(
                    before_median,
                    after_median,
                )
                improvement_text = ""
                if not article_compact and np.isfinite(improvement):
                    improvement_text = (
                        f"; improvement {100.0 * improvement:+.1f}%"
                    )
                note_lines.append(
                    (
                        f"{spec['short_label']}: "
                        "{:.3f} / {:.3f}{}".format(
                            before_median,
                            after_median,
                            improvement_text,
                        )
                    )
                )

        if all_values:
            finite_values = np.concatenate(
                [arr for arr in all_values if arr.size > 0]
            )
        else:
            finite_values = np.array([], dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        positive_values = finite_values[finite_values > 0]
        use_log_scale = (
            positive_values.size > 0
            and np.nanmax(positive_values) / np.nanmin(positive_values) > 20
        )
        note_line_count = max(len(note_lines), 1)
        bottom_margin_fraction = min(0.36, 0.10 + 0.035 * note_line_count)
        top_margin_fraction = 0.18
        y_label = "Robust QC distance"
        if use_log_scale:
            current_ax.set_yscale("log")
            y_label = "Robust QC distance (log scale)"
            log_min = np.log10(np.nanmin(positive_values))
            log_max = np.log10(np.nanmax(positive_values))
            log_span = max(log_max - log_min, 0.5)
            y_lower = 10 ** (log_min - log_span * bottom_margin_fraction)
            y_upper = 10 ** (log_max + log_span * top_margin_fraction)
            current_ax.set_ylim(y_lower, y_upper)
        elif finite_values.size > 0:
            data_upper = np.nanpercentile(finite_values, 98)
            data_upper = max(data_upper, np.nanmax(finite_values) * 0.85)
            data_upper = data_upper if data_upper > 0 else 1.0
            y_lower = -data_upper * bottom_margin_fraction
            y_upper = data_upper * (1.0 + top_margin_fraction)
            current_ax.set_ylim(y_lower, y_upper)

        from matplotlib.lines import Line2D

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=self.pal[stage],
                marker="o",
                linestyle="",
                markeredgecolor="k",
                markeredgewidth=0.25,
                markersize=6,
                label=stage,
            )
            for stage in stage_order
        ]
        if show_legend:
            current_ax.legend(handles=legend_handles, loc="best")
            self._format_single_legend(
                ax=current_ax,
                loc="best",
                bbox_to_anchor=None,
                group_title="QC structure distance stage",
                max_item_rows=6,
            )

        current_ax.set_xlim(-0.55, len(distance_specs) - 0.45)
        current_ax.set_xticks(range(len(distance_specs)))
        current_ax.set_xticklabels([spec["label"] for spec in distance_specs])
        self._apply_standard_format(
            ax=current_ax,
            title="QC Structure Distance Change",
            xlabel="QC distance metric",
            ylabel=y_label,
            append_stage=False,
        )
        clean_note_lines = [line for line in note_lines if line]
        if clean_note_lines:
            al.add_auto_annotation(
                ax=current_ax,
                text="\n".join(clean_note_lines),
                fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                bbox=pu.ai_ready_text_bbox(
                    pad=0.25 if article_compact else 0.35
                ),
            )
        return fig if ax is None else current_ax

    def _plot_sample_structure_preservation(
        self,
        ax_geom: plt.Axes | None = None,
        max_features: int = 5000,
        compact_style: bool = False,
    ) -> plt.Axes | plt.Figure:
        """
        Plot score-aligned sample-structure preservation after normalization.
        """
        if ax_geom is None:
            created_fig, ax_geom = plt.subplots(figsize=(4, 4))
        else:
            created_fig = None

        structure_metrics: dict[str, float] | None = None
        auto_summary = self.norm.attrs.get("selection", {}).get(
            "candidate_summary"
        )
        if auto_summary:
            selected_method = str(
                self.norm.attrs.get("selection", {}).get(
                    "selected_method", ""
                )
            )
            selected_row = next(
                (
                    row
                    for row in auto_summary
                    if row.get("selected") is True
                    or str(row.get("method", "")) == selected_method
                ),
                None,
            )
            if selected_row is not None:
                structure_metrics = selected_row

        plot_sample_structure_change_map(
            ax=ax_geom,
            raw_obj=self.raw,
            transformed_obj=self.norm,
            structure_metrics=structure_metrics,
            seed=int(
                self.norm.attrs.get("global_seed", DEFAULT_RANDOM_SEED)
            ),
            max_features=max_features,
            scale_log_ratio_tol=self.norm._SAMPLE_SCALE_LOG_RATIO_TOL,
            scale_rel_delta_tol=self.norm._SAMPLE_SCALE_REL_DELTA_TOL,
            title="Sample Structure Change Map",
            compact_style=compact_style,
        )
        if created_fig is not None:
            plt.tight_layout()
            return created_fig
        return ax_geom

    def _plot_ecdf_overlay(
        self, metrics: dict[str, Any] | None = None, ax: plt.Axes | None = None
    ) -> plt.Figure | plt.Axes:
        """Plot Empirical Cumulative Distribution Function (eCDF) overlay.

        Visualizes intensity alignment with a legend in the upper left and
        QA metrics text box in the lower right.
        """
        import matplotlib.lines as mlines

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))
        else:
            fig = ax.figure
        pu.mark_preserve_alpha(ax)

        handles = []
        for label, obj in self.stages:
            log_d = su._extract_log2_target(obj)
            if log_d is None or log_d.empty:
                continue

            for col in log_d.columns:
                vals = log_d[col].dropna().values
                if len(vals) == 0:
                    continue

                vals_sorted = np.sort(vals)
                p = np.linspace(0, 1, len(vals_sorted))
                z = 2 if label == "After Norm" else 1
                ax.plot(
                    vals_sorted,
                    p,
                    color=self.pal[label],
                    alpha=0.2,
                    linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
                    zorder=z,
                )

            ax.plot([], [], color=self.pal[label], label=label, linewidth=2)
            # Create handles for the legend
            handles.append(
                mlines.Line2D(
                    [], [], color=self.pal[label], label=label, linewidth=2
                )
            )

        self._apply_standard_format(
            ax=ax,
            title="eCDF Distribution Alignment",
            xlabel="Log2 Intensity",
            ylabel="Cumulative Probability",
            append_stage=False,
        )

        # Inject Metrics Text Box (Lower Right)
        if metrics and "eCDF" in metrics:
            lines = ["Dist. Alignment (W / KS)"]
            for label in ["Before Norm", "After Norm"]:
                m_dict = metrics["eCDF"].get(label, {})
                if m_dict:
                    lines.append(
                        f"{label}: {m_dict.get('Wasserstein', 0):.2f} / "
                        f"{m_dict.get('KS', 0):.3f}"
                    )

            ax.text(
                0.96,
                0.02,
                "\n".join(lines),
                transform=ax.transAxes,
                fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                verticalalignment="bottom",
                horizontalalignment="right",
                clip_on=False,
                bbox=pu.ai_ready_text_bbox(pad=0.4),
                zorder=10,
            )

        # Force Legend in Upper Left
        if handles:
            ax.legend(handles=handles)
            self._format_single_legend(
                ax=ax,
                group_title="Distribution alignment stage",
                loc="upper left",
                bbox_to_anchor=None,
            )

        return fig if ax is None else ax
