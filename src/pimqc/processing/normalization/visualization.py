"""Diagnostic visualizations for normalization evaluation and selection.

MetaboVisualizerNormalizer renders RLE alignment, QC variance stabilization,
QC structure, candidate score components, and sample-preservation diagnostics.
It supports compact manuscript panels and full dashboards while reading, rather
than modifying, the normalized stage object and its stored metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from loguru import logger
from typing import Any

from ...core import model
from ...visualization import base as visualizer_classes
from ...visualization import plot_utils as pu
from ...statistics import metrics as su


from .analysis import MetaboIntNormalizer


class MetaboVisualizerNormalizer(visualizer_classes.BaseMetaboVisualizer):
    """2-Stage Visualization Suite (Before vs After Normalization).

    Generates high-contrast diagnostic plots evaluating the efficacy of the
    global variance stabilization and normalization preprocessing.
    """

    def __init__(
        self, raw_obj: model.MetaboInt, norm_obj: model.MetaboInt
    ) -> None:
        """Initialize with pre- and post-normalization datasets."""
        super().__init__(metabo_obj=norm_obj)
        self.raw = raw_obj
        self.norm = norm_obj
        self.stages = [("Before Norm", self.raw), ("After Norm", self.norm)]
        self.pal = {
            "Before Norm": pu.NEUTRAL_COLOR,
            "After Norm": pu.PRIMARY_ACCENT_COLOR,
        }

    @staticmethod
    def _normalization_score_component_style() -> tuple[
        list[str],
        dict[str, str],
        dict[str, str],
    ]:
        """Return the score-component ordering, labels, and colors."""
        score_cols = [
            "rle_alignment_change_score",
            "variance_stabilization_score",
            "qc_structure_change_score",
            "sample_structure_score",
        ]
        label_map = {
            "rle_alignment_change_score": "QC RLE alignment change",
            "variance_stabilization_score": "QC variance stabilization",
            "qc_structure_change_score": "QC structure distance change",
            "sample_structure_score": "Sample structure preservation",
        }
        color_map = {
            "rle_alignment_change_score": pu.get_equivalent_hex(
                pu.PRIMARY_ACCENT_COLOR, alpha=1.0
            ),
            "variance_stabilization_score": pu.get_equivalent_hex(
                pu.PRIMARY_ACCENT_COLOR, alpha=0.67
            ),
            "qc_structure_change_score": pu.get_equivalent_hex(
                pu.PRIMARY_ACCENT_COLOR, alpha=0.33
            ),
            "sample_structure_score": pu.get_equivalent_hex(
                "tab:gray", alpha=0.5
            ),
        }
        return score_cols, label_map, color_map

    def plot_normalization_score_summary(
        self,
        auto_summary: list[dict[str, Any]] | pd.DataFrame | None = None,
        figsize: tuple[float, float] = (4.0, 4.0),
        show_legend: bool = True,
    ) -> object | None:
        """Plot Auto normalization weighted score components as stacked bars."""
        if auto_summary is None:
            auto_summary = self.norm.attrs.get("normalization_auto_summary")
        if not auto_summary:
            return None

        try:
            import patchworklib as pw
            import matplotlib.patches as mpatches
        except ImportError as e:
            logger.warning(f"Skipping Auto normalization stacked bar plot: {e}")
            return None

        summary_df = pd.DataFrame(auto_summary).copy()
        if summary_df.empty:
            return None

        score_cols, label_map, color_map = (
            self._normalization_score_component_style()
        )
        contribution_weights = MetaboIntNormalizer._AUTO_SCORE_COMPONENT_WEIGHTS
        bar_edgecolor = "k"
        bar_linewidth = 0.5
        for col in ["overall_score", *score_cols]:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")

        plot_df = summary_df.copy()
        plot_df["status"] = plot_df["status"].fillna("failed")
        scoreable_mask = (
            plot_df["status"].eq("ok") & plot_df["overall_score"].notna()
        )
        if not scoreable_mask.any():
            return None

        def _method_label(method: object) -> str:
            """Create compact display labels for Auto normalization methods."""
            method_str = str(method)
            display_map = {
                "ROBUST_LOG_ONLY": "Robust Log2",
                "TIC": "TIC",
                "MEDIAN": "Median",
                "PQN": "PQN",
                "MDFC": "MDFC",
                "QUANTILE": "Quantile",
                "VSN": "VSN",
            }
            return display_map.get(method_str.upper(), method_str)

        plot_df["_sort_score"] = plot_df["overall_score"].fillna(-1.0)
        plot_df = plot_df.sort_values(
            by=["_sort_score", "method"], ascending=[False, True]
        ).reset_index(drop=True)

        contribution_cols: list[str] = []
        for col in score_cols:
            contribution_col = f"contribution_{col}"
            contribution_cols.append(contribution_col)
            plot_df[contribution_col] = 0.0

        for idx, row in plot_df.iterrows():
            if row["status"] != "ok":
                continue

            available_weight = 0.0
            for col in score_cols:
                value = row[col]
                if np.isfinite(value):
                    available_weight += contribution_weights[col]

            if available_weight <= 0:
                continue

            raw_sum = 0.0
            for col in score_cols:
                value = row[col]
                if np.isfinite(value):
                    contribution = (
                        np.clip(float(value), 0.0, 1.0)
                        * contribution_weights[col]
                        / available_weight
                    )
                    plot_df.loc[idx, f"contribution_{col}"] = contribution
                    raw_sum += contribution

            overall_score = row["overall_score"]
            if np.isfinite(overall_score) and raw_sum > 0:
                scale_factor = float(np.clip(overall_score, 0.0, 1.0)) / raw_sum
                for contribution_col in contribution_cols:
                    plot_df.loc[idx, contribution_col] *= scale_factor

        ax = pw.Brick(figsize=figsize, label="auto_norm_stacked_bar")
        y_pos = np.arange(len(plot_df))
        left = np.zeros(len(plot_df), dtype=float)

        for col in score_cols:
            contribution_col = f"contribution_{col}"
            values = plot_df[contribution_col].to_numpy(dtype=float)
            left_start = left.copy()
            ax.barh(
                y_pos,
                values,
                left=left,
                height=0.62,
                color=color_map[col],
                edgecolor=bar_edgecolor,
                linewidth=bar_linewidth,
                label=label_map[col],
                zorder=3,
            )
            for y_idx, row in enumerate(plot_df.itertuples()):
                score_value = su.finite_or_nan(getattr(row, col))
                if values[y_idx] < 0.09 or not np.isfinite(score_value):
                    continue
                face_color = color_map[col]
                ax.text(
                    left_start[y_idx] + values[y_idx] / 2.0,
                    y_idx,
                    f"{score_value:.2f}",
                    va="center",
                    ha="center",
                    fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                    color=pu.get_contrast_color(face_color),
                    clip_on=True,
                    zorder=4,
                )
            left += values

        x_data_max = float(np.nanmax(left)) if left.size else 1.0
        if not np.isfinite(x_data_max) or x_data_max <= 0:
            x_data_max = 1.0
        x_upper = min(1.05, max(0.20, x_data_max * 1.08, x_data_max + 0.04))

        for i, row in plot_df.iterrows():
            total_score = row["overall_score"]
            if row["status"] == "ok" and np.isfinite(total_score):
                label = f"{float(total_score):.3f}"
                ax.text(
                    min(float(left[i]) + x_upper * 0.015, x_upper * 0.965),
                    y_pos[i],
                    label,
                    va="center",
                    ha="left",
                    fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                    color="0.15",
                )
            else:
                ax.text(
                    0.015,
                    y_pos[i],
                    "failed",
                    va="center",
                    ha="left",
                    fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                    color="0.45",
                    style="italic",
                )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [
                f"* {_method_label(row.method)}"
                if bool(getattr(row, "selected", False))
                else _method_label(row.method)
                for row in plot_df.itertuples()
            ]
        )

        if show_legend:
            handles = [
                mpatches.Patch(
                    facecolor=color_map[col],
                    edgecolor=bar_edgecolor,
                    linewidth=bar_linewidth,
                    label=label_map[col],
                )
                for col in score_cols
            ]
            ax.legend(handles=handles, loc="lower right", bbox_to_anchor=None)
            self._format_single_legend(
                ax,
                loc="lower right",
                bbox_to_anchor=None,
                group_title="Normalization score components",
                max_item_rows=6,
            )

        ax.set_xlim(0, x_upper)
        ax.set_ylim(-0.5, len(plot_df) - 0.5)
        ax.invert_yaxis()
        self._apply_standard_format(
            ax=ax,
            title="Auto Normalization Method Selection",
            xlabel="Weighted contribution to overall score",
            append_stage=False,
        )
        return ax

    def plot_normalization_preservation_scorecard(
        self,
        auto_summary: list[dict[str, Any]] | pd.DataFrame | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Axes | None:
        """Plot candidate-level sample-structure preservation scores."""
        if auto_summary is None:
            auto_summary = self.norm.attrs.get("normalization_auto_summary")
        if not auto_summary:
            return None

        try:
            import patchworklib as pw
        except ImportError as e:
            logger.warning(f"Skipping Auto normalization scorecard plot: {e}")
            return None

        current_ax = (
            pw.Brick(
                figsize=pu.dashboard_brick_size(4.8, 4.0, 8.0),
                label="normalization_preservation_scorecard",
            )
            if ax is None
            else ax
        )
        score_cols = [
            "sample_structure_trustworthiness",
            "sample_structure_rank_preservation",
            "sample_structure_scale_preservation",
        ]
        metric_labels = [
            "Trustworthiness",
            "Distance-rank\npreservation",
            "Distance-scale\npreservation",
        ]

        def _method_label(method: object) -> str:
            """Create compact display labels for Auto normalization methods."""
            method_str = str(method)
            display_map = {
                "ROBUST_LOG_ONLY": "Robust Log2",
                "TIC": "TIC",
                "MEDIAN": "Median",
                "PQN": "PQN",
                "MDFC": "MDFC",
                "QUANTILE": "Quantile",
                "VSN": "VSN",
            }
            return display_map.get(method_str.upper(), method_str)

        score_df = pd.DataFrame(auto_summary).copy()
        if score_df.empty:
            current_ax.axis("off")
            return current_ax

        for col in ["overall_score", *score_cols]:
            score_df[col] = pd.to_numeric(score_df[col], errors="coerce")
        score_df = score_df.loc[score_df["status"].eq("ok")].copy()
        score_df = score_df.dropna(subset=score_cols, how="all")
        score_df = score_df.sort_values(
            by=["overall_score", "method"], ascending=[False, True]
        ).reset_index(drop=True)
        if score_df.empty:
            current_ax.axis("off")
            return current_ax

        matrix = score_df[score_cols].to_numpy(dtype=float)
        cmap = pu.score_heatmap_cmap()
        annot_size = pu.heatmap_annotation_fontsize(
            current_ax,
            n_rows=matrix.shape[0],
            n_cols=matrix.shape[1],
            default_size=pu.DEFAULT_SCORE_HEATMAP_ANNOTATION_FONTSIZE,
            max_size=pu.DEFAULT_SCORE_HEATMAP_ANNOTATION_FONTSIZE,
            min_size=4.0,
        )
        current_ax.imshow(
            np.ma.masked_invalid(matrix),
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
        )
        current_ax.set_xticks(np.arange(len(score_cols)))
        current_ax.set_xticklabels(metric_labels)
        current_ax.set_yticks(np.arange(len(score_df)))
        current_ax.set_yticklabels(
            [
                f"* {_method_label(row.method)}"
                if bool(getattr(row, "selected", False))
                else _method_label(row.method)
                for row in score_df.itertuples()
            ]
        )
        current_ax.set_xticks(np.arange(-0.5, len(score_cols), 1), minor=True)
        current_ax.set_yticks(np.arange(-0.5, len(score_df), 1), minor=True)
        current_ax.grid(
            which="minor",
            color="k",
            linestyle="-",
            linewidth=pu.DEFAULT_HEATMAP_CELL_LINEWIDTH,
        )
        current_ax.tick_params(which="minor", bottom=False, left=False)

        for y_idx in range(matrix.shape[0]):
            for x_idx in range(matrix.shape[1]):
                value = matrix[y_idx, x_idx]
                if not np.isfinite(value):
                    label = "NA"
                    color = "0.35"
                else:
                    label = f"{value:.2f}"
                    color = pu.get_contrast_color(cmap(value))
                current_ax.text(
                    x_idx,
                    y_idx,
                    label,
                    ha="center",
                    va="center",
                    fontsize=annot_size,
                    color=color,
                )

        self._apply_standard_format(
            current_ax,
            title="Candidate Preservation Scorecard",
            xlabel="",
            ylabel="",
            append_stage=False,
        )
        pu.rotate_xticks_if_overlapping(current_ax)
        current_ax.tick_params(axis="both", length=0)
        return current_ax

    @staticmethod
    def _add_metric_note(
        ax: plt.Axes,
        lines: list[str],
        loc: str = "lower right",
        fontsize: float = pu.DEFAULT_ANNOTATION_FONTSIZE,
        pad: float = 0.35,
    ) -> None:
        """Place a compact metric note in a diagnostic panel."""
        clean_lines = [line for line in lines if line]
        if not clean_lines:
            return
        if loc == "upper right":
            x_pos, y_pos = 0.96, 0.98
            verticalalignment = "top"
            horizontalalignment = "right"
        elif loc == "upper left":
            x_pos, y_pos = 0.04, 0.98
            verticalalignment = "top"
            horizontalalignment = "left"
        elif loc == "lower left":
            x_pos, y_pos = 0.04, 0.02
            verticalalignment = "bottom"
            horizontalalignment = "left"
        else:
            x_pos, y_pos = 0.96, 0.02
            verticalalignment = "bottom"
            horizontalalignment = "right"
        ax.text(
            x_pos,
            y_pos,
            "\n".join(clean_lines),
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment=verticalalignment,
            horizontalalignment=horizontalalignment,
            clip_on=False,
            bbox=pu.ai_ready_text_bbox(pad=pad),
            zorder=10,
        )

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
                self._add_metric_note(current_ax, note_lines, loc="lower right")
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
                    current_ax.text(
                        0.96,
                        0.02,
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

            variance_metrics = (
                MetaboIntNormalizer._calc_qc_variance_stabilization_values(
                    log_d,
                    qc_cols=qc_cols,
                )
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
        self._add_metric_note(
            current_ax,
            note_lines,
            loc="upper right",
            fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
            pad=0.25 if article_compact else 0.35,
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
            qc_structure = MetaboIntNormalizer._calc_qc_structure_values(
                log_d,
                qc_cols=qc_cols,
                max_features=max_features,
                seed=int(self.norm.attrs.get("global_seed", 123)),
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
            int(self.norm.attrs.get("global_seed", 123))
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
        self._add_metric_note(
            current_ax,
            note_lines,
            loc="lower right",
            fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
            pad=0.25 if article_compact else 0.35,
        )
        self._apply_standard_format(
            ax=current_ax,
            title="QC Structure Distance Change",
            xlabel="QC distance metric",
            ylabel=y_label,
            append_stage=False,
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
        auto_summary = self.norm.attrs.get("normalization_auto_summary")
        if auto_summary:
            selected_method = str(
                self.norm.attrs.get("auto_selected_method", "")
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

        pu.plot_sample_structure_change_map(
            ax=ax_geom,
            raw_obj=self.raw,
            transformed_obj=self.norm,
            structure_metrics=structure_metrics,
            seed=int(self.norm.attrs.get("global_seed", 123)),
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

    def plot_normalization_dashboard_legend(
        self,
        ax: plt.Axes,
        fontsize: float = pu.DEFAULT_LEGEND_FONTSIZE,
        title_fontsize: float = pu.DEFAULT_LEGEND_TITLE_FONTSIZE,
        article_compact: bool = False,
        layout_cols: int = 1,
    ) -> plt.Axes:
        """Draw grouped score-component and stage legends for the dashboard."""
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches

        legend_linewidth = pu.DEFAULT_AXIS_LINEWIDTH
        line_width = pu.DEFAULT_GUIDE_LINEWIDTH
        marker_size = pu.DEFAULT_LEGEND_MARKER_SIZE
        score_cols, label_map, color_map = (
            self._normalization_score_component_style()
        )
        score_handles = [
            mpatches.Patch(
                facecolor=color_map[col],
                edgecolor="k",
                linewidth=legend_linewidth,
                label=label_map[col],
            )
            for col in score_cols
        ]
        rle_handles = [
            mpatches.Patch(
                facecolor=self.pal["Before Norm"],
                edgecolor="k",
                linewidth=legend_linewidth,
                label="Before Norm",
            ),
            mpatches.Patch(
                facecolor=self.pal["After Norm"],
                edgecolor="k",
                linewidth=legend_linewidth,
                label="After Norm",
            ),
        ]
        variance_handles = [
            mlines.Line2D(
                [],
                [],
                color=self.pal["Before Norm"],
                linestyle="--",
                marker="o",
                linewidth=line_width,
                markersize=marker_size,
                label="Before Norm",
            ),
            mlines.Line2D(
                [],
                [],
                color=self.pal["After Norm"],
                linestyle="-",
                marker="o",
                linewidth=line_width,
                markersize=marker_size,
                label="After Norm",
            ),
        ]
        distance_handles = [
            mlines.Line2D(
                [0],
                [0],
                color=self.pal["Before Norm"],
                marker="o",
                linestyle="",
                markeredgecolor="k",
                markeredgewidth=0.5 if article_compact else 0.25,
                markersize=marker_size,
                label="Before Norm",
            ),
            mlines.Line2D(
                [0],
                [0],
                color=self.pal["After Norm"],
                marker="o",
                linestyle="",
                markeredgecolor="k",
                markeredgewidth=0.5 if article_compact else 0.25,
                markersize=marker_size,
                label="After Norm",
            ),
        ]

        self._plot_grouped_standalone_legends(
            ax=ax,
            legend_groups=[
                ("Normalization score components", score_handles),
                ("QC RLE alignment stage", rle_handles),
                ("QC variance stabilization stage", variance_handles),
                ("QC structure distance stage", distance_handles),
            ],
            loc="upper left",
            start_bbox=(0.0, 1.0),
            row_gap=0.035,
            layout_cols=layout_cols,
            column_gap=0.12,
            max_item_rows=6,
            borderaxespad=0.0,
            handlelength=1.0 if article_compact else 1.8,
            handletextpad=0.3 if article_compact else 0.8,
            labelspacing=0.25 if article_compact else 0.5,
            borderpad=0.3 if article_compact else 0.4,
            fontsize=fontsize,
            title_fontsize=title_fontsize,
        )
        if article_compact:
            self._apply_article_legend_style(
                ax=ax,
                fontsize=fontsize,
                title_fontsize=title_fontsize,
            )
        return ax

    def plot_normalization_article_legend(self, ax: plt.Axes) -> plt.Axes:
        """
        Draw right-side grouped legends for the normalization article panel.
        """
        return self.plot_normalization_dashboard_legend(
            ax=ax,
            fontsize=pu.ARTICLE_LEGEND_FONTSIZE,
            title_fontsize=pu.ARTICLE_LEGEND_TITLE_FONTSIZE,
            article_compact=True,
            layout_cols=1,
        )

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

        # 1. Inject Metrics Text Box (Lower Right)
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

        # 2. Force Legend in Upper Left
        if handles:
            ax.legend(handles=handles)
            self._format_single_legend(
                ax=ax,
                group_title="Distribution alignment stage",
                loc="upper left",
                bbox_to_anchor=None,
            )

        return fig if ax is None else ax

    def plot_normalization_dashboard(self) -> object | None:
        """
        Combine score-aligned normalization diagnostics into a PW dashboard.
        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping summary grid.")
            return None

        pw.clear()

        auto_summary = self.norm.attrs.get("normalization_auto_summary")
        is_auto = bool(auto_summary)

        if is_auto:
            layout_width = 13.7
            ax_auto = self.plot_normalization_score_summary(
                auto_summary=auto_summary,
                figsize=pu.dashboard_brick_size(5.5, 4.0, layout_width),
                show_legend=False,
            )
            if ax_auto is None:
                ax_auto = pw.Brick(
                    figsize=pu.dashboard_brick_size(4.5, 4.0, layout_width),
                    label="Auto_Score_Spacer",
                )
                ax_auto.axis("off")

            ax_scorecard = pw.Brick(
                figsize=pu.dashboard_brick_size(4.8, 4.0, layout_width),
                label="Norm_Preservation_Scorecard",
            )
            self.plot_normalization_preservation_scorecard(
                auto_summary=auto_summary,
                ax=ax_scorecard,
            )
            ax_legend = pw.Brick(
                figsize=pu.dashboard_brick_size(3.4, 4.0, layout_width),
                label="normalization_dashboard_legend",
            )
            self.plot_normalization_dashboard_legend(ax=ax_legend)
            row1 = ax_auto | ax_scorecard | ax_legend

            ax_qc_variance = pw.Brick(
                figsize=pu.dashboard_brick_size(3.0, 4.0, layout_width),
                label="QC_Variance",
            )
            self._plot_qc_variance_stabilization(
                ax=ax_qc_variance,
                show_legend=False,
                article_compact=True,
            )
            ax_qc_structure = pw.Brick(
                figsize=pu.dashboard_brick_size(3.0, 4.0, layout_width),
                label="QC_Structure",
            )
            self._plot_qc_structure_improvement(
                ax=ax_qc_structure,
                show_legend=False,
                article_compact=True,
            )
            ax_sample_structure = pw.Brick(
                figsize=pu.dashboard_brick_size(3.0, 4.0, layout_width),
                label="Sample_Structure",
            )
            self._plot_sample_structure_preservation(
                ax_geom=ax_sample_structure, compact_style=True
            )
            ax_qc_alignment = pw.Brick(
                figsize=pu.dashboard_brick_size(3.0, 4.0, layout_width),
                label="QC_Alignment",
            )
            self._plot_qc_rle_boxplot(
                ax=ax_qc_alignment,
                show_legend=False,
                article_compact=True,
            )
            row2 = (
                ax_qc_alignment
                | ax_qc_variance
                | ax_qc_structure
                | ax_sample_structure
            )

            return row1 / row2

        layout_width = 8.0
        ax_qc_alignment = pw.Brick(
            figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
            label="QC_Alignment",
        )
        self._plot_qc_rle_boxplot(ax=ax_qc_alignment, article_compact=True)
        ax_qc_variance = pw.Brick(
            figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
            label="QC_Variance",
        )
        self._plot_qc_variance_stabilization(
            ax=ax_qc_variance, article_compact=True
        )
        row1 = ax_qc_alignment | ax_qc_variance

        ax_qc_structure = pw.Brick(
            figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
            label="QC_Structure",
        )
        self._plot_qc_structure_improvement(
            ax=ax_qc_structure, article_compact=True
        )
        ax_sample_structure = pw.Brick(
            figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
            label="Sample_Structure",
        )
        self._plot_sample_structure_preservation(
            ax_geom=ax_sample_structure, compact_style=True
        )
        row2 = ax_qc_structure | ax_sample_structure

        return row1 / row2

    def plot_normalization_article_dashboard(self) -> object | None:
        """Create a compact AUTO normalization panel for manuscript figures."""
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning(
                "patchworklib not found. Skipping normalization article panel."
            )
            return None

        auto_summary = self.norm.attrs.get("normalization_auto_summary")
        if not auto_summary:
            return None

        pw.clear()
        panel_height = pu.ARTICLE_PANEL_HEIGHT_IN

        summary_ax = self.plot_normalization_score_summary(
            auto_summary=auto_summary,
            figsize=pu.article_brick_size(1.42, panel_height),
            show_legend=False,
        )
        if summary_ax is None:
            summary_ax = pw.Brick(
                figsize=pu.article_brick_size(1.42, panel_height),
                label="article_normalization_summary",
            )
            summary_ax.axis("off")
        self._apply_article_panel_format(
            summary_ax,
            title="Auto Normalization Method Selection",
        )

        rle_ax = pw.Brick(
            figsize=pu.article_brick_size(1.20, panel_height),
            label="article_normalization_rle",
        )
        self._plot_qc_rle_boxplot(
            ax=rle_ax,
            show_legend=False,
            article_compact=True,
        )
        self._apply_article_panel_format(
            rle_ax,
            title="QC RLE\nAlignment Change",
        )

        variance_ax = pw.Brick(
            figsize=pu.article_brick_size(1.20, panel_height),
            label="article_normalization_variance",
        )
        self._plot_qc_variance_stabilization(
            ax=variance_ax,
            show_legend=False,
            article_compact=True,
        )
        self._apply_article_panel_format(
            variance_ax,
            title="QC Variance Stabilization",
        )
        variance_ax.set_xlabel("Mean QC log2 Intensity")
        variance_ax.set_ylabel("QC dispersion")

        structure_ax = pw.Brick(
            figsize=pu.article_brick_size(1.20, panel_height),
            label="article_normalization_structure",
        )
        self._plot_qc_structure_improvement(
            ax=structure_ax,
            show_legend=False,
            article_compact=True,
        )
        self._apply_article_panel_format(
            structure_ax,
            title="QC Structure Distance Change",
        )
        structure_ax.set_ylabel("QC distance (log scale)")

        legend_ax = pw.Brick(
            figsize=pu.article_brick_size(1.30, panel_height),
            label="article_normalization_legend",
        )
        self.plot_normalization_article_legend(ax=legend_ax)
        return summary_ax | rle_ax | variance_ax | structure_ax | legend_ax
