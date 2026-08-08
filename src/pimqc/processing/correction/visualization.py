"""Diagnostic visualizations for signal-correction evaluation and selection.

MetaboVisualizerCorrector displays candidate scorecards, QC-RSD comparisons,
method-specific correction diagnostics, internal-standard trends, and selected
method summaries. It formats these outputs for standalone review and for the
reporting layer without changing corrected intensity values.
"""

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from loguru import logger
from typing import Any, Callable, Iterator

from ...visualization import plot_utils as pu
from ...core import model
from ...visualization import base as visualizer_classes
from ...statistics import metrics as su

FitPredictCallable = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
CorrectionModel = (
    RandomForestRegressor
    | TransformedTargetRegressor
    | Pipeline
    | FitPredictCallable
)


from .analysis import MetaboIntCorrector
from .algorithms import _format_correction_method_label


class MetaboVisualizerCorrector(visualizer_classes.BaseMetaboVisualizer):
    """Visualization suite matching original alpha output styles."""

    def __init__(self, corr_obj: MetaboIntCorrector) -> None:
        """Initialize with a computed MetaboIntCorrector object."""
        super().__init__(metabo_obj=corr_obj)
        self.corr = corr_obj

    # =========================================================================
    # Evaluation & Diagnostic Plotters
    # =========================================================================
    def plot_rsd_standalone_legend(
        self,
        ax: plt.Axes | None = None,
        show_cv: bool = True,
        loc: str = "center left",
        bbox_to_anchor: tuple[float, float] = (0.1, 0.5),
        legend_cols: int | None = None,
    ) -> plt.Axes:
        """Create a standalone legend for RSD plots using explicit Patches."""
        if ax is None:
            try:
                import patchworklib as pw
            except ImportError:
                raise ImportError("patchworklib is required for this plot.")
            current_ax = pw.Brick(figsize=(1.5, 4.0), label="rsd_legend")
        else:
            current_ax = ax

        current_ax.axis("off")

        from matplotlib.patches import Patch

        c_base = pu.get_equivalent_hex("tab:gray", alpha=1.0)
        c_cv = pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=0.33)
        c_full = pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=1.0)

        legend_elements = [
            Patch(
                facecolor=c_base,
                edgecolor="k",
                linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
                label="Baseline",
            )
        ]
        if show_cv:
            legend_elements.append(
                Patch(
                    facecolor=c_cv,
                    edgecolor="k",
                    linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
                    linestyle="--",
                    label="OOF model",
                )
            )
        legend_elements.append(
            Patch(
                facecolor=c_full,
                edgecolor="k",
                linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
                label="Full model",
            )
        )

        current_ax.legend(handles=legend_elements)

        self._format_single_legend(
            ax=current_ax,
            group_title="Correction evaluation",
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            legend_cols=legend_cols,
            borderaxespad=0.0,
        )

        # Keep patchworklib-compatible legends attached to the legend axis.
        if hasattr(current_ax.figure, "legends"):
            for leg in list(current_ax.figure.legends):
                current_ax.add_artist(leg)
            current_ax.figure.legends.clear()

        return current_ax

    def _collect_corr_rsd_series(
        self,
        stage_dfs: dict[str, pd.DataFrame],
        stage_oof_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> list[np.ndarray]:
        """Collect finite feature-wise QC RSD arrays from correction stages."""
        rsd_arrays: list[np.ndarray] = []
        for df_obj in stage_dfs.values():
            rsd = self.corr.extract_qc_rsd_series(df_obj)
            if not rsd.empty:
                values = rsd.to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size:
                    rsd_arrays.append(values)

        for df_obj in (stage_oof_dfs or {}).values():
            rsd = self.corr.extract_qc_rsd_series(df_obj)
            if not rsd.empty:
                values = rsd.to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size:
                    rsd_arrays.append(values)

        return rsd_arrays

    @staticmethod
    def _boxplot_visible_limits(
        values: np.ndarray,
    ) -> tuple[float, float] | None:
        """
        Return Tukey boxplot whisker limits for finite raw-ratio RSD values.
        """
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return None

        q1, q3 = np.nanpercentile(finite_values, [25, 75])
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= 0:
            return float(np.nanmin(finite_values)), float(
                np.nanmax(finite_values)
            )

        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        visible_values = finite_values[
            (finite_values >= lower_fence) & (finite_values <= upper_fence)
        ]
        if visible_values.size == 0:
            visible_values = finite_values
        return float(np.nanmin(visible_values)), float(
            np.nanmax(visible_values)
        )

    @staticmethod
    def _resolve_corr_rsd_ylim_from_values(
        rsd_arrays: list[np.ndarray],
        top_margin: float = 0.32,
    ) -> tuple[float, float] | None:
        """Resolve QC RSD y-axis range from visible boxplot whisker limits."""
        visible_limits = [
            MetaboVisualizerCorrector._boxplot_visible_limits(values)
            for values in rsd_arrays
        ]
        visible_limits = [
            limits for limits in visible_limits if limits is not None
        ]
        if not visible_limits:
            return None

        data_min = min(limit[0] for limit in visible_limits)
        data_max = max(limit[1] for limit in visible_limits)
        lower = min(0.0, data_min)
        span = max(data_max - lower, abs(data_max) * 0.10, 0.02)
        upper = data_max + max(span * top_margin, 0.02)
        if upper <= lower:
            upper = lower + 0.1
        return lower, upper

    def _resolve_corr_rsd_ylim(
        self,
        stage_dfs: dict[str, pd.DataFrame],
        stage_oof_dfs: dict[str, pd.DataFrame] | None = None,
        top_margin: float = 0.32,
    ) -> tuple[float, float] | None:
        """Resolve a QC RSD y-axis range with room for top annotations."""
        rsd_arrays = self._collect_corr_rsd_series(stage_dfs, stage_oof_dfs)
        return self._resolve_corr_rsd_ylim_from_values(
            rsd_arrays=rsd_arrays,
            top_margin=top_margin,
        )

    def _resolve_dashboard_corr_rsd_ylim(
        self,
        results_store: dict[str, dict[str, Any]],
    ) -> tuple[float, float] | None:
        """
        Resolve a shared QC RSD y-axis range for all correction candidates.
        """
        all_rsd_arrays: list[np.ndarray] = []
        for result in results_store.values():
            stage_dfs = result.get("stage_dfs", {})
            stage_oof_dfs = result.get("stage_oof_dfs", {})
            all_rsd_arrays.extend(
                self._collect_corr_rsd_series(stage_dfs, stage_oof_dfs)
            )

        if not all_rsd_arrays:
            return None

        return self._resolve_corr_rsd_ylim_from_values(all_rsd_arrays)

    def plot_corr_rsd(
        self,
        stage_dfs: dict[str, pd.DataFrame],
        stage_oof_dfs: dict[str, pd.DataFrame],
        ax: plt.Axes | None = None,
        show_legend: bool = True,
        y_limits: tuple[float, float] | None = None,
        article_compact: bool = False,
    ) -> plt.Axes:
        """Plot dual-mode RSD boxplots with dynamic width and annotations."""
        box_data = []
        positions = []
        box_colors = []
        box_styles = []
        tick_pos = []
        tick_labels = []
        medians_text = []

        c_base = pu.get_equivalent_hex("tab:gray", alpha=1.0)
        c_cv = pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=0.33)
        c_full = pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=1.0)

        # Retrieve the key of the final stage to mark the selection metric
        stage_keys = list(stage_dfs.keys())
        last_stage_key = stage_keys[-1] if stage_keys else None

        orig_df = stage_dfs.get("Original")
        if orig_df is not None:
            orig_rsd = self.corr.extract_qc_rsd_series(orig_df)
            box_data.append(orig_rsd.values)
            positions.append(1.0)
            box_colors.append(c_base)
            box_styles.append("-")
            tick_pos.append(1.0)
            tick_labels.append("Before\ncorrection")
            medians_text.append(
                f"Before correction: {orig_rsd.median() * 100:.2f}%"
            )

        current_x = 2.6  # Adjusted starting position for wider spacing
        for stage_name, df in stage_dfs.items():
            if stage_name == "Original":
                continue

            clean_name = stage_name.replace("\n", " ")
            has_cv = stage_name in stage_oof_dfs
            full_rsd = self.corr.extract_qc_rsd_series(df)
            is_last = stage_name == last_stage_key

            if has_cv:
                cv_rsd = self.corr.extract_qc_rsd_series(
                    stage_oof_dfs[stage_name]
                )
                box_data.extend([cv_rsd.values, full_rsd.values])

                # Widened symmetrical offset for fatter boxes (0.28 vs 0.22)
                positions.extend([current_x - 0.28, current_x + 0.28])

                box_colors.extend([c_cv, c_full])
                box_styles.extend(["--", "-"])

                # Append asterisk strictly to the CV metric if it's the final
                # stage
                prefix = "* " if is_last else ""
                medians_text.append(
                    f"{prefix}{clean_name} (OOF): {cv_rsd.median() * 100:.2f}%"
                )
                medians_text.append(
                    f"{clean_name} (Full): {full_rsd.median() * 100:.2f}%"
                )
            else:
                box_data.append(full_rsd.values)
                positions.append(current_x)
                box_colors.append(c_full)
                box_styles.append("-")

                # Append asterisk strictly to the global model metric.
                prefix = "* " if is_last else ""
                medians_text.append(
                    f"{prefix}{clean_name} (Full): "
                    f"{full_rsd.median() * 100:.2f}%"
                )

            tick_pos.append(current_x)
            formatted_label = stage_name.replace(" ", "\n")
            tick_labels.append(formatted_label)
            current_x += 1.6  # Adjusted step for wider boxes

        if ax is None:
            try:
                import patchworklib as pw
            except ImportError:
                raise ImportError("patchworklib is required for this plot.")
            pw.clear()
            fig_width = max(4.0, len(stage_dfs) * 1.2 + 2)
            current_ax = pw.Brick(figsize=(fig_width, 4.0), label="rsd_box")
        else:
            current_ax = ax

        box_linewidth = 0.5 if article_compact else 1.0
        median_linewidth = 0.7 if article_compact else 1.5
        box_width = 0.38 if article_compact else 0.50
        bp = current_ax.boxplot(
            box_data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
        )

        for i in range(len(box_data)):
            bp["boxes"][i].set_facecolor(box_colors[i])
            bp["boxes"][i].set_edgecolor("k")
            bp["boxes"][i].set_linewidth(box_linewidth)
            bp["boxes"][i].set_linestyle(box_styles[i])

            bp["medians"][i].set_color("k")
            bp["medians"][i].set_linewidth(median_linewidth)
            bp["medians"][i].set_linestyle(box_styles[i])

            for j in range(2):
                idx = i * 2 + j
                bp["whiskers"][idx].set_color("k")
                bp["whiskers"][idx].set_linewidth(box_linewidth)
                bp["whiskers"][idx].set_linestyle(box_styles[i])

                bp["caps"][idx].set_color("k")
                bp["caps"][idx].set_linewidth(box_linewidth)
                bp["caps"][idx].set_linestyle(box_styles[i])

        current_ax.set_xticks(tick_pos)
        current_ax.set_xticklabels(tick_labels)

        if show_legend:
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(
                    facecolor=c_base,
                    edgecolor="k",
                    linewidth=box_linewidth,
                    label="Baseline",
                ),
                Patch(
                    facecolor=c_cv,
                    edgecolor="k",
                    linewidth=box_linewidth,
                    linestyle="--",
                    label="OOF model",
                ),
                Patch(
                    facecolor=c_full,
                    edgecolor="k",
                    linewidth=box_linewidth,
                    label="Full model",
                ),
            ]
            current_ax.legend(handles=legend_elements)
            self._format_single_legend(
                ax=current_ax,
                group_title="Correction evaluation",
                loc="lower right",
                bbox_to_anchor=None,
                max_item_rows=6,
            )

        resolved_y_limits = y_limits or self._resolve_corr_rsd_ylim(
            stage_dfs=stage_dfs,
            stage_oof_dfs=stage_oof_dfs,
        )
        if resolved_y_limits is not None:
            current_ax.set_ylim(*resolved_y_limits)

        if article_compact:
            # Keep every evaluated stage. Selecting only the first and last
            # values hid the intra-batch result whenever an inter-batch stage
            # was also available.
            compact_lines = list(dict.fromkeys(medians_text))
            compact_lines = [
                line.replace("Before correction", "Before")
                .replace("Intra-batch corrected", "Intra")
                .replace("Inter-batch corrected", "Inter")
                .replace(" (Global)", " Global")
                for line in compact_lines
            ]
            annot_text = "Median QC-RSD\n" + "\n".join(compact_lines)
        else:
            annot_text = "Median QC RSD:\n" + "\n".join(medians_text)
        current_ax.text(
            0.96,
            0.98,
            annot_text,
            transform=current_ax.transAxes,
            fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
            verticalalignment="top",
            horizontalalignment="right",
            clip_on=False,
            bbox=pu.ai_ready_text_bbox(pad=0.25 if article_compact else 0.4),
            zorder=10,
        )

        self._apply_standard_format(
            current_ax, ylabel="QC RSD (%)", append_stage=False
        )
        pu.change_axis_format(current_ax, "percentage", "y")

        return current_ax

    def plot_correction_score_summary(
        self,
        results_store: dict[str, dict[str, Any]],
        best_method: str,
        ax: plt.Axes | None = None,
        show_legend: bool = True,
    ) -> plt.Axes:
        """Plot weighted AUTO correction score components."""
        try:
            import patchworklib as pw
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError("patchworklib is required for this plot.")

        if ax is None:
            current_ax = pw.Brick(
                figsize=(9.0, 3.0), label="correction_eval_summary"
            )
        else:
            current_ax = ax

        summary_rows = []
        for method, result in results_store.items():
            method_label = _format_correction_method_label(method)
            summary_rows.append(
                {
                    "method": method,
                    "label": method_label,
                    "selected": method == best_method,
                    "eval_rsd": result.get("eval_rsd"),
                    "median_qc_rsd_improvement_score": result.get(
                        "median_qc_rsd_improvement_score"
                    ),
                    "featurewise_qc_rsd_improvement_score": result.get(
                        "featurewise_qc_rsd_improvement_score"
                    ),
                    "sample_structure_score": result.get(
                        "sample_structure_score"
                    ),
                    "auto_score": result.get("auto_score"),
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.replace([np.inf, -np.inf], np.nan)
        summary_df = summary_df.dropna(subset=["auto_score"])
        summary_df = summary_df.sort_values(
            by=["auto_score", "label"], ascending=[False, True]
        ).reset_index(drop=True)

        if summary_df.empty:
            current_ax.axis("off")
            return current_ax

        score_cols = [
            "median_qc_rsd_improvement_score",
            "featurewise_qc_rsd_improvement_score",
            "sample_structure_score",
        ]
        weights = {
            "median_qc_rsd_improvement_score": 0.35,
            "featurewise_qc_rsd_improvement_score": 0.35,
            "sample_structure_score": 0.30,
        }
        label_map = {
            "median_qc_rsd_improvement_score": "Median QC-RSD improvement",
            "featurewise_qc_rsd_improvement_score": (
                "Feature-wise QC-RSD improvement"
            ),
            "sample_structure_score": "Sample structure preservation",
        }
        color_map = {
            "median_qc_rsd_improvement_score": pu.get_equivalent_hex(
                pu.PRIMARY_ACCENT_COLOR, alpha=1.0
            ),
            "featurewise_qc_rsd_improvement_score": pu.get_equivalent_hex(
                pu.PRIMARY_ACCENT_COLOR, alpha=0.67
            ),
            "sample_structure_score": pu.get_equivalent_hex(
                "tab:gray", alpha=0.6
            ),
        }

        y_pos = np.arange(len(summary_df))
        left = np.zeros(len(summary_df), dtype=float)
        for score_col in score_cols:
            left_start = left.copy()
            values = []
            for _, row in summary_df.iterrows():
                available_weight = sum(
                    weights[col]
                    for col in score_cols
                    if np.isfinite(su.finite_or_nan(row.get(col)))
                )
                if available_weight <= 0:
                    values.append(0.0)
                    continue
                score_value = np.clip(
                    su.finite_or_nan(row.get(score_col)), 0.0, 1.0
                )
                values.append(
                    score_value * weights[score_col] / available_weight
                )

            values_arr = np.asarray(values, dtype=float)
            current_ax.barh(
                y_pos,
                values_arr,
                left=left,
                color=color_map[score_col],
                edgecolor="k",
                linewidth=0.5,
                height=0.58,
                label=label_map[score_col],
            )
            for y_idx, row in enumerate(summary_df.itertuples()):
                score_value = su.finite_or_nan(getattr(row, score_col))
                if values_arr[y_idx] < 0.11 or not np.isfinite(score_value):
                    continue
                face_color = color_map[score_col]
                current_ax.text(
                    left_start[y_idx] + values_arr[y_idx] / 2.0,
                    y_idx,
                    f"{score_value:.2f}",
                    va="center",
                    ha="center",
                    fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                    color=pu.get_contrast_color(face_color),
                    clip_on=True,
                )
            left += values_arr

        y_labels = [
            f"* {row.label}" if bool(row.selected) else str(row.label)
            for row in summary_df.itertuples()
        ]
        current_ax.set_yticks(y_pos)
        current_ax.set_yticklabels(y_labels)
        current_ax.invert_yaxis()

        x_upper = float(np.nanmax(left)) if left.size else 1.0
        x_upper = min(1.08, max(x_upper + 0.08, x_upper * 1.10, 0.20))
        current_ax.set_xlim(0, x_upper)
        for y_idx, row in enumerate(summary_df.itertuples()):
            score = su.finite_or_nan(row.auto_score)
            current_ax.text(
                min(float(left[y_idx]) + 0.015, x_upper * 0.97),
                y_idx,
                f"{score:.3f}",
                va="center",
                ha="left",
                fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
            )

        self._apply_standard_format(
            current_ax,
            title="Auto Correction Method Selection",
            xlabel="Weighted contribution to overall score",
            append_stage=False,
        )
        if show_legend:
            legend_handles = [
                mpatches.Patch(
                    facecolor=color_map[col],
                    edgecolor="k",
                    linewidth=0.5,
                    label=label_map[col],
                )
                for col in score_cols
            ]
            current_ax.legend(handles=legend_handles)
            self._format_single_legend(
                ax=current_ax,
                group_title="Correction score components",
                loc="lower right",
                bbox_to_anchor=None,
                max_item_rows=6,
            )
        current_ax.tick_params(axis="y", length=0)

        return current_ax

    def plot_correction_dashboard_legend(
        self,
        ax: plt.Axes,
        show_cv: bool = True,
        fontsize: float = pu.DEFAULT_LEGEND_FONTSIZE,
        title_fontsize: float = pu.DEFAULT_LEGEND_TITLE_FONTSIZE,
        article_compact: bool = False,
    ) -> plt.Axes:
        """Draw grouped score-component and correction-mode legends."""
        import matplotlib.patches as mpatches

        legend_linewidth = pu.DEFAULT_AXIS_LINEWIDTH

        score_handles = [
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex(
                    pu.PRIMARY_ACCENT_COLOR, alpha=1.0
                ),
                edgecolor="k",
                linewidth=legend_linewidth,
                label="Median QC-RSD improvement",
            ),
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex(
                    pu.PRIMARY_ACCENT_COLOR, alpha=0.67
                ),
                edgecolor="k",
                linewidth=legend_linewidth,
                label="Feature-wise QC-RSD improvement",
            ),
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=0.6),
                edgecolor="k",
                linewidth=legend_linewidth,
                label="Sample structure preservation",
            ),
        ]

        mode_handles = [
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=1.0),
                edgecolor="k",
                linewidth=legend_linewidth,
                label="Baseline",
            )
        ]
        if show_cv:
            mode_handles.append(
                mpatches.Patch(
                    facecolor=pu.get_equivalent_hex(
                        pu.PRIMARY_ACCENT_COLOR, alpha=0.33
                    ),
                    edgecolor="k",
                    linewidth=legend_linewidth,
                    linestyle="--",
                    label="OOF model",
                )
            )
        mode_handles.append(
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex(
                    pu.PRIMARY_ACCENT_COLOR, alpha=1.0
                ),
                edgecolor="k",
                linewidth=legend_linewidth,
                label="Full model",
            )
        )

        self._plot_grouped_standalone_legends(
            ax=ax,
            legend_groups=[
                ("Correction score components", score_handles),
                ("QC-RSD evaluation stage", mode_handles),
            ],
            loc="upper left",
            start_bbox=(0.0, 1.0),
            row_gap=0.04,
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

    def plot_featurewise_qc_rsd_improvement_ecdf(
        self,
        result: dict[str, Any],
        ax: plt.Axes | None = None,
        article_compact: bool = False,
    ) -> plt.Axes:
        """Plot ECDF of paired feature-wise QC-RSD relative improvement."""
        try:
            import patchworklib as pw
        except ImportError:
            raise ImportError("patchworklib is required for this plot.")

        current_ax = (
            pw.Brick(figsize=(4.0, 4.0), label="featurewise_qc_rsd_ecdf")
            if ax is None
            else ax
        )

        raw_values = result.get("featurewise_qc_rsd_improvement_values")
        values = pd.Series(raw_values, dtype=float).replace(
            [np.inf, -np.inf], np.nan
        )
        values = values.dropna()
        if values.empty:
            current_ax.text(
                0.5,
                0.5,
                "No paired QC-RSD values",
                transform=current_ax.transAxes,
                ha="center",
                va="center",
                fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                bbox=pu.ai_ready_text_bbox(),
                zorder=10,
            )
            self._apply_standard_format(
                current_ax,
                title="Feature-wise QC-RSD Improvement",
                xlabel="Feature-wise QC-RSD relative improvement",
                ylabel="Cumulative feature fraction",
                append_stage=False,
            )
            return current_ax

        sorted_values = np.sort(values.to_numpy(dtype=float))
        cumulative = np.arange(1, sorted_values.size + 1) / sorted_values.size
        current_ax.step(
            sorted_values,
            cumulative,
            where="post",
            color=pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=1.0),
            linewidth=1.1 if article_compact else 1.8,
        )
        current_ax.axvline(
            0.0,
            color="0.35",
            linestyle="--",
            linewidth=0.6 if article_compact else 1.0,
            zorder=2,
        )
        current_ax.axhline(
            0.5,
            color="0.70",
            linestyle=":",
            linewidth=0.6 if article_compact else 1.0,
            zorder=1,
        )

        x_low, x_high = np.nanpercentile(sorted_values, [1.0, 99.0])
        x_span = max(float(x_high - x_low), 0.1)
        current_ax.set_xlim(
            float(x_low - x_span * 0.08), float(x_high + x_span * 0.08)
        )
        current_ax.set_ylim(0.0, 1.02)

        featurewise_score = su.finite_or_nan(
            result.get("featurewise_qc_rsd_improvement_score")
        )
        featurewise_median = su.finite_or_nan(
            result.get("featurewise_qc_rsd_improvement_median")
        )
        note_lines = []
        if np.isfinite(featurewise_score):
            note_lines.append(
                f"Score: {featurewise_score:.3f}"
                if article_compact
                else f"Winsorized score: {featurewise_score:.3f}"
            )
        if np.isfinite(featurewise_median):
            note_lines.append(
                f"Median: {featurewise_median:.1%}"
                if article_compact
                else f"Median improvement: {featurewise_median:.1%}"
            )
        if note_lines:
            current_ax.text(
                0.04,
                0.96,
                "\n".join(note_lines),
                transform=current_ax.transAxes,
                ha="left",
                va="top",
                fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                color="0.25",
                bbox=pu.ai_ready_text_bbox(
                    pad=0.25 if article_compact else 0.4
                ),
                zorder=10,
            )

        self._apply_standard_format(
            current_ax,
            title="Feature-wise QC-RSD Improvement",
            xlabel="Feature-wise QC-RSD relative improvement",
            ylabel="Cumulative feature fraction",
            append_stage=False,
        )
        pu.change_axis_format(current_ax, "percentage", "x")
        return current_ax

    def plot_correction_preservation_scorecard(
        self,
        results_store: dict[str, dict[str, Any]],
        best_method: str,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot actual-sample structure metrics used by AUTO correction."""
        try:
            import patchworklib as pw
        except ImportError:
            raise ImportError("patchworklib is required for this plot.")

        if ax is None:
            current_ax = pw.Brick(
                figsize=(3.6, 4.0), label="correction_preservation_scorecard"
            )
        else:
            current_ax = ax

        rows = []
        for method, result in results_store.items():
            method_label = _format_correction_method_label(method)
            rows.append(
                {
                    "method": method,
                    "label": method_label,
                    "selected": method == best_method,
                    "sample_structure_score": result.get(
                        "sample_structure_score"
                    ),
                    "Trustworthiness": result.get(
                        "sample_structure_metrics", {}
                    ).get("sample_structure_trustworthiness"),
                    "Distance rank preservation": result.get(
                        "sample_structure_metrics", {}
                    ).get("sample_structure_rank_preservation"),
                    "Distance scale preservation": result.get(
                        "sample_structure_metrics", {}
                    ).get("sample_structure_scale_preservation"),
                    "auto_score": result.get("auto_score"),
                }
            )

        summary_df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
        metric_cols = [
            "Trustworthiness",
            "Distance rank preservation",
            "Distance scale preservation",
        ]
        metric_labels = [
            "Trustworthiness",
            "Distance-rank\npreservation",
            "Distance-scale\npreservation",
        ]
        for col in ["auto_score", "sample_structure_score", *metric_cols]:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")
        summary_df = summary_df.dropna(subset=metric_cols, how="all")
        summary_df = summary_df.sort_values(
            by=["auto_score", "label"], ascending=[False, True]
        ).reset_index(drop=True)

        if summary_df.empty:
            current_ax.axis("off")
            return current_ax

        matrix = summary_df[metric_cols].to_numpy(dtype=float)
        cmap = pu.score_heatmap_cmap()
        annot_size = pu.heatmap_annotation_fontsize(
            current_ax,
            n_rows=matrix.shape[0],
            n_cols=matrix.shape[1],
            default_size=pu.DEFAULT_SCORE_HEATMAP_ANNOTATION_FONTSIZE,
            max_size=pu.DEFAULT_SCORE_HEATMAP_ANNOTATION_FONTSIZE,
            min_size=4.0,
        )

        masked_matrix = np.ma.masked_invalid(matrix)
        current_ax.imshow(
            masked_matrix,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
        )
        current_ax.set_xticks(np.arange(len(metric_cols)))
        current_ax.set_xticklabels(metric_labels)
        current_ax.set_yticks(np.arange(len(summary_df)))
        current_ax.set_yticklabels(
            [
                f"* {row.label}" if bool(row.selected) else str(row.label)
                for row in summary_df.itertuples()
            ]
        )
        current_ax.set_xticks(np.arange(-0.5, len(metric_cols), 1), minor=True)
        current_ax.set_yticks(np.arange(-0.5, len(summary_df), 1), minor=True)
        grid_lw = pu.DEFAULT_HEATMAP_CELL_LINEWIDTH
        current_ax.grid(
            which="minor", color="k", linestyle="-", linewidth=grid_lw
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

    def plot_correction_article_legend(
        self,
        ax: plt.Axes,
        show_oof: bool,
    ) -> plt.Axes:
        """Draw right-side grouped legends for the correction article panel."""
        return self.plot_correction_dashboard_legend(
            ax=ax,
            show_cv=show_oof,
            fontsize=pu.ARTICLE_LEGEND_FONTSIZE,
            title_fontsize=pu.ARTICLE_LEGEND_TITLE_FONTSIZE,
            article_compact=True,
        )

    def plot_correction_article_dashboard(
        self,
        results_store: dict[str, dict[str, Any]],
        best_method: str,
    ) -> object | None:
        """
        Create a compact score-aligned correction panel for manuscript figures.
        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning(
                "patchworklib not found. Skipping correction article panel."
            )
            return None

        if best_method not in results_store:
            return None

        pw.clear()
        best_result = results_store[best_method]
        panel_height = pu.ARTICLE_PANEL_HEIGHT_IN

        summary_ax = pw.Brick(
            figsize=pu.article_brick_size(1.85, panel_height),
            label="article_correction_summary",
        )
        self.plot_correction_score_summary(
            results_store=results_store,
            best_method=best_method,
            ax=summary_ax,
            show_legend=False,
        )
        self._apply_article_panel_format(
            summary_ax,
            title="Auto Correction Method Selection",
        )

        rsd_ax = pw.Brick(
            figsize=pu.article_brick_size(1.70, panel_height),
            label="article_correction_qc_rsd",
        )
        self.plot_corr_rsd(
            stage_dfs=best_result["stage_dfs"],
            stage_oof_dfs=best_result.get("stage_oof_dfs", {}),
            ax=rsd_ax,
            show_legend=False,
            article_compact=True,
        )
        self._apply_article_panel_format(
            rsd_ax,
            title="QC-RSD Distribution",
        )

        ecdf_ax = pw.Brick(
            figsize=pu.article_brick_size(1.70, panel_height),
            label="article_correction_featurewise",
        )
        self.plot_featurewise_qc_rsd_improvement_ecdf(
            result=best_result,
            ax=ecdf_ax,
            article_compact=True,
        )
        self._apply_article_panel_format(
            ecdf_ax,
            title="Feature-wise QC-RSD Improvement",
        )
        ecdf_ax.set_xlabel("QC-RSD relative improvement")
        ecdf_ax.set_ylabel("Cumulative fraction")

        legend_ax = pw.Brick(
            figsize=pu.article_brick_size(1.30, panel_height),
            label="article_correction_legend",
        )
        self.plot_correction_article_legend(
            ax=legend_ax,
            show_oof=bool(best_result.get("stage_oof_dfs")),
        )
        return summary_ax | rsd_ax | ecdf_ax | legend_ax

    def plot_correction_dashboard(
        self,
        results_store: dict[str, dict[str, Any]],
        best_method: str,
        include_auto_summary: bool = True,
    ) -> object | None:
        """Combine correction selection and selected-method diagnostics."""
        try:
            import patchworklib as pw
        except ImportError:
            raise ImportError("patchworklib is required for this plot.")

        pw.clear()
        if not results_store:
            return None

        row1 = None
        if include_auto_summary:
            layout_width = 12.0
            summary_brick = pw.Brick(
                figsize=pu.dashboard_brick_size(4.5, 4.0, layout_width),
                label="correction_eval_summary",
            )
            self.plot_correction_score_summary(
                results_store=results_store,
                best_method=best_method,
                ax=summary_brick,
                show_legend=False,
            )
            structure_brick = pw.Brick(
                figsize=pu.dashboard_brick_size(4.7, 4.0, layout_width),
                label="correction_preservation_scorecard",
            )
            self.plot_correction_preservation_scorecard(
                results_store=results_store,
                best_method=best_method,
                ax=structure_brick,
            )
            legend_brick = pw.Brick(
                figsize=pu.dashboard_brick_size(2.8, 4.0, layout_width),
                label="correction_dashboard_legend",
            )
            self.plot_correction_dashboard_legend(ax=legend_brick)
            row1 = summary_brick | structure_brick | legend_brick

        if best_method not in results_store:
            return row1

        best_result = results_store[best_method]
        layout_width = 12.0
        selected_rsd = pw.Brick(
            figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
            label="selected_correction_qc_rsd",
        )
        self.plot_corr_rsd(
            stage_dfs=best_result["stage_dfs"],
            stage_oof_dfs=best_result.get("stage_oof_dfs", {}),
            ax=selected_rsd,
            show_legend=not include_auto_summary,
            article_compact=True,
        )
        selected_rsd.set_title(
            "QC-RSD Distribution",
            fontsize=pu.DEFAULT_TITLE_FONTSIZE,
            fontweight="bold",
        )

        featurewise_ecdf = pw.Brick(
            figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
            label="selected_featurewise_qc_rsd_ecdf",
        )
        self.plot_featurewise_qc_rsd_improvement_ecdf(
            result=best_result,
            ax=featurewise_ecdf,
            article_compact=True,
        )

        sample_structure = pw.Brick(
            figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
            label="selected_correction_sample_structure",
        )
        final_stage_df = list(best_result["stage_dfs"].values())[-1]
        pu.plot_sample_structure_change_map(
            ax=sample_structure,
            raw_obj=self.corr,
            transformed_obj=final_stage_df,
            structure_metrics=best_result.get("sample_structure_metrics", {}),
            seed=int(self.corr.attrs.get("global_seed", 123)),
            title="Sample Structure Change Map",
            compact_style=True,
        )

        row2 = selected_rsd | featurewise_ecdf | sample_structure
        return row1 / row2 if row1 is not None else row2

    def plot_correction_candidate_grid(
        self, results_store: dict[str, dict[str, Any]], best_method: str
    ) -> object | None:
        """Plot all AUTO correction candidates as a QC-RSD appendix grid."""
        try:
            import patchworklib as pw
        except ImportError:
            raise ImportError("patchworklib is required for this plot.")

        pw.clear()
        if not results_store:
            return None

        panel_width = 3.7
        panel_height = 4.0
        layout_width = panel_width * 3.0
        bricks: dict[str, object] = {}
        method_rows = [
            ["QC-RLSC", "robust QC-RLSC", "QC-SVR"],
            ["SERRF", "RUV-III", "WaveICA 2.0"],
        ]
        detail_methods = [method for row in method_rows for method in row]
        shared_y_limits = self._resolve_dashboard_corr_rsd_ylim(results_store)

        for method in detail_methods:
            if method not in results_store:
                continue
            res = results_store[method]
            stage_dfs = res["stage_dfs"]
            stage_oof_dfs = res.get("stage_oof_dfs", {})
            safe_label = re.sub(r"[^A-Za-z0-9_]+", "_", f"rsd_box_{method}")

            b = pw.Brick(
                figsize=pu.dashboard_brick_size(
                    panel_width, panel_height, layout_width
                ),
                label=safe_label,
            )

            self.plot_corr_rsd(
                stage_dfs=stage_dfs,
                stage_oof_dfs=stage_oof_dfs,
                ax=b,
                show_legend=False,
                y_limits=shared_y_limits,
                article_compact=True,
            )

            method_label = _format_correction_method_label(method)
            title = (
                f"* {method_label}" if method == best_method else method_label
            )
            b.set_title(
                title,
                fontsize=pu.DEFAULT_TITLE_FONTSIZE,
                fontweight="bold",
            )
            bricks[method] = b

        plot_rows = []
        for row_methods in method_rows:
            row_bricks = [
                bricks[method] for method in row_methods if method in bricks
            ]
            if not row_bricks:
                continue
            row = row_bricks[0]
            for brick in row_bricks[1:]:
                row = row | brick
            plot_rows.append(row)

        if not plot_rows:
            return None

        legend_brick = pw.Brick(
            figsize=pu.dashboard_brick_size(
                panel_width * 3.0, 0.55, layout_width
            ),
            label="correction_mode_legend",
        )
        self.plot_rsd_standalone_legend(
            ax=legend_brick,
            show_cv=True,
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            legend_cols=3,
        )

        grid_pw = plot_rows[0]
        for row in plot_rows[1:]:
            grid_pw = grid_pw / row

        return grid_pw / legend_brick

    def _plot_standalone_is_legend(
        self,
        ax: plt.Axes,
        sample_type: str,
        batch: str,
        qc_label: str,
        actual_label: str,
        has_baseline: bool,
    ) -> plt.Axes:
        """Render a standalone multi-group legend for IS scatters."""
        import matplotlib.lines as mlines

        ax.axis("off")
        legend_handles = []
        legend_labels = []
        group_titles = [sample_type, batch]

        # Group 1: Sample Type
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
                markersize=6,
                markeredgecolor="k",
                markeredgewidth=0.5,
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
                markersize=6,
                markeredgecolor="k",
                markeredgewidth=0.5,
                label=actual_label,
            )
        )
        legend_labels.append(actual_label)

        # Group 2: Batch (Reusing BaseVisualizer properties)
        legend_handles.append(mlines.Line2D([], [], color="none", label=batch))
        legend_labels.append(batch)

        for b_val in getattr(self, "all_batches", []):
            m_style = getattr(self, "style_map", {}).get(b_val, "o")
            legend_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color="tab:gray",
                    marker=m_style,
                    linestyle="none",
                    markersize=6,
                    markeredgecolor="k",
                    markeredgewidth=0.5,
                    label=str(b_val),
                )
            )
            legend_labels.append(str(b_val))

        # Group 3: Model Baseline (Rendered only if prediction exists)
        if has_baseline:
            group_titles.append("Model")
            legend_handles.append(
                mlines.Line2D([], [], color="none", label="Model")
            )
            legend_labels.append("Model")
            legend_handles.append(
                mlines.Line2D(
                    [], [], color="k", ls="-", lw=1.5, label="Fitted Baseline"
                )
            )
            legend_labels.append("Fitted Baseline")

        # =====================================================================
        # Initialize a standard Matplotlib legend before applying shared
        # styling.
        # before passing it to the multi-legend layout formatter engine.
        # =====================================================================
        ax.legend(legend_handles, legend_labels)

        self._format_multi_legends(
            ax=ax,
            group_titles=group_titles,
            loc="upper left",
            start_bbox=(0.0, 0.95),
            row_gap=0.04,
            layout_cols=1,
            column_gap=0.1,
            max_item_rows=6,
        )

        # Prevent Patchworklib from discarding figure-level legends
        if hasattr(ax.figure, "legends"):
            for leg in list(ax.figure.legends):
                ax.add_artist(leg)
            ax.figure.legends.clear()

        return ax

    def _get_is_shared_ylim(
        self,
        stage_dfs: dict[str, model.MetaboInt],
        pred_df: model.MetaboInt | None,
        feat: str,
        boundary: str,
    ) -> tuple[float, float] | None:
        """
        Calculate one y-axis range for one internal standard across stages.
        """
        y_values: list[float] = []
        boundary_helper = model.MetaboInt()

        for df in stage_dfs.values():
            try:
                plot_data = df.int_order_info(feat_type="IS").reset_index()
            except Exception:
                continue

            if feat not in plot_data.columns:
                continue

            feature_values = pd.to_numeric(plot_data[feat], errors="coerce")
            finite_values = feature_values[np.isfinite(feature_values)]
            if finite_values.empty:
                continue

            y_values.extend(finite_values.astype(float).tolist())

            try:
                boundaries = boundary_helper.calculate_boundaries(
                    finite_values, boundary
                )
            except Exception:
                boundaries = ()
            y_values.extend(
                float(value)
                for value in boundaries
                if np.isfinite(float(value))
            )

        if pred_df is not None:
            try:
                pred_info = pred_df.int_order_info(feat_type="IS").reset_index()
            except Exception:
                pred_info = pd.DataFrame()

            if feat in pred_info.columns:
                pred_values = pd.to_numeric(pred_info[feat], errors="coerce")
                finite_pred = pred_values[np.isfinite(pred_values)]
                y_values.extend(finite_pred.astype(float).tolist())

        if not y_values:
            return None

        finite_y = np.asarray(y_values, dtype=float)
        finite_y = finite_y[np.isfinite(finite_y)]
        if finite_y.size == 0:
            return None

        y_min = float(np.min(finite_y))
        y_max = float(np.max(finite_y))
        if np.isclose(y_min, y_max):
            y_pad = max(abs(y_min) * 0.05, 1.0)
        else:
            y_pad = (y_max - y_min) * 0.08
        return y_min - y_pad, y_max + y_pad

    @staticmethod
    def _get_is_shared_yticks(
        ylim: tuple[float, float] | None,
    ) -> np.ndarray | None:
        """Resolve one set of y ticks shared by all IS scatter stages."""
        if ylim is None:
            return None

        locator = mticker.MaxNLocator(
            nbins=4, min_n_ticks=3, steps=[1, 2, 2.5, 5, 10]
        )
        ticks = locator.tick_values(ylim[0], ylim[1])
        ticks = ticks[np.isfinite(ticks)]
        ticks = ticks[(ticks >= ylim[0]) & (ticks <= ylim[1])]

        if ticks.size < 3:
            ticks = np.linspace(ylim[0], ylim[1], num=4)

        return ticks

    @staticmethod
    def _apply_is_shared_y_axis(
        ax: plt.Axes,
        ylim: tuple[float, float] | None,
        yticks: np.ndarray | None,
    ) -> None:
        """Apply shared y limits, ticks, formatter, and tick styling."""
        if ylim is not None:
            ax.set_ylim(ylim)
        if yticks is not None:
            ax.yaxis.set_major_locator(mticker.FixedLocator(yticks))

        pu.change_axis_format(ax, "scientific notation", "y")
        pu.change_fontsize(ax, axis=pu.DEFAULT_FORMAT_AXIS)
        pu.change_weight(ax, axis=pu.DEFAULT_FORMAT_AXIS)
        offset_text = ax.yaxis.get_offset_text()
        offset_text.set_fontsize(pu.DEFAULT_AXIS_TICK_FONTSIZE)
        offset_text.set_weight(pu.DEFAULT_AXIS_TICK_WEIGHT)

    def plot_is_int_order_scatter(
        self,
        stage_dfs: dict[str, model.MetaboInt],
        pred_df: model.MetaboInt | None,
        valid: list[str],
        sample_type: str,
        batch: str,
        inject_order: str,
        qc_label: str,
        actual_label: str,
        boundary: str,
    ) -> Iterator[tuple[object, object]]:
        """Dynamically assemble IS scatters using a data-driven 2/3+1 grid.

        Yields figures iteratively to prevent Matplotlib memory leaks and
        registry collisions during sequential batch saving.
        """
        try:
            import patchworklib as pw
            import seaborn as sns
        except ImportError:
            return

        if not valid:
            return

        has_baseline = pred_df is not None

        for feat in valid:
            pw.clear()
            bricks = []
            shared_ylim = self._get_is_shared_ylim(
                stage_dfs=stage_dfs,
                pred_df=pred_df,
                feat=feat,
                boundary=boundary,
            )
            shared_yticks = self._get_is_shared_yticks(shared_ylim)

            for stage_name, df in stage_dfs.items():
                brick = pw.Brick(figsize=pu.dashboard_brick_size(6.5, 2.0, 9.0))

                # Directly reuse existing base plotter for individual panels
                self.plot_single_is_scatter(
                    df=df,
                    feat=feat,
                    sample_type=sample_type,
                    batch=batch,
                    inject_order=inject_order,
                    qc_label=qc_label,
                    actual_label=actual_label,
                    ylabel=stage_name,
                    boundary=boundary,
                    ax=brick,
                    ylim=shared_ylim,
                    yticks=shared_yticks,
                )

                # Overlay prediction lines strictly for the Original stage
                if stage_name == "Original" and has_baseline:
                    pred_info = pred_df.int_order_info(
                        feat_type="IS"
                    ).reset_index()

                    for batch_id in pred_info[batch].unique():
                        b_pred = pred_info[pred_info[batch] == batch_id]
                        sns.lineplot(
                            data=b_pred,
                            x=inject_order,
                            y=feat,
                            color="k",
                            linestyle="-",
                            ax=brick,
                            zorder=3,
                        )
                    if shared_ylim is not None:
                        self._apply_is_shared_y_axis(
                            ax=brick, ylim=shared_ylim, yticks=shared_yticks
                        )

                # Strip internal standard legends to favor the global brick
                if brick.get_legend():
                    brick.get_legend().remove()

                bricks.append(brick)

            # Assemble left column iteratively via patchworklib
            if not bricks:
                continue

            left_col = bricks[0]
            for b in bricks[1:]:
                left_col = left_col / b

            # Assemble right column (Standalone Legend Brick)
            leg_h = len(bricks) * 2.0
            leg_brick = pw.Brick(
                figsize=pu.dashboard_brick_size(2.5, leg_h, 9.0)
            )

            self._plot_standalone_is_legend(
                ax=leg_brick,
                sample_type=sample_type,
                batch=batch,
                qc_label=qc_label,
                actual_label=actual_label,
                has_baseline=has_baseline,
            )

            # Yield immediately to allow saving before the next iteration
            yield feat, left_col | leg_brick

    def plot_single_is_scatter(
        self,
        df: model.MetaboInt,
        feat: str,
        sample_type: str,
        batch: str,
        inject_order: str,
        qc_label: str,
        actual_label: str,
        ylabel: str,
        boundary: str,
        ax: plt.Axes | None = None,
        ylim: tuple[float, float] | None = None,
        yticks: np.ndarray | None = None,
    ) -> object:
        """Plot a single scatter panel with calculated boundaries."""
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(7.5, 2.5))
        else:
            current_ax = ax
            fig = current_ax.figure

        plot_data = df.int_order_info(feat_type="IS").reset_index()
        plot_data[sample_type] = pd.Categorical(
            plot_data[sample_type],
            categories=[actual_label, qc_label],
            ordered=True,
        )
        plot_data = plot_data.sort_values(sample_type)

        sns.scatterplot(
            data=plot_data,
            x=inject_order,
            y=feat,
            hue=sample_type,
            style=batch,
            s=40,
            edgecolor="k",
            palette=self.pal,
            hue_order=[qc_label, actual_label],
            markers=self.style_map,
            style_order=self.all_batches,
            ax=current_ax,
        )

        solid_line, lower_limit, upper_limit = (
            model.MetaboInt().calculate_boundaries(plot_data[feat], boundary)
        )
        for y, linestyle in zip(
            [solid_line, lower_limit, upper_limit], ["-", "--", "--"]
        ):
            current_ax.axhline(y, color="k", linestyle=linestyle)

        self._apply_is_shared_y_axis(
            ax=current_ax,
            ylim=ylim,
            yticks=yticks,
        )

        # Enable append_stage=True and feed the precise pipeline stage attribute
        self._apply_standard_format(
            current_ax,
            title=feat,
            xlabel=inject_order,
            ylabel=ylabel,
            append_stage=False,
            custom_stage=df.attrs.get("pipeline_stage", ""),
        )
        self._apply_is_shared_y_axis(
            ax=current_ax,
            ylim=ylim,
            yticks=yticks,
        )
        return fig

    def plot_pred_baseline_is(
        self,
        raw: model.MetaboInt,
        pred: model.MetaboInt | None,
        valid: list[str],
        sample_type: str,
        batch: str,
        inject_order: str,
        qc_label: str,
        actual_label: str,
        method: str = "QC-RLSC",
    ) -> object | None:
        """Assemble IS fitted-baseline panels with a single shared legend."""
        try:
            import patchworklib as pw
        except ImportError:
            return None

        if not valid:
            return None

        pw.clear()
        plot_bricks = []
        panel_cols = 1 if len(valid) == 1 else 2
        layout_width = 6.5 * panel_cols + 2.5
        panel_size = pu.dashboard_brick_size(6.5, 2.0, layout_width)

        pred_info = None
        global_model_methods = {"SERRF", "RUV-III", "WAVEICA 2.0"}
        if pred is not None and method.upper() not in global_model_methods:
            pred_info = pred.int_order_info(feat_type="IS").reset_index()

        for n, feat in enumerate(valid):
            ax = pw.Brick(figsize=panel_size, label=f"pred_base_is_{n}")
            plot_data = raw.int_order_info(feat_type="IS").reset_index()

            plot_data[sample_type] = pd.Categorical(
                plot_data[sample_type],
                categories=[actual_label, qc_label],
                ordered=True,
            )
            plot_data = plot_data.sort_values(sample_type)

            sns.scatterplot(
                data=plot_data,
                x=inject_order,
                y=feat,
                hue=sample_type,
                style=batch,
                s=40,
                edgecolor="k",
                palette=self.pal,
                hue_order=[qc_label, actual_label],
                markers=self.style_map,
                style_order=self.all_batches,
                ax=ax,
            )

            if pred_info is not None and feat in pred_info.columns:
                for batch_id in pred_info[batch].unique():
                    sns.lineplot(
                        data=pred_info[pred_info[batch] == batch_id],
                        x=inject_order,
                        y=feat,
                        color="k",
                        ax=ax,
                    )
            self._apply_standard_format(
                ax,
                title=feat,
                xlabel=inject_order,
                ylabel="Intensity",
                append_stage=False,
            )
            pu.change_axis_format(ax, "scientific notation", "y")
            pu.change_fontsize(ax, axis="y")
            pu.change_weight(ax, axis="y")
            offset_text = ax.yaxis.get_offset_text()
            offset_text.set_fontsize(pu.DEFAULT_AXIS_TICK_FONTSIZE)
            offset_text.set_weight(pu.DEFAULT_AXIS_TICK_WEIGHT)

            if ax.get_legend():
                ax.get_legend().remove()
            plot_bricks.append(ax)

        row_bricks = []
        for row_start in range(0, len(plot_bricks), panel_cols):
            row_items = plot_bricks[row_start : row_start + panel_cols]
            if panel_cols == 2 and len(row_items) == 1:
                spacer = pw.Brick(
                    figsize=panel_size, label=f"pred_base_is_spacer_{n}"
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

        legend_height = max(panel_size[1], len(row_bricks) * panel_size[1])
        leg_brick = pw.Brick(
            figsize=pu.dashboard_brick_size(2.5, legend_height, layout_width),
            label="pred_base_is_legend",
        )
        self._plot_standalone_is_legend(
            ax=leg_brick,
            sample_type=sample_type,
            batch=batch,
            qc_label=qc_label,
            actual_label=actual_label,
            has_baseline=pred_info is not None,
        )

        return plot_grid | leg_brick
