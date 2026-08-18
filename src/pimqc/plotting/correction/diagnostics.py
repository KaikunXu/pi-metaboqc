"""Diagnostic panels for signal-correction results.

The module renders QC-RSD distributions and feature-wise improvement curves;
candidate scorecards and dashboard composition live in sibling modules.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections.abc import Callable
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from typing import Any

from ...statistics import metrics as su
from .. import annotation_layout as al
from .. import plot_utils as pu

FitPredictCallable = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
CorrectionModel = (
    RandomForestRegressor
    | TransformedTargetRegressor
    | Pipeline
    | FitPredictCallable
)


class CorrectionDiagnosticsMixin:
    """Render RSD distributions and feature-wise diagnostics."""

    # Correction evaluation panels
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
            CorrectionDiagnosticsMixin._boxplot_visible_limits(values)
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
        self._apply_standard_format(
            current_ax, ylabel="QC RSD (%)", append_stage=False
        )
        pu.change_axis_format(current_ax, "percentage", "y")
        al.add_auto_annotation(
            ax=current_ax,
            text=annot_text,
            fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
            bbox=pu.ai_ready_text_bbox(pad=0.25 if article_compact else 0.4),
        )

        return current_ax

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
        self._apply_standard_format(
            current_ax,
            title="Feature-wise QC-RSD Improvement",
            xlabel="Feature-wise QC-RSD relative improvement",
            ylabel="Cumulative feature fraction",
            append_stage=False,
        )
        pu.change_axis_format(current_ax, "percentage", "x")
        if note_lines:
            al.add_auto_annotation(
                ax=current_ax,
                text="\n".join(note_lines),
                occupancy_arrays=[np.column_stack((sorted_values, cumulative))],
                fontsize=pu.DEFAULT_ANNOTATION_FONTSIZE,
                color="0.25",
                bbox=pu.ai_ready_text_bbox(
                    pad=0.25 if article_compact else 0.4
                ),
            )
        return current_ax
