"""Internal-standard diagnostics for signal correction.

The module renders injection-order traces, per-standard comparisons, and
shared-axis candidate panels without participating in correction fitting.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from collections.abc import Callable, Iterator
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from ...core import model
from .. import plot_utils as pu

FitPredictCallable = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
CorrectionModel = (
    RandomForestRegressor
    | TransformedTargetRegressor
    | Pipeline
    | FitPredictCallable
)


class CorrectionInternalStandardMixin:
    """Render internal-standard correction diagnostics."""

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

        # Initialize the Matplotlib legend before applying shared styling.
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
