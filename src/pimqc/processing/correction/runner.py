"""Export and visualize completed signal-correction results.

``CorrectionStageRunner`` delegates numerical work and candidate selection to
``MetaboIntCorrector``, then writes only the selected matrices and renders the
stage-specific dashboard and internal-standard diagnostics.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from loguru import logger

from ...io import ensure_directory
from ..stage import StageResult, StageRunner
from .algorithms import (
    _format_correction_method_file_label,
    _format_correction_method_label,
)
from ...plotting.correction import CorrectionPlotter

if TYPE_CHECKING:
    from ...core.model import MetaboInt

    # Resolve the quoted generic forward reference for static analyzers while
    # avoiding the runtime analysis/runner import cycle.


class CorrectionStageRunner(
    StageRunner["MetaboIntCorrector", dict[str, "MetaboInt"]]
):
    """Run correction while keeping artifact handling outside the processor."""

    def compute(self) -> StageResult[dict[str, MetaboInt]]:
        """Evaluate candidates and return the selected correction stages."""
        return self.processor.transform_correction()

    def export(self, result: StageResult[dict[str, MetaboInt]]) -> None:
        """Write selected correction stages and the fitted QC baseline."""
        assert self.output_dir is not None
        method = result.metadata["selected_method"]
        file_label = _format_correction_method_file_label(method)

        # Export only the stages selected for propagation; the original matrix
        # remains available through the processor and is never duplicated.
        for stage_name, frame in result.data.items():
            clean_name = stage_name.replace("\n", " ")
            if method in {"SERRF", "RUV-III", "WaveICA 2.0"}:
                file_name = f"{method}.csv"
            else:
                prefix = clean_name.replace(" corrected", "")
                file_name = f"{prefix.replace(' ', '_')}_{file_label}.csv"
            frame.to_csv(self.output_dir / file_name)

        predicted = result.metadata["selected_pred_df"]
        if predicted is not None:
            predicted.to_csv(self.output_dir / f"QC_Fit_Base_{file_label}.csv")

    def render(self, result: StageResult[dict[str, MetaboInt]]) -> None:
        """Render the selected-method and candidate correction diagnostics."""
        assert self.output_dir is not None
        processor = result.render_context["processor"]
        plotter = CorrectionPlotter(processor)
        label = result.metadata["selected_label"]
        method = result.metadata["selected_method"]
        file_label = _format_correction_method_file_label(method)
        dashboard_label = file_label.replace(" ", "_")
        is_auto = result.metadata["is_auto"]

        logger.info("Assembling correction diagnostic dashboard...")
        dashboard = plotter.plot_correction_dashboard(
            result.candidates,
            label,
            include_auto_summary=is_auto and len(result.candidates) > 1,
        )
        if dashboard is not None:
            path = self.output_dir / (
                f"Correction_Dashboard_{dashboard_label}.svg"
            )
            plotter.save_and_show_pw(
                pw_obj=dashboard,
                width="60%",
                file_path=str(path),
            )

        if is_auto and len(result.candidates) > 1:
            # Candidate comparison is an AUTO-only artifact; fixed-method runs
            # retain the smaller selected-method dashboard.
            candidate_dashboard = plotter.plot_correction_candidate_dashboard(
                results_store=result.candidates,
                selected_method=label,
            )
            if candidate_dashboard is not None:
                path = self.output_dir / (
                    f"Correction_Candidate_Dashboard_{dashboard_label}.svg"
                )
                plotter.save_and_show_pw(
                    pw_obj=candidate_dashboard,
                    width="60%",
                    file_path=str(path),
                )
                logger.info(f"Correction candidate dashboard saved as: {path}")

        self._render_internal_standards(result, plotter, file_label)
        display_label = _format_correction_method_label(label)
        logger.success(f"Signal drift correction ({display_label}) completed.")

    def _render_internal_standards(
        self,
        result: StageResult[dict[str, MetaboInt]],
        plotter: CorrectionPlotter,
        file_label: str,
    ) -> None:
        """Render internal-standard diagnostics using explicit stage context."""
        processor = result.render_context["processor"]
        if not len(processor.valid_is):
            return

        assert self.output_dir is not None
        metadata = result.metadata
        method = metadata["selected_method"]
        display_label = _format_correction_method_label(
            metadata["selected_label"]
        )
        predicted = metadata["selected_pred_df"]
        boundary = processor.attrs.get("boundary", "IQR")
        # Reconstruct the visual stage sequence without adding Original to the
        # exported StageResult payload.
        stage_dfs = {"Original": processor, **result.data}
        directory = ensure_directory(
            self.output_dir / "Internal_Standard_Scatters"
        )

        logger.info(f"Generating IS plots for {display_label}...")
        for feature, figure in plotter.plot_is_int_order_scatter(
            stage_dfs,
            predicted,
            processor.valid_is,
            metadata["sample_type_col"],
            metadata["batch_col"],
            metadata["inject_order_col"],
            metadata["qc_label"],
            metadata["actual_label"],
            boundary,
        ):
            safe_feature = re.sub(r"[^a-zA-Z0-9]", "_", feature)
            path = directory / f"IS_Scatter_{safe_feature}_{file_label}.svg"
            plotter.save_and_show_pw(
                pw_obj=figure,
                file_path=str(path),
                show_plot=False,
            )

        if method in {"SERRF", "RUV-III", "WaveICA 2.0"} or predicted is None:
            logger.info(
                f"Bypassing IS baseline prediction for {display_label}."
            )
            return

        baseline = plotter.plot_pred_baseline_is(
            processor,
            predicted,
            processor.valid_is,
            metadata["sample_type_col"],
            metadata["batch_col"],
            metadata["inject_order_col"],
            metadata["qc_label"],
            metadata["actual_label"],
            method=method,
        )
        if baseline is not None:
            path = self.output_dir / f"Pred_Base_IS_{file_label}.svg"
            plotter.save_and_show_pw(
                pw_obj=baseline,
                file_path=str(path),
                show_plot=False,
            )
