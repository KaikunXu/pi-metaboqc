"""Export and visualize completed missing-value imputation results.

The runner keeps benchmark arrays and candidate metrics in ``StageResult`` and
uses them to render the selected method or AUTO diagnostics. The processor
therefore remains responsible for transformation rather than file operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from ..stage import StageResult, StageRunner
from ...plotting.imputation import ImputationPlotter

if TYPE_CHECKING:
    from .analysis import MetaboIntImputer


class ImputationStageRunner(
    StageRunner["MetaboIntImputer", "MetaboIntImputer"]
):
    """Run imputation while keeping calculation artifacts in StageResult."""

    def compute(self) -> StageResult["MetaboIntImputer"]:
        """Compute the completed matrix and retain benchmark arrays."""
        return self.processor.transform_imputation()

    def export(self, result: StageResult["MetaboIntImputer"]) -> None:
        """Write the completed matrix using the selected method label."""
        assert self.output_dir is not None
        filename = (
            "Imputed_Data_NotRequired.csv"
            if result.metadata["skipped"]
            else f"Imputed_Data_{result.metadata['selected_label']}.csv"
        )
        result.data.to_csv(self.output_dir / filename)

    def render(self, result: StageResult["MetaboIntImputer"]) -> None:
        """Render selected-method and candidate diagnostics."""
        if result.metadata["skipped"] or not result.metadata["idx_mar_count"]:
            return
        assert self.output_dir is not None
        logger.info("Generating diagnostic plots for imputation...")
        processor = result.render_context["processor"]
        plotter = ImputationPlotter(
            raw_obj=processor,
            imp_obj=result.data,
        )
        selected_method = result.metadata["selected_method"]
        selected_label = result.metadata["selected_label"]
        if result.metadata["is_auto"]:
            # AUTO retains all benchmark candidates for comparative rendering.
            dashboard = plotter.plot_imputation_auto_dashboard(
                result.candidates,
                selected_method=selected_method,
            )
        else:
            metrics, true_values, predicted_values = result.candidates[
                selected_method
            ]
            dashboard = plotter.plot_imputation_method_dashboard(
                metrics=metrics,
                true_vals=true_values,
                pred_vals=predicted_values,
                method_name=selected_label,
            )
        if dashboard is not None:
            path = self.output_dir / (
                f"Imputation_Dashboard_{selected_label}.svg"
            )
            plotter.save_and_show_pw(
                pw_obj=dashboard,
                file_path=str(path),
                width="60%" if result.metadata["is_auto"] else "45%",
            )
            logger.info(f"Imputation dashboard saved as: {path}")

        appendix = (
            # The appendix requires the complete candidate cache and therefore
            # has no meaningful fixed-method equivalent.
            plotter.plot_imputation_nrmse_appendix_dashboard(result.candidates)
            if result.metadata["is_auto"]
            and result.metadata["has_candidate_cache"]
            else None
        )
        if appendix is not None:
            path = self.output_dir / (
                f"Imputation_Candidate_Dashboard_{selected_label}.svg"
            )
            plotter.save_and_show_pw(
                pw_obj=appendix,
                file_path=str(path),
                width="60%",
            )
            logger.info(f"Imputer candidate NRMSE grid saved as: {path}")
