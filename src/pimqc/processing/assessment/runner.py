"""Export and render completed quality-assessment diagnostics.

The runner keeps assessment calculations side-effect free and preserves the
existing QA table, panel, legend, control-chart, and dashboard artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from loguru import logger

from ..stage import StageResult, StageRunner
from .analysis import AssessmentDiagnostics
from ...plotting.assessment import AssessmentPlotter

if TYPE_CHECKING:
    from .analysis import MetaboIntAssessor


class AssessmentStageRunner(
    StageRunner["MetaboIntAssessor", AssessmentDiagnostics]
):
    """Run assessment while separating computation from artifact generation."""

    def __init__(
        self,
        processor: "MetaboIntAssessor",
        output_dir: str | Path | None,
        *,
        legend_mode: str = "external",
        runtime_overrides: Mapping[str, object] | None = None,
        allowed_override_keys: frozenset[str] | set[str] | None = None,
    ) -> None:
        """Initialize the assessment lifecycle and legend strategy."""
        super().__init__(
            processor,
            output_dir,
            runtime_overrides=runtime_overrides,
            allowed_override_keys=allowed_override_keys,
        )
        self.legend_mode = legend_mode

    def compute(self) -> StageResult[AssessmentDiagnostics]:
        """Compute all tables and plot inputs without filesystem side effects."""
        return self.processor.compute_assessment()

    def export(self, result: StageResult[AssessmentDiagnostics]) -> None:
        """Write the combined sample-level QA diagnostic table."""
        if result.metadata["skipped"]:
            return
        assert self.output_dir is not None
        result.candidates.to_csv(
            self.output_dir / "QA_Diagnostics_Outliers.csv",
            encoding="utf-8-sig",
            na_rep="NA",
        )

    def render(self, result: StageResult[AssessmentDiagnostics]) -> None:
        """Render the same QA figures and dashboard as the legacy workflow."""
        if result.metadata["skipped"]:
            return
        assert self.output_dir is not None

        diagnostics = result.data
        metadata = result.metadata
        output_dir = str(self.output_dir)
        processor = result.render_context["processor"]
        plotter = AssessmentPlotter(processor)
        legend_mode = plotter._validate_legend_mode(self.legend_mode)
        title_mode = "stage" if legend_mode == "external" else "full"

        plotter.save_and_close_fig(
            fig=plotter.plot_qc_corr_heatmap(
                corr_matrix=diagnostics.qc_correlation,
                corr_mask=metadata["qc_mask"],
                batches=metadata["qc_batches"],
                method=metadata["correlation_method"],
                cluster="none",
                show_colorbar=legend_mode != "external",
                title_mode=title_mode,
            ),
            file_path=os.path.join(output_dir, "QC_Correlation_Heatmap"),
            save_format=plotter.QA_PANEL_SAVE_FORMAT,
            bbox_inches="tight",
            pad_inches=0.04,
        )

        plotter.save_and_close_fig(
            fig=plotter.plot_batch_corr_heatmap(
                batch_corr_matrix=diagnostics.batch_qc_correlation,
                method=metadata["correlation_method"],
                show_colorbar=legend_mode != "external",
                title_mode=title_mode,
            ),
            file_path=os.path.join(output_dir, "Batch_Correlation_Heatmap"),
            save_format=plotter.QA_PANEL_SAVE_FORMAT,
            bbox_inches="tight",
            pad_inches=0.04,
        )

        plotter.save_and_close_fig(
            fig=plotter.plot_pca_scatter(
                pca_df=diagnostics.pca["pca_scatter"],
                pca_var=diagnostics.pca["pca_variance"],
                pca_diagnostics=diagnostics.pca["diagnostics"],
                sample_type=metadata["sample_type"],
                batch=metadata["batch"],
                qc_label=metadata["qc_label"],
                actual_label=metadata["actual_label"],
                legend_mode=legend_mode,
                title_mode=title_mode,
            ),
            file_path=os.path.join(output_dir, "PCA_Scatter_QC_Sample"),
            save_format=plotter.QA_PANEL_SAVE_FORMAT,
            bbox_inches="tight",
            pad_inches=0.04,
        )

        plotter.save_and_close_fig(
            fig=plotter.plot_sd_od_scatter(
                metrics_df=diagnostics.pca["metrics_df"],
                sd_limit=diagnostics.pca["sd_limit"],
                od_limit=diagnostics.pca["od_limit"],
                is_flags=diagnostics.internal_standard_flags,
                orf_flags=diagnostics.outlier_reference_flags,
                show_legend=legend_mode == "local",
                legend_mode=legend_mode,
                title_mode=title_mode,
                annotate_thresholds=legend_mode == "external",
            ),
            file_path=os.path.join(output_dir, "Outlier_Scatter"),
            save_format=plotter.QA_PANEL_SAVE_FORMAT,
            bbox_inches="tight",
            pad_inches=0.04,
        )

        plotter.save_and_close_fig(
            fig=plotter.plot_rsd_bar(
                rsd_data=diagnostics.rsd_distribution,
                qc_label=metadata["qc_label"],
                actual_label=metadata["actual_label"],
                legend_mode=legend_mode,
                title_mode=title_mode,
            ),
            file_path=os.path.join(output_dir, "RSD_Barplot"),
            save_format=plotter.QA_PANEL_SAVE_FORMAT,
            bbox_inches="tight",
            pad_inches=0.04,
        )

        if legend_mode == "external":
            corr_legend_prefix = (
                "Batch_Correlation_Heatmap"
                if processor.attrs.get("is_multi_batch", False)
                else "QC_Correlation_Heatmap"
            )
            plotter.save_and_close_fig(
                fig=plotter.plot_correlation_colorbar_legend(
                    method=metadata["correlation_method"]
                ),
                file_path=os.path.join(
                    output_dir,
                    f"{corr_legend_prefix}_Legend",
                ),
                save_format=plotter.QA_LEGEND_SAVE_FORMAT,
                bbox_inches="tight",
                pad_inches=0.03,
            )
            plotter.save_and_close_fig(
                fig=plotter.plot_rsd_standalone_legend(
                    qc_label=metadata["qc_label"],
                    actual_label=metadata["actual_label"],
                ),
                file_path=os.path.join(output_dir, "RSD_Barplot_Legend"),
                save_format=plotter.QA_LEGEND_SAVE_FORMAT,
                bbox_inches="tight",
                pad_inches=0.03,
            )
            plotter.save_and_close_fig(
                fig=plotter.plot_pca_diagnostics_legend(
                    pca_df=diagnostics.pca["pca_scatter"],
                    sample_type=metadata["sample_type"],
                    batch=metadata["batch"],
                    qc_label=metadata["qc_label"],
                    actual_label=metadata["actual_label"],
                ),
                file_path=os.path.join(
                    output_dir,
                    "PCA_Scatter_QC_Sample_Legend",
                ),
                save_format=plotter.QA_LEGEND_SAVE_FORMAT,
                bbox_inches="tight",
                pad_inches=0.03,
            )
            plotter.save_and_close_fig(
                fig=plotter.plot_outlier_standalone_legend(
                    metrics_df=diagnostics.pca["metrics_df"],
                    sd_limit=diagnostics.pca["sd_limit"],
                    od_limit=diagnostics.pca["od_limit"],
                    is_flags=diagnostics.internal_standard_flags,
                    orf_flags=diagnostics.outlier_reference_flags,
                    complete_categories=True,
                    include_bar_diagnostics=False,
                    include_thresholds=False,
                ),
                file_path=os.path.join(
                    output_dir,
                    "Outlier_Scatter_Legend",
                ),
                save_format=plotter.QA_LEGEND_SAVE_FORMAT,
                bbox_inches="tight",
                pad_inches=0.03,
            )

        if metadata["valid_is"]:
            is_grid = plotter.plot_ref_shewhart_chart(
                ref_data=diagnostics.internal_standard_data,
                valid_feats=metadata["valid_is"],
                sample_type=metadata["sample_type"],
                batch=metadata["batch"],
                inject_order=metadata["inject_order"],
                qc_label=metadata["qc_label"],
                actual_label=metadata["actual_label"],
                bound_type=metadata["boundary_type"],
                ref_type="IS",
            )
            plotter.save_and_show_pw(
                pw_obj=is_grid,
                show_plot=False,
                file_path=os.path.join(output_dir, "IS_Shewhart_Chart"),
            )

        if metadata["valid_orf"]:
            orf_grid = plotter.plot_ref_shewhart_chart(
                ref_data=diagnostics.outlier_reference_data,
                valid_feats=metadata["valid_orf"],
                sample_type=metadata["sample_type"],
                batch=metadata["batch"],
                inject_order=metadata["inject_order"],
                qc_label=metadata["qc_label"],
                actual_label=metadata["actual_label"],
                bound_type=metadata["boundary_type"],
                ref_type="ORF",
            )
            plotter.save_and_show_pw(
                pw_obj=orf_grid,
                show_plot=False,
                file_path=os.path.join(output_dir, "ORF_Shewhart_Chart"),
            )

        dashboard = plotter.plot_assessment_dashboard(
            pca_res=diagnostics.pca,
            rsd_data=diagnostics.rsd_distribution,
            batch_corr=diagnostics.batch_qc_correlation,
            corr_mat=diagnostics.qc_correlation,
            qc_mask=metadata["qc_mask"],
            batches=metadata["qc_batches"],
            method=metadata["correlation_method"],
            sample_type=metadata["sample_type"],
            batch=metadata["batch"],
            qc_label=metadata["qc_label"],
            actual_label=metadata["actual_label"],
            is_flags=diagnostics.internal_standard_flags,
            orf_flags=diagnostics.outlier_reference_flags,
        )
        grid_path = os.path.join(output_dir, "QA_Summary_Dashboard.svg")
        plotter.save_and_show_pw(pw_obj=dashboard, file_path=grid_path)

        logger.info(f"Assessor summary dashboard saved as: {grid_path}")
        logger.success("Data quality assessment completed.")
