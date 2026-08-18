"""Regression tests for the assessment StageRunner lifecycle."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from pimqc.processing.assessment import (
    AssessmentDiagnostics,
    AssessmentStageRunner,
    MetaboIntAssessor,
)


def _prepared_assessor() -> tuple[MetaboIntAssessor, pd.DataFrame]:
    """Build an assessor with deterministic cached diagnostic calculations."""
    columns = pd.MultiIndex.from_tuples(
        [
            ("QC1", "QC", "B1", 1),
            ("QC2", "QC", "B1", 2),
            ("S1", "Sample", "B1", 3),
            ("S2", "Sample", "B1", 4),
        ],
        names=["Sample Name", "Sample Type", "Batch", "Inject Order"],
    )
    assessor = MetaboIntAssessor(
        [[10.0, 11.0, 9.0, 12.0], [20.0, 19.0, 22.0, 21.0]],
        index=["F1", "F2"],
        columns=columns,
    )

    qc_columns = columns[:2]
    correlation = pd.DataFrame(
        np.eye(2),
        index=qc_columns,
        columns=qc_columns,
    )
    batch_correlation = pd.DataFrame([[1.0]], index=["B1"], columns=["B1"])
    metrics_df = pd.DataFrame(
        {
            "OD": [0.1, 0.2, 0.3, 0.4],
            "SD": [0.2, 0.3, 0.4, 0.5],
            "Category": ["Normal"] * 4,
            "is_od_outlier": [False] * 4,
            "is_sd_outlier": [False] * 4,
        },
        index=columns,
    )
    outliers = pd.DataFrame(
        {
            ("SPE-DModX", "SPE-DModX"): metrics_df["OD"],
            ("SPE-DModX", "Outliers (SPE-DModX)"): False,
            ("HT2", "Hotelling T2 Score"): metrics_df["SD"],
            ("HT2", "Outliers (HT2)"): False,
        },
        index=columns,
    )
    pca_result = {
        "pca_scatter": pd.DataFrame(
            {"PC1": [0.0, 0.1, 0.2, 0.3], "PC2": [0.3, 0.2, 0.1, 0.0]},
            index=columns,
        ),
        "pca_variance": pd.Series({"PC1": 0.6, "PC2": 0.3}),
        "outliers": outliers,
        "metrics_df": metrics_df,
        "sd_limit": 1.0,
        "od_limit": 1.0,
        "diagnostics": {
            "relative_dispersion": 0.1,
            "batch_silhouette": None,
            "centrality_shift": 0.2,
        },
    }

    assessor.__dict__["qc_corr_matrix"] = correlation
    assessor.__dict__["batch_qc_corr_matrix"] = batch_correlation
    assessor.__dict__["pca_res"] = pca_result
    assessor.__dict__["rsd_distribution"] = {
        "qc": {"0-10%": 2, "10-20%": 0, "20-30%": 0, ">30%": 0},
        "actual": {
            "0-10%": 2,
            "10-20%": 0,
            "20-30%": 0,
            ">30%": 0,
        },
    }
    assessor.__dict__["assessment_metrics"] = {"status": "prepared"}
    return assessor, outliers


def test_assessment_compute_only_has_no_artifact_side_effects() -> None:
    """An output-free run returns diagnostics without exporting or plotting."""
    assessor, expected_outliers = _prepared_assessor()

    with (
        patch.object(pd.DataFrame, "to_csv") as to_csv,
        patch(
            "pimqc.processing.assessment.runner.AssessmentPlotter"
        ) as visualizer,
    ):
        result = AssessmentStageRunner(assessor, output_dir=None).run()

    assert isinstance(result.data, AssessmentDiagnostics)
    pd.testing.assert_frame_equal(result.data.outliers, expected_outliers)
    assert result.metrics == {"status": "prepared"}
    to_csv.assert_not_called()
    visualizer.assert_not_called()


def test_run_without_output_returns_complete_stage_result() -> None:
    """The public API exposes diagnostics when artifact output is disabled."""
    assessor, expected_outliers = _prepared_assessor()

    with (
        patch.object(pd.DataFrame, "to_csv") as to_csv,
        patch(
            "pimqc.processing.assessment.runner.AssessmentPlotter"
        ) as visualizer,
    ):
        result = assessor.run_assessment()

    assert isinstance(result.data, AssessmentDiagnostics)
    pd.testing.assert_frame_equal(result.candidates, expected_outliers)
    assert result.metrics == {"status": "prepared"}
    assert result.metadata["skipped"] is False
    to_csv.assert_not_called()
    visualizer.assert_not_called()


def test_assessment_export_preserves_diagnostic_filename(
    tmp_path: Path,
) -> None:
    """The runner keeps the established QA CSV name and serialization options."""
    assessor, _ = _prepared_assessor()
    runner = AssessmentStageRunner(assessor, output_dir=tmp_path)
    runner.render = Mock()

    with patch.object(pd.DataFrame, "to_csv") as to_csv:
        runner.run()

    to_csv.assert_called_once_with(
        tmp_path / "QA_Diagnostics_Outliers.csv",
        encoding="utf-8-sig",
        na_rep="NA",
    )


def test_assessment_render_preserves_panel_and_dashboard_names(
    tmp_path: Path,
) -> None:
    """Rendering retains the established report-facing artifact names."""
    assessor, _ = _prepared_assessor()
    runner = AssessmentStageRunner(assessor, output_dir=tmp_path)
    result = runner.compute()
    result.render_context["processor"] = assessor
    visualizer = Mock()
    visualizer.QA_PANEL_SAVE_FORMAT = "svg"
    visualizer.QA_LEGEND_SAVE_FORMAT = "svg"
    visualizer._validate_legend_mode.return_value = "external"

    with patch(
        "pimqc.processing.assessment.runner.AssessmentPlotter",
        return_value=visualizer,
    ):
        runner.render(result)

    panel_names = {
        Path(call.kwargs["file_path"]).name
        for call in visualizer.save_and_close_fig.call_args_list
    }
    assert panel_names == {
        "QC_Correlation_Heatmap",
        "Batch_Correlation_Heatmap",
        "PCA_Scatter_QC_Sample",
        "Outlier_Scatter",
        "RSD_Barplot",
        "QC_Correlation_Heatmap_Legend",
        "RSD_Barplot_Legend",
        "PCA_Scatter_QC_Sample_Legend",
        "Outlier_Scatter_Legend",
    }
    visualizer.save_and_show_pw.assert_called_once()
    assert (
        Path(visualizer.save_and_show_pw.call_args.kwargs["file_path"]).name
        == "QA_Summary_Dashboard.svg"
    )


def test_empty_run_assessment_does_not_create_output_dir(
    tmp_path: Path,
) -> None:
    """The public runner avoids empty artifacts for an empty assessment."""
    assessor = MetaboIntAssessor(pd.DataFrame())
    output_dir = tmp_path / "not-created"

    result = assessor.run_assessment(
        output_dir=str(output_dir),
        corr_method="Pearson",
    )

    assert isinstance(result.data, AssessmentDiagnostics)
    assert result.metadata["skipped"] is True
    assert result.metrics == {}
    assert assessor.attrs["corr_method"] == "Pearson"
    assert not output_dir.exists()
