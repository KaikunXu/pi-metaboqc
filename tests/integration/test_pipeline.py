"""Exercise complete pipeline orchestration in memory and with artifacts.

The synthetic and bundled-data scenarios traverse all processing stages while
patching expensive file writers. These are deliberately slow integration tests
because their purpose is to validate stage composition, result contracts, and
artifact dispatch rather than individual numerical kernels.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from pimqc.pipeline import PipelineResult, run_pipeline

PipelineData = tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]
pytestmark = pytest.mark.slow


@patch("pimqc.pipeline.ensure_directory")
@patch("pandas.DataFrame.to_csv")
@patch("pimqc.plotting.base.BasePlotter.save_and_close_fig")
@patch("pimqc.plotting.base.BasePlotter.save_and_show_pw")
@patch("pimqc.pipeline.ru.NarrativeStatsReporter")
@patch("pimqc.pipeline.ru.VisualAssetReporter")
def test_run_pipeline_with_artifact_dispatch(
    mock_visual_reporter: Mock,
    mock_narrative_reporter: Mock,
    mock_save_dashboard: Mock,
    mock_save_figure: Mock,
    mock_to_csv: Mock,
    mock_ensure_directory: Mock,
    synthetic_pipeline_data: PipelineData,
    tmp_path: Path,
) -> None:
    """Return structured results and dispatch every artifact category."""
    metadata, intensity, parameters = synthetic_pipeline_data
    mock_ensure_directory.side_effect = lambda path: Path(path)

    result = run_pipeline(
        meta_df=metadata,
        int_df=intensity,
        params=parameters,
        output_dir=str(tmp_path / "synthetic_pipeline"),
    )

    assert isinstance(result, PipelineResult)
    assert result.data is result.stage_tables["normalized"]
    assert result.stage_results["normalization"].data is result.data
    assert result.assessments["normalization"].metrics
    assert (
        result.pipeline_metrics["normalization"]
        is result.stage_results["normalization"].metrics
    )
    assert result.report_input.pipeline_metrics is result.pipeline_metrics
    assert mock_to_csv.call_count > 0
    assert mock_save_figure.call_count > 0
    assert mock_narrative_reporter.called
    assert mock_visual_reporter.called
    assert mock_save_dashboard.call_count > 0


@patch("pimqc.pipeline.ensure_directory")
@patch("pandas.DataFrame.to_csv")
@patch("pimqc.plotting.base.BasePlotter.save_and_close_fig")
@patch("pimqc.plotting.base.BasePlotter.save_and_show_pw")
@patch("pimqc.pipeline.ru.NarrativeStatsReporter")
@patch("pimqc.pipeline.ru.VisualAssetReporter")
def test_run_pipeline_in_memory_has_no_artifact_side_effects(
    mock_visual_reporter: Mock,
    mock_narrative_reporter: Mock,
    mock_save_dashboard: Mock,
    mock_save_figure: Mock,
    mock_to_csv: Mock,
    mock_ensure_directory: Mock,
    synthetic_pipeline_data: PipelineData,
) -> None:
    """Return all pipeline products without creating filesystem artifacts."""
    metadata, intensity, parameters = synthetic_pipeline_data

    result = run_pipeline(
        meta_df=metadata,
        int_df=intensity,
        params=parameters,
        output_dir=None,
    )

    assert isinstance(result, PipelineResult)
    assert result.output_dir is None
    assert result.report_generated is False
    assert result.data is result.stage_tables["normalized"]
    assert result.assessments["raw_dataset"].data.qc_correlation.size > 0
    assert result.report_input.asset_manifest == {}
    mock_ensure_directory.assert_not_called()
    mock_to_csv.assert_not_called()
    mock_save_figure.assert_not_called()
    mock_save_dashboard.assert_not_called()
    mock_visual_reporter.assert_not_called()
    mock_narrative_reporter.assert_not_called()


@patch("pimqc.pipeline.ensure_directory")
@patch("pandas.DataFrame.to_csv")
@patch("pimqc.plotting.base.BasePlotter.save_and_close_fig")
@patch("pimqc.plotting.base.BasePlotter.save_and_show_pw")
@patch("pimqc.pipeline.ru.NarrativeStatsReporter")
@patch("pimqc.pipeline.ru.VisualAssetReporter")
def test_run_pipeline_with_bundled_project_data(
    mock_visual_reporter: Mock,
    mock_narrative_reporter: Mock,
    mock_save_dashboard: Mock,
    mock_save_figure: Mock,
    mock_to_csv: Mock,
    mock_ensure_directory: Mock,
    real_project_data: PipelineData,
    tmp_path: Path,
) -> None:
    """Confirm the complete pipeline accepts the distributed demo dataset."""
    metadata, intensity, parameters = real_project_data
    mock_ensure_directory.side_effect = lambda path: Path(path)

    result = run_pipeline(
        meta_df=metadata,
        int_df=intensity,
        params=parameters,
        output_dir=str(tmp_path / "real_pipeline"),
    )

    assert isinstance(result, PipelineResult)
    assert result.data is result.stage_tables["normalized"]
    assert mock_to_csv.call_count > 0
    assert mock_narrative_reporter.called
    assert mock_visual_reporter.called
    assert mock_save_figure.call_count > 0
    assert mock_save_dashboard.call_count > 0
    assert (
        result.pipeline_metrics["missing_value_imputation"]["selection"][
            "requested_method"
        ]
        == "Auto"
    )
    selected_method = result.pipeline_metrics["missing_value_imputation"][
        "selection"
    ]["selected_method"]
    candidate_results = result.pipeline_metrics["missing_value_imputation"][
        "selection"
    ]["candidate_results"]
    selected_result = next(
        candidate
        for candidate in candidate_results
        if candidate["method"] == selected_method
    )
    assert "jsd_total" in selected_result
    assert "wasserstein_normalized" in selected_result
