"""Tests for the structured report input and Markdown rendering contract."""

import json
from pathlib import Path

from pimqc.reporting import NarrativeStatsReporter, ReportInput


def _report_input() -> ReportInput:
    """Return the smallest complete report input accepted by both templates."""
    return ReportInput(
        pipeline_metrics={
            "raw_dataset": {
                "mode": "POS",
                "pi-metaboqc_version": "test-version",
                "features": {"total": 2, "internal_standards": []},
                "samples": {"total": 3, "qc": 1, "blank": 1, "actual": 1},
                "batches": {"batch_count": 1, "batch_distribution": {}},
            },
            "high_mv_feature_filtering": {
                "sample_wise": {
                    "thresholds": {"sample_mv_tol": 0.42},
                    "feature_retention": {
                        "total_checked": 3,
                        "retained_count": 3,
                        "retention_rate_pct": 100.0,
                    },
                },
                "feature_wise": {
                    "filtering_level": "Group",
                    "thresholds": {"mnar_group_mv_tol": 0.73},
                    "missing_classification": {
                        "mar_count": 1,
                        "mnar_total": 1,
                    },
                    "feature_retention": {
                        "pre_mv_filter_count": 2,
                        "retention_rate_pct": 100.0,
                    },
                },
            },
            "signal_correction": {
                "correction_status": "Completed",
                "overall_performance": {},
                "stages_executed": [
                    {
                        "stage_name": "Global correction",
                        "algorithm": "LOESS",
                        "parameters": {"loess_span": 0.61},
                    }
                ],
            },
            "low_quality_feature_filtering": {
                "feature_retention": {
                    "pre_stage2": {"mar_count": 1, "mnar_count": 1},
                    "post_blank_check": {
                        "mar_count": 1,
                        "mnar_count": 1,
                    },
                    "post_rsd_check": {
                        "mar_count": 1,
                        "mnar_count": 1,
                    },
                },
                "thresholds": {"qc_rsd_tol": 0.25},
                "filtering_breakdown": {},
            },
            "missing_value_imputation": {
                "imputation_status": "Completed",
                "strategies": {
                    "mnar_method": "row-wise",
                    "mnar_fraction": 0.5,
                },
                "feature_distribution": {"mar_count": 1, "mnar_count": 1},
                "selection": {
                    "requested_method": "KNN",
                    "selected_method": "KNN",
                    "selected_label": "KNN",
                    "is_auto": False,
                    "candidate_results": [
                        {
                            "method": "KNN",
                            "selected": True,
                            "status": "ok",
                            "nrmse_low": 0.12,
                            "nrmse_total": 0.08,
                            "jsd_total": 0.125,
                            "wasserstein_normalized": 0.375,
                        }
                    ],
                },
            },
            "normalization": {
                "strategies": {
                    "normalization_method": "QUANTILE",
                    "log_transform_active": True,
                },
                "selection": {
                    "requested_method": "QUANTILE",
                    "selected_method": "QUANTILE",
                    "selected_label": "QUANTILE",
                    "is_auto": False,
                },
            },
        },
        qa_metrics={},
        metadata={"date": "2026-08-18 12:00", "mode": "POS"},
        resolved_config={"MetaboInt": {"mode": "POS"}},
        asset_manifest={"pca": "assets/02_PCA_Scatter_Dashboard.svg"},
    )


def test_report_input_writes_a_portable_json_snapshot(tmp_path: Path) -> None:
    """Persist report state without retaining DataFrame or processor objects."""
    output_path = _report_input().write_json(tmp_path / "Report_Input.json")

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["metadata"]["mode"] == "POS"
    assert payload["resolved_config"]["MetaboInt"]["mode"] == "POS"
    assert payload["asset_manifest"]["pca"].endswith(".svg")


def test_reporter_renders_markdown_from_report_input(tmp_path: Path) -> None:
    """Use the structured report contract as the only narrative input."""
    reporter = NarrativeStatsReporter(base_dir=str(tmp_path))

    reporter.generate_markdown(_report_input(), report_folder="report")

    report_dir = tmp_path / "report"
    comprehensive = (report_dir / "Report_Comprehensive.md").read_text(
        encoding="utf-8"
    )
    assert (report_dir / "Report_Input.json").is_file()
    assert "Ionization Mode:** POS" in comprehensive
    assert "test-version" in comprehensive
    assert "0.42" in comprehensive
    assert "LOESS" in comprehensive
    assert "KNN" in comprehensive
    assert "QUANTILE" in comprehensive
    assert "**Deterministic Substitution for MNAR Features**\n\nFor" in comprehensive
    assert "**Configured MAR Imputation Method**\n\nFor" in comprehensive
    assert "Jensen-Shannon distance of **0.125**" in comprehensive
    assert "normalized Wasserstein distance of **0.375**" in comprehensive


def test_imputation_dashboards_follow_selection_evidence(tmp_path: Path) -> None:
    """Place candidate dashboard before the selected-method dashboard."""
    report_input = _report_input()
    selection = report_input.pipeline_metrics["missing_value_imputation"][
        "selection"
    ]
    selection["requested_method"] = "AUTO"
    selection["is_auto"] = True

    reporter = NarrativeStatsReporter(base_dir=str(tmp_path))
    reporter.generate_markdown(report_input, report_folder="report")

    comprehensive = (tmp_path / "report" / "Report_Comprehensive.md").read_text(
        encoding="utf-8"
    )
    assert comprehensive.index("Masked-Value Distribution Fidelity") < (
        comprehensive.index("MAR Imputation Candidate Comparison")
    )
    assert comprehensive.index("MAR Imputation Candidate Comparison") < (
        comprehensive.index("MAR Imputation Dashboard: KNN")
    )
    assert comprehensive.index("MAR Imputation Candidate Dashboard") < (
        comprehensive.index("MAR Imputation Dashboard: KNN")
    )
    assert "Imputation_Candidate_Dashboard_KNN.svg" in comprehensive
    assert "KNN (selected)" in comprehensive


def test_auto_normalization_renders_candidate_comparison(tmp_path: Path) -> None:
    """Render a compact AUTO comparison from the unified candidate contract."""
    report_input = _report_input()
    selection = report_input.pipeline_metrics["normalization"]["selection"]
    selection.update(
        {
            "requested_method": "AUTO",
            "is_auto": True,
            "selected_score": 0.71,
            "selection_margin": 0.05,
            "candidate_results": [
                {
                    "method": "ROBUST_LOG_ONLY",
                    "selected": False,
                    "status": "ok",
                    "overall_score": 0.5,
                    "rle_alignment_change_score": 0.5,
                    "variance_stabilization_score": 0.5,
                    "qc_structure_change_score": 0.5,
                    "sample_structure_score": 0.5,
                },
                {
                    "method": "PQN",
                    "selected": True,
                    "status": "ok",
                    "overall_score": 0.71,
                    "rle_alignment_change_score": 1.0,
                    "variance_stabilization_score": 0.5,
                    "qc_structure_change_score": 0.86,
                    "sample_structure_score": 0.46,
                },
            ],
        }
    )

    reporter = NarrativeStatsReporter(base_dir=str(tmp_path))
    reporter.generate_markdown(report_input, report_folder="report")

    comprehensive = (tmp_path / "report" / "Report_Comprehensive.md").read_text(
        encoding="utf-8"
    )
    assert "Normalization Candidate Comparison" in comprehensive
    assert "PQN (selected)" in comprehensive


def test_reporter_does_not_render_missing_metrics_as_zero(tmp_path: Path) -> None:
    """Show unavailable correction and VSN metrics without numeric fallbacks."""
    report_input = _report_input()
    report_input.pipeline_metrics["signal_correction"]["stages_executed"].append(
        {
            "stage_name": "Final correction",
            "algorithm": "LOESS",
            "parameters": {},
        }
    )
    report_input.pipeline_metrics["normalization"]["strategies"][
        "normalization_method"
    ] = "VSN"

    reporter = NarrativeStatsReporter(base_dir=str(tmp_path))
    reporter.generate_markdown(report_input, report_folder="report")

    report_dir = tmp_path / "report"
    brief = (report_dir / "Report_Brief.md").read_text(encoding="utf-8")
    comprehensive = (report_dir / "Report_Comprehensive.md").read_text(
        encoding="utf-8"
    )

    assert "QC RSD metrics are unavailable" in brief
    assert "without a reportable QC RSD summary" in comprehensive
    assert "0.000e+00" not in comprehensive
    assert "structural scale factor of **N/A**" in comprehensive
