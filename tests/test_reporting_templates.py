"""Tests for package-owned report template discovery."""

from pimqc.reporting.utils import NarrativeStatsReporter


def test_report_templates_load_from_the_package_root(tmp_path) -> None:
    reporter = NarrativeStatsReporter(base_dir=str(tmp_path))

    assert reporter.env.get_template("report_brief.md.j2") is not None
    assert reporter.env.get_template("report_comprehensive.md.j2") is not None
