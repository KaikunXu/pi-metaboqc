"""Validate the demo data, configuration files, and report templates.

These checks cross package boundaries: they load distributed files through the
public configuration and dataset APIs, then verify that report templates remain
discoverable from an arbitrary output directory. Keeping them together makes
resource packaging failures distinguishable from isolated parser tests.
"""

from pathlib import Path

import pandas as pd

from pimqc.core import MetaboInt
from pimqc.dataset.builder import build_dataset
from pimqc.io import load_pipeline_config
from pimqc.reporting.utils import NarrativeStatsReporter


def test_demo_json_configuration_matches_toml() -> None:
    """Load both distributed formats into the same validated mapping."""
    demo_directory = (
        Path(__file__).parents[2] / "src" / "pimqc" / "resources" / "demo"
    )

    toml_config = load_pipeline_config(
        str(demo_directory / "pipeline_parameters.toml")
    )
    json_config = load_pipeline_config(
        str(demo_directory / "pipeline_parameters.json")
    )

    assert json_config == toml_config


def test_build_dataset_with_bundled_project_data(
    real_project_data: tuple[pd.DataFrame, pd.DataFrame, dict[str, object]],
) -> None:
    """Build a populated MetaboInt with every configured metadata level."""
    metadata, intensity, parameters = real_project_data

    dataset = build_dataset(
        meta_info=metadata,
        int_df=intensity,
        pipeline_params=parameters,
    )

    assert isinstance(dataset, MetaboInt)
    assert not dataset.empty
    assert parameters["MetaboInt"]["batch"] in dataset.columns.names
    assert parameters["MetaboInt"]["sample_type"] in dataset.columns.names


def test_report_templates_load_from_package_resources(tmp_path: Path) -> None:
    """Resolve both templates independently of the current output directory."""
    reporter = NarrativeStatsReporter(base_dir=str(tmp_path))

    assert reporter.env.get_template("report_brief.md.j2") is not None
    assert reporter.env.get_template("report_comprehensive.md.j2") is not None
