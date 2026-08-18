"""Provide test data shared across unit, integration, and reference suites.

The root configuration selects Matplotlib's non-interactive backend before
test modules import plotting code, so integration tests do not require a Tcl/Tk
desktop runtime. R and rpy2 initialization is scoped to
``tests/reference/conftest.py``.
"""

from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Assign suite markers from the functional test directory structure.

    Args:
        items: Collected pytest items to classify in place.
    """
    for item in items:
        path_parts = Path(str(item.path)).parts
        if "reference" in path_parts:
            item.add_marker(pytest.mark.reference)
        elif "integration" in path_parts:
            item.add_marker(pytest.mark.integration)
        elif "quality" in path_parts:
            item.add_marker(pytest.mark.quality)
        else:
            item.add_marker(pytest.mark.unit)


@pytest.fixture
def real_project_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Load isolated copies of the bundled demo tables and configuration.

    Function scope prevents a stage that mutates a dataframe in place from
    leaking state into a later integration or reference test.
    """
    from pimqc.io import load_pipeline_config

    data_dir = (
        Path(__file__).parents[1] / "src" / "pimqc" / "resources" / "demo"
    )

    meta_path = data_dir / "project_meta.csv"
    int_path = data_dir / "project_intensity.csv"
    param_path = data_dir / "pipeline_parameters.toml"

    meta_df = pd.read_csv(meta_path)
    int_df = pd.read_csv(int_path, index_col=0)

    pipeline_params = load_pipeline_config(str(param_path))

    return meta_df, int_df, pipeline_params
