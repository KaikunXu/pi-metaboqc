# tests/test_dataset_builder.py
"""
Script purpose: Verify dataset construction with bundled project data.

This module exercises build_dataset() with the real demo metadata, intensity
matrix, and pipeline configuration loaded by the shared fixture. The test
checks that a non-empty MetaboInt object is produced and that configured sample
annotation levels are present in the resulting column MultiIndex.
"""

import pandas as pd

from pimqc.dataset_builder import build_dataset
from pimqc.core_classes import MetaboInt


def test_build_dataset_with_real_data(
    real_project_data: tuple[pd.DataFrame, pd.DataFrame, dict[str, object]],
) -> None:
    """Test building MetaboInt object using actual project files."""
    meta_df, int_df, params = real_project_data

    obj = build_dataset(meta_info=meta_df, int_df=int_df, pipeline_params=params)

    assert isinstance(obj, MetaboInt)
    assert not obj.empty
    assert params["MetaboInt"]["batch"] in obj.columns.names
    assert params["MetaboInt"]["sample_type"] in obj.columns.names
