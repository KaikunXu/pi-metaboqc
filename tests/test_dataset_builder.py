"""Pytest module for dataset_builder.py using real project data."""

import pytest
import pandas as pd
from pimqc.dataset_builder import build_dataset
from pimqc.core_classes import MetaboInt

def test_build_dataset_with_real_data(real_project_data):
    """Test building MetaboInt object using actual project files."""
    meta_df, int_df, params = real_project_data
    
    obj = build_dataset(
        meta_info=meta_df,
        int_df=int_df,
        pipeline_params=params,
        mode=params["MetaboInt"]["mode"],
        sample_name=params["MetaboInt"]["sample_name"],
        sample_type=params["MetaboInt"]["sample_type"]
    )
    
    assert isinstance(obj, MetaboInt)
    assert not obj.empty
    assert params["MetaboInt"]["batch"] in obj.columns.names
    assert params["MetaboInt"]["sample_type"] in obj.columns.names