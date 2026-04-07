"""Pytest module for MetaboIntQA using real project data."""

import os
import pytest
from pathlib import Path
from pimqc.dataset_builder import build_dataset
from pimqc.data_quality_assessment import MetaboIntQA

@pytest.fixture
def real_qa_object(real_project_data):
    """Construct a MetaboIntQA object from real project data."""
    meta_df, int_df, params = real_project_data
    
    base_obj = build_dataset(
        meta_info=meta_df,
        int_df=int_df,
        pipeline_params=params,
        mode=params["MetaboInt"]["mode"]
    )
    return MetaboIntQA(base_obj, pipeline_params=params)

def test_real_data_pca_results(real_qa_object):
    """Verify PCA results on real project data."""
    qa = real_qa_object
    results = qa.pca_results
    
    assert "pca_scatter" in results
    assert "outliers" in results
    assert not results["pca_scatter"].isnull().all().all()

def test_real_data_execute_qa_pipeline(real_qa_object, tmp_path):
    """Test full QA pipeline execution with real data and file I/O."""
    qa = real_qa_object
    output_dir = tmp_path / "real_qa_results"
    
    qa.execute_qa(output_dir=str(output_dir))
    
    mode = qa.attrs.get("mode", "POS")
    expected_files = [
        f"PCA_Scatter_{mode}.csv",
        f"QC_Corr_Heatmap_{mode}.pdf",
        f"QC_AS_PCA_Scatter_{mode}.pdf"
    ]
    
    for file_name in expected_files:
        assert (output_dir / file_name).exists()