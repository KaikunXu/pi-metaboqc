# tests/test_pipeline.py
"""
End-to-end tests for the pi-metaboqc pipeline.

This module validates the execution of the entire metabolomics data 
quality control workflow using a realistically scaled synthetic dataset, 
ensuring structural transformations and proper I/O isolation.
"""

import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from src.pimqc.pipeline import run_pipeline
from src.pimqc.core_classes import MetaboInt


@pytest.fixture
def dummy_pipeline_data():
    """Generate realistically scaled synthetic data and pipeline parameters.
    
    Constructs a 2-batch MS experiment with stable baseline variance.
    Injects extreme artifacts (blank > QC, RSD > 100%, Global MV > 85%) 
    to robustly trigger all filtering nodes without erasing normal features.
    """
    np.random.seed(42)

    # ====================================================================
    # 1. Synthesize Metadata (60 Samples: 40 Bio, 10 QC, 10 Blank)
    # ====================================================================
    meta_records = []
    sample_idx = 1
    inj_order = 1

    for batch in ["Batch1", "Batch2"]:
        # Instrument equilibrium: 5 Blanks at the beginning
        for _ in range(5):
            meta_records.append(
                [f"S{sample_idx:03d}", "Blank", np.nan, batch, inj_order]
            )
            sample_idx += 1
            inj_order += 1
            
        # Initial QC
        meta_records.append(
            [f"S{sample_idx:03d}", "QC", np.nan, batch, inj_order]
        )
        sample_idx += 1
        inj_order += 1
        
        # Analytical run: 4 blocks of (5 Samples + 1 QC)
        for block in range(4):
            for s in range(5):
                grp = "GroupA" if (block * 5 + s) % 2 == 0 else "GroupB"
                meta_records.append(
                    [f"S{sample_idx:03d}", "Sample", grp, batch, inj_order]
                )
                sample_idx += 1
                inj_order += 1
                
            meta_records.append(
                [f"S{sample_idx:03d}", "QC", np.nan, batch, inj_order]
            )
            sample_idx += 1
            inj_order += 1

    df_meta = pd.DataFrame(
        meta_records, 
        columns=[
            "Sample Name", "Sample Type", "Bio Group", "Batch", "Inject Order"
        ]
    )

    # ====================================================================
    # 2. Base Intensity Matrix Initialization
    # ====================================================================
    n_samps = len(df_meta)  # Expected: 60
    features = [f"Feature_{i:03d}" for i in range(1, 101)]  # Expected: 100
    n_feats = len(features)

    # [CRITICAL FIX]: Reduce sigma to 0.1 so base RSD is ~10% (passes 0.3 tol)
    base_int = np.random.lognormal(
        mean=np.log(1e5), sigma=0.1, size=(n_feats, n_samps)
    )
    df_int = pd.DataFrame(
        base_int, index=features, columns=df_meta["Sample Name"]
    )

    # Boolean masks for vectorized operations
    is_blank = (df_meta["Sample Type"] == "Blank").values
    is_qc = (df_meta["Sample Type"] == "QC").values
    is_ga = (df_meta["Bio Group"] == "GroupA").values
    is_gb = (df_meta["Bio Group"] == "GroupB").values
    is_b1 = (df_meta["Batch"] == "Batch1").values
    is_b2 = (df_meta["Batch"] == "Batch2").values

    # Restrict blank signal to typical MS noise levels (~10^3) with low variance
    df_int.loc[:, is_blank] = np.random.lognormal(
        mean=np.log(1e3), sigma=0.1, size=(n_feats, sum(is_blank))
    )

    # ====================================================================
    # 3. Simulate Systemic Experimental Effects
    # ====================================================================
    # 3.1 Batch Effect & Inject Order Drift
    for b_mask, drift_rate, b_effect in [
        (is_b1, -0.005, 1.0), (is_b2, -0.008, 0.6)
    ]:
        io = df_meta.loc[b_mask, "Inject Order"].values
        drift_factor = np.exp(drift_rate * io) * b_effect
        df_int.loc[:, b_mask] = df_int.loc[:, b_mask] * drift_factor

    # 3.2 High Blank Contamination (Triggers Stage 2.1 Blank Filter)
    blank_fail_feats = features[10:15]  # Feature_011 to Feature_015
    qc_mean = df_int.loc[blank_fail_feats, is_qc].mean(axis=1).values
    # [CRITICAL FIX]: Blank extremely larger than QC (150%) to ensure it drops
    df_int.loc[blank_fail_feats, is_blank] = qc_mean[:, None] * 1.5

    # 3.3 High QC RSD (Triggers Stage 2.2 RSD Filter for MAR features)
    rsd_fail_feats = features[20:25]  # Feature_021 to Feature_025
    qc_mask_b1 = is_qc & is_b1
    # [CRITICAL FIX]: Extreme variance (up to 3x) to guarantee RSD failure
    df_int.loc[rsd_fail_feats, qc_mask_b1] *= np.random.uniform(
        0.1, 3.0, size=(5, sum(qc_mask_b1))
    )

    # ====================================================================
    # 4. Simulate Missing Value Topologies (MAR & MNAR)
    # ====================================================================
    # 4.1 Missing At Random (MAR)
    mar_feats = features[30:50]
    for feat in mar_feats:
        drop_idx = np.random.choice(
            n_samps, int(n_samps * 0.15), replace=False
        )
        df_int.iloc[df_int.index.get_loc(feat), drop_idx] = np.nan

    # 4.2 Missing Not At Random (MNAR) - Biological LOD Truncation
    mnar_feats = features[60:70]
    for feat in mnar_feats:
        ga_idx = np.where(is_ga)[0]
        gb_idx = np.where(is_gb)[0]
        qc_idx = np.where(is_qc)[0]
        
        drop_ga = np.random.choice(
            ga_idx, int(len(ga_idx) * 0.90), replace=False
        )
        drop_gb = np.random.choice(
            gb_idx, int(len(gb_idx) * 0.05), replace=False
        )
        drop_qc = np.random.choice(
            qc_idx, int(len(qc_idx) * 0.50), replace=False
        )
        
        all_drops = np.concatenate([drop_ga, drop_gb, drop_qc])
        df_int.iloc[df_int.index.get_loc(feat), all_drops] = np.nan
        
        rem_ga = list(set(ga_idx) - set(drop_ga))
        if rem_ga:
            df_int.iloc[df_int.index.get_loc(feat), rem_ga] *= 0.05

    # 4.3 High Global Missingness (Triggers Stage 1 MV Filter)
    high_mv_feats = features[80:85]  # Feature_081 to Feature_085
    for feat in high_mv_feats:
        drop_idx = np.random.choice(
            n_samps, int(n_samps * 0.85), replace=False
        )
        df_int.iloc[df_int.index.get_loc(feat), drop_idx] = np.nan

    # ====================================================================
    # 5. Define Pipeline Parameters (Single Source of Truth)
    # ====================================================================
    params = {
        "MetaboInt": {
            "mode": "POS",
            "batch": "Batch",
            "sample_type": "Sample Type",
            "bio_group": "Bio Group",
            "sample_name": "Sample Name",
            "inject_order": "Inject Order",
            "sample_dict": {"QC sample": "QC", "Blank sample": "Blank"}
        },
        "MetaboIntFilter": {
            "mv_group_tol": 0.8,
            "mv_qc_tol": 0.8,
            "mv_global_tol": 0.8,
            "blank_qc_ratio": 0.8,  # Raised to 0.8 to be robust against Log2 transforms
            "rsd_qc_tol": 0.3
        },
        "MetaboIntCorrector": {
            "corr_method": "svr"
        },
        "MetaboIntImputer": {
            "method": "probabilistic",
            "knn_neighbors": 3
        },
        "MetaboIntNormalizer": {
            "col_method": "median",
            "row_method": "auto_scale"
        }
    }
    
    return df_meta, df_int, params


# ========================================================================
# End-to-End Test (Execution with I/O Isolation)
# ========================================================================
@patch("src.pimqc.pipeline.iu._load_json_file")
@patch("src.pimqc.pipeline.iu._check_dir_exists")
@patch("src.pimqc.pipeline.iu._zip_folder")
@patch("pandas.DataFrame.to_csv")
@patch("src.pimqc.visualizer_classes.BaseMetaboVisualizer.save_and_close_fig")
@patch("src.pimqc.visualizer_classes.BaseMetaboVisualizer.save_and_show_pw")
def test_run_pipeline_e2e(
    mock_save_pw, 
    mock_save_fig, 
    mock_to_csv, 
    mock_zip_folder, 
    mock_check_dir, 
    mock_load_json, 
    dummy_pipeline_data
):
    """Test the full pi-metaboqc pipeline execution without side effects."""
    
    # Unpack synthetic data
    meta_df, int_df, mock_params = dummy_pipeline_data
    
    # Mock the JSON loader to return our synthetic dictionary
    mock_load_json.return_value = mock_params
    
    # Execute the pipeline using a dummy directory
    output_dir = "dummy_output_directory"
    final_data = run_pipeline(
        meta_df=meta_df,
        int_df=int_df,
        params_path="dummy_path/pipeline_parameters.json",
        output_dir=output_dir,
        compress_output=True
    )
    
    # ==========================================
    # Assertions
    # ==========================================
    
    # 1. Structural Verification: Output should be a valid MetaboInt instance
    assert final_data is not None, "Pipeline failed to return a dataset."
    assert isinstance(final_data, MetaboInt), "Output is not a MetaboInt object."
    
    # 2. Mathematical Invariant: No missing values should remain post-imputation
    assert final_data.isna().sum().sum() == 0, "Missing values detected in output."
    
    # 3. Filtering Verification: Ensure artifact features were removed
    # Feature_081 was injected with >85% global MV (Stage 1 drop)
    assert "Feature_081" not in final_data.index, "Stage 1 MV filter failed."
    
    # Feature_011 was injected with 150% Blank Contamination (Stage 2 drop)
    assert "Feature_011" not in final_data.index, "Stage 2 Blank filter failed."
    
    # Feature_021 was injected with high variance in B1 QC (Stage 2 drop)
    assert "Feature_021" not in final_data.index, "Stage 2 RSD filter failed."
    
    # Ensure normal features survive
    assert "Feature_001" in final_data.index, "Normal feature was incorrectly filtered."
    
    # 4. Pipeline Finalization: Compression must be triggered
    mock_zip_folder.assert_called_once()
    
    # 5. Output Integrity: Ensure visual and tabular reports were generated
    assert mock_to_csv.call_count > 0, "CSV exports were bypassed."
    assert mock_save_fig.call_count > 0, "Plot generation was bypassed."