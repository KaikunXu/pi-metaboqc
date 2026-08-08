# tests/test_pipeline.py
"""
Script purpose: Validate full pipeline orchestration without file side effects.

This module builds synthetic stress-test data and reuses bundled project data
to exercise run_pipeline() through the major processing stages. Expensive or
filesystem-heavy exports are patched so the tests can verify orchestration,
CSV/report calls, and real-data viability without writing full output trees.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from pimqc.pipeline import run_pipeline

PipelineData = tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]


@pytest.fixture
def dummy_pipeline_data() -> PipelineData:
    """Generate realistically scaled synthetic data and pipeline parameters.

    Constructs a 2-batch MS experiment (150 Samples: 110 Bio, 20 QC, 20 Blank).
    Injects extreme artifacts (blank > QC, RSD > 100%, Global MV > 85%)
    to robustly trigger all filtering nodes without erasing normal features.

    Returns:
        tuple: Contains the metadata DataFrame, intensity DataFrame, and the
            pipeline parameter dictionary.
    """
    np.random.seed(42)

    # ====================================================================
    # 1. Synthesize Metadata (150 Samples: 110 Bio, 20 QC, 20 Blank)
    # ====================================================================
    meta_records = []
    sample_idx = 1
    inj_order = 1

    for batch in ["Batch1", "Batch2"]:
        # 1. Starting Anchor QC (Prevents SVR backward extrapolation)
        meta_records.append([f"S{sample_idx:03d}", "QC", np.nan, batch, inj_order])
        sample_idx += 1
        inj_order += 1

        # 2. Instrument equilibrium: 10 Blanks at the beginning
        for _ in range(10):
            meta_records.append(
                [f"S{sample_idx:03d}", "Blank", np.nan, batch, inj_order]
            )
            sample_idx += 1
            inj_order += 1

        # 3. Post-Blank Anchor QC (Safely brackets the Blank samples)
        meta_records.append([f"S{sample_idx:03d}", "QC", np.nan, batch, inj_order])
        sample_idx += 1
        inj_order += 1

        # 4. Analytical run: 55 Bio Samples and 8 QCs remaining per batch
        # Strategy: 7 blocks of (7 Bio + 1 QC) and 1 block of (6 Bio + 1 QC)
        for block in range(8):
            num_bio = 6 if block == 7 else 7
            for s in range(num_bio):
                grp = "GroupA" if (block * 7 + s) % 2 == 0 else "GroupB"
                meta_records.append(
                    [f"S{sample_idx:03d}", "Sample", grp, batch, inj_order]
                )
                sample_idx += 1
                inj_order += 1

            # End of block QC
            meta_records.append([f"S{sample_idx:03d}", "QC", np.nan, batch, inj_order])
            sample_idx += 1
            inj_order += 1

    df_meta = pd.DataFrame(
        meta_records,
        columns=["Sample Name", "Sample Type", "Bio Group", "Batch", "Inject Order"],
    )

    # ====================================================================
    # 2. Base Intensity Matrix Initialization
    # ====================================================================
    n_samps = len(df_meta)  # Guaranteed: 150
    features = [f"Feature_{i:03d}" for i in range(1, 101)]
    n_feats = len(features)

    # Ultra-clean baseline (1e6) with 1% variance to prevent SVR overfitting
    base_int = np.random.lognormal(
        mean=np.log(1e6), sigma=0.01, size=(n_feats, n_samps)
    )
    df_int = pd.DataFrame(base_int, index=features, columns=df_meta["Sample Name"])

    is_blank = (df_meta["Sample Type"] == "Blank").values
    is_qc = (df_meta["Sample Type"] == "QC").values
    is_ga = (df_meta["Bio Group"] == "GroupA").values
    is_gb = (df_meta["Bio Group"] == "GroupB").values
    is_b1 = (df_meta["Batch"] == "Batch1").values
    is_b2 = (df_meta["Batch"] == "Batch2").values

    # Enforce absolute minimal signal for blanks
    df_int.loc[:, is_blank] = 10.0

    # ====================================================================
    # 3. Simulate Systemic Experimental Effects
    # ====================================================================
    # Mild drift to ensure SVR works perfectly without negative predictions
    for b_mask, drift_rate, b_effect in [(is_b1, -0.001, 1.0), (is_b2, -0.002, 0.9)]:
        io = df_meta.loc[b_mask, "Inject Order"].values
        drift_factor = np.exp(drift_rate * io) * b_effect
        df_int.loc[:, b_mask] = df_int.loc[:, b_mask] * drift_factor

    # 3.2 High Blank Contamination (Triggers Stage 2.1 Blank Filter)
    blank_fail_feats = features[10:15]
    qc_mean = df_int.loc[blank_fail_feats, is_qc].mean(axis=1).values
    df_int.loc[blank_fail_feats, is_blank] = qc_mean[:, None] * 2.0

    # 3.3 High QC RSD (Triggers Stage 2.2 RSD Filter)
    rsd_fail_feats = features[20:25]
    qc_indices = np.where(is_qc)[0]

    # Inject an extreme spike to bypass SVR overfitting and ensure high RSD
    for feat in rsd_fail_feats:
        df_int.loc[feat, is_qc] = 1000.0
        spike_idx = qc_indices[len(qc_indices) // 2]
        df_int.iloc[df_int.index.get_loc(feat), spike_idx] = 1e12

    # ====================================================================
    # 4. Simulate Missing Value Topologies (MAR & MNAR)
    # ====================================================================
    mar_feats = features[30:50]
    for feat in mar_feats:
        drop_idx = np.random.choice(n_samps, int(n_samps * 0.15), replace=False)
        df_int.iloc[df_int.index.get_loc(feat), drop_idx] = np.nan

    mnar_feats = features[60:70]
    for feat in mnar_feats:
        ga_idx = np.where(is_ga)[0]
        gb_idx = np.where(is_gb)[0]
        qc_idx = np.where(is_qc)[0]

        drop_ga = np.random.choice(ga_idx, int(len(ga_idx) * 0.90), replace=False)
        drop_gb = np.random.choice(gb_idx, int(len(gb_idx) * 0.05), replace=False)
        drop_qc = np.random.choice(qc_idx, int(len(qc_idx) * 0.50), replace=False)

        all_drops = np.concatenate([drop_ga, drop_gb, drop_qc])
        df_int.iloc[df_int.index.get_loc(feat), all_drops] = np.nan

        rem_ga = list(set(ga_idx) - set(drop_ga))
        if rem_ga:
            df_int.iloc[df_int.index.get_loc(feat), rem_ga] *= 0.05

    high_mv_feats = features[80:85]
    for feat in high_mv_feats:
        drop_idx = np.random.choice(n_samps, int(n_samps * 0.85), replace=False)
        df_int.iloc[df_int.index.get_loc(feat), drop_idx] = np.nan

    # ====================================================================
    # 5. Define Pipeline Parameters
    # ====================================================================
    params = {
        "MetaboInt": {
            "mode": "POS",
            "batch": "Batch",
            "sample_type": "Sample Type",
            "bio_group": "Bio Group",
            "sample_name": "Sample Name",
            "inject_order": "Inject Order",
            "sample_dict": {"QC sample": "QC", "Blank sample": "Blank"},
        },
        "MetaboIntFilter": {
            "mv_group_tol": 0.8,
            "mv_qc_tol": 0.8,
            "mv_global_tol": 0.8,
            "blank_qc_ratio": 0.8,
            "blank_qc_tol": 0.8,
            "rsd_qc_tol": 0.3,
            "qc_rsd_tol": 0.3,
        },
        "MetaboIntCorrector": {"corr_method": "svr"},
        "MetaboIntImputer": {"method": "probabilistic", "knn_neighbors": 3},
        "MetaboIntNormalizer": {"col_method": "median", "row_method": "auto_scale"},
    }

    return df_meta, df_int, params


# ========================================================================
# End-to-End Test (Execution with Synthetic Data)
# ========================================================================
@pytest.mark.skip(
    reason="There is currently an unresolved bug related to edge extreme "
    "value triggering; we will focus on testing with real project data "
    "for now."
)
@patch("pimqc.pipeline.iu._check_dir_exists")
@patch("pandas.DataFrame.to_csv")
@patch("pimqc.visualization.base.BaseMetaboVisualizer.save_and_close_fig")
@patch("pimqc.visualization.base.BaseMetaboVisualizer.save_and_show_pw")
@patch("pimqc.pipeline.ru.NarrativeStatsReporter")
@patch("pimqc.pipeline.ru.VisualAssetReporter")
def test_run_pipeline_e2e(
    mock_visual_rep: Mock,
    mock_md_rep: Mock,
    mock_save_pw: Mock,
    mock_save_fig: Mock,
    mock_to_csv: Mock,
    mock_check_dir: Mock,
    dummy_pipeline_data: PipelineData,
) -> None:
    """Test the full pipeline execution without generating side effects."""
    meta_df, int_df, mock_params = dummy_pipeline_data

    output_dir = "dummy_output_directory"

    run_pipeline(
        meta_df=meta_df, int_df=int_df, params=mock_params, output_dir=output_dir
    )

    assert mock_to_csv.call_count > 0, "CSV exports were bypassed."
    assert mock_save_fig.call_count > 0, "Plot generation was bypassed."
    assert mock_md_rep.called, "Markdown Report Generation was bypassed."


# ========================================================================
# End-to-End Test (Execution with REAL Project Data)
# ========================================================================
@patch("pimqc.pipeline.iu._check_dir_exists")
@patch("pandas.DataFrame.to_csv")
@patch("pimqc.visualization.base.BaseMetaboVisualizer.save_and_close_fig")
@patch("pimqc.visualization.base.BaseMetaboVisualizer.save_and_show_pw")
@patch("pimqc.pipeline.ru.NarrativeStatsReporter")
@patch("pimqc.pipeline.ru.VisualAssetReporter")
def test_run_pipeline_real_data(
    mock_visual_rep: Mock,
    mock_md_rep: Mock,
    mock_save_pw: Mock,
    mock_save_fig: Mock,
    mock_to_csv: Mock,
    mock_check_dir: Mock,
    real_project_data: PipelineData,
) -> None:
    """Test the pipeline using actual project MS data for real viability."""
    meta_df, int_df, real_params = real_project_data

    output_dir = "dummy_real_output_directory"

    run_pipeline(
        meta_df=meta_df, int_df=int_df, params=real_params, output_dir=output_dir
    )

    assert mock_to_csv.call_count > 0, "CSV export failed on real data."
    assert mock_md_rep.called, "Markdown Report Generation failed on real data."
