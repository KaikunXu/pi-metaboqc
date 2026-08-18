"""Build deterministic datasets used by pipeline integration tests.

The synthetic fixture represents two analytical batches containing biological
samples, pooled QCs, and blanks. It deliberately includes drift, blank
contamination, unstable QC features, and several missing-value mechanisms so a
single pipeline run exercises every processing stage without relying on an
external dataset.
"""

import numpy as np
import pandas as pd
import pytest

from pimqc.constants import DEFAULT_RANDOM_SEED


@pytest.fixture
def synthetic_pipeline_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    """Return deterministic metadata, intensities, and pipeline parameters."""
    # Keep all synthetic draws local to the fixture so collection and library
    # code never mutate NumPy's application-wide random state.
    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)

    # Build two 75-injection batches with bracketing QCs and ten blanks each.
    metadata_records: list[list[object]] = []
    sample_index = 1
    injection_order = 1
    for batch in ("Batch1", "Batch2"):
        metadata_records.append(
            [f"S{sample_index:03d}", "QC", np.nan, batch, injection_order]
        )
        sample_index += 1
        injection_order += 1

        for _ in range(10):
            metadata_records.append(
                [
                    f"S{sample_index:03d}",
                    "Blank",
                    np.nan,
                    batch,
                    injection_order,
                ]
            )
            sample_index += 1
            injection_order += 1

        metadata_records.append(
            [f"S{sample_index:03d}", "QC", np.nan, batch, injection_order]
        )
        sample_index += 1
        injection_order += 1

        for block in range(8):
            biological_count = 6 if block == 7 else 7
            for offset in range(biological_count):
                group = "GroupA" if (block * 7 + offset) % 2 == 0 else "GroupB"
                metadata_records.append(
                    [
                        f"S{sample_index:03d}",
                        "Sample",
                        group,
                        batch,
                        injection_order,
                    ]
                )
                sample_index += 1
                injection_order += 1
            metadata_records.append(
                [f"S{sample_index:03d}", "QC", np.nan, batch, injection_order]
            )
            sample_index += 1
            injection_order += 1

    metadata = pd.DataFrame(
        metadata_records,
        columns=[
            "Sample Name",
            "Sample Type",
            "Bio Group",
            "Batch",
            "Inject Order",
        ],
    )

    # Start with stable signals, then introduce stage-specific failure modes.
    features = [f"Feature_{index:03d}" for index in range(1, 101)]
    intensity = pd.DataFrame(
        rng.lognormal(
            mean=np.log(1e6),
            sigma=0.01,
            size=(len(features), len(metadata)),
        ),
        index=features,
        columns=metadata["Sample Name"],
    )
    is_blank = metadata["Sample Type"].eq("Blank").to_numpy()
    is_qc = metadata["Sample Type"].eq("QC").to_numpy()
    is_group_a = metadata["Bio Group"].eq("GroupA").to_numpy()
    is_group_b = metadata["Bio Group"].eq("GroupB").to_numpy()

    intensity.loc[:, is_blank] = 10.0
    for batch, drift_rate, batch_effect in (
        ("Batch1", -0.001, 1.0),
        ("Batch2", -0.002, 0.9),
    ):
        batch_mask = metadata["Batch"].eq(batch).to_numpy()
        orders = metadata.loc[batch_mask, "Inject Order"].to_numpy()
        intensity.loc[:, batch_mask] *= (
            np.exp(drift_rate * orders) * batch_effect
        )

    blank_fail_features = features[10:15]
    qc_mean = intensity.loc[blank_fail_features, is_qc].mean(axis=1).to_numpy()
    intensity.loc[blank_fail_features, is_blank] = qc_mean[:, None] * 2.0

    qc_indices = np.flatnonzero(is_qc)
    for feature in features[20:25]:
        intensity.loc[feature, is_qc] = 1000.0
        spike_index = qc_indices[len(qc_indices) // 2]
        intensity.iloc[intensity.index.get_loc(feature), spike_index] = 1e12

    for feature in features[30:50]:
        drop_indices = rng.choice(
            len(metadata),
            int(len(metadata) * 0.15),
            replace=False,
        )
        intensity.iloc[intensity.index.get_loc(feature), drop_indices] = np.nan

    for feature in features[60:70]:
        group_a_indices = np.flatnonzero(is_group_a)
        group_b_indices = np.flatnonzero(is_group_b)
        drop_group_a = rng.choice(
            group_a_indices,
            int(len(group_a_indices) * 0.90),
            replace=False,
        )
        drop_group_b = rng.choice(
            group_b_indices,
            int(len(group_b_indices) * 0.05),
            replace=False,
        )
        drop_qc = rng.choice(
            qc_indices,
            int(len(qc_indices) * 0.50),
            replace=False,
        )
        drop_indices = np.concatenate((drop_group_a, drop_group_b, drop_qc))
        intensity.iloc[intensity.index.get_loc(feature), drop_indices] = np.nan

        retained_group_a = np.setdiff1d(group_a_indices, drop_group_a)
        intensity.iloc[intensity.index.get_loc(feature), retained_group_a] *= (
            0.05
        )

    for feature in features[80:85]:
        drop_indices = rng.choice(
            len(metadata),
            int(len(metadata) * 0.85),
            replace=False,
        )
        intensity.iloc[intensity.index.get_loc(feature), drop_indices] = np.nan

    parameters: dict[str, object] = {
        "MetaboInt": {
            "mode": "POS",
            "batch": "Batch",
            "sample_type": "Sample Type",
            "bio_group": "Bio Group",
            "sample_name": "Sample Name",
            "inject_order": "Inject Order",
            "sample_dict": {
                "QC sample": "QC",
                "Actual sample": "Sample",
                "Blank sample": "Blank",
            },
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
        "MetaboIntNormalizer": {
            "col_method": "median",
            "row_method": "auto_scale",
        },
    }
    return metadata, intensity, parameters
