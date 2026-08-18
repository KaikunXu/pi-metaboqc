"""Regression tests for assessment visualization annotations."""

import numpy as np
import pandas as pd

from pimqc.plotting.assessment import AssessmentPlotter


def test_single_batch_pca_annotation_omits_batch_silhouette() -> None:
    """Hide batch silhouette when the PCA data contain only one batch."""
    annotation = AssessmentPlotter._format_pca_diagnostics_annotation(
        pca_diagnostics={
            "relative_dispersion": 0.25,
            "batch_silhouette": np.nan,
            "centrality_shift": 0.1,
        },
        batches=pd.Series(["Batch 1", "Batch 1"]),
    )

    assert annotation.splitlines() == [
        "Relative Dispersion: 0.2500",
        "Centrality Shift: 0.1000",
    ]


def test_multi_batch_pca_annotation_keeps_batch_silhouette() -> None:
    """Keep batch silhouette for multi-batch PCA data, including N/A."""
    annotation = AssessmentPlotter._format_pca_diagnostics_annotation(
        pca_diagnostics={
            "relative_dispersion": 0.25,
            "batch_silhouette": np.nan,
            "centrality_shift": 0.1,
        },
        batches=pd.Series(["Batch 1", "Batch 2"]),
    )

    assert annotation.splitlines() == [
        "Relative Dispersion: 0.2500",
        "Batch Silhouette: N/A",
        "Centrality Shift: 0.1000",
    ]
