"""Regression tests for normalization plotter initialization contracts.

Normalization plotting is assembled from focused plotting components, but the
public plotter must still initialize the shared before/after palette. This test
protects the constructor contract that previously regressed during module
splitting without coupling the suite to private mixin decorators.
"""

import pandas as pd

from pimqc.plotting import plot_utils as pu
from pimqc.plotting.normalization import NormalizationPlotter


def _minimal_normalization_frame() -> pd.DataFrame:
    """Return a matrix with the metadata levels required by the plotter."""
    columns = pd.MultiIndex.from_tuples(
        [("QC", "Batch 1", 1, "Group 1")],
        names=["Sample Type", "Batch", "Inject Order", "Bio Group"],
    )
    frame = pd.DataFrame([[10.0]], index=["Feature 1"], columns=columns)
    frame.attrs["pipeline_parameters"] = {}
    return frame


def test_normalization_plotter_uses_shared_palette_constants() -> None:
    """Initialize split plotting components with the public palette values."""
    plotter = NormalizationPlotter(
        _minimal_normalization_frame(),
        _minimal_normalization_frame(),
    )

    assert plotter.pal == {
        "Before Norm": pu.NEUTRAL_COLOR,
        "After Norm": pu.PRIMARY_ACCENT_COLOR,
    }
