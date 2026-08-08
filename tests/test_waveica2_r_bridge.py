# tests/test_waveica2_r_bridge.py
"""
Script purpose: Compare Python WaveICA2 correction with an R reference.

This bridge test sources the official WaveICA_2.0.R and unbiased_stICA.R files
from an external source directory, then checks that the native Python
WaveICA2Corrector remains close to the R implementation while reducing
injection-order structure.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from scipy.stats import spearmanr

from pimqc.processing.correction import WaveICA2Corrector


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WAVEICA2_R_SOURCE_DIR = (
    REPO_ROOT / "tests" / "r_references" / "WaveICA_2.0" / "R"
)


def _require_r_package(package: str) -> None:
    """Skip the bridge test when an optional R package is unavailable."""
    r_require = ro.r("function(pkg) requireNamespace(pkg, quietly=TRUE)")
    available = bool(r_require(package)[0])
    if not available:
        pytest.skip(f"R package '{package}' is not installed.")


def _make_waveica_matrix() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Create a matrix with QC samples and a smooth injection-order artifact."""
    rng = np.random.default_rng(20260611)
    n_features, n_samples = 14, 32
    order = np.arange(1, n_samples + 1, dtype=float)
    qc_mask = np.array([(idx % 4) == 0 for idx in range(n_samples)])
    order_scaled = (order - order.mean()) / np.ptp(order)
    artifact = 0.32 * order_scaled + 0.10 * np.sin(order / 4.0)
    artifact += np.where(qc_mask, 0.07 * np.cos(order / 5.0), 0.0)

    base = rng.lognormal(mean=8.0, sigma=0.25, size=n_features)
    weights = rng.uniform(0.7, 1.3, size=n_features)
    arr = base[:, None] * (1.0 + weights[:, None] * artifact[None, :])
    arr *= rng.normal(1.0, 0.01, size=arr.shape)
    arr = np.clip(arr, a_min=1e-6, a_max=None)

    shuffle_idx = rng.permutation(n_samples)
    arr = arr[:, shuffle_idx]
    order = order[shuffle_idx]
    qc_mask = qc_mask[shuffle_idx]

    index = [f"Met_{idx:02d}" for idx in range(n_features)]
    columns = [f"S{idx:02d}" for idx in range(n_samples)]
    return pd.DataFrame(arr, index=index, columns=columns), order, qc_mask


def _median_abs_order_corr(df: pd.DataFrame, order: np.ndarray) -> float:
    """Summarize absolute feature-wise Spearman association with order."""
    vals = []
    for _, row in df.iterrows():
        corr = spearmanr(row.to_numpy(dtype=float), order).correlation
        if corr is not None and np.isfinite(corr):
            vals.append(abs(float(corr)))
    return float(np.nanmedian(vals))


def _median_qc_rsd(df: pd.DataFrame, qc_mask: np.ndarray) -> float:
    """Calculate median feature RSD across QC samples."""
    qc_df = df.iloc[:, qc_mask]
    means = qc_df.mean(axis=1)
    rsd = qc_df.std(axis=1, ddof=1).div(means.replace(0.0, np.nan))
    return float(np.nanmedian(rsd.to_numpy()))


def _get_waveica2_source_dir() -> Path:
    """Return the official WaveICA 2.0 R source directory for bridge tests."""
    source_dir = Path(
        os.environ.get("WAVEICA2_R_SOURCE_DIR", DEFAULT_WAVEICA2_R_SOURCE_DIR)
    )
    required_files = ("unbiased_stICA.R", "WaveICA_2.0.R")
    missing_files = [
        file_name
        for file_name in required_files
        if not (source_dir / file_name).exists()
    ]
    if missing_files:
        pytest.skip(
            "Official WaveICA 2.0 R source files are unavailable: "
            + ", ".join(str(source_dir / file_name) for file_name in missing_files)
        )
    return source_dir


def run_r_original_waveica2(
    df_input: pd.DataFrame,
    order: np.ndarray,
    n_components: int = 4,
    cutoff: float = 0.10,
) -> pd.DataFrame:
    """Execute the official WaveICA 2.0 R implementation through rpy2."""
    for package in ("waveslim", "ica", "JADE", "corpcor", "mgcv"):
        _require_r_package(package)

    source_dir = _get_waveica2_source_dir()
    ro.r("options(mc.cores=1)")
    ro.r["source"](str(source_dir / "unbiased_stICA.R").replace("\\", "/"))
    ro.r["source"](str(source_dir / "WaveICA_2.0.R").replace("\\", "/"))
    r_waveica_func = ro.globalenv["WaveICA_2.0"]

    sort_idx = np.argsort(order, kind="mergesort")
    inverse_idx = np.argsort(sort_idx)
    sorted_order = order[sort_idx]
    sorted_sample_by_peak = df_input.iloc[:, sort_idx].T

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_result = r_waveica_func(
            sorted_sample_by_peak,
            "haar",
            ro.FloatVector([float(x) for x in sorted_order]),
            0.0,
            float(cutoff),
            int(n_components),
        )

    result_items = dict(zip(list(r_result.names()), list(r_result)))
    r_arr_sorted = np.asarray(result_items["data_wave"], dtype=float)
    r_arr = r_arr_sorted[inverse_idx, :]
    return pd.DataFrame(r_arr.T, index=df_input.index, columns=df_input.columns)


def test_waveica2_matches_original_r_reference() -> None:
    """Check that Python WaveICA2 is close to the original R algorithm."""
    df_input, order, qc_mask = _make_waveica_matrix()
    raw_corr = _median_abs_order_corr(df_input, order)
    raw_qc_rsd = _median_qc_rsd(df_input, qc_mask)

    py_df = WaveICA2Corrector(
        n_components=4,
        cutoff=0.10,
        n_levels=None,
        spline_knots=5,
        max_iter=1000,
        random_state=123,
    ).fit_transform(df_input, order_array=order)["WaveICA 2.0"][0]
    r_df = run_r_original_waveica2(
        df_input,
        order=order,
        n_components=4,
        cutoff=0.10,
    )

    assert py_df.shape == df_input.shape
    assert r_df.shape == df_input.shape
    assert np.isfinite(py_df.to_numpy()).all()
    assert np.isfinite(r_df.to_numpy()).all()
    assert float(py_df.min().min()) > 0.0
    assert float(r_df.min().min()) > 0.0

    py_corr = _median_abs_order_corr(py_df, order)
    py_qc_rsd = _median_qc_rsd(py_df, qc_mask)
    assert py_corr < raw_corr
    assert py_qc_rsd < raw_qc_rsd

    py_flat = py_df.to_numpy(dtype=float).ravel()
    r_flat = r_df.to_numpy(dtype=float).ravel()
    matrix_corr = spearmanr(py_flat, r_flat).correlation
    relative_mae = float(np.mean(np.abs(py_flat - r_flat)) / np.mean(np.abs(r_flat)))

    assert matrix_corr is not None and matrix_corr > 0.995
    assert relative_mae < 0.03
