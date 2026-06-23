# tests/test_bpca_r_bridge.py
"""
Script purpose: Compare the Python BPCA imputer with pcaMethods BPCA.

This bridge test targets the BPCA implementation specifically. The local
Python code is an independent port rather than a byte-for-byte wrapper, so the
test compares the Python output with pcaMethods on a low-rank masked matrix at
two levels: full-matrix structural agreement and missing-entry reconstruction
quality relative to a simple median baseline.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from scipy.stats import spearmanr

from pimqc.imputation import MetaboIntImputer


def _require_r_package(package: str) -> None:
    """Skip the bridge test when an optional R package is unavailable."""
    r_require = ro.r("function(pkg) requireNamespace(pkg, quietly=TRUE)")
    available = bool(r_require(package)[0])
    if not available:
        pytest.skip(f"R package '{package}' is not installed.")


def _make_low_rank_log_matrix() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Create a positive log-scale matrix with structured missing values."""
    rng = np.random.default_rng(20260611)
    n_features, n_samples = 30, 14
    sample_axis = np.linspace(-1.0, 1.0, n_samples)
    seasonal_axis = np.cos(np.linspace(0.0, 2.0 * np.pi, n_samples))

    load_1 = rng.normal(0.0, 0.8, size=n_features)
    load_2 = rng.normal(0.0, 0.5, size=n_features)
    true_arr = (
        8.0
        + load_1[:, None] * sample_axis[None, :]
        + load_2[:, None] * seasonal_axis[None, :]
        + rng.normal(0.0, 0.05, size=(n_features, n_samples))
    )
    true_arr = np.clip(true_arr, a_min=0.1, a_max=None)

    missing_mask = rng.random(true_arr.shape) < 0.18
    missing_mask[0, :] = False
    missing_mask[:, 0] = False
    masked_arr = true_arr.copy()
    masked_arr[missing_mask] = np.nan

    index = [f"Met_{i:02d}" for i in range(n_features)]
    columns = [f"S{i:02d}" for i in range(n_samples)]
    true_df = pd.DataFrame(true_arr, index=index, columns=columns)
    masked_df = pd.DataFrame(masked_arr, index=index, columns=columns)
    return true_df, masked_df, missing_mask


def _row_median_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with feature medians for baseline NRMSE."""
    return df.apply(lambda row: row.fillna(row.median()), axis=1)


def _nrmse(truth: np.ndarray, estimate: np.ndarray) -> float:
    """Calculate normalized RMSE on flattened arrays."""
    rmse = float(np.sqrt(np.mean((truth - estimate) ** 2)))
    denom = float(np.nanmax(truth) - np.nanmin(truth))
    return rmse / max(denom, 1e-12)


def _relative_mae(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Calculate mean absolute error relative to the reference magnitude."""
    valid = np.isfinite(reference) & np.isfinite(estimate)
    mae = float(np.mean(np.abs(reference[valid] - estimate[valid])))
    denom = float(np.mean(np.abs(reference[valid])))
    return mae / max(denom, 1e-12)


def run_r_bpca(
    df_input: pd.DataFrame,
    n_components: int = 2,
    max_iter: int = 100,
    threshold: float = 1e-4,
) -> pd.DataFrame:
    """Execute pcaMethods BPCA through rpy2."""
    _require_r_package("pcaMethods")
    r_script = """
    function(df, n_components, max_iter, threshold) {
        suppressWarnings(suppressPackageStartupMessages(library(pcaMethods)))
        mat <- as.matrix(df)
        storage.mode(mat) <- "double"
        mat[is.nan(mat)] <- NA
        suppressMessages(capture.output(
            fit <- pca(
                t(mat),
                method="bpca",
                nPcs=as.integer(n_components),
                maxSteps=as.integer(max_iter),
                threshold=as.numeric(threshold),
                center=TRUE,
                scale="none"
            )
        ))
        res <- t(completeObs(fit))
        dimnames(res) <- dimnames(mat)
        return(as.data.frame(res))
    }
    """
    r_bpca_func = ro.r(r_script)
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df_result = r_bpca_func(df_input, n_components, max_iter, threshold)
    r_df_result.index = df_input.index
    r_df_result.columns = df_input.columns
    return r_df_result


def test_bpca_matches_pcamethods_matrix_structure() -> None:
    """Check that Python BPCA tracks pcaMethods at matrix level."""
    true_df, masked_df, missing_mask = _make_low_rank_log_matrix()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        py_df = MetaboIntImputer.impute_by_bpca(
            masked_df, n_components=2, max_iter=100, threshold=1e-4
        )
    r_df = run_r_bpca(masked_df, n_components=2, max_iter=100, threshold=1e-4)

    observed_mask = ~missing_mask
    np.testing.assert_allclose(
        py_df.to_numpy()[observed_mask],
        masked_df.to_numpy()[observed_mask],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        r_df.to_numpy()[observed_mask],
        masked_df.to_numpy()[observed_mask],
        atol=1e-12,
    )
    assert not py_df.isna().any().any()
    assert not r_df.isna().any().any()
    assert float(py_df.min().min()) >= 0.0

    py_flat = py_df.to_numpy(dtype=float).ravel()
    r_flat = r_df.to_numpy(dtype=float).ravel()
    matrix_corr = spearmanr(py_flat, r_flat).correlation
    assert matrix_corr is not None and matrix_corr > 0.90
    assert _relative_mae(r_flat, py_flat) < 0.02

    baseline_df = _row_median_baseline(masked_df)
    truth = true_df.to_numpy()[missing_mask]
    py_est = py_df.to_numpy()[missing_mask]
    r_est = r_df.to_numpy()[missing_mask]
    baseline_est = baseline_df.to_numpy()[missing_mask]

    baseline_nrmse = _nrmse(truth, baseline_est)
    py_nrmse = _nrmse(truth, py_est)
    r_nrmse = _nrmse(truth, r_est)
    assert py_nrmse < baseline_nrmse * 1.10
    assert r_nrmse < baseline_nrmse

    data_range = float(np.nanmax(true_df.to_numpy()) - np.nanmin(true_df.to_numpy()))
    missing_mae = float(np.mean(np.abs(py_est - r_est)))
    assert missing_mae / max(data_range, 1e-12) < 0.12

    observed_values = masked_df.to_numpy()[~missing_mask]
    lower = float(np.nanpercentile(observed_values, 1))
    upper = float(np.nanpercentile(observed_values, 99))
    assert float(np.nanmedian(py_est)) > lower
    assert float(np.nanmedian(py_est)) < upper
    assert float(np.nanmedian(r_est)) > lower
    assert float(np.nanmedian(r_est)) < upper
