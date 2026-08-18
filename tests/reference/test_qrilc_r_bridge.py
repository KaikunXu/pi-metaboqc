"""
Script purpose: Compare the Python QRILC imputer with imputeLCMD QRILC.

This bridge test targets QRILC specifically. The Python implementation follows
imputeLCMD's column-wise quantile-regression logic, but QRILC is stochastic, so
the test compares distribution-level agreement with the original R output
rather than requiring identical random draws.
"""

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from scipy.stats import spearmanr

from pimqc.constants import DEFAULT_RANDOM_SEED
from pimqc.processing.imputation import MetaboIntImputer

from .helpers import relative_mae, require_r_package


def _make_left_censored_log_matrix() -> tuple[pd.DataFrame, np.ndarray]:
    """Create a log-scale matrix with low-abundance missing values."""
    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
    n_features, n_samples = 24, 10
    locs = np.linspace(5.5, 9.5, n_features)
    arr = rng.normal(locs[:, None], 0.25, size=(n_features, n_samples))
    arr = np.clip(arr, a_min=0.1, a_max=None)

    missing_mask = np.zeros_like(arr, dtype=bool)
    for i in range(n_features):
        low_idx = np.argsort(arr[i])[:2]
        missing_mask[i, low_idx] = True

    masked = arr.copy()
    masked[missing_mask] = np.nan
    index = [f"Met_{i:02d}" for i in range(n_features)]
    columns = [f"S{i:02d}" for i in range(n_samples)]
    return pd.DataFrame(masked, index=index, columns=columns), missing_mask


def run_r_qrilc(
    df_input: pd.DataFrame,
    tune_sigma: float = 1.0,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Execute imputeLCMD QRILC through rpy2."""
    require_r_package("imputeLCMD")
    r_script = """
    function(df, tune_sigma, seed) {
        suppressWarnings(suppressPackageStartupMessages(library(imputeLCMD)))
        set.seed(as.integer(seed))
        mat <- as.matrix(df)
        storage.mode(mat) <- "double"
        mat[is.nan(mat)] <- NA
        res <- impute.QRILC(mat, tune.sigma=as.numeric(tune_sigma))[[1]]
        dimnames(res) <- dimnames(mat)
        return(as.data.frame(res))
    }
    """
    r_qrilc_func = ro.r(r_script)
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df_result = r_qrilc_func(df_input, tune_sigma, seed)
    r_df_result.index = df_input.index
    r_df_result.columns = df_input.columns
    return r_df_result


def test_qrilc_matches_imputelcmd_distributional_reference() -> None:
    """Check that Python QRILC tracks the imputeLCMD distribution."""
    masked_df, missing_mask = _make_left_censored_log_matrix()

    py_df = MetaboIntImputer.impute_by_qrilc(
        masked_df,
        tune_sigma=1.0,
        global_seed=DEFAULT_RANDOM_SEED,
    )
    r_df = run_r_qrilc(
        masked_df,
        tune_sigma=1.0,
        seed=DEFAULT_RANDOM_SEED,
    )

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

    py_values = py_df.to_numpy()
    r_values = r_df.to_numpy()
    masked_values = masked_df.to_numpy()

    py_missing = py_values[missing_mask]
    r_missing = r_values[missing_mask]
    observed_values = masked_values[observed_mask]
    py_all = py_values.ravel()
    r_all = r_values.ravel()
    matrix_corr = spearmanr(py_all, r_all).correlation

    assert np.nanmedian(py_missing) < np.nanpercentile(observed_values, 25)
    assert np.nanmedian(r_missing) < np.nanpercentile(observed_values, 25)
    assert matrix_corr is not None and matrix_corr > 0.95
    assert relative_mae(r_all, py_all) < 0.03
    assert relative_mae(r_missing, py_missing) < 0.25
    np.testing.assert_allclose(
        np.nanpercentile(py_missing, [10, 50, 90]),
        np.nanpercentile(r_missing, [10, 50, 90]),
        atol=0.4,
    )
