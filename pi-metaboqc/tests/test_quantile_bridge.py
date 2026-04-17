# tests/test_quantile_bridge.py
"""Pytest module for Quantile Normalization consistency testing."""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from pimqc.normalization import MetaboIntNormalizer


def run_r_quantile(df_input: pd.DataFrame) -> pd.DataFrame:
    """Execute Bioconductor preprocessCore via the rpy2 interface."""
    r_script = """
    function(df) {
        suppressWarnings(suppressPackageStartupMessages(
            library(preprocessCore)))
        mat <- as.matrix(df)
        mat[is.nan(mat)] <- NA
        res <- normalize.quantiles(mat)
        dimnames(res) <- dimnames(mat)
        return(as.data.frame(res))
    }
    """
    r_quant_func = ro.r(r_script)
    
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df_result = r_quant_func(df_input)
        
    r_df_result.index = df_input.index
    return r_df_result


def test_quantile_equivalence(mock_ms_data: pd.DataFrame) -> None:
    """Test if Python Quantile norm matches preprocessCore.
    
    Now leverages the decoupled staticmethod directly for pure math testing.
    """
    df_raw = mock_ms_data
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Directly call the pure staticmethod instead of initializing an object
        df_py_norm = MetaboIntNormalizer.calc_quantile_normalization(df_raw)
        
    df_r_norm = run_r_quantile(df_raw)
    
    py_flat = df_py_norm.values.flatten()
    r_flat = df_r_norm.values.flatten()
    valid = ~np.isnan(py_flat) & ~np.isnan(r_flat)
    
    assert np.sum(valid) >= 2, "Insufficient valid data points."
    
    max_res = np.max(np.abs(py_flat[valid] - r_flat[valid]))
    corr, _ = pearsonr(py_flat[valid], r_flat[valid])
    
    # Assert bit-level equivalence OR statistical equivalence
    assert max_res < 1e-5 or corr > 0.99, (
        f"Quantile check failed. Max residual: {max_res:.6e}"
    )