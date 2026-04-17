"""Pytest module for VSN implementation consistency testing."""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from pimqc.normalization import MetaboIntNormalizer


def run_r_vsn(df_input: pd.DataFrame) -> pd.DataFrame:
    """Execute Bioconductor vsn2 via the rpy2 interface."""
    r_script = """
    function(df) {
        suppressWarnings(suppressPackageStartupMessages(library(vsn)))
        mat <- as.matrix(df)
        mat[is.nan(mat)] <- NA
        fit <- vsn2(mat, verbose=FALSE)
        res <- predict(fit, mat)
        return(as.data.frame(res))
    }
    """
    r_vsn_func = ro.r(r_script)
    
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df_result = r_vsn_func(df_input)
        
    r_df_result.index = df_input.index
    return r_df_result


def test_vsn_equivalence(mock_ms_data: pd.DataFrame) -> None:
    """Test if Python VSN is statistically equivalent to R vsn2.
    
    Note: Python implementation adds a constant `pure_shift` to align high 
    abundance data with log2. Pearson correlation evaluates the structural 
    equivalence, which remains invariant to this constant shift.
    """
    df_raw = mock_ms_data
    
    # Suppress numpy division warnings during log1p
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Unpack the tuple (df, vsn_meta) returned by the new staticmethod
        df_py_norm, _ = MetaboIntNormalizer.calc_vsn_normalization(df_raw)
        
    df_r_norm = run_r_vsn(df_raw)
    
    py_flat = df_py_norm.values.flatten()
    r_flat = df_r_norm.values.flatten()
    valid = ~np.isnan(py_flat) & ~np.isnan(r_flat)
    
    assert np.sum(valid) >= 2, "Insufficient valid data points."
    
    corr, _ = pearsonr(py_flat[valid], r_flat[valid])
    
    # Assert statistical equivalence
    assert corr > 0.99, f"VSN correlation below threshold: {corr:.6f}"