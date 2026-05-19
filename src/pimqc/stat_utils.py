# src/pimqc/stat_utils.py
"""Shared statistical and mathematical utility functions."""

import math
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from numba import njit, prange
from loguru import logger


# ====================================================================
# Numba Compiled Fast Engine
# ====================================================================
@njit(parallel=True, fastmath=True)
def _numba_gaussian_kde(data, grid, bandwidth):
    """Extremely fast parallel KDE using Silverman's rule of thumb."""
    n = len(data)
    m = len(grid)
    kde_values = np.zeros(m)
    c = 1.0 / (bandwidth * math.sqrt(2.0 * math.pi))
    
    for i in prange(m):
        grid_val = grid[i]
        kernel_sum = 0.0
        for j in range(n):
            u = (grid_val - data[j]) / bandwidth
            kernel_sum += math.exp(-0.5 * u * u)
        kde_values[i] = (kernel_sum / n) * c
    return kde_values


# ====================================================================
# Data Transformation & Scaling Operators
# ====================================================================
def robust_log2_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Perform log2 transform with LOD offset (half of min positive)."""
    out_df = df.copy()
    if (out_df <= 0).any().any():
        min_pos = out_df[out_df > 0].min().min()
        offset = min_pos / 2.0
        out_df = out_df.clip(lower=offset)
    return np.log2(out_df)


def calc_auto_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Auto Scaling (Z-score) feature-wise."""
    # Compute mean and std for the provided subset
    row_means = df.mean(axis=1)
    row_stds = df.std(axis=1, ddof=1).replace(0, 1)
    return df.sub(row_means, axis=0).div(row_stds, axis=0)


def calc_pareto_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Pareto Scaling feature-wise."""
    # Divide by square root of standard deviation
    row_means = df.mean(axis=1)
    row_stds = df.std(axis=1, ddof=1).replace(0, 1)
    return df.sub(row_means, axis=0).div(np.sqrt(row_stds), axis=0)


def apply_feature_scaling(
    df: pd.DataFrame, method: str = "None") -> pd.DataFrame:
    """Unified entry for applying feature-wise scaling.
    
    Args:
        df: Feature matrix (Features as rows, Samples as columns).
        method: Scaling strategy ("Auto-scaling", "Pareto-scaling", "None").
    """
    method_upper = str(method).upper()
    if method_upper == "AUTO-SCALING":
        return calc_auto_scaling(df)
    elif method_upper == "PARETO-SCALING":
        return calc_pareto_scaling(df)
    return df

# ====================================================================
# Statistical Similarity Metrics
# ====================================================================
def calc_jsd_similarity(arr1, arr2, grid_points=200):
    """Calculate Jensen-Shannon Divergence between two arrays via KDE."""
    a = arr1[~np.isnan(arr1)]
    b = arr2[~np.isnan(arr2)]
    if len(a) == 0 or len(b) == 0:
        return {"jsd": 0.0}

    grid = np.linspace(
        min(a.min(), b.min()), max(a.max(), b.max()), grid_points)
    bw_a = 1.06 * np.std(a) * (len(a) ** -0.2) if np.std(a) > 0 else 0.1
    bw_b = 1.06 * np.std(b) * (len(b) ** -0.2) if np.std(b) > 0 else 0.1

    p = _numba_gaussian_kde(a, grid, bw_a)
    q = _numba_gaussian_kde(b, grid, bw_b)
    p /= (p.sum() + 1e-10)
    q /= (q.sum() + 1e-10)
    
    return {"jsd": float(jensenshannon(p, q))}

# ====================================================================
# Extract target data for imputation and normalization
# ====================================================================
def _extract_log2_target(obj, auto_log_for_vis=True):
    """
    Extract data and ensure Log2 scale for visualization.
    
    Fixed: Quantile normalization is now correctly treated as linear scale.
    Only VSN or explicit log flags bypass the transient transformation. 
    """
    if obj is None:
        return None
    target_cols = obj.columns.difference(obj._blank.columns)
    data = obj[target_cols].astype(float)
    
    is_logged = obj.attrs.get("is_logged", False)
    norm_method = str(obj.attrs.get("norm_method", "None")).upper()

    # Only skip log if already formally logged or processed via VSN glog
    if is_logged or norm_method == "VSN":
        return data
    
    if auto_log_for_vis:
        return robust_log2_transform(data)
    return data