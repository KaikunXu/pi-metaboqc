# src/pimqc/stat_utils.py
"""
Script purpose: Provide shared statistical transformations and metrics.

This module contains numerical helpers used by multiple processing stages,
including robust log2 transformation, auto and Pareto scaling, feature-scaling
dispatch, fast Numba-backed Gaussian KDE estimation, and Jensen-Shannon
distribution similarity. It also extracts comparable log-scale vectors from
paired datasets for normalization and imputation diagnostics.
The functions are kept independent from MetaboInt so they can be reused in
PCA, assessment, and processing modules.
"""

import math

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.spatial.distance import jensenshannon


# ====================================================================
# Numba Compiled Fast Engine
# ====================================================================
@njit(parallel=True, fastmath=True)
def _numba_gaussian_kde(
    data: np.ndarray, grid: np.ndarray, bandwidth: float
) -> np.ndarray:
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


def apply_feature_scaling(df: pd.DataFrame, method: str = "None") -> pd.DataFrame:
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
def calc_jsd_similarity(
    arr1: np.ndarray, arr2: np.ndarray, grid_points: int = 200
) -> dict[str, float]:
    """Calculate Jensen-Shannon Divergence between two arrays via KDE."""
    a = arr1[~np.isnan(arr1)]
    b = arr2[~np.isnan(arr2)]
    if len(a) == 0 or len(b) == 0:
        return {"jsd": 0.0}

    grid = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), grid_points)
    bw_a = 1.06 * np.std(a) * (len(a) ** -0.2) if np.std(a) > 0 else 0.1
    bw_b = 1.06 * np.std(b) * (len(b) ** -0.2) if np.std(b) > 0 else 0.1

    p = _numba_gaussian_kde(a, grid, bw_a)
    q = _numba_gaussian_kde(b, grid, bw_b)
    p /= p.sum() + 1e-10
    q /= q.sum() + 1e-10

    return {"jsd": float(jensenshannon(p, q))}


# ====================================================================
# Extract target data for imputation and normalization
# ====================================================================
def _extract_log2_target(
    obj: pd.DataFrame | None, auto_log_for_vis: bool = True
) -> pd.DataFrame | None:
    """Extract data and strictly enforce Log2 scale for relative comparisons.

    Refined logic: Verifies if the object has actually completed normalization
    rather than just checking the requested 'norm_method' string. This prevents
    pre-normalization linear data from bypassing the transient log transform
    when the global workflow is configured for VSN.
    """
    if obj is None:
        return None

    target_cols = obj.columns.difference(obj._blank.columns)
    data = obj[target_cols].astype(float)

    is_logged = obj.attrs.get("is_logged", False)
    norm_method = str(obj.attrs.get("norm_method", "None")).upper()
    current_stage = str(obj.attrs.get("pipeline_stage", "Unknown"))

    # Strictly verify that the normalization has actually been executed
    is_post_norm = current_stage == "Normalization"

    # Bypass transient log transform ONLY IF:
    # 1. Explicitly flagged as is_logged OR
    # 2. It has actually completed the VSN normalization stage (glog space)
    if is_logged or (norm_method == "VSN" and is_post_norm):
        return data

    if auto_log_for_vis:
        from . import stat_utils as su

        return su.robust_log2_transform(data)

    return data
