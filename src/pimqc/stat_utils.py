# src/pimqc/stat_utils.py
"""
Purpose of script: Shared statistical and mathematical utility functions.
"""

import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon

# ====================================================================
# Statistical Similarity Metrics
# ====================================================================
def calc_jsd_similarity(arr1, arr2, grid_points=200):
    """Calculate Jensen-Shannon Divergence between two arrays via KDE.
    
    Args:
        arr1 (np.ndarray): First data array (can contain NaNs).
        arr2 (np.ndarray): Second data array (can contain NaNs).
        grid_points (int): Number of points for KDE discretization.
        
    Returns:
        dict: A dictionary containing the computed JSD value.
    """
    a = arr1[~np.isnan(arr1)]
    b = arr2[~np.isnan(arr2)]
    
    if len(a) == 0 or len(b) == 0:
        return {"jsd": 0.0}

    # Define shared grid for probability mass calculation
    grid_min = min(a.min(), b.min())
    grid_max = max(a.max(), b.max())
    grid = np.linspace(grid_min, grid_max, grid_points)

    # Estimate densities and convert to probability vectors
    kde_a = gaussian_kde(a)(grid)
    kde_b = gaussian_kde(b)(grid)
    
    # Normalize to ensure sum equals 1 for jensenshannon function
    p = kde_a / (kde_a.sum() + 1e-10)
    q = kde_b / (kde_b.sum() + 1e-10)
    
    return {"jsd": float(jensenshannon(p, q))}

# ====================================================================
# Module-level Helper Functions
# ====================================================================
def _extract_log2_target(obj, check_is_scaled=False):
    """Extract target columns and conditionally apply log2 transform.
    
    Intercepts VSN, Auto, and Pareto scaled data to prevent fatal 
    log2(negative) generating NaNs.
    """
    if obj is None:
        return None
        
    target_cols = obj.columns.difference(obj._blank.columns)
    
    if check_is_scaled:
        f_norm = obj.attrs.get("feature_wise_norm", "None").lower()
        if f_norm in ("vsn", "auto", "pareto"):
            return obj[target_cols].astype(float)
            
    return calc_log2_transform(obj[target_cols])

def calc_log2_transform(df):
    """Perform log2 transformation after replacing zeros with NaNs."""
    return np.log2(df.astype(float).replace({0: np.nan}))
