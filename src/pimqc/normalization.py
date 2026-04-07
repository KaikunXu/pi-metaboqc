# src/pimqc/normalization.py
"""
Purpose of script: Data normalization for MetaboInt.
"""

import os
import copy
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from . import core_classes
from . import io_utils as iu


class MetaboIntNorm(core_classes.MetaboInt):
    """Normalization class for metabolomics intensity data.

    This class handles three categories of normalization:
    1. Sample-wise (Column) normalization: TIC, Median, and PQN.
    2. Feature-wise (Row) normalization: VSN, Auto scaling, and Pareto scaling.
    3. Standalone Quantile normalization: NaN-compatible interpolation logic.
    """

    # Register custom attributes for pandas metadata propagation
    _metadata: List[str] = ["attrs"]

    def __init__(
        self,
        *args: Any,
        pipeline_params: Optional[Dict[str, Any]] = None,
        col_norm: str = "PQN",
        row_norm: str = "VSN",
        quantile_norm: bool = False,
        **kwargs: Any
    ) -> None:
        """Initialize MetaboIntNorm with hierarchical parameter loading.

        Args:
            *args: Variable length arguments passed to pandas DataFrame.
            pipeline_params: Configuration dictionary for the pipeline.
            col_norm: Method for sample-wise normalization.
            row_norm: Method for feature-wise normalization.
            quantile_norm: Whether to apply standalone quantile normalization.
            **kwargs: Arbitrary keyword arguments for pandas DataFrame.
        """
        # 1. Initialize parent (loads "MetaboInt" level parameters)
        super().__init__(*args, pipeline_params=pipeline_params, **kwargs)

        # 2. Define subclass-specific default parameters
        norm_configs: Dict[str, Any] = {
            "col_norm": col_norm,
            "row_norm": row_norm,
            "quantile_norm": quantile_norm
        }

        # 3. Incrementally load "MetaboIntNorm" block from pipeline_params
        if pipeline_params and "MetaboIntNorm" in pipeline_params:
            norm_configs.update(pipeline_params["MetaboIntNorm"])

        # 4. Merge child parameters into the existing attrs dictionary
        self.attrs.update(norm_configs)

    @property
    def _constructor(self) -> type:
        """Override pandas constructor to return MetaboIntNorm type.

        Returns:
            type: The MetaboIntNorm class.
        """
        return MetaboIntNorm

    def __finalize__(
        self, other: Any, method: Optional[str] = None, **kwargs: Any
    ) -> "MetaboIntNorm":
        """Explicitly deepcopy custom attributes during object creation.

        Args:
            other: The original object to copy metadata from.
            method: The internal pandas method calling finalize.
            **kwargs: Additional keyword arguments.

        Returns:
            MetaboIntNorm: The new instance with preserved metadata.
        """
        super().__finalize__(other, method=method, **kwargs)
        if hasattr(other, "attrs"):
            self.attrs = copy.deepcopy(other.attrs)
        return self

    # ====================================================================
    # Category 1: Column (Sample) Dimension Normalization
    # ====================================================================

    def _col_norm_tic(self, df: "MetaboIntNorm") -> "MetaboIntNorm":
        """Apply Total Ion Current (TIC) normalization column-wise."""
        col_sums = df.sum(axis="index")
        return df.div(col_sums, axis="columns") * col_sums.median()

    def _col_norm_median(self, df: "MetaboIntNorm") -> "MetaboIntNorm":
        """Apply Median normalization column-wise."""
        col_medians = df.median(axis="index")
        return df.div(col_medians, axis="columns") * col_medians.median()

    def _col_norm_pqn(self, df: "MetaboIntNorm") -> "MetaboIntNorm":
        """Apply Probabilistic Quotient Normalization (PQN) column-wise."""
        df_safe = df.replace({0: np.nan})
        ref_spectrum = df_safe.median(axis="columns")
        quotients = df_safe.div(ref_spectrum, axis="index")
        median_quotients = quotients.median(axis="index")
        return df_safe.div(median_quotients, axis="columns").fillna(0)

    # ====================================================================
    # Category 2: Row (Feature) Dimension Normalization
    # ====================================================================

    def _estimate_vsn_params(
        self, 
        data: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Estimate VSN parameters via mathematically exact Profile Likelihood."""
        rows, cols = data.shape

        def log_likelihood(params: np.ndarray) -> float:
            a_vec = params[:-1]
            b = params[-1]

            z = a_vec + b * data
            transformed = np.arcsinh(z)
            
            log_jacobian = np.log(b) - 0.5 * np.log1p(z**2)

            row_means = np.nanmean(transformed, axis=1, keepdims=True)
            residuals_sq = (transformed - row_means) ** 2
            
            valid_mask = ~np.isnan(residuals_sq)
            N = np.sum(valid_mask)
            
            if N == 0:
                return 1e10
                
            sigma_sq = np.nanmean(residuals_sq)
            if sigma_sq <= 1e-16:
                return 1e10

            ll = np.nansum(log_jacobian) - (N / 2.0) * np.log(sigma_sq)
            
            return -ll

        a_init = np.zeros(cols)
        b_init = 1.0 / np.nanmedian(data)
        x0 = np.concatenate([a_init, [b_init]])

        bounds = [(None, None)] * cols + [(1e-12, None)]

        res = minimize(
            log_likelihood, 
            x0=x0, 
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-9}
        )
        return res.x[:-1], res.x[-1]

    def _row_norm_vsn(self, df: "MetaboIntNorm") -> "MetaboIntNorm":
        """Apply mathematically pure VSN matching Bioconductor vsn2 behavior."""
        data_arr = df.to_numpy(dtype=np.float64)
        a_vec, b = self._estimate_vsn_params(data_arr)

        # 1. asinh(a + bx) / ln(2) - log2(2b)
        shift_constant = np.log2(2 * b)
        normed_arr = (
            np.arcsinh(a_vec + b * data_arr) / np.log(2)) - shift_constant

        # 2. Pure Anchor Shift
        log2_data = np.log2(np.where(data_arr > 0, data_arr, np.nan))
        valid = ~np.isnan(log2_data) & ~np.isnan(normed_arr)
        
        if np.any(valid):
            y_val = normed_arr[valid]
            x_val = log2_data[valid]
            
            threshold = np.percentile(x_val, 50)
            high_mask = x_val > threshold
            
            pure_shift = np.median(x_val[high_mask] - y_val[high_mask])
            normed_arr += pure_shift

        self.attrs["vsn_offsets"] = a_vec.tolist()
        self.attrs["vsn_scale"] = float(b)
        self.attrs["vsn_shift"] = float(pure_shift)

        res_df = df.copy()
        res_df.iloc[:, :] = normed_arr
        return res_df

    def _row_norm_auto(self, df: "MetaboIntNorm") -> "MetaboIntNorm":
        """Apply Auto Scaling (Z-score) row-wise."""
        row_means = df.mean(axis="columns")
        row_stds = df.std(axis="columns", ddof=1)
        return df.sub(row_means, axis="index").div(row_stds, axis="index") 

    def _row_norm_pareto(self, df: "MetaboIntNorm") -> "MetaboIntNorm":
        """Apply Pareto Scaling row-wise."""
        row_means = df.mean(axis="columns")
        row_stds = np.sqrt(df.std(axis="columns", ddof=1))
        return df.sub(row_means, axis="index").div(row_stds, axis="index")

    # ====================================================================
    # Category 3: Quantile Normalization (NaN Compatible)
    # ====================================================================

    def _quantile_norm(self, df: "MetaboIntNorm") -> "MetaboIntNorm":
        """Quantile normalizes a DataFrame with missing value handling.

        Replicates the methodology of original R preprocessCore using
        vectorized numpy interpolation to project values to target space.
        
        Inspired with: 
            https://github.com/huosan0123/quantile_normalization_python
        """
        origin_arr = df.to_numpy(dtype=np.float64)
        rows, cols = origin_arr.shape

        # Sort independently. NaNs move to the end.
        sorted_arr = np.sort(origin_arr, axis=0)
        non_nas = rows - np.isnan(sorted_arr).sum(axis=0)

        # Compute reference target distribution
        row_means = np.zeros(rows, dtype=np.float64)
        x_target = np.linspace(0, 1, rows)

        for j in range(cols):
            non_na = non_nas[j]
            if non_na == 0:
                continue
            y = sorted_arr[:non_na, j]
            x = np.linspace(0, 1, non_na)
            row_means += np.interp(x_target, x, y)

        row_means /= float(cols)

        # Map original values to the target distribution
        normed_arr = np.full((rows, cols), np.nan, dtype=np.float64)
        for j in range(cols):
            non_na = non_nas[j]
            if non_na < 2:
                if non_na == 1:
                    valid = ~np.isnan(origin_arr[:, j])
                    normed_arr[valid, j] = row_means[rows // 2]
                continue

            col_data = origin_arr[:, j]
            valid = ~np.isnan(col_data)
            ranks = stats.rankdata(col_data[valid], method="average")
            rank_percentiles = (ranks - 1.0) / (non_na - 1.0)
            interp_vals = np.interp(rank_percentiles, x_target, row_means)
            normed_arr[valid, j] = interp_vals

        # Use copy() and iloc to maintain subclass structure and metadata
        res_df = df.copy()
        res_df.iloc[:, :] = normed_arr
        return res_df

    # ====================================================================
    # Core Execution Logic
    # ====================================================================

    def apply_normalization(self) -> "MetaboIntNorm":
        """Execute the configured sequence of normalizations.

        Returns:
            MetaboIntNorm: The fully normalized object with metadata intact.
        """
        result_df = self.copy()

        if self.attrs.get("quantile_norm", False):
            return self._quantile_norm(result_df)

        # Step 1: Column (Sample) Normalization
        col_norm = self.attrs.get("col_norm")
        if col_norm in ("TIC", "tic"):
            result_df = self._col_norm_tic(result_df)
        elif col_norm in ("Median", "median"):
            result_df = self._col_norm_median(result_df)
        elif col_norm in ("PQN", "pqn"):
            result_df = self._col_norm_pqn(result_df)

        # Step 2: Row (Feature) Normalization
        row_norm = self.attrs.get("row_norm")
        if row_norm in ("VSN", "vsn"):
            result_df = self._row_norm_vsn(result_df)
        elif row_norm in ("Auto", "auto"):
            result_df = self._row_norm_auto(result_df)
        elif row_norm in ("Pareto", "pareto"):
            result_df = self._row_norm_pareto(result_df)

        return result_df

    @iu._exe_time
    def execute_norm(self, output_dir: str) -> "MetaboIntNorm":
        """Execute normalization workflow and save outputs to disk.

        Args:
            output_dir: Directory path to save normalized data results.

        Returns:
            MetaboIntNorm: The normalized metabolomics data object.
        """
        iu._check_dir_exists(dir_path=output_dir, handle="makedirs")

        mode = self.attrs.get("mode", "POS")
        is_quantile = self.attrs.get("quantile_norm", False)

        if is_quantile:
            suffix = f"Quantile_{mode}"
            logger.info("Applying Standalone Quantile Normalization.")
        else:
            cn = self.attrs.get("col_norm", "None")
            rn = self.attrs.get("row_norm", "None")
            suffix = f"Col_{cn}_Row_{rn}_{mode}"
            logger.info(f"Applying Normalization | Col: {cn} | Row: {rn}")

        normalized_data = self.apply_normalization()

        file_path = os.path.join(output_dir, f"Normalized_Data_{suffix}.csv")
        normalized_data.to_csv(
            path_or_buf=file_path,
            na_rep="NA",
            encoding="utf-8-sig"
        )

        return normalized_data