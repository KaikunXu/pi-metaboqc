"""Implement SERRF-style random-forest signal correction.

The engine constructs feature-wise predictors from correlated reference
features, uses cross-validation for QC diagnostics, and projects fitted
corrections onto eligible samples. Scoring, export, and plotting remain
external.
"""

import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from ...constants import DEFAULT_RANDOM_SEED
from ...runtime import joblib_execution_context, joblib_progress


class SERRFCorrector:
    """Pure mathematical engine for Hybrid SERRF correction."""

    def __init__(
        self,
        n_estimators: int = 100,
        cv_folds: int = 5,
        n_corr_features: int = 10,
        random_state: int = DEFAULT_RANDOM_SEED,
        n_jobs: int = -1,
        joblib_backend: str = "loky",
        joblib_batch_size: Union[str, int] = "auto",
    ) -> None:
        """Initialize SERRF model and parallel-execution settings.

        Args:
            n_estimators: Number of trees fitted per feature.
            cv_folds: Folds used for QC out-of-fold prediction.
            n_corr_features: Correlated features supplied as predictors.
            random_state: Seed for deterministic model fitting.
            n_jobs: Number of parallel feature workers.
            joblib_backend: Joblib backend used for parallel execution.
            joblib_batch_size: Joblib task batching policy.
        """
        self.n_estimators = n_estimators
        self.cv_folds = cv_folds
        self.n_corr_features = n_corr_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.joblib_backend = str(joblib_backend).lower()
        self.joblib_batch_size = joblib_batch_size

    def _prepare_base_features(
        self, batch_array: np.ndarray, order_array: np.ndarray
    ) -> np.ndarray:
        x_df = pd.DataFrame({"Order": order_array, "Batch": batch_array})
        x_df["Order"] = pd.to_numeric(x_df["Order"], errors="coerce")
        x_encoded = pd.get_dummies(x_df, columns=["Batch"], drop_first=False)
        return np.nan_to_num(x_encoded.values, nan=0.0)

    def _process_single_feature(
        self,
        feat_idx: int,
        y_mat: np.ndarray,
        x_base: np.ndarray,
        is_qc: np.ndarray,
        top_idx_row: np.ndarray | None,
        blank_mask: np.ndarray | None,
        blank_predictor_values: np.ndarray | None,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """Worker function using pre-sliced indices for memory efficiency."""
        y_all = y_mat[:, feat_idx]

        # Feature Construction using pre-computed indices
        if top_idx_row is not None:
            x_corr = np.nan_to_num(y_mat[:, top_idx_row], nan=0.0)
            if blank_mask is not None and blank_predictor_values is not None:
                # SERRF is trained only on QC rows.  Blank rows should receive a
                # frozen technical prediction rather than supplying their
                # low-intensity correlated-feature values to the forest.
                x_corr[blank_mask, :] = np.nan_to_num(
                    blank_predictor_values[blank_mask, :][:, top_idx_row],
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            x_current = np.hstack([x_base, x_corr])
        else:
            x_current = x_base

        # Extract and validate QC
        y_qc = y_all[is_qc]
        valid_qc = ~np.isnan(y_qc) & (y_qc > 0)

        if valid_qc.sum() < self.cv_folds:
            return feat_idx, y_all, y_all

        x_qc_valid = x_current[is_qc][valid_qc]
        y_qc_valid = y_qc[valid_qc]

        # K-Fold Out-Of-Fold Prediction
        kf = KFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        y_pred_qc = np.zeros(len(y_qc_valid))

        rf_oof = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=1,
        )
        for train_idx, test_idx in kf.split(x_qc_valid):
            rf_oof.fit(x_qc_valid[train_idx], y_qc_valid[train_idx])
            y_pred_qc[test_idx] = rf_oof.predict(x_qc_valid[test_idx])

        # Predict Expected Baselines for ALL Samples
        rf_full = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=1,
        )
        rf_full.fit(x_qc_valid, y_qc_valid)
        y_pred_all_full = rf_full.predict(x_current)

        # Global Assembly for OOF Baseline
        y_pred_all_oof = y_pred_all_full.copy()
        pred_qc_oof_arr = y_all[is_qc].copy()
        pred_qc_oof_arr[valid_qc] = y_pred_qc
        pred_qc_oof_arr[~valid_qc] = np.nanmedian(y_qc_valid)
        y_pred_all_oof[is_qc] = pred_qc_oof_arr

        y_pred_all_full[y_pred_all_full <= 0] = 1e-6
        y_pred_all_oof[y_pred_all_oof <= 0] = 1e-6

        qc_median = np.nanmedian(y_qc_valid)
        res_full = (y_all / y_pred_all_full) * qc_median
        res_oof = (y_all / y_pred_all_oof) * qc_median

        return feat_idx, res_full, res_oof

    def fit_transform(
        self,
        intensity_df: pd.DataFrame,
        batch_array: np.ndarray,
        qc_mask: np.ndarray,
        order_array: np.ndarray,
        corr_mat: Optional[np.ndarray] = None,
        blank_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Fit feature-wise SERRF models and return corrected matrices.

        Args:
            intensity_df: Feature-by-sample intensity matrix.
            batch_array: Batch label for each sample.
            qc_mask: Boolean mask identifying pooled-QC samples.
            order_array: Injection order for each sample.
            corr_mat: Optional feature-correlation matrix.
            blank_mask: Optional mask excluding blanks from fitted predictors.

        Returns:
            Mapping containing full-fit and out-of-fold corrected matrices.

        Raises:
            ValueError: If the masks are invalid or QCs are insufficient.
        """

        logger.info("Initializing High-Performance Hybrid SERRF Corrector...")
        y_mat = intensity_df.T.to_numpy(dtype=float, copy=False)
        x_base = self._prepare_base_features(batch_array, order_array)
        n_features = y_mat.shape[1]
        if blank_mask is None:
            blank_mask = np.zeros(y_mat.shape[0], dtype=bool)
        else:
            blank_mask = np.asarray(blank_mask, dtype=bool)
            if blank_mask.shape != qc_mask.shape:
                raise ValueError("blank_mask must match the sample dimension.")

        if sum(qc_mask) < self.cv_folds:
            raise ValueError("Insufficient QCs for configured CV.")

        # Extract top features globally to avoid memory leak in parallel workers
        top_indices = None
        blank_predictor_values = None
        if self.n_corr_features > 0 and corr_mat is not None:
            # Prevent feature from selecting itself as highly correlated
            np.fill_diagonal(corr_mat, -1.0)
            top_indices = np.argsort(corr_mat, axis=1)[
                :, -self.n_corr_features :
            ]
            if blank_mask.any():
                blank_predictor_values = self._interpolate_qc_reference_values(
                    y_mat=y_mat,
                    batch_array=batch_array,
                    order_array=order_array,
                    qc_mask=qc_mask,
                )

        y_corrected = np.zeros_like(y_mat)
        y_corrected_oof = np.zeros_like(y_mat)

        actual_cores = (
            (os.cpu_count() or 1) if self.n_jobs == -1 else self.n_jobs
        )
        safe_n_jobs = max(1, int(actual_cores / 2))
        joblib_backend = self.joblib_backend
        if joblib_backend not in {"threading", "loky"}:
            logger.warning(
                f"Unsupported serrf_backend='{joblib_backend}'. "
                "Falling back to 'loky'."
            )
            joblib_backend = "loky"
        joblib_batch_size = self.joblib_batch_size
        if isinstance(joblib_batch_size, str):
            if joblib_batch_size.lower() == "auto":
                joblib_batch_size = "auto"
            else:
                joblib_batch_size = int(joblib_batch_size)

        # Use the configured joblib context for SERRF feature models.
        with joblib_execution_context(joblib_backend):
            with joblib_progress(total=n_features, desc="SERRF"):
                results = Parallel(
                    n_jobs=safe_n_jobs,
                    batch_size=joblib_batch_size,
                )(
                    delayed(self._process_single_feature)(
                        feat_idx,
                        y_mat,
                        x_base,
                        qc_mask,
                        None if top_indices is None else top_indices[feat_idx],
                        blank_mask,
                        blank_predictor_values,
                    )
                    for feat_idx in range(n_features)
                )

        if results:
            feat_order = [feat_idx for feat_idx, _, _ in results]
            y_corrected[:, feat_order] = np.column_stack(
                [res_full for _, res_full, _ in results]
            )
            y_corrected_oof[:, feat_order] = np.column_stack(
                [res_oof for _, _, res_oof in results]
            )

        res_df_full = pd.DataFrame(
            y_corrected.T,
            index=intensity_df.index,
            columns=intensity_df.columns,
        )
        res_df_oof = pd.DataFrame(
            y_corrected_oof.T,
            index=intensity_df.index,
            columns=intensity_df.columns,
        )
        return {"SERRF": (res_df_full, res_df_oof)}

    @staticmethod
    def _interpolate_qc_reference_values(
        y_mat: np.ndarray,
        batch_array: np.ndarray,
        order_array: np.ndarray,
        qc_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Build per-feature QC-only reference values at each injection order.
        """
        reference = y_mat.copy()
        orders = np.asarray(order_array, dtype=float)
        for batch in pd.unique(batch_array):
            batch_mask = batch_array == batch
            batch_qc = batch_mask & qc_mask
            x_qc = orders[batch_qc]
            if x_qc.size == 0:
                continue

            order_idx = np.argsort(x_qc, kind="mergesort")
            x_qc = x_qc[order_idx]
            qc_values = y_mat[batch_qc, :][order_idx, :]
            target_orders = orders[batch_mask]
            for feat_idx in range(y_mat.shape[1]):
                y_qc = qc_values[:, feat_idx]
                valid = np.isfinite(y_qc) & (y_qc > 0)
                if valid.sum() == 0:
                    continue
                if valid.sum() == 1:
                    reference[batch_mask, feat_idx] = y_qc[valid][0]
                else:
                    # np.interp holds endpoint values outside the QC range,
                    # giving a deterministic frozen-model fallback.
                    reference[batch_mask, feat_idx] = np.interp(
                        target_orders, x_qc[valid], y_qc[valid]
                    )
        return reference
