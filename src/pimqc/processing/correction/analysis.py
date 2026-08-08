"""Signal-correction orchestration, candidate evaluation, and selection.

MetaboIntCorrector prepares QC, batch, and injection-order inputs; applies a
configured correction method or evaluates AUTO candidates; and records metrics
for QC precision and sample-structure preservation. It writes corrected stage
matrices and audit artifacts while delegating numerical kernels and plotting.
"""

import os
import re
import math
import copy
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from joblib import Parallel, delayed
from loguru import logger
from typing import Any, Callable, Dict, Optional, Tuple, Union

from ...io import utils as iu
from ...core import model
from ...config import resolve_stage_config
from ...statistics import metrics as su
from ...statistics import selection as selection_utils

FitPredictCallable = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
CorrectionModel = (
    RandomForestRegressor
    | TransformedTargetRegressor
    | Pipeline
    | FitPredictCallable
)


from .algorithms import (
    _format_correction_method_file_label,
    _format_correction_method_label,
    _normalize_correction_method,
    _numba_batch_qc_rlsc,
    _numba_batch_qc_rlsc_per_feature_spans,
    _parse_correction_candidate,
    _select_batch_loess_spans,
    fit_predict_intra_batch_safely,
)


class RegressionCorrector:
    """Pure mathematical engine for regression-based drift correction."""

    def __init__(self, method: str, **kwargs: object) -> None:
        """
        Initialize purely with mathematical hyperparameters.
        No domain knowledge (like sample types or columns) is permitted here.
        """
        self.method = method.upper()
        self.params = kwargs

    def _build_correction_pipeline(self) -> CorrectionModel:
        """Construct the inner pipeline with forced single-thread policy."""
        if self.method in ("RF", "RANDOM FOREST", "QC-RFSC"):
            return RandomForestRegressor(
                n_estimators=self.params.get("rf_n_tree", 200),
                random_state=self.params.get("global_seed", 123),
                n_jobs=1,
            )
        elif self.method in ("SVR", "QC-SVR", "QC-SVRC"):
            # SVR is inherently single-threaded in scikit-learn
            base_svr = make_pipeline(
                StandardScaler(),
                SVR(
                    kernel=self.params.get("svr_kernel", "rbf"),
                    C=self.params.get("svr_c", 10),
                    gamma=self.params.get("svr_gamma", 1.0),
                ),
            )
            return TransformedTargetRegressor(
                regressor=base_svr, transformer=StandardScaler()
            )
        else:
            raise ValueError(f"Unsupported pipeline method: '{self.method}'.")

    def fit_transform(
        self,
        intensity_df: pd.DataFrame,
        batch_array: np.ndarray,
        qc_mask: np.ndarray,
        order_array: np.ndarray,
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Execute core mathematical fitting using pure arrays and masks."""
        stages_output = {}
        logger.info(
            "Phase 1: Executing Intra-batch drift correction with "
            f"{self.method}..."
        )

        # Force initialization to float64 to accept fractional predictions
        pred_df = intensity_df.copy().astype(float)
        oof_pred_df = intensity_df.copy().astype(float)
        unique_batches = np.unique(batch_array)

        # Restore full core utilization for lightweight threading
        n_jobs_conf = self.params.get("n_jobs", -1)

        # Intra-batch processing
        for batch_id in unique_batches:
            b_mask = batch_array == batch_id
            b_data = intensity_df.loc[:, b_mask]
            b_qc_mask = qc_mask[b_mask]
            b_orders = order_array[b_mask]

            if self.method in ("LOESS", "LOWESS", "QC-RLSC"):
                # Numba execution remains internal
                loess_span = self.params.get("loess_span", 0.3)
                robust = bool(
                    self.params.get(
                        "robust", self.params.get("rlsc_robust", True)
                    )
                )
                robust_iterations = max(
                    1,
                    int(
                        self.params.get(
                            "robust_iterations",
                            self.params.get("rlsc_robust_iterations", 3),
                        )
                    ),
                )
                max_iter = robust_iterations if robust else 0
                cv_folds = self.params.get("cv_folds", 5)
                seed = self.params.get("global_seed", 123)
                loess_degree = max(
                    1,
                    min(
                        2,
                        int(
                            self.params.get(
                                "loess_degree",
                                self.params.get("rlsc_loess_degree", 1),
                            )
                        ),
                    ),
                )
                span_selection = str(
                    self.params.get(
                        "rlsc_span_selection",
                        self.params.get("loess_span_selection", "fixed"),
                    )
                ).lower()
                use_gcv = span_selection == "gcv"
                span_grid_raw = self.params.get(
                    "rlsc_span_grid",
                    self.params.get("loess_span_grid", (0.3, 0.5, 0.7)),
                )
                span_grid = tuple(
                    sorted(
                        {
                            float(span)
                            for span in span_grid_raw
                            if 0.0 < float(span) <= 1.0
                        }
                    )
                )
                if not span_grid:
                    span_grid = (float(loess_span),)
                min_qc = max(
                    3,
                    int(
                        self.params.get(
                            "rlsc_min_qc",
                            self.params.get("loess_min_qc", 7),
                        )
                    ),
                )

                def _fit_batch(train_qc_mask: np.ndarray) -> np.ndarray:
                    if use_gcv:
                        selected_spans = _select_batch_loess_spans(
                            data=b_data.values,
                            qc_mask=train_qc_mask,
                            injection_orders=b_orders,
                            span_grid=span_grid,
                            max_iter=max_iter,
                            degree=loess_degree,
                            cv_folds=cv_folds,
                            fallback_span=float(loess_span),
                            min_qc=min_qc,
                            random_state=seed,
                        )
                        return _numba_batch_qc_rlsc_per_feature_spans(
                            data=b_data.values,
                            qc_mask=train_qc_mask,
                            injection_orders=b_orders,
                            loess_spans=selected_spans,
                            max_iter=max_iter,
                            degree=loess_degree,
                        )
                    return _numba_batch_qc_rlsc(
                        data=b_data.values,
                        qc_mask=train_qc_mask,
                        injection_orders=b_orders,
                        loess_span=loess_span,
                        max_iter=max_iter,
                        degree=loess_degree,
                    )

                pred_matrix = _fit_batch(b_qc_mask)
                pred_df.loc[:, b_mask] = pred_matrix

                oof_matrix = pred_matrix.copy()
                qc_indices = np.where(b_qc_mask)[0]
                if len(qc_indices) >= max(3, cv_folds):
                    kf = KFold(
                        n_splits=cv_folds, shuffle=True, random_state=seed
                    )
                    for train_idx, test_idx in kf.split(qc_indices):
                        train_qc_mask = b_qc_mask.copy()
                        train_qc_mask[qc_indices[test_idx]] = False
                        fold_pred = _fit_batch(train_qc_mask)
                        test_qcs = qc_indices[test_idx]
                        oof_matrix[:, test_qcs] = fold_pred[:, test_qcs]
                oof_pred_df.loc[:, b_mask] = oof_matrix
            else:
                feat_idx = b_data.index
                mat_vals = b_data.to_numpy(dtype=float, copy=False)
                joblib_backend = str(
                    self.params.get("regression_backend", "loky")
                ).lower()
                if joblib_backend not in {"threading", "loky"}:
                    logger.warning(
                        f"Unsupported regression_backend='{joblib_backend}'. "
                        "Falling back to 'loky'."
                    )
                    joblib_backend = "loky"
                joblib_batch_size = self.params.get(
                    "regression_batch_size", "auto"
                )
                if isinstance(joblib_batch_size, str):
                    if joblib_batch_size.lower() == "auto":
                        joblib_batch_size = "auto"
                    else:
                        joblib_batch_size = int(joblib_batch_size)
                batch_col_pos = np.where(b_mask)[0]

                tasks = (
                    delayed(self._fit_predict_feature)(
                        feat_idx[i], mat_vals[i, :], b_qc_mask, b_orders
                    )
                    for i in range(len(feat_idx))
                )

                # Use the configured joblib context for parallel regressions.
                with iu.tqdm_joblib_env(
                    total=len(feat_idx), desc=f"SC [{batch_id}]"
                ):
                    results = Parallel(
                        n_jobs=n_jobs_conf,
                        backend=joblib_backend,
                        batch_size=joblib_batch_size,
                    )(tasks)

                if results:
                    pred_df.iloc[:, batch_col_pos] = np.vstack(
                        [p_full for _, p_full, _ in results]
                    )
                    oof_pred_df.iloc[:, batch_col_pos] = np.vstack(
                        [p_oof for _, _, p_oof in results]
                    )

        # Mathematical division with broadcasted QC means
        pred_df[pred_df <= 0] = np.nan
        oof_pred_df[oof_pred_df <= 0] = np.nan

        qc_intensity = intensity_df.loc[:, qc_mask]
        batch_qc_means = qc_intensity.T.groupby(batch_array[qc_mask]).mean().T

        base_bc = pd.DataFrame(
            index=intensity_df.index, columns=intensity_df.columns
        )
        for b_id in unique_batches:
            b_mask = batch_array == b_id
            bc_blk = pd.concat([batch_qc_means[b_id]] * np.sum(b_mask), axis=1)
            base_bc.loc[:, b_mask] = bc_blk.values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            intra_full = base_bc * (intensity_df / pred_df)
            intra_oof = base_bc * (intensity_df / oof_pred_df)

        stages_output["Intra-batch corrected"] = (intra_full, intra_oof)

        # Inter-batch Median Alignment
        if len(unique_batches) > 1:
            logger.info("Phase 2: Executing Inter-batch median alignment...")
            inter_full = intra_full.copy()
            inter_oof = intra_oof.copy()

            intra_qc = intra_full.loc[:, qc_mask]
            bt_qc_mean = intra_qc.T.groupby(batch_array[qc_mask]).mean().T
            global_mean = intra_qc.mean(axis=1)

            intra_qc_oof = intra_oof.loc[:, qc_mask]
            bt_qc_mean_oof = (
                intra_qc_oof.T.groupby(batch_array[qc_mask]).mean().T
            )
            global_mean_oof = intra_qc_oof.mean(axis=1)

            for b_id in unique_batches:
                b_mask = batch_array == b_id
                inter_full.loc[:, b_mask] = inter_full.loc[:, b_mask].multiply(
                    global_mean / bt_qc_mean[b_id], axis=0
                )
                inter_oof.loc[:, b_mask] = inter_oof.loc[:, b_mask].multiply(
                    global_mean_oof / bt_qc_mean_oof[b_id], axis=0
                )
            stages_output["Inter-batch corrected"] = (inter_full, inter_oof)

        return stages_output

    def _fit_predict_feature(
        self,
        feat_idx: object,
        raw_vals: np.ndarray,
        qc_mask: np.ndarray,
        inject_order_array: np.ndarray,
    ) -> Tuple[object, np.ndarray, np.ndarray]:
        """Core worker for fitting a single feature via sklearn pipeline."""
        x_all = inject_order_array.reshape(-1, 1)
        qc_x = x_all[qc_mask]
        qc_y = raw_vals[qc_mask]
        valid = ~np.isnan(qc_y)

        if valid.sum() < 3:
            nan_arr = np.full(len(inject_order_array), np.nan)
            return feat_idx, nan_arr, nan_arr

        model = self._build_correction_pipeline()
        cv_folds = self.params.get("cv_folds", 5)
        seed = self.params.get("global_seed", 123)

        try:
            pred_all_full, pred_qc_oof = fit_predict_intra_batch_safely(
                base_model=model,
                x_qc=qc_x[valid],
                y_qc=qc_y[valid],
                x_all=x_all,
                cv_folds=cv_folds,
                random_state=seed,
            )
            pred_y_oof = pred_all_full.copy()
            pred_y_qc_oof = np.full(len(qc_x), np.nanmedian(qc_y[valid]))
            pred_y_qc_oof[valid] = pred_qc_oof
            pred_y_oof[qc_mask] = pred_y_qc_oof

            return feat_idx, pred_all_full, pred_y_oof
        except Exception as e:
            logger.debug(
                f"Feature {feat_idx} fit failed via {self.method}: {e}"
            )
            nan_arr = np.full(len(inject_order_array), np.nan)
            return feat_idx, nan_arr, nan_arr


# ==============================================================================
# Engine 2: SERRFCorrector
# ==============================================================================
class SERRFCorrector:
    """Pure mathematical engine for Hybrid SERRF correction."""

    def __init__(
        self,
        n_estimators: int = 100,
        cv_folds: int = 5,
        n_corr_features: int = 10,
        random_state: int = 123,
        n_jobs: int = -1,
        joblib_backend: str = "loky",
        joblib_batch_size: Union[str, int] = "auto",
    ) -> None:
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

        # 1. Feature Construction using pre-computed indices
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

        # 2. Extract and validate QC
        y_qc = y_all[is_qc]
        valid_qc = ~np.isnan(y_qc) & (y_qc > 0)

        if valid_qc.sum() < self.cv_folds:
            return feat_idx, y_all, y_all

        x_qc_valid = x_current[is_qc][valid_qc]
        y_qc_valid = y_qc[valid_qc]

        # 3. K-Fold Out-Of-Fold Prediction
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

        # 4. Predict Expected Baselines for ALL Samples
        rf_full = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=1,
        )
        rf_full.fit(x_qc_valid, y_qc_valid)
        y_pred_all_full = rf_full.predict(x_current)

        # 5. Global Assembly for OOF Baseline
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
        with iu.tqdm_joblib_env(total=n_features, desc="SERRF"):
            results = Parallel(
                n_jobs=safe_n_jobs,
                backend=joblib_backend,
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


# ==============================================================================
# Engine 3: WaveICA2Corrector
# ==============================================================================
class WaveICA2Corrector:
    """
    Native Python WaveICA 2.0 correction using order-associated stICA removal.

    This implementation ports the public WaveICA_2.0 R workflow more closely
    than the previous lightweight approximation: samples are ordered by
    injection order, each feature is decomposed with periodic Haar MODWT,
    scale-wise coefficients are factorized with the unbiased stICA/Jade-style
    joint diagonalization routine, order-associated components are removed, and
    the matrix is reconstructed with inverse MODWT plus the original feature
    mean, matching the R wrapper's reconstruction convention.

    """

    def __init__(
        self,
        n_components: int = 10,
        cutoff: float = 0.1,
        n_levels: Optional[int] = None,
        spline_knots: int = 5,
        max_iter: int = 1000,
        random_state: int = 123,
    ) -> None:
        self.n_components = max(1, int(n_components))
        self.cutoff = float(cutoff)
        self.n_levels = n_levels
        self.spline_knots = max(3, int(spline_knots))
        self.max_iter = max(50, int(max_iter))
        self.random_state = int(random_state)
        self.selected_component_counts: list[int] = []
        self.selected_component_r2: list[np.ndarray] = []
        self.blank_proxy_extrapolation_cells = 0
        self.blank_source_extrapolation_count = 0

    @staticmethod
    def _fill_missing_by_feature_median(data: np.ndarray) -> np.ndarray:
        """Fill missing feature values before matrix factorization."""
        filled = data.copy().astype(float)
        nan_mask = np.isnan(filled)
        if not nan_mask.any():
            return filled

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            col_medians = np.nanmedian(filled, axis=0)
            global_median = np.nanmedian(filled)

        if not np.isfinite(global_median):
            global_median = 0.0
        col_medians = np.where(
            np.isfinite(col_medians), col_medians, global_median
        )
        row_idx, col_idx = np.where(nan_mask)
        filled[row_idx, col_idx] = col_medians[col_idx]
        return filled

    def _decompose(self, data: np.ndarray) -> list[np.ndarray]:
        """Compute periodic Haar MODWT coefficients along the sample axis."""
        n_samples = data.shape[0]
        max_levels = max(1, int(np.floor(np.log2(max(n_samples, 2)))))
        if self.n_levels is not None:
            max_levels = max(1, min(int(self.n_levels), max_levels))

        coeffs = []
        smooth = data.copy()
        for level_idx in range(max_levels):
            lag = 2**level_idx
            lagged = np.roll(smooth, shift=lag, axis=0)
            detail = 0.5 * (smooth - lagged)
            smooth_next = 0.5 * (smooth + lagged)
            coeffs.append(detail)
            smooth = smooth_next
        coeffs.append(smooth)
        return coeffs

    @staticmethod
    def _reconstruct(coeffs: list[np.ndarray]) -> np.ndarray:
        """Reconstruct a matrix from periodic Haar MODWT coefficients."""
        smooth = coeffs[-1].copy()
        for level_idx in range(len(coeffs) - 2, -1, -1):
            detail = coeffs[level_idx]
            lag = 2**level_idx
            smooth = 0.5 * (
                smooth
                + detail
                + np.roll(smooth, shift=-lag, axis=0)
                - np.roll(detail, shift=-lag, axis=0)
            )
        return smooth

    @staticmethod
    def _jade_cumulant_matrices(x: np.ndarray) -> np.ndarray:
        """Calculate JADE cumulant-like matrices used by unbiased stICA."""
        n_rows, n_cols = x.shape
        n_mats = n_rows * (n_rows + 1) // 2
        mats = np.zeros((n_rows, n_rows, n_mats), dtype=float)
        cov = np.atleast_2d(np.cov(x.T, rowvar=False))
        scale = 1.0 / float(n_cols)

        mat_idx = 0
        for p_idx in range(n_rows):
            prod = x[p_idx] * x[p_idx]
            c_mat = (x * (prod * scale)[None, :]) @ x.T
            e_mat = np.zeros((n_rows, n_rows), dtype=float)
            e_mat[p_idx, p_idx] = 1.0
            mats[:, :, mat_idx] = (
                c_mat
                - cov @ e_mat @ cov
                - np.trace(e_mat @ cov) * cov
                - cov @ e_mat.T @ cov
            )
            mat_idx += 1

            for q_idx in range(p_idx):
                prod = x[p_idx] * x[q_idx]
                c_mat = (x * (prod * scale)[None, :]) @ x.T * np.sqrt(2.0)
                e_mat = np.zeros((n_rows, n_rows), dtype=float)
                e_mat[p_idx, q_idx] = 1.0 / np.sqrt(2.0)
                e_mat[q_idx, p_idx] = e_mat[p_idx, q_idx]
                mats[:, :, mat_idx] = (
                    c_mat
                    - cov @ e_mat @ cov
                    - np.trace(e_mat @ cov) * cov
                    - cov @ e_mat.T @ cov
                )
                mat_idx += 1

        return mats

    def _joint_diagonalize(self, mats: np.ndarray) -> np.ndarray:
        """
        Approximately jointly diagonalize symmetric matrices by Jacobi
        rotations.
        """
        n_rows = mats.shape[0]
        rot = np.eye(n_rows, dtype=float)
        work = mats.copy()
        eps = 1e-6

        for _ in range(self.max_iter):
            changed = False
            for p_idx in range(n_rows - 1):
                for q_idx in range(p_idx + 1, n_rows):
                    g0 = work[p_idx, p_idx, :] - work[q_idx, q_idx, :]
                    g1 = work[p_idx, q_idx, :] + work[q_idx, p_idx, :]
                    ton = float(np.dot(g0, g0) - np.dot(g1, g1))
                    toff = float(2.0 * np.dot(g0, g1))
                    denom = ton + math.sqrt(ton * ton + toff * toff)
                    if abs(denom) <= 1e-15 and abs(toff) <= 1e-15:
                        continue

                    theta = 0.5 * math.atan2(toff, denom)
                    c_val = math.cos(theta)
                    s_val = math.sin(theta)
                    if abs(s_val) <= eps:
                        continue

                    changed = True
                    left_p = (
                        c_val * work[p_idx, :, :] + s_val * work[q_idx, :, :]
                    )
                    left_q = (
                        -s_val * work[p_idx, :, :] + c_val * work[q_idx, :, :]
                    )
                    work[p_idx, :, :] = left_p
                    work[q_idx, :, :] = left_q

                    col_p = (
                        c_val * work[:, p_idx, :] + s_val * work[:, q_idx, :]
                    )
                    col_q = (
                        -s_val * work[:, p_idx, :] + c_val * work[:, q_idx, :]
                    )
                    work[:, p_idx, :] = col_p
                    work[:, q_idx, :] = col_q

                    rot_p = c_val * rot[:, p_idx] + s_val * rot[:, q_idx]
                    rot_q = -s_val * rot[:, p_idx] + c_val * rot[:, q_idx]
                    rot[:, p_idx] = rot_p
                    rot[:, q_idx] = rot_q

            if not changed:
                break
        return rot

    def _unbiased_stica(
        self,
        x: np.ndarray,
        n_components: int,
        alpha: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Port the WaveICA unbiased_stICA factorization X = A @ B.T."""
        n_features, n_samples = x.shape
        safe_k = min(int(n_components), n_features, n_samples)
        if safe_k < 1:
            raise ValueError(
                "WaveICA 2.0 stICA requires at least one component."
            )
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("WaveICA 2.0 alpha must be in [0, 1].")

        col_centered = x - np.mean(x, axis=0, keepdims=True)
        x_centered = col_centered - np.mean(col_centered, axis=1, keepdims=True)

        u_mat, s_vals, vt_mat = np.linalg.svd(x_centered, full_matrices=False)
        u_mat = u_mat[:, :safe_k]
        s_vals = s_vals[:safe_k]
        v_mat = vt_mat[:safe_k, :].T

        d_alpha = np.diag(s_vals**alpha)
        d_one_minus_alpha = np.diag(s_vals ** (1.0 - alpha))

        b_t = d_one_minus_alpha @ v_mat.T
        if alpha == 1.0:
            b_t = v_mat.T
        a_t = d_alpha @ u_mat.T
        if alpha == 0.0:
            a_t = u_mat.T

        n_cumulants = safe_k * (safe_k + 1) // 2
        mats = np.zeros((safe_k, safe_k, 2 * n_cumulants), dtype=float)
        mats[:, :, :n_cumulants] = self._jade_cumulant_matrices(b_t)
        mats[:, :, n_cumulants:] = self._jade_cumulant_matrices(a_t)

        first_norm = np.mean(
            np.sqrt(np.sum(mats[:, :, :n_cumulants] ** 2, axis=(0, 1)))
        )
        second_norm = np.mean(
            np.sqrt(np.sum(mats[:, :, n_cumulants:] ** 2, axis=(0, 1)))
        )
        if first_norm > 0:
            mats[:, :, :n_cumulants] *= alpha / first_norm
        else:
            mats[:, :, :n_cumulants] = 0.0
        if second_norm > 0:
            mats[:, :, n_cumulants:] *= (1.0 - alpha) / second_norm
        else:
            mats[:, :, n_cumulants:] = 0.0

        worth_v = self._joint_diagonalize(mats)
        wo_mat = worth_v.T
        wo_inv = np.linalg.pinv(wo_mat)

        a0 = u_mat @ d_alpha @ wo_inv
        b0 = v_mat @ d_one_minus_alpha @ wo_mat.T
        if alpha == 1.0:
            b0 = v_mat @ wo_mat.T
        if alpha == 0.0:
            a0 = u_mat @ wo_inv

        mean_cols = np.mean(x, axis=0, keepdims=True).T
        mean_rows = np.mean(x, axis=1, keepdims=True)
        mean_b = np.linalg.pinv(a0) @ mean_rows
        mean_a = np.linalg.pinv(b0) @ mean_cols

        b_fin = b0 + np.tile(mean_b.T, (n_samples, 1))
        a_fin = a0 + np.tile(mean_a.T, (n_features, 1))
        return a_fin, b_fin

    def _order_r2(
        self, component: np.ndarray, order_array: np.ndarray
    ) -> float:
        """Approximate mgcv GAM R2 for a component against injection order."""
        valid = np.isfinite(component) & np.isfinite(order_array)
        if valid.sum() < 4:
            return 0.0

        x = np.asarray(order_array[valid], dtype=float).reshape(-1, 1)
        y = np.asarray(component[valid], dtype=float)
        order_idx = np.argsort(x[:, 0])
        x = x[order_idx]
        y = y[order_idx]

        if np.nanstd(y) <= 1e-12:
            return 0.0

        try:
            n_knots = min(self.spline_knots, max(3, valid.sum() - 2))
            degree = min(3, n_knots - 1)
            basis = SplineTransformer(
                n_knots=n_knots,
                degree=degree,
                include_bias=False,
            ).fit_transform(x)
            y_hat = LinearRegression().fit(basis, y).predict(basis)
        except Exception:
            x_vec = x[:, 0]
            coefs = np.polyfit(x_vec, y, deg=1)
            y_hat = np.polyval(coefs, x_vec)

        sst = float(np.sum((y - np.mean(y)) ** 2))
        if sst <= 1e-12:
            return 0.0
        sse = float(np.sum((y - y_hat) ** 2))
        return float(np.clip(1.0 - sse / sst, 0.0, 1.0))

    @staticmethod
    def _interpolate_nonblank_proxy(
        raw: np.ndarray,
        batch_array: np.ndarray,
        order_array: np.ndarray,
        fit_mask: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Replace Blank rows with non-Blank, batch-local feature references."""
        proxy = raw.copy()
        blank_mask = ~fit_mask
        if not blank_mask.any():
            return proxy, 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            global_reference = np.nanmedian(raw[fit_mask, :], axis=0)
        global_reference = np.where(
            np.isfinite(global_reference), global_reference, 0.0
        )
        extrapolation_cells = 0
        orders = np.asarray(order_array, dtype=float)

        for batch in pd.unique(batch_array):
            batch_mask = batch_array == batch
            batch_fit = batch_mask & fit_mask
            batch_blank = batch_mask & blank_mask
            if not batch_blank.any():
                continue

            x_target = orders[batch_blank]
            x_fit = orders[batch_fit]
            if x_fit.size == 0:
                proxy[batch_blank, :] = global_reference
                extrapolation_cells += int(batch_blank.sum() * raw.shape[1])
                continue

            sort_idx = np.argsort(x_fit, kind="mergesort")
            x_fit = x_fit[sort_idx]
            fit_values = raw[batch_fit, :][sort_idx, :]
            for feat_idx in range(raw.shape[1]):
                y_fit = fit_values[:, feat_idx]
                valid = np.isfinite(y_fit)
                if valid.sum() == 0:
                    proxy[batch_blank, feat_idx] = global_reference[feat_idx]
                    extrapolation_cells += int(batch_blank.sum())
                    continue
                if valid.sum() == 1:
                    proxy[batch_blank, feat_idx] = y_fit[valid][0]
                    extrapolation_cells += int(batch_blank.sum())
                    continue

                valid_x = x_fit[valid]
                proxy[batch_blank, feat_idx] = np.interp(
                    x_target, valid_x, y_fit[valid]
                )
                extrapolation_cells += int(
                    np.sum((x_target < valid_x[0]) | (x_target > valid_x[-1]))
                )
        return proxy, extrapolation_cells

    def _predict_source_at_blank_orders(
        self,
        source: np.ndarray,
        order_array: np.ndarray,
        fit_mask: np.ndarray,
    ) -> np.ndarray:
        """Predict one frozen ICA source at Blank injection orders."""
        predicted = source.copy()
        blank_mask = ~fit_mask
        if not blank_mask.any():
            return predicted

        x_fit = np.asarray(order_array[fit_mask], dtype=float)
        y_fit = np.asarray(source[fit_mask], dtype=float)
        valid = np.isfinite(x_fit) & np.isfinite(y_fit)
        x_target = np.asarray(order_array[blank_mask], dtype=float)
        if valid.sum() == 0:
            predicted[blank_mask] = 0.0
            self.blank_source_extrapolation_count += int(blank_mask.sum())
            return predicted
        if valid.sum() == 1:
            predicted[blank_mask] = y_fit[valid][0]
            self.blank_source_extrapolation_count += int(blank_mask.sum())
            return predicted

        x_fit = x_fit[valid]
        y_fit = y_fit[valid]
        sort_idx = np.argsort(x_fit, kind="mergesort")
        x_fit = x_fit[sort_idx]
        y_fit = y_fit[sort_idx]
        outside = (x_target < x_fit[0]) | (x_target > x_fit[-1])
        self.blank_source_extrapolation_count += int(outside.sum())

        try:
            n_knots = min(self.spline_knots, max(3, len(x_fit) - 2))
            degree = min(3, n_knots - 1)
            transformer = SplineTransformer(
                n_knots=n_knots,
                degree=degree,
                include_bias=False,
                extrapolation="constant",
            )
            basis_fit = transformer.fit_transform(x_fit.reshape(-1, 1))
            basis_target = transformer.transform(x_target.reshape(-1, 1))
            predicted[blank_mask] = (
                LinearRegression().fit(basis_fit, y_fit).predict(basis_target)
            )
        except Exception:
            predicted[blank_mask] = np.interp(x_target, x_fit, y_fit)
        return predicted

    def _remove_order_components(
        self,
        coeff: np.ndarray,
        order_array: np.ndarray,
        fit_mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove stICA components whose scores are explained by injection order.
        """
        n_samples, n_features = coeff.shape
        if fit_mask is None:
            fit_mask = np.ones(n_samples, dtype=bool)
        else:
            fit_mask = np.asarray(fit_mask, dtype=bool)
        n_fit_samples = int(np.sum(fit_mask))
        safe_k = min(self.n_components, n_fit_samples, n_features)
        if safe_k < 2 or n_fit_samples < 4:
            self.selected_component_counts.append(0)
            self.selected_component_r2.append(np.array([], dtype=float))
            return coeff, np.zeros_like(coeff)

        try:
            mixing, sources = self._unbiased_stica(
                # Fit the ICA loading/source basis on non-Blank rows only.
                # Blank positions are filled later by frozen time prediction.
                x=coeff[fit_mask, :].T,
                n_components=safe_k,
                alpha=0.0,
            )
            r2_vals = np.array(
                [
                    self._order_r2(sources[:, i], order_array[fit_mask])
                    for i in range(safe_k)
                ]
            )
            selected = np.where(r2_vals >= self.cutoff)[0]
            self.selected_component_counts.append(int(len(selected)))
            self.selected_component_r2.append(r2_vals)

            if len(selected) == 0:
                return coeff, np.zeros_like(coeff)

            source_values = np.zeros((n_samples, safe_k), dtype=float)
            source_values[fit_mask, :] = sources
            for component_idx in selected:
                source_values[:, component_idx] = (
                    self._predict_source_at_blank_orders(
                        source_values[:, component_idx], order_array, fit_mask
                    )
                )
            artifact = (mixing[:, selected] @ source_values[:, selected].T).T
            return coeff - artifact, artifact
        except Exception as e:
            logger.debug(f"WaveICA 2.0 coefficient correction failed: {e}")
            self.selected_component_counts.append(0)
            self.selected_component_r2.append(np.array([], dtype=float))
            return coeff, np.zeros_like(coeff)

    def fit_transform(
        self,
        intensity_df: pd.DataFrame,
        order_array: np.ndarray,
        batch_array: Optional[np.ndarray] = None,
        blank_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
        """Execute WaveICA 2.0 correction."""
        logger.info("Executing WaveICA 2.0 correction...")

        order_array = np.asarray(order_array, dtype=float)
        if order_array.shape[0] != intensity_df.shape[1]:
            raise ValueError(
                "WaveICA 2.0 requires one injection order value per sample."
            )
        if batch_array is None:
            batch_array = np.zeros(intensity_df.shape[1], dtype=int)
        batch_array = np.asarray(batch_array)
        if batch_array.shape[0] != intensity_df.shape[1]:
            raise ValueError("WaveICA 2.0 requires one batch value per sample.")
        if blank_mask is None:
            blank_mask = np.zeros(intensity_df.shape[1], dtype=bool)
        blank_mask = np.asarray(blank_mask, dtype=bool)
        if blank_mask.shape[0] != intensity_df.shape[1]:
            raise ValueError("blank_mask must match the sample dimension.")

        sort_idx = np.argsort(order_array, kind="mergesort")
        inverse_idx = np.argsort(sort_idx)
        sorted_order = order_array[sort_idx]
        sorted_batch = batch_array[sort_idx]
        sorted_blank_mask = blank_mask[sort_idx]
        fit_mask = ~sorted_blank_mask
        sorted_df = intensity_df.iloc[:, sort_idx]

        raw = sorted_df.T.values.astype(float)
        nan_mask = np.isnan(raw)
        proxy, self.blank_proxy_extrapolation_cells = (
            self._interpolate_nonblank_proxy(
                raw=raw,
                batch_array=sorted_batch,
                order_array=sorted_order,
                fit_mask=fit_mask,
            )
        )
        filled = self._fill_missing_by_feature_median(proxy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fit_means = np.nanmean(raw[fit_mask, :], axis=0)
        fit_means = np.where(np.isfinite(fit_means), fit_means, 0.0)

        coeffs = self._decompose(filled)
        self.blank_source_extrapolation_count = 0
        corrected_pairs = [
            self._remove_order_components(coeff, sorted_order, fit_mask)
            for coeff in coeffs
        ]
        cleaned_coeffs = [pair[0] for pair in corrected_pairs]
        artifact_coeffs = [pair[1] for pair in corrected_pairs]
        corrected = self._reconstruct(cleaned_coeffs) + fit_means[None, :]
        if sorted_blank_mask.any():
            artifact = self._reconstruct(artifact_coeffs)
            corrected[sorted_blank_mask, :] = (
                raw[sorted_blank_mask, :]
                - artifact[sorted_blank_mask, :]
                + fit_means[None, :]
            )
            logger.info(
                "WaveICA 2.0 Blank policy: {} Blank samples received frozen "
                "artifact predictions ({} proxy fallback cells; {} source "
                "boundary fallbacks).",
                int(sorted_blank_mask.sum()),
                self.blank_proxy_extrapolation_cells,
                self.blank_source_extrapolation_count,
            )

        corrected = np.clip(corrected, a_min=1e-6, a_max=None)
        corrected[nan_mask] = np.nan
        corrected = corrected[inverse_idx, :]

        res_df_full = pd.DataFrame(
            corrected.T, index=intensity_df.index, columns=intensity_df.columns
        )
        return {"WaveICA 2.0": (res_df_full, None)}


# ==============================================================================
# Engine 4: RUVCorrector
# ==============================================================================
class RUVCorrector:
    """Pure mathematical engine for RUV-III correction via SVD projection."""

    def __init__(self, k: int = 3) -> None:
        self.k = k

    def fit_transform(
        self,
        intensity_df: pd.DataFrame,
        qc_mask: np.ndarray,
        control_features: pd.Index,
        blank_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Tuple[pd.DataFrame, Optional[pd.DataFrame]]]:

        logger.info(f"Executing RUV-III (k={self.k})...")
        if control_features.empty:
            raise ValueError("RUV-III requires at least one control feature.")

        Y_raw = intensity_df.T.values.astype(np.float64)
        n_samples, n_features = Y_raw.shape
        qc_mask = np.asarray(qc_mask, dtype=bool)
        if blank_mask is None:
            blank_mask = np.zeros(n_samples, dtype=bool)
        else:
            blank_mask = np.asarray(blank_mask, dtype=bool)
            if blank_mask.shape != qc_mask.shape:
                raise ValueError("blank_mask must match the sample dimension.")
        fit_mask = ~blank_mask
        if not np.any(fit_mask):
            raise ValueError("RUV-III requires at least one non-Blank sample.")

        Y_safe = np.clip(Y_raw, a_min=0, a_max=None)
        Y = np.log1p(Y_safe)

        # Define zero-only features from the fitting set so Blank background
        # cannot alter the RUV feature space.
        zero_mask = np.all(Y_safe[fit_mask, :] == 0, axis=0)

        nan_mask = np.isnan(Y)
        if nan_mask.any():
            logger.warning(
                "NaNs detected. Applying non-Blank median imputation..."
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                col_medians = np.nanmedian(Y[fit_mask, :], axis=0)
            col_medians[np.isnan(col_medians)] = 0.0
            nan_rows, nan_cols = np.where(nan_mask)
            Y[nan_rows, nan_cols] = col_medians[nan_cols]

        # Blank rows are held out of all fitted quantities.  QC rows share one
        # group, while each non-QC biological row remains its own RUV group.
        fit_indices = np.flatnonzero(fit_mask)
        group_ids = []
        uid_counter = 1
        for row_idx in fit_indices:
            is_qc = qc_mask[row_idx]
            if is_qc:
                group_ids.append(0)
            else:
                group_ids.append(uid_counter)
                uid_counter += 1

        n_groups = len(set(group_ids))
        M = np.zeros((len(fit_indices), n_groups), dtype=np.float64)
        for row_idx, g_id in enumerate(group_ids):
            M[row_idx, g_id] = 1.0

        group_sizes = M.T @ M
        Y_fit = Y[fit_mask, :]
        group_means = np.linalg.solve(group_sizes, M.T @ Y_fit)
        Y0 = Y_fit - (M @ group_means)

        # Exclude zero-variance control features dynamically
        ctl_mask = intensity_df.index.isin(control_features) & ~zero_mask
        Y0_ctl = Y0[:, ctl_mask]

        U, S, Vt = np.linalg.svd(Y0_ctl, full_matrices=False)
        safe_k = min(self.k, Y0_ctl.shape[0], Y0_ctl.shape[1])
        alpha_ctl = Vt[:safe_k, :]

        W_fit = Y_fit[:, ctl_mask] @ alpha_ctl.T
        W_means = np.linalg.solve(group_sizes, M.T @ W_fit)
        W0 = W_fit - (M @ W_means)
        alpha_full = np.linalg.lstsq(W0, Y0, rcond=None)[0]

        # Freeze the fitted correction centre.  Blanks are projected with the
        # same alpha matrices and centre, but never update either quantity.
        correction_fit = W_fit @ alpha_full
        correction_center = np.mean(correction_fit, axis=0)
        W_all = Y[:, ctl_mask] @ alpha_ctl.T
        correction = W_all @ alpha_full - correction_center

        # Protect absolute zero features from being negatively corrected
        correction[:, zero_mask] = 0.0

        Y_corr_log = Y - correction
        Y_corrected = np.expm1(Y_corr_log)
        Y_corrected = np.clip(Y_corrected, a_min=1e-6, a_max=None)

        # Restore original strict zeros
        Y_corrected[:, zero_mask] = 1e-6

        if nan_mask.any():
            Y_corrected[nan_mask] = np.nan

        res_df_full = intensity_df.copy()
        res_df_full.iloc[:, :] = Y_corrected.T

        return {"RUV-III": (res_df_full, None)}


# ==============================================================================
# Main Dispatcher Class: MetaboIntCorrector
# ==============================================================================
class MetaboIntCorrector(model.MetaboInt):
    """
    Quality control-based signal drift correction dispatcher.

    Orchestrates the dynamic execution routing between RegressionCorrector,
    SERRFCorrector, WaveICA2Corrector, and RUVCorrector. Extracts
    domain-specific metadata into pure mathematical arrays to preserve engine
    purity. Manages file exports, dual-mode RSD tracking, and downstream
    visualization diagnostics.
    """

    _metadata = ["attrs"]

    def __init__(
        self,
        *args: object,
        pipeline_params: Optional[Dict[str, Any]] = None,
        base_est: Optional[str] = None,
        loess_span: Optional[float] = None,
        loess_degree: Optional[int] = None,
        rlsc_span_selection: Optional[str] = None,
        rlsc_span_grid: Optional[list[float]] = None,
        rlsc_min_qc: Optional[int] = None,
        rlsc_robust: Optional[bool] = None,
        rlsc_robust_iterations: Optional[int] = None,
        rf_n_tree: Optional[int] = None,
        serrf_n_tree: Optional[int] = None,
        serrf_corr_features: Optional[int] = None,
        serrf_backend: Optional[str] = None,
        serrf_batch_size: Optional[Union[str, int]] = None,
        svr_kernel: Optional[str] = None,
        svr_c: Optional[Union[float, int]] = None,
        svr_gamma: Optional[Union[str, float]] = None,
        ruv_k: Optional[int] = None,
        waveica_components: Optional[int] = None,
        waveica_cutoff: Optional[float] = None,
        waveica_levels: Optional[int] = None,
        waveica_spline_knots: Optional[int] = None,
        waveica_max_iter: Optional[int] = None,
        regression_backend: Optional[str] = None,
        regression_batch_size: Optional[Union[str, int]] = None,
        cv_folds: Optional[int] = None,
        n_jobs: Optional[int] = None,
        **kwargs: object,
    ) -> None:
        """Initialize the signal drift correction dispatcher."""
        super().__init__(*args, pipeline_params=pipeline_params, **kwargs)

        sc_configs = resolve_stage_config(
            pipeline_params,
            "MetaboIntCorrector",
            {
                "base_est": "QC-RLSC",
                "loess_span": 0.5,
                "loess_degree": 1,
                "rlsc_span_selection": "fixed",
                "rlsc_span_grid": [0.3, 0.5, 0.7],
                "rlsc_min_qc": 7,
                "rlsc_robust": True,
                "rlsc_robust_iterations": 3,
                "rf_n_tree": 200,
                "serrf_n_tree": 100,
                "serrf_corr_features": 5,
                "serrf_backend": "loky",
                "serrf_batch_size": "auto",
                "svr_kernel": "rbf",
                "svr_c": 10,
                "svr_gamma": 1.0,
                "cv_folds": 3,
                "ruv_k": 3,
                "waveica_components": 10,
                "waveica_cutoff": 0.1,
                "waveica_levels": None,
                "waveica_spline_knots": 5,
                "waveica_max_iter": 1000,
                "regression_backend": "loky",
                "regression_batch_size": "auto",
                "n_jobs": getattr(iu, "__max_threading__", -1),
                "global_seed": 123,
            },
            {
                "base_est": base_est,
                "loess_span": loess_span,
                "loess_degree": loess_degree,
                "rlsc_span_selection": rlsc_span_selection,
                "rlsc_span_grid": rlsc_span_grid,
                "rlsc_min_qc": rlsc_min_qc,
                "rlsc_robust": rlsc_robust,
                "rlsc_robust_iterations": rlsc_robust_iterations,
                "rf_n_tree": rf_n_tree,
                "serrf_n_tree": serrf_n_tree,
                "serrf_corr_features": serrf_corr_features,
                "serrf_backend": serrf_backend,
                "serrf_batch_size": serrf_batch_size,
                "svr_kernel": svr_kernel,
                "svr_c": svr_c,
                "svr_gamma": svr_gamma,
                "cv_folds": cv_folds,
                "ruv_k": ruv_k,
                "waveica_components": waveica_components,
                "waveica_cutoff": waveica_cutoff,
                "waveica_levels": waveica_levels,
                "waveica_spline_knots": waveica_spline_knots,
                "waveica_max_iter": waveica_max_iter,
                "regression_backend": regression_backend,
                "regression_batch_size": regression_batch_size,
                "n_jobs": n_jobs,
            },
        )

        # 4. Integrate unified properties into internal attributes dictionary
        self.attrs.update(sc_configs)

    @property
    def _constructor(self) -> type["MetaboIntCorrector"]:
        """Override constructor to return MetaboIntCorrector."""
        return MetaboIntCorrector

    def __finalize__(
        self,
        other: object,
        method: Optional[str] = None,
        **kwargs: object,
    ) -> "MetaboIntCorrector":
        """Explicitly preserve custom attributes during operations."""
        super().__finalize__(other, method=method, **kwargs)
        for name in self._metadata:
            if hasattr(other, name):
                setattr(self, name, copy.deepcopy(getattr(other, name)))
        return self

    # =========================================================================
    # Domain Preprocessing & Statistical Core Methods
    # =========================================================================
    @staticmethod
    def extract_qc_rsd_series(df_obj: model.MetaboInt) -> pd.Series:
        """Extracts the RSD series for QC samples across all features."""
        if hasattr(df_obj, "_qc") and not df_obj._qc.empty:
            qc_data = df_obj._qc.astype(float)
        else:
            sample_type_col = df_obj.attrs.get("sample_type", "Sample Type")
            qc_label = df_obj.attrs.get("sample_dict", {}).get(
                "QC sample", "QC"
            )
            mask = df_obj.columns.get_level_values(sample_type_col) == qc_label
            qc_data = df_obj.loc[:, mask].astype(float)

        return (qc_data.std(axis=1, ddof=1) / qc_data.mean(axis=1)).dropna()

    @staticmethod
    def calculate_median_qc_rsd(df_obj: model.MetaboInt) -> float:
        """Calculates the scalar median RSD of QC samples."""
        rsd_series = MetaboIntCorrector.extract_qc_rsd_series(df_obj)
        if rsd_series.empty:
            return float("nan")
        return float(rsd_series.median())

    @staticmethod
    def calculate_featurewise_qc_rsd_improvement(
        before_obj: model.MetaboInt,
        after_obj: model.MetaboInt,
    ) -> dict[str, Any]:
        """Calculate paired feature-wise QC-RSD improvement diagnostics."""
        before_rsd = MetaboIntCorrector.extract_qc_rsd_series(before_obj)
        after_rsd = MetaboIntCorrector.extract_qc_rsd_series(after_obj)
        common_idx = before_rsd.index.intersection(after_rsd.index, sort=False)
        if common_idx.empty:
            return {
                "score": float("nan"),
                "median": float("nan"),
                "values": pd.Series(dtype=float),
            }

        before_vals = pd.to_numeric(before_rsd.loc[common_idx], errors="coerce")
        after_vals = pd.to_numeric(after_rsd.loc[common_idx], errors="coerce")
        valid = (
            np.isfinite(before_vals.to_numpy(dtype=float))
            & np.isfinite(after_vals.to_numpy(dtype=float))
            & (before_vals.to_numpy(dtype=float) > np.finfo(float).eps)
        )
        if not np.any(valid):
            return {
                "score": float("nan"),
                "median": float("nan"),
                "values": pd.Series(dtype=float),
            }

        before_vals = before_vals.iloc[np.flatnonzero(valid)]
        after_vals = after_vals.iloc[np.flatnonzero(valid)]
        signed_improvement = (before_vals - after_vals) / before_vals
        signed_improvement = signed_improvement.replace(
            [np.inf, -np.inf], np.nan
        )
        signed_improvement = signed_improvement.dropna()
        if signed_improvement.empty:
            return {
                "score": float("nan"),
                "median": float("nan"),
                "values": pd.Series(dtype=float),
            }

        clipped_improvement = signed_improvement.clip(lower=0.0, upper=1.0)
        winsor_low, winsor_high = np.nanpercentile(
            clipped_improvement.to_numpy(dtype=float), [5.0, 95.0]
        )
        winsorized = clipped_improvement.clip(
            lower=winsor_low, upper=winsor_high
        )
        return {
            "score": float(np.nanmean(winsorized.to_numpy(dtype=float))),
            "median": float(
                np.nanmedian(signed_improvement.to_numpy(dtype=float))
            ),
            "values": signed_improvement,
        }

    def _calculate_qc_baseline_means(
        self, batch_col: str, sample_type_col: str, qc_label: str
    ) -> pd.DataFrame:
        """Calculate batch-wise QC mean to reverse-engineer visual baselines."""
        qc_df = self.loc[
            :, self.columns.get_level_values(sample_type_col) == qc_label
        ]
        batch_levels = qc_df.columns.get_level_values(batch_col)
        int_base = qc_df.T.groupby(batch_levels).mean().T

        base_int_bc = pd.DataFrame(index=self.index, columns=self.columns)
        for batch in self.columns.get_level_values(batch_col).unique():
            mask = self.columns.get_level_values(batch_col) == batch
            bc_block = pd.concat([int_base[batch]] * mask.sum(), axis=1)
            base_int_bc.loc[:, mask] = bc_block.values

        return base_int_bc

    def _prepare_serrf_correlation_matrix(self) -> Optional[np.ndarray]:
        """Domain logic: Compute Spearman correlation on QCs."""
        if hasattr(self, "_qc") and not self._qc.empty:
            logger.info("Calculating Spearman correlation on features...")

            # _qc shape: (n_features, n_qc_samples)
            qc_df = self._qc.reindex(self.index).astype(float)

            # Rank across samples (axis=1) to prevent indexing errors
            rank_arr = qc_df.rank(axis=1).values

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # Calculates feature-to-feature correlation mapping
                corr_mat = np.abs(np.corrcoef(rank_arr))
                corr_mat = np.nan_to_num(corr_mat, nan=-1.0)
            return corr_mat
        return None

    def _prepare_ruv_control_features(
        self, empirical_ratio: float = 0.05
    ) -> pd.Index:
        """Domain logic: Fuse predefined and actual empirical controls."""
        is_list = getattr(self, "valid_is", [])
        orf_list = getattr(self, "valid_orf", [])
        base_controls = set(is_list + orf_list)

        empirical_controls = []
        if hasattr(self, "_actual_sample") and not self._actual_sample.empty:
            actual_data = self._actual_sample.astype(float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                r_series = actual_data.std(axis=1, ddof=1)
                r_series = r_series / actual_data.mean(axis=1)

            valid_rsd = r_series.replace([np.inf, -np.inf], np.nan).dropna()
            n_empirical = max(10, int(len(self) * empirical_ratio))
            empirical_controls = valid_rsd.nsmallest(n_empirical).index.tolist()

        combined_controls = base_controls.union(empirical_controls)
        valid_ctl = pd.Index(list(combined_controls)).intersection(self.index)

        logger.info(
            f"RUV-III Control Features: {len(valid_ctl)} total "
            f"({len(base_controls)} predefined, "
            f"{len(empirical_controls)} empirical)."
        )
        return valid_ctl

    def _evaluate_correction_candidates(
        self,
        methods_to_run: list,
        batch_array: np.ndarray,
        qc_mask: np.ndarray,
        blank_mask: np.ndarray,
        order_array: np.ndarray,
        batch_col: str,
        sample_type_col: str,
        qc_label: str,
    ) -> Dict[str, Any]:
        """
        Evaluate configured correction candidates and collect selection metrics.
        """
        results_store = {}
        for raw_method in methods_to_run:
            method, candidate_label, candidate_params = (
                _parse_correction_candidate(raw_method)
            )
            logger.info(f"--- Evaluating Method: {candidate_label} ---")

            # 1. Route to specific engine
            if method == "SERRF":
                corr_mat = self._prepare_serrf_correlation_matrix()
                engine = SERRFCorrector(
                    n_estimators=self.attrs.get("serrf_n_tree", 100),
                    cv_folds=self.attrs.get("cv_folds", 5),
                    n_corr_features=self.attrs.get("serrf_corr_features", 10),
                    random_state=self.attrs.get("global_seed", 123),
                    n_jobs=self.attrs.get("n_jobs", -1),
                    joblib_backend=self.attrs.get("serrf_backend", "loky"),
                    joblib_batch_size=self.attrs.get(
                        "serrf_batch_size", "auto"
                    ),
                )
                stages_output = engine.fit_transform(
                    intensity_df=self,
                    batch_array=batch_array,
                    qc_mask=qc_mask,
                    order_array=order_array,
                    corr_mat=corr_mat,
                    blank_mask=blank_mask,
                )
            elif method == "RUV-III":
                ctrl_features = self._prepare_ruv_control_features()
                engine = RUVCorrector(k=self.attrs.get("ruv_k", 3))
                stages_output = engine.fit_transform(
                    intensity_df=self,
                    qc_mask=qc_mask,
                    control_features=ctrl_features,
                    blank_mask=blank_mask,
                )
            elif method == "WaveICA 2.0":
                engine = WaveICA2Corrector(
                    n_components=self.attrs.get("waveica_components", 10),
                    cutoff=self.attrs.get("waveica_cutoff", 0.1),
                    n_levels=self.attrs.get("waveica_levels"),
                    spline_knots=self.attrs.get("waveica_spline_knots", 5),
                    max_iter=self.attrs.get("waveica_max_iter", 1000),
                    random_state=self.attrs.get("global_seed", 123),
                )
                stages_output = engine.fit_transform(
                    intensity_df=self,
                    order_array=order_array,
                    batch_array=batch_array,
                    blank_mask=blank_mask,
                )
            else:
                candidate_attrs = dict(self.attrs)
                candidate_attrs.update(candidate_params)
                engine = RegressionCorrector(method=method, **candidate_attrs)
                stages_output = engine.fit_transform(
                    intensity_df=self,
                    batch_array=batch_array,
                    qc_mask=qc_mask,
                    order_array=order_array,
                )

            # 2. Extract DataFrames and calculate RSD tracking
            raw_rsd = MetaboIntCorrector.calculate_median_qc_rsd(self)
            rsd_hist_oof = {"Original": raw_rsd}
            rsd_hist_full = {"Original": raw_rsd}
            stage_dfs = {"Original": self}
            stage_oof_dfs = {}

            for stage_name, (full_df, oof_df) in stages_output.items():
                clean_name = stage_name.replace("\n", " ")
                final_df = self._constructor(full_df).__finalize__(self)
                final_df.attrs["pipeline_stage"] = "Correction"
                final_df.attrs["qc_rsd_baseline"] = raw_rsd

                curr_full_rsd = MetaboIntCorrector.calculate_median_qc_rsd(
                    final_df
                )
                rsd_hist_full[clean_name] = curr_full_rsd
                final_df.attrs["qc_rsd_current_full"] = curr_full_rsd

                if oof_df is not None:
                    oof_wrap = self._constructor(oof_df).__finalize__(self)
                    oof_wrap.attrs["pipeline_stage"] = "Correction"
                    curr_oof_rsd = MetaboIntCorrector.calculate_median_qc_rsd(
                        oof_wrap
                    )
                    rsd_hist_oof[clean_name] = curr_oof_rsd
                    final_df.attrs["qc_rsd_current_oof"] = curr_oof_rsd
                    stage_oof_dfs[clean_name] = oof_wrap
                else:
                    rsd_hist_oof[clean_name] = None
                    final_df.attrs["qc_rsd_current_oof"] = None

                stage_dfs[clean_name] = final_df

            for name, df in stage_dfs.items():
                if name != "Original":
                    df.attrs["rsd_history_oof"] = rsd_hist_oof
                    df.attrs["rsd_history_full"] = rsd_hist_full

            # 3. Algebraically Reverse-Engineer Smooth Fit Baseline Matrix
            pred_df = None
            if "Intra-batch corrected" in stage_dfs and method not in (
                "SERRF",
                "RUV-III",
                "WaveICA 2.0",
            ):
                try:
                    base_int_bc = self._calculate_qc_baseline_means(
                        batch_col, sample_type_col, qc_label
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        raw_pred_df = base_int_bc * (
                            self / stage_dfs["Intra-batch corrected"]
                        )
                    pred_df = self._constructor(raw_pred_df).__finalize__(self)
                except Exception as e:
                    logger.debug(f"Baseline back-calc failed: {e}")

            # 4. Extract final RSD to evaluate performance
            final_stage = list(stage_dfs.keys())[-1]
            final_full = rsd_hist_full[final_stage]
            final_oof = rsd_hist_oof.get(final_stage)
            full_only_methods = ("RUV-III", "WaveICA 2.0")
            eval_rsd = final_full if method in full_only_methods else final_oof
            if eval_rsd is None:
                eval_rsd = final_full

            median_qc_rsd_improvement_score = float(
                np.clip(
                    su.relative_change_lower_better(raw_rsd, eval_rsd),
                    0.0,
                    1.0,
                )
            )
            final_corrected_df = stage_dfs[final_stage]
            eval_corrected_df = (
                stage_oof_dfs.get(final_stage)
                if method not in full_only_methods
                else final_corrected_df
            )
            if eval_corrected_df is None:
                eval_corrected_df = final_corrected_df
            featurewise_improvement = (
                MetaboIntCorrector.calculate_featurewise_qc_rsd_improvement(
                    before_obj=self,
                    after_obj=eval_corrected_df,
                )
            )
            featurewise_qc_rsd_improvement_score = su.finite_or_nan(
                featurewise_improvement.get("score")
            )
            structure_metrics = su.calc_sample_structure_preservation(
                raw_obj=self,
                transformed_obj=final_corrected_df,
                max_features=5000,
                seed=int(self.attrs.get("global_seed", 123)),
            )
            sample_structure_score = su.finite_or_nan(
                structure_metrics.get("sample_structure_composite_preservation")
            )
            sample_structure_score = (
                float(np.clip(sample_structure_score, 0.0, 1.0))
                if np.isfinite(sample_structure_score)
                else float("nan")
            )
            auto_score = su.weighted_mean_score(
                [
                    (median_qc_rsd_improvement_score, 0.35),
                    (featurewise_qc_rsd_improvement_score, 0.35),
                    (sample_structure_score, 0.30),
                ],
            )

            results_store[candidate_label] = {
                "method": method,
                "candidate_label": candidate_label,
                "candidate_params": candidate_params,
                "stage_dfs": stage_dfs,
                "stage_oof_dfs": stage_oof_dfs,
                "pred_df": pred_df,
                "final_rsd_full": final_full,
                "final_rsd_oof": final_oof,
                "eval_rsd": eval_rsd,
                "median_qc_rsd_improvement_score": (
                    median_qc_rsd_improvement_score
                ),
                "featurewise_qc_rsd_improvement_score": (
                    featurewise_qc_rsd_improvement_score
                ),
                "featurewise_qc_rsd_improvement_median": (
                    featurewise_improvement.get("median")
                ),
                "featurewise_qc_rsd_improvement_values": (
                    featurewise_improvement.get("values")
                ),
                "sample_structure_score": sample_structure_score,
                "sample_structure_metrics": structure_metrics,
                "auto_score": auto_score,
            }

            log_rsd = eval_rsd
            if log_rsd is not None:
                logger.info(
                    f"{candidate_label} Eval QC RSD: {log_rsd * 100:.2f}%"
                )

        return results_store

    def _select_best_correction_method(
        self, results_store: Dict[str, Any]
    ) -> str:
        """
        Identify the optimal correction method using Auto score.

        The AUTO score combines median QC-RSD improvement, feature-wise
        QC-RSD improvement, and actual-sample structure preservation.
        RUV-III and WaveICA 2.0 use global-model QC-RSD evaluation; methods
        with OOF support use the OOF metric.
        """
        if not results_store:
            return ""

        rank_rows = []
        for method, result in results_store.items():
            auto_score = su.finite_or_nan(result.get("auto_score"))
            rank_rows.append(
                {
                    "method": method,
                    # Preserve the prior fallback policy for unavailable scores.
                    "auto_score": (
                        auto_score
                        if np.isfinite(auto_score)
                        else np.finfo(float).min
                    ),
                    "eval_rsd": self._get_correction_eval_rsd(method, result),
                }
            )

        ranked = selection_utils.rank_candidates(
            pd.DataFrame(rank_rows),
            score_column="auto_score",
            tie_breakers=(("eval_rsd", True), ("method", True)),
        )
        return str(ranked.iloc[0]["method"])

    @staticmethod
    def _get_correction_eval_rsd(method: str, result: dict[str, Any]) -> float:
        """Return the QC-RSD metric used for correction-method selection."""
        cached_eval = su.finite_or_nan(result.get("eval_rsd"))
        if np.isfinite(cached_eval):
            return cached_eval

        canonical_method = _normalize_correction_method(
            result.get("method", method)
        )
        if canonical_method in ("RUV-III", "WaveICA 2.0"):
            return float(result.get("final_rsd_full", float("inf")))

        eval_rsd = result.get("final_rsd_oof")
        if eval_rsd is None:
            eval_rsd = result.get("final_rsd_full", float("inf"))
        return float(eval_rsd)

    # =========================================================================
    # Core Pipeline Execution Flow
    # =========================================================================
    @iu._exe_time
    def execute_signal_correction(self, output_dir: str) -> Dict[str, Any]:
        """Execute complete signal correction workflow dynamically."""
        # Cast the internal matrix to float for safe in-place regression
        # updates.
        self._update_inplace(self.astype(float))

        self.attrs["pipeline_stage"] = "Original"
        iu._check_dir_exists(output_dir, handle="makedirs")

        # Extract domain context strictly into mathematical arrays
        sample_type_col = self.attrs.get("sample_type", "Sample Type")
        batch_col = self.attrs.get("batch", "Batch")
        inject_order_col = self.attrs.get("inject_order", "Inject Order")

        sample_dict = self.attrs.get("sample_dict", {})
        qc_label = sample_dict.get("QC sample", "QC")
        actual_label = sample_dict.get("Actual sample", "Sample")
        blank_label = sample_dict.get("Blank sample", "Blank")

        qc_mask = self.columns.get_level_values(sample_type_col) == qc_label
        blank_mask = (
            self.columns.get_level_values(sample_type_col) == blank_label
        )
        batch_array = self.columns.get_level_values(batch_col).values
        order_array = self.columns.get_level_values(inject_order_col).values

        if blank_mask.any():
            logger.info(
                "Blank policy: {} Blank samples are excluded from "
                "model fitting "
                "where supported and retained as frozen-model corrections "
                "in output.",
                int(blank_mask.sum()),
            )

        req_method = _normalize_correction_method(
            self.attrs.get("base_est", "QC-RLSC")
        )
        self.attrs["is_auto_mode"] = req_method == "AUTO"

        if req_method == "AUTO":
            methods_to_run = [
                "SERRF",
                "RUV-III",
                "WaveICA 2.0",
                {
                    "method": "QC-RLSC",
                    "label": "QC-RLSC",
                    "params": {"robust": False},
                },
                {
                    "method": "QC-RLSC",
                    "label": "robust QC-RLSC",
                    "params": {"robust": True},
                },
                "QC-SVR",
            ]
            logger.info("AUTO mode enabled. Evaluating multiple methods.")
        else:
            methods_to_run = [req_method]

        # ---------------------------------------------------------------------
        # 1. Computation & Evaluation Phase
        # ---------------------------------------------------------------------
        results_store = self._evaluate_correction_candidates(
            methods_to_run=methods_to_run,
            batch_array=batch_array,
            qc_mask=qc_mask,
            blank_mask=blank_mask,
            order_array=order_array,
            batch_col=batch_col,
            sample_type_col=sample_type_col,
            qc_label=qc_label,
        )

        # ---------------------------------------------------------------------
        # 2. Selection Phase
        # ---------------------------------------------------------------------
        best_label = self._select_best_correction_method(results_store)
        best_result = results_store[best_label]
        best_method = best_result.get(
            "method",
            _normalize_correction_method(best_label),
        )

        if req_method == "AUTO":
            best_rsd = self._get_correction_eval_rsd(
                method=best_label, result=best_result
            )
            best_score = su.finite_or_nan(best_result.get("auto_score"))

            logger.success(
                "Auto selection: "
                f"{_format_correction_method_label(best_label)} is optimal "
                f"(score = {best_score:.3f}, "
                f"Eval QC RSD = {best_rsd * 100:.2f}%)."
            )
            # Update metric tracker to reflect dynamically chosen algorithm
            self.attrs["base_est"] = best_method
            self.attrs["correction_method_label"] = best_label
            selected_params = best_result.get("candidate_params", {})
            if best_method == "QC-RLSC":
                self.attrs["rlsc_robust"] = selected_params.get(
                    "robust", self.attrs.get("rlsc_robust", True)
                )
                self.attrs["rlsc_robust_iterations"] = selected_params.get(
                    "robust_iterations",
                    self.attrs.get("rlsc_robust_iterations", 3),
                )

            # Ensure the propagated DataFrames carry the resolved name
            for df in best_result["stage_dfs"].values():
                df.attrs["base_est"] = best_method
                df.attrs["correction_method_label"] = best_label

        # ---------------------------------------------------------------------
        # 3. Visualization Routing Phase
        # ---------------------------------------------------------------------
        vis = MetaboVisualizerCorrector(self)
        best_method_file_label = _format_correction_method_file_label(
            best_method
        )
        best_dashboard_file_label = best_method_file_label.replace(" ", "_")

        logger.info("Assembling correction diagnostic dashboard...")
        grid_obj = vis.plot_correction_dashboard(
            results_store,
            best_label,
            include_auto_summary=req_method == "AUTO"
            and len(results_store) > 1,
        )
        if grid_obj is not None:
            vis.save_and_show_pw(
                pw_obj=grid_obj,
                width="60%",
                file_path=os.path.join(
                    output_dir,
                    f"Correction_Dashboard_{best_dashboard_file_label}.svg",
                ),
            )

        if req_method == "AUTO" and len(results_store) > 1:
            candidate_obj = vis.plot_correction_candidate_grid(
                results_store=results_store,
                best_method=best_label,
            )
            if candidate_obj is not None:
                candidate_path = os.path.join(
                    output_dir,
                    "Correction_Candidate_Dashboard_"
                    f"{best_dashboard_file_label}.svg",
                )
                vis.save_and_show_pw(
                    pw_obj=candidate_obj,
                    width="60%",
                    file_path=candidate_path,
                )
                logger.info(
                    f"Correction candidate dashboard saved as: {candidate_path}"
                )

            # article_obj = vis.plot_correction_article_dashboard(
            #     results_store=results_store,
            #     best_method=best_method,
            # )
            # if article_obj is not None:
            #     article_path = os.path.join(
            #         output_dir,
            # f"Correction_Article_Dashboard_{best_dashboard_file_label}.svg",
            #     )
            #     vis.save_and_show_pw(
            #         pw_obj=article_obj,
            #         width="60%",
            #         file_path=article_path,
            #     )
            #     logger.info(
            #         f"Correction article dashboard saved as: {article_path}"
            #     )

        best_method_label = _format_correction_method_label(best_label)

        # ---------------------------------------------------------------------
        # 4. File Export Phase (Exclusive to the optimal method)
        # ---------------------------------------------------------------------
        best_res = best_result
        self.stage_dfs = best_res["stage_dfs"]
        best_pred_df = best_res["pred_df"]

        for stage_name, df in self.stage_dfs.items():
            if stage_name == "Original":
                continue

            clean_name = stage_name.replace("\n", " ")
            if best_method == "SERRF":
                file_name = "SERRF.csv"
            elif best_method == "RUV-III":
                file_name = "RUV-III.csv"
            elif best_method == "WaveICA 2.0":
                file_name = "WaveICA 2.0.csv"
            else:
                prefix = clean_name.replace(" corrected", "")
                prefix = prefix.replace(" ", "_")
                file_name = f"{prefix}_{best_method_file_label}.csv"

            df.to_csv(os.path.join(output_dir, file_name))

        if best_pred_df is not None:
            best_pred_df.to_csv(
                os.path.join(
                    output_dir, f"QC_Fit_Base_{best_method_file_label}.csv"
                )
            )

        pipe_params = self.attrs.get("pipeline_parameters", {})
        bound_type = pipe_params.get("MetaboInt", {}).get("boundary", "IQR")

        if len(self.valid_is) > 0:
            logger.info(f"Generating IS plots for {best_method_label}...")

            is_dir = os.path.join(output_dir, "Internal_Standard_Scatters")
            iu._check_dir_exists(is_dir, handle="makedirs")

            # Consume the generator: Create -> Save -> Clear iteratively
            for feat, fig in vis.plot_is_int_order_scatter(
                self.stage_dfs,
                best_pred_df,
                self.valid_is,
                sample_type_col,
                batch_col,
                inject_order_col,
                qc_label,
                actual_label,
                bound_type,
            ):
                safe_feat = re.sub(r"[^a-zA-Z0-9]", "_", feat)
                save_path = os.path.join(
                    is_dir,
                    f"IS_Scatter_{safe_feat}_{best_method_file_label}.svg",
                )
                vis.save_and_show_pw(
                    pw_obj=fig, file_path=save_path, show_plot=False
                )

            if best_method not in (
                "SERRF",
                "RUV-III",
                "WaveICA 2.0",
            ) and (best_pred_df is not None):
                fig_pred = vis.plot_pred_baseline_is(
                    self,
                    best_pred_df,
                    self.valid_is,
                    sample_type_col,
                    batch_col,
                    inject_order_col,
                    qc_label,
                    actual_label,
                    method=best_method,
                )
                if fig_pred is not None:
                    vis.save_and_show_pw(
                        pw_obj=fig_pred,
                        file_path=os.path.join(
                            output_dir,
                            f"Pred_Base_IS_{best_method_file_label}.svg",
                        ),
                        show_plot=False,
                    )
            else:
                logger.info(
                    f"Bypassing IS baseline prediction for {best_method_label}."
                )

        logger.success(
            f"Signal drift correction ({best_method_label}) completed."
        )
        return {k: v for k, v in self.stage_dfs.items() if k != "Original"}

    @property
    def correction_metrics(self) -> Dict[str, Any]:
        """Extracts comprehensive multi-stage correction metrics."""
        stage = self.attrs.get("pipeline_stage", "Unknown")
        rsd_base = self.attrs.get("qc_rsd_baseline")
        rsd_curr_oof = self.attrs.get("qc_rsd_current_oof")
        rsd_curr_full = self.attrs.get("qc_rsd_current_full")
        hist_oof = self.attrs.get("rsd_history_oof", {})
        hist_full = self.attrs.get("rsd_history_full", {})
        method = _normalize_correction_method(
            self.attrs.get("base_est", "Unknown")
        )

        metrics = {
            "correction_status": stage,
            "is_auto_mode": self.attrs.get("is_auto_mode", False),
            "overall_performance": {
                "median_qc_rsd_baseline": rsd_base,
                "median_qc_rsd_current_oof": rsd_curr_oof,
                "median_qc_rsd_current_full": rsd_curr_full,
                "relative_noise_reduction_oof": None,
                "relative_noise_reduction_full": None,
            },
            "stages_executed": [],
        }

        if rsd_base is not None and rsd_base > 0:
            if rsd_curr_oof is not None:
                oof_reduction = (rsd_base - rsd_curr_oof) / rsd_base
                metrics["overall_performance"][
                    "relative_noise_reduction_oof"
                ] = oof_reduction
            if rsd_curr_full is not None:
                full_reduction = (rsd_base - rsd_curr_full) / rsd_base
                metrics["overall_performance"][
                    "relative_noise_reduction_full"
                ] = full_reduction

        for stage_name in hist_oof.keys():
            if stage_name == "Original":
                continue

            alg_identifier = method
            if "Inter-batch" in stage_name:
                alg_identifier = "QC Median Alignment"

            # Dynamically build parameter dict based on the executed algorithm
            stage_params = {}
            if alg_identifier != "QC Median Alignment":
                if alg_identifier in ("QC-RLSC", "LOESS"):
                    stage_params["loess_span"] = self.attrs.get("loess_span")
                    stage_params["loess_degree"] = self.attrs.get(
                        "loess_degree"
                    )
                    stage_params["rlsc_span_selection"] = self.attrs.get(
                        "rlsc_span_selection"
                    )
                    stage_params["rlsc_span_grid"] = self.attrs.get(
                        "rlsc_span_grid"
                    )
                    stage_params["rlsc_min_qc"] = self.attrs.get("rlsc_min_qc")
                    stage_params["rlsc_robust"] = self.attrs.get("rlsc_robust")
                    stage_params["rlsc_robust_iterations"] = self.attrs.get(
                        "rlsc_robust_iterations"
                    )
                elif alg_identifier in ("QC-RFSC", "RF"):
                    stage_params["n_estimators"] = self.attrs.get("rf_n_tree")
                elif alg_identifier == "QC-SVR":
                    stage_params["svr_kernel"] = self.attrs.get("svr_kernel")
                    stage_params["svr_c"] = self.attrs.get("svr_c")
                    stage_params["svr_gamma"] = self.attrs.get("svr_gamma")
                elif alg_identifier == "SERRF":
                    stage_params["n_estimators"] = self.attrs.get(
                        "serrf_n_tree"
                    )
                    stage_params["n_corr_features"] = self.attrs.get(
                        "serrf_corr_features"
                    )
                    stage_params["backend"] = self.attrs.get("serrf_backend")
                    stage_params["batch_size"] = self.attrs.get(
                        "serrf_batch_size"
                    )
                elif alg_identifier == "RUV-III":
                    stage_params["ruv_k"] = self.attrs.get("ruv_k")
                elif alg_identifier == "WaveICA 2.0":
                    stage_params["n_components"] = self.attrs.get(
                        "waveica_components"
                    )
                    stage_params["cutoff"] = self.attrs.get("waveica_cutoff")
                    stage_params["n_levels"] = self.attrs.get("waveica_levels")
                    stage_params["spline_knots"] = self.attrs.get(
                        "waveica_spline_knots"
                    )
                    stage_params["max_iter"] = self.attrs.get(
                        "waveica_max_iter"
                    )

                if alg_identifier not in (
                    "RUV-III",
                    "WaveICA 2.0",
                ):
                    stage_params["cv_folds"] = self.attrs.get("cv_folds")

            metrics["stages_executed"].append(
                {
                    "stage_name": stage_name,
                    "algorithm": alg_identifier,
                    "parameters": stage_params,
                    "stage_qc_rsd_oof": hist_oof.get(stage_name),
                    "stage_qc_rsd_full": hist_full.get(stage_name),
                }
            )

        return metrics


# ==============================================================================
# Main Visualizer Class: MetaboVisualizerCorrector
# ==============================================================================

from .visualization import MetaboVisualizerCorrector
