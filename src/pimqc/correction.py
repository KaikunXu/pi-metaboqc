# src/pimqc/correction.py
"""
Script purpose: Execute technical signal correction and method selection.

execute_signal_correction() casts the matrix to float, builds QC, batch, and
injection-order arrays, then evaluates the configured correction method or an
AUTO panel of QC-RLSC, QC-RFSC, QC-SVR, SERRF, RUV-III, and WaveICA 2.0. It compares
candidate outputs with QC RSD metrics, selects the best method when needed, and
records the selected stage data in the MetaboInt attributes.
WaveICA 2.0 removes injection-order-associated independent components from
multiscale signals and is evaluated with full-data QC RSD in AUTO mode.
The workflow exports corrected matrices, QC fitted baselines, RSD evaluation
figures, optional AUTO comparison grids, and internal-standard diagnostic
scatter plots for the chosen correction strategy.
"""

import os
import re
import math
import copy
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.pipeline import Pipeline, make_pipeline
from joblib import Parallel, delayed
from loguru import logger
from numba import njit, prange
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

from . import io_utils as iu
from . import plot_utils as pu
from . import core_classes
from . import visualizer_classes

FitPredictCallable = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
CorrectionModel = (
    RandomForestRegressor | TransformedTargetRegressor | Pipeline | FitPredictCallable
)


def _format_correction_method_label(method: str) -> str:
    """Return the display label for correction method identifiers."""
    return _normalize_correction_method(method)


def _normalize_correction_method(method: str) -> str:
    """Normalize correction method aliases to canonical public names."""
    method_text = str(method).strip()
    method_upper = method_text.upper()
    method_compact = re.sub(r"[\s_.-]+", "", method_upper)

    if method_upper == "AUTO":
        return "AUTO"
    if method_compact in ("QCRLSC", "RLSC", "LOESS"):
        return "QC-RLSC"
    if method_compact in ("QCRFSC", "RFSC", "RF"):
        return "QC-RFSC"
    if method_compact in ("QCSVR", "QCSVRC", "SVR"):
        return "QC-SVR"
    if method_compact == "SERRF":
        return "SERRF"
    if method_compact in ("RUV", "RUVIII", "RUV3"):
        return "RUV-III"
    if method_compact in ("WAVEICA2", "WAVEICA20"):
        return "WaveICA 2.0"
    return method_text


def _format_correction_method_file_label(method: str) -> str:
    """Return a filesystem-friendly label for correction method identifiers."""
    return re.sub(r'[<>:"/\\|?*]', "-", _format_correction_method_label(method))


# ==============================================================================
# Cross-Validation Engine for Robust Drift Correction to Prevent Overfitting
# ==============================================================================
def fit_predict_intra_batch_safely(
    base_model: CorrectionModel,
    x_qc: np.ndarray,
    y_qc: np.ndarray,
    x_all: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return both continuous full baseline and OOF baseline for metrics."""
    n_qc = len(y_qc)
    pred_qc_oof = np.zeros(n_qc)
    safe_folds = min(cv_folds, n_qc)

    def _run_model(
        x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
    ) -> np.ndarray:
        if hasattr(base_model, "fit") and hasattr(base_model, "predict"):
            model = clone(base_model)
            model.fit(x_train, y_train)
            return model.predict(x_test)
        return base_model(x_train, y_train, x_test)

    if safe_folds < 3:
        pred_all_full = _run_model(x_qc, y_qc, x_all)
        pred_qc_oof = _run_model(x_qc, y_qc, x_qc)
        return (
            np.clip(pred_all_full, a_min=1e-6, a_max=None),
            np.clip(pred_qc_oof, a_min=1e-6, a_max=None),
        )

    kf = KFold(n_splits=safe_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(x_qc):
        pred_qc_oof[test_idx] = _run_model(
            x_train=x_qc[train_idx], y_train=y_qc[train_idx], x_test=x_qc[test_idx]
        )

    pred_all_full = _run_model(x_train=x_qc, y_train=y_qc, x_test=x_all)

    return (
        np.clip(pred_all_full, a_min=1e-6, a_max=None),
        np.clip(pred_qc_oof, a_min=1e-6, a_max=None),
    )


# ==============================================================================
# Numba JIT Engines for Fast Robust QC-RLSC
# ==============================================================================
@njit(fastmath=True)
def _tricube_kernel(x: float) -> float:
    abs_x = abs(x)
    if abs_x >= 1.0:
        return 0.0
    return (1.0 - abs_x**3) ** 3


@njit(fastmath=True)
def _bisquare_weight(res: float, s: float) -> float:
    if s <= 1e-9:
        return 1.0 if abs(res) <= 1e-9 else 0.0
    v = res / (6.0 * s)
    if abs(v) >= 1.0:
        return 0.0
    return (1.0 - v * v) ** 2


@njit(fastmath=True)
def _numba_loess_1d_core(
    x: np.ndarray,
    y: np.ndarray,
    x_pred: np.ndarray,
    loess_frac: float,
    delta: np.ndarray,
) -> np.ndarray:
    n = len(x)
    m = len(x_pred)
    y_pred = np.zeros(m)

    k = int(math.ceil(n * loess_frac))
    if k < 2:
        k = 2
    if k > n:
        k = n

    for i in range(m):
        x0 = x_pred[i]
        diffs = np.abs(x - x0)
        sorted_diffs = np.sort(diffs)
        h = sorted_diffs[k - 1]

        if h <= 0.0:
            h = 1e-9

        sum_w = 0.0
        sum_wx = 0.0
        sum_wy = 0.0

        for j in range(n):
            w = _tricube_kernel((x[j] - x0) / h) * delta[j]
            sum_w += w
            sum_wx += w * x[j]
            sum_wy += w * y[j]

        if sum_w <= 0:
            y_pred[i] = np.mean(y)
            continue

        x_bar = sum_wx / sum_w
        y_bar = sum_wy / sum_w

        numerator = 0.0
        denominator = 0.0

        for j in range(n):
            w = _tricube_kernel((x[j] - x0) / h) * delta[j]
            dev_x = x[j] - x_bar
            numerator += w * dev_x * (y[j] - y_bar)
            denominator += w * dev_x * dev_x

        if denominator <= 0:
            beta1 = 0.0
        else:
            beta1 = numerator / denominator

        beta0 = y_bar - beta1 * x_bar
        y_pred[i] = beta0 + beta1 * x0

    return y_pred


@njit(fastmath=True)
def _numba_loess_robust(
    x: np.ndarray, y: np.ndarray, x_pred: np.ndarray, loess_frac: float, max_iter: int
) -> np.ndarray:
    n = len(x)
    delta = np.ones(n)
    for iteration in range(max_iter):
        y_fit = _numba_loess_1d_core(x, y, x, loess_frac, delta)
        residuals = np.abs(y - y_fit)
        s = np.median(residuals)
        for j in range(n):
            delta[j] = _bisquare_weight(residuals[j], s)
    return _numba_loess_1d_core(x, y, x_pred, loess_frac, delta)


@njit(parallel=True, fastmath=True)
def _numba_batch_qc_rlsc(
    data: np.ndarray,
    qc_mask: np.ndarray,
    injection_orders: np.ndarray,
    loess_frac: float,
    max_iter: int,
) -> np.ndarray:
    rows, cols = data.shape
    predicted_matrix = np.zeros((rows, cols))
    x_qc_all = injection_orders[qc_mask]

    for i in prange(rows):
        row_data = data[i, :]
        y_qc_all = row_data[qc_mask]

        valid_mask = ~np.isnan(y_qc_all)
        valid_count = np.sum(valid_mask)

        if valid_count < 3:
            mean_val = 0.0
            if valid_count > 0:
                mean_val = np.sum(y_qc_all[valid_mask]) / valid_count
            else:
                mean_val = np.nan
            predicted_matrix[i, :] = mean_val
            continue

        clean_x = x_qc_all[valid_mask]
        clean_y = y_qc_all[valid_mask]

        predicted_matrix[i, :] = _numba_loess_robust(
            clean_x, clean_y, injection_orders, loess_frac, max_iter
        )

    return predicted_matrix


# ==============================================================================
# Engine 1: RegressionCorrector
# ==============================================================================
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
            f"Phase 1: Executing Intra-batch drift correction with {self.method}..."
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
                loess_frac = self.params.get("loess_frac", 0.3)
                max_iter = self.params.get("max_iter", 3)
                cv_folds = self.params.get("cv_folds", 5)
                seed = self.params.get("global_seed", 123)

                pred_matrix = _numba_batch_qc_rlsc(
                    data=b_data.values,
                    qc_mask=b_qc_mask,
                    injection_orders=b_orders,
                    loess_frac=loess_frac,
                    max_iter=max_iter,
                )
                pred_df.loc[:, b_mask] = pred_matrix

                oof_matrix = pred_matrix.copy()
                qc_indices = np.where(b_qc_mask)[0]
                if len(qc_indices) >= max(3, cv_folds):
                    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
                    for train_idx, test_idx in kf.split(qc_indices):
                        train_qc_mask = b_qc_mask.copy()
                        train_qc_mask[qc_indices[test_idx]] = False
                        fold_pred = _numba_batch_qc_rlsc(
                            data=b_data.values,
                            qc_mask=train_qc_mask,
                            injection_orders=b_orders,
                            loess_frac=loess_frac,
                            max_iter=max_iter,
                        )
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
                joblib_batch_size = self.params.get("regression_batch_size", "auto")
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

                # [FIX]: Use patched joblib context manager
                with iu.tqdm_joblib_env(total=len(feat_idx), desc=f"SC [{batch_id}]"):
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

        base_bc = pd.DataFrame(index=intensity_df.index, columns=intensity_df.columns)
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
            bt_qc_mean_oof = intra_qc_oof.T.groupby(batch_array[qc_mask]).mean().T
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
            logger.debug(f"Feature {feat_idx} fit failed via {self.method}: {e}")
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
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """Worker function using pre-sliced indices for memory efficiency."""
        y_all = y_mat[:, feat_idx]

        # 1. Feature Construction using pre-computed indices
        if top_idx_row is not None:
            x_corr = np.nan_to_num(y_mat[:, top_idx_row], nan=0.0)
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
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        y_pred_qc = np.zeros(len(y_qc_valid))

        rf_oof = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=1
        )
        for train_idx, test_idx in kf.split(x_qc_valid):
            rf_oof.fit(x_qc_valid[train_idx], y_qc_valid[train_idx])
            y_pred_qc[test_idx] = rf_oof.predict(x_qc_valid[test_idx])

        # 4. Predict Expected Baselines for ALL Samples
        rf_full = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=1
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
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:

        logger.info("Initializing High-Performance Hybrid SERRF Corrector...")
        y_mat = intensity_df.T.to_numpy(dtype=float, copy=False)
        x_base = self._prepare_base_features(batch_array, order_array)
        n_features = y_mat.shape[1]

        if sum(qc_mask) < self.cv_folds:
            raise ValueError("Insufficient QCs for configured CV.")

        # Extract top features globally to avoid memory leak in parallel workers
        top_indices = None
        if self.n_corr_features > 0 and corr_mat is not None:
            # Prevent feature from selecting itself as highly correlated
            np.fill_diagonal(corr_mat, -1.0)
            top_indices = np.argsort(corr_mat, axis=1)[:, -self.n_corr_features :]

        y_corrected = np.zeros_like(y_mat)
        y_corrected_oof = np.zeros_like(y_mat)

        actual_cores = (os.cpu_count() or 1) if self.n_jobs == -1 else self.n_jobs
        safe_n_jobs = max(1, int(actual_cores / 2))
        joblib_backend = self.joblib_backend
        if joblib_backend not in {"threading", "loky"}:
            logger.warning(
                f"Unsupported serrf_backend='{joblib_backend}'. Falling back to 'loky'."
            )
            joblib_backend = "loky"
        joblib_batch_size = self.joblib_batch_size
        if isinstance(joblib_batch_size, str):
            if joblib_batch_size.lower() == "auto":
                joblib_batch_size = "auto"
            else:
                joblib_batch_size = int(joblib_batch_size)

        # [FIX]: Use patched joblib context manager for SERRF
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
            y_corrected.T, index=intensity_df.index, columns=intensity_df.columns
        )
        res_df_oof = pd.DataFrame(
            y_corrected_oof.T, index=intensity_df.index, columns=intensity_df.columns
        )
        return {"SERRF": (res_df_full, res_df_oof)}


# ==============================================================================
# Engine 3: WaveICA2Corrector
# ==============================================================================
class WaveICA2Corrector:
    """Native Python WaveICA 2.0 correction using order-associated stICA removal.

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
        col_medians = np.where(np.isfinite(col_medians), col_medians, global_median)
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
        """Approximately jointly diagonalize symmetric matrices by Jacobi rotations."""
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
                    left_p = c_val * work[p_idx, :, :] + s_val * work[q_idx, :, :]
                    left_q = -s_val * work[p_idx, :, :] + c_val * work[q_idx, :, :]
                    work[p_idx, :, :] = left_p
                    work[q_idx, :, :] = left_q

                    col_p = c_val * work[:, p_idx, :] + s_val * work[:, q_idx, :]
                    col_q = -s_val * work[:, p_idx, :] + c_val * work[:, q_idx, :]
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
            raise ValueError("WaveICA 2.0 stICA requires at least one component.")
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

    def _order_r2(self, component: np.ndarray, order_array: np.ndarray) -> float:
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

    def _remove_order_components(
        self, coeff: np.ndarray, order_array: np.ndarray
    ) -> np.ndarray:
        """Remove stICA components whose scores are explained by injection order."""
        n_samples, n_features = coeff.shape
        safe_k = min(self.n_components, n_samples, n_features)
        if safe_k < 2 or n_samples < 4:
            self.selected_component_counts.append(0)
            self.selected_component_r2.append(np.array([], dtype=float))
            return coeff

        try:
            mixing, sources = self._unbiased_stica(
                x=coeff.T,
                n_components=safe_k,
                alpha=0.0,
            )
            r2_vals = np.array(
                [self._order_r2(sources[:, i], order_array) for i in range(safe_k)]
            )
            selected = np.where(r2_vals >= self.cutoff)[0]
            self.selected_component_counts.append(int(len(selected)))
            self.selected_component_r2.append(r2_vals)

            if len(selected) == 0:
                return coeff

            artifact = (mixing[:, selected] @ sources[:, selected].T).T
            return coeff - artifact
        except Exception as e:
            logger.debug(f"WaveICA 2.0 coefficient correction failed: {e}")
            self.selected_component_counts.append(0)
            self.selected_component_r2.append(np.array([], dtype=float))
            return coeff

    def fit_transform(
        self,
        intensity_df: pd.DataFrame,
        order_array: np.ndarray,
    ) -> Dict[str, Tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
        """Execute WaveICA 2.0 correction."""
        logger.info("Executing WaveICA 2.0 correction...")

        order_array = np.asarray(order_array, dtype=float)
        if order_array.shape[0] != intensity_df.shape[1]:
            raise ValueError(
                "WaveICA 2.0 requires one injection order value per sample."
            )

        sort_idx = np.argsort(order_array, kind="mergesort")
        inverse_idx = np.argsort(sort_idx)
        sorted_order = order_array[sort_idx]
        sorted_df = intensity_df.iloc[:, sort_idx]

        raw = sorted_df.T.values.astype(float)
        nan_mask = np.isnan(raw)
        filled = self._fill_missing_by_feature_median(raw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            raw_means = np.nanmean(raw, axis=0)
        raw_means = np.where(np.isfinite(raw_means), raw_means, 0.0)

        coeffs = self._decompose(filled)
        cleaned_coeffs = [
            self._remove_order_components(coeff, sorted_order) for coeff in coeffs
        ]
        corrected = self._reconstruct(cleaned_coeffs) + raw_means[None, :]

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
    ) -> Dict[str, Tuple[pd.DataFrame, Optional[pd.DataFrame]]]:

        logger.info(f"Executing RUV-III (k={self.k})...")
        if control_features.empty:
            raise ValueError("RUV-III requires at least one control feature.")

        Y_raw = intensity_df.T.values.astype(np.float64)
        n_samples, n_features = Y_raw.shape

        Y_safe = np.clip(Y_raw, a_min=0, a_max=None)
        Y = np.log1p(Y_safe)

        # Mask out features that are completely zero to prevent SVD artifacts
        zero_mask = np.all(Y_safe == 0, axis=0)

        nan_mask = np.isnan(Y)
        if nan_mask.any():
            logger.warning("NaNs detected. Applying median imputation...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                col_medians = np.nanmedian(Y, axis=0)
            col_medians[np.isnan(col_medians)] = 0.0
            nan_rows, nan_cols = np.where(nan_mask)
            Y[nan_rows, nan_cols] = col_medians[nan_cols]

        group_ids = []
        uid_counter = 1
        for is_qc in qc_mask:
            if is_qc:
                group_ids.append(0)
            else:
                group_ids.append(uid_counter)
                uid_counter += 1

        n_groups = len(set(group_ids))
        M = np.zeros((n_samples, n_groups), dtype=np.float64)
        for row_idx, g_id in enumerate(group_ids):
            M[row_idx, g_id] = 1.0

        group_sizes = M.T @ M
        group_means = np.linalg.solve(group_sizes, M.T @ Y)
        Y0 = Y - (M @ group_means)

        # Exclude zero-variance control features dynamically
        ctl_mask = intensity_df.index.isin(control_features) & ~zero_mask
        Y0_ctl = Y0[:, ctl_mask]

        U, S, Vt = np.linalg.svd(Y0_ctl, full_matrices=False)
        safe_k = min(self.k, Y0_ctl.shape[0], Y0_ctl.shape[1])
        alpha_ctl = Vt[:safe_k, :]

        W = Y[:, ctl_mask] @ alpha_ctl.T
        W_means = np.linalg.solve(group_sizes, M.T @ W)
        W0 = W - (M @ W_means)
        alpha_full = np.linalg.lstsq(W0, Y0, rcond=None)[0]

        correction = W @ alpha_full
        correction -= np.mean(correction, axis=0)

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
class MetaboIntCorrector(core_classes.MetaboInt):
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
        loess_frac: Optional[float] = None,
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

        # 1. Establish robust configuration defaults
        sc_configs = {
            "base_est": "QC-RLSC",
            "loess_frac": 0.5,
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
        }

        # 2. Extract pipeline configuration layer from TOML
        if pipeline_params and "MetaboIntCorrector" in pipeline_params:
            sc_configs.update(pipeline_params["MetaboIntCorrector"])

        # 3. Explicit functional kwargs override TOML configurations
        local_args = locals()
        explicit_params = [
            "base_est",
            "loess_frac",
            "rf_n_tree",
            "serrf_n_tree",
            "serrf_corr_features",
            "serrf_backend",
            "serrf_batch_size",
            "svr_kernel",
            "svr_c",
            "svr_gamma",
            "cv_folds",
            "ruv_k",
            "waveica_components",
            "waveica_cutoff",
            "waveica_levels",
            "waveica_spline_knots",
            "waveica_max_iter",
            "regression_backend",
            "regression_batch_size",
            "n_jobs",
        ]
        for param in explicit_params:
            if local_args[param] is not None:
                sc_configs[param] = local_args[param]

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
    def extract_qc_rsd_series(df_obj: core_classes.MetaboInt) -> pd.Series:
        """Extracts the RSD series for QC samples across all features."""
        if hasattr(df_obj, "_qc") and not df_obj._qc.empty:
            qc_data = df_obj._qc.astype(float)
        else:
            sample_type_col = df_obj.attrs.get("sample_type", "Sample Type")
            qc_label = df_obj.attrs.get("sample_dict", {}).get("QC sample", "QC")
            mask = df_obj.columns.get_level_values(sample_type_col) == qc_label
            qc_data = df_obj.loc[:, mask].astype(float)

        return (qc_data.std(axis=1, ddof=1) / qc_data.mean(axis=1)).dropna()

    @staticmethod
    def calculate_median_qc_rsd(df_obj: core_classes.MetaboInt) -> float:
        """Calculates the scalar median RSD of QC samples."""
        rsd_series = MetaboIntCorrector.extract_qc_rsd_series(df_obj)
        if rsd_series.empty:
            return float("nan")
        return float(rsd_series.median())

    def _calculate_qc_baseline_means(
        self, batch_col: str, sample_type_col: str, qc_label: str
    ) -> pd.DataFrame:
        """Calculate batch-wise QC mean to reverse-engineer visual baselines."""
        qc_df = self.loc[:, self.columns.get_level_values(sample_type_col) == qc_label]
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

    def _prepare_ruv_control_features(self, empirical_ratio: float = 0.05) -> pd.Index:
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

    def _evaluate_all_methods(
        self,
        methods_to_run: list,
        batch_array: np.ndarray,
        qc_mask: np.ndarray,
        order_array: np.ndarray,
        batch_col: str,
        sample_type_col: str,
        qc_label: str,
    ) -> Dict[str, Any]:
        """Execute core computation and metrics extraction for all methods."""
        results_store = {}
        for raw_method in methods_to_run:
            method = _normalize_correction_method(raw_method)
            logger.info(f"--- Evaluating Method: {method} ---")

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
                    joblib_batch_size=self.attrs.get("serrf_batch_size", "auto"),
                )
                stages_output = engine.fit_transform(
                    intensity_df=self,
                    batch_array=batch_array,
                    qc_mask=qc_mask,
                    order_array=order_array,
                    corr_mat=corr_mat,
                )
            elif method == "RUV-III":
                ctrl_features = self._prepare_ruv_control_features()
                engine = RUVCorrector(k=self.attrs.get("ruv_k", 3))
                stages_output = engine.fit_transform(
                    intensity_df=self, qc_mask=qc_mask, control_features=ctrl_features
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
                )
            else:
                engine = RegressionCorrector(method=method, **self.attrs)
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

                curr_full_rsd = MetaboIntCorrector.calculate_median_qc_rsd(final_df)
                rsd_hist_full[clean_name] = curr_full_rsd
                final_df.attrs["qc_rsd_current_full"] = curr_full_rsd

                if oof_df is not None:
                    oof_wrap = self._constructor(oof_df).__finalize__(self)
                    oof_wrap.attrs["pipeline_stage"] = "Correction"
                    curr_oof_rsd = MetaboIntCorrector.calculate_median_qc_rsd(oof_wrap)
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

            results_store[method] = {
                "stage_dfs": stage_dfs,
                "stage_oof_dfs": stage_oof_dfs,
                "pred_df": pred_df,
                "final_rsd_full": final_full,
                "final_rsd_oof": final_oof,
            }

            full_only_methods = ("RUV-III", "WaveICA 2.0")
            log_rsd = final_full if method in full_only_methods else final_oof
            if log_rsd is not None:
                logger.info(f"{method} Eval QC RSD: {log_rsd * 100:.2f}%")

        return results_store

    def _find_best_correction(self, results_store: Dict[str, Any]) -> str:
        """
        Identify the optimal correction method based on QC RSD.

        RUV-III and WaveICA 2.0 evaluate using the full dataset RSD.
        All other methods rely on the Out-Of-Fold (OOF) cross-validated
        RSD to prevent overfitting.
        """
        best_method = ""
        min_rsd = float("inf")

        for method, res in results_store.items():
            if method in ("RUV-III", "WaveICA 2.0"):
                eval_rsd = res.get("final_rsd_full", float("inf"))
            else:
                eval_rsd = res.get("final_rsd_oof")
                if eval_rsd is None:
                    eval_rsd = res.get("final_rsd_full", float("inf"))

            if eval_rsd < min_rsd:
                min_rsd = eval_rsd
                best_method = method

        return best_method

    # =========================================================================
    # Core Pipeline Execution Flow
    # =========================================================================
    @iu._exe_time
    def execute_signal_correction(self, output_dir: str) -> Dict[str, Any]:
        """Execute complete signal correction workflow dynamically."""
        # [NEW FIX] Preemptively cast the entire internal matrix to float
        # to guarantee safe inplace assignments during regression steps.
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

        qc_mask = self.columns.get_level_values(sample_type_col) == qc_label
        batch_array = self.columns.get_level_values(batch_col).values
        order_array = self.columns.get_level_values(inject_order_col).values

        req_method = _normalize_correction_method(self.attrs.get("base_est", "QC-RLSC"))
        self.attrs["is_auto_mode"] = req_method == "AUTO"

        if req_method == "AUTO":
            methods_to_run = [
                "SERRF",
                "RUV-III",
                "WaveICA 2.0",
                "QC-RLSC",
                "QC-RFSC",
                "QC-SVR",
            ]
            logger.info("AUTO mode enabled. Evaluating multiple methods.")
        else:
            methods_to_run = [req_method]

        # ---------------------------------------------------------------------
        # 1. Computation & Evaluation Phase
        # ---------------------------------------------------------------------
        results_store = self._evaluate_all_methods(
            methods_to_run=methods_to_run,
            batch_array=batch_array,
            qc_mask=qc_mask,
            order_array=order_array,
            batch_col=batch_col,
            sample_type_col=sample_type_col,
            qc_label=qc_label,
        )

        # ---------------------------------------------------------------------
        # 2. Selection Phase
        # ---------------------------------------------------------------------
        best_method = self._find_best_correction(results_store)

        if req_method == "AUTO":
            if best_method in ("RUV-III", "WaveICA 2.0"):
                best_rsd = results_store[best_method]["final_rsd_full"]
            else:
                best_rsd = results_store[best_method]["final_rsd_oof"]

            logger.success(
                f"Auto selection: {_format_correction_method_label(best_method)} is optimal "
                f"(Eval QC RSD = {best_rsd * 100:.2f}%)."
            )
            # Update metric tracker to reflect dynamically chosen algorithm
            self.attrs["base_est"] = best_method

            # Ensure the propagated DataFrames carry the resolved name
            for df in results_store[best_method]["stage_dfs"].values():
                df.attrs["base_est"] = best_method

        # ---------------------------------------------------------------------
        # 3. Visualization Routing Phase
        # ---------------------------------------------------------------------
        vis = MetaboVisualizerCorrector(self)

        # Auto mode: Generate the comprehensive comparison grid
        if req_method == "AUTO" and len(results_store) > 1:
            logger.info("Assembling evaluation grid for all methods...")
            grid_obj = vis.plot_correction_dashboard(results_store, best_method)

            if grid_obj is not None:
                vis.save_and_show_pw(
                    pw_obj=grid_obj,
                    width="60%",
                    file_path=os.path.join(output_dir, "QC_RSD_Evaluation_Grid.svg"),
                )

        # Standard behavior: Generate the specific plot for the chosen method
        best_method_label = _format_correction_method_label(best_method)
        best_method_file_label = _format_correction_method_file_label(best_method)
        logger.info(f"Generating specific RSD plot for: {best_method_label}")
        m_res = results_store[best_method]
        fig_rsd = vis.plot_corr_rsd(
            stage_dfs=m_res["stage_dfs"], stage_oof_dfs=m_res.get("stage_oof_dfs", {})
        )
        vis.save_and_show_pw(
            pw_obj=fig_rsd,
            width="20%",
            file_path=os.path.join(output_dir, f"QC_RSD_{best_method_file_label}.svg"),
        )

        # ---------------------------------------------------------------------
        # 4. File Export Phase (Exclusive to the optimal method)
        # ---------------------------------------------------------------------
        best_res = results_store[best_method]
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
                file_name = f"{prefix}_{best_method}.csv"

            df.to_csv(os.path.join(output_dir, file_name))

        if best_pred_df is not None:
            best_pred_df.to_csv(
                os.path.join(output_dir, f"QC_Fit_Base_{best_method_file_label}.csv")
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
                    is_dir, f"IS_Scatter_{safe_feat}_{best_method_file_label}.svg"
                )
                vis.save_and_show_pw(pw_obj=fig, file_path=save_path, show_plot=False)

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
                            output_dir, f"Pred_Base_IS_{best_method_file_label}.svg"
                        ),
                        show_plot=False,
                    )
            else:
                logger.info(
                    f"Bypassing IS baseline prediction for {best_method_label}."
                )

        logger.success(f"Signal drift correction ({best_method_label}) completed.")
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
        method = _normalize_correction_method(self.attrs.get("base_est", "Unknown"))

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
                metrics["overall_performance"]["relative_noise_reduction_oof"] = (
                    oof_reduction
                )
            if rsd_curr_full is not None:
                full_reduction = (rsd_base - rsd_curr_full) / rsd_base
                metrics["overall_performance"]["relative_noise_reduction_full"] = (
                    full_reduction
                )

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
                    stage_params["loess_frac"] = self.attrs.get("loess_frac")
                elif alg_identifier in ("QC-RFSC", "RF"):
                    stage_params["n_estimators"] = self.attrs.get("rf_n_tree")
                elif alg_identifier == "QC-SVR":
                    stage_params["svr_kernel"] = self.attrs.get("svr_kernel")
                    stage_params["svr_c"] = self.attrs.get("svr_c")
                    stage_params["svr_gamma"] = self.attrs.get("svr_gamma")
                elif alg_identifier == "SERRF":
                    stage_params["n_estimators"] = self.attrs.get("serrf_n_tree")
                    stage_params["n_corr_features"] = self.attrs.get(
                        "serrf_corr_features"
                    )
                    stage_params["backend"] = self.attrs.get("serrf_backend")
                    stage_params["batch_size"] = self.attrs.get("serrf_batch_size")
                elif alg_identifier == "RUV-III":
                    stage_params["ruv_k"] = self.attrs.get("ruv_k")
                elif alg_identifier == "WaveICA 2.0":
                    stage_params["n_components"] = self.attrs.get("waveica_components")
                    stage_params["cutoff"] = self.attrs.get("waveica_cutoff")
                    stage_params["n_levels"] = self.attrs.get("waveica_levels")
                    stage_params["spline_knots"] = self.attrs.get(
                        "waveica_spline_knots"
                    )
                    stage_params["max_iter"] = self.attrs.get("waveica_max_iter")

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
class MetaboVisualizerCorrector(visualizer_classes.BaseMetaboVisualizer):
    """Visualization suite matching original alpha output styles."""

    def __init__(self, corr_obj: MetaboIntCorrector) -> None:
        """Initialize with a computed MetaboIntCorrector object."""
        super().__init__(metabo_obj=corr_obj)
        self.corr = corr_obj

    # =========================================================================
    # Evaluation & Diagnostic Plotters
    # =========================================================================
    def plot_rsd_standalone_legend(
        self,
        ax: plt.Axes | None = None,
        show_cv: bool = True,
        bbox_to_anchor: tuple[float, float] = (0.1, 0.5),
    ) -> plt.Axes:
        """Create a standalone legend for RSD plots using explicit Patches."""
        if ax is None:
            try:
                import patchworklib as pw
            except ImportError:
                raise ImportError("patchworklib is required for this plot.")
            current_ax = pw.Brick(figsize=(1.5, 4.0), label="rsd_legend")
        else:
            current_ax = ax

        current_ax.axis("off")

        from matplotlib.patches import Patch

        c_base = pu.get_equivalent_hex("tab:gray", alpha=1)
        c_cv = pu.get_equivalent_hex("tab:red", alpha=0.5)
        c_full = pu.get_equivalent_hex("tab:red", alpha=1)

        # Explicitly construct legend patches in memory.
        # Since the updated _format_single_legend can now extract handles
        # from an active legend before clearing it, we no longer need
        # empty proxy bars or invisible zero-dimension shapes.
        legend_elements = [
            Patch(facecolor=c_base, edgecolor="k", linewidth=1.0, label="Baseline")
        ]
        if show_cv:
            legend_elements.append(
                Patch(
                    facecolor=c_cv,
                    edgecolor="k",
                    linewidth=1.0,
                    linestyle="--",
                    label="OOF Model",
                )
            )
        legend_elements.append(
            Patch(facecolor=c_full, edgecolor="k", linewidth=1.0, label="Global model")
        )

        # Step 1: Bind the explicit handles to a temporary legend object
        current_ax.legend(handles=legend_elements)

        # Step 2: Pass to the updated robust base formatter for styling
        self._format_single_legend(
            ax=current_ax,
            group_title="Correction Mode",
            loc="center left",
            bbox_to_anchor=bbox_to_anchor,
        )

        # --- CRITICAL PATCHWORKLIB FIX ---
        # Bring the promoted figure-level legend back down to the Axes layer
        # to guarantee that patchworklib doesn't drop it during layout binding.
        if hasattr(current_ax.figure, "legends"):
            for leg in list(current_ax.figure.legends):
                current_ax.add_artist(leg)
            current_ax.figure.legends.clear()

        return current_ax

    def plot_corr_rsd(
        self,
        stage_dfs: dict[str, pd.DataFrame],
        stage_oof_dfs: dict[str, pd.DataFrame],
        ax: plt.Axes | None = None,
        show_legend: bool = True,
    ) -> plt.Axes:
        """Plot dual-mode RSD boxplots with dynamic width and annotations."""
        box_data = []
        positions = []
        box_colors = []
        box_styles = []
        tick_pos = []
        tick_labels = []
        medians_text = []

        c_base = pu.get_equivalent_hex("tab:gray", alpha=1)
        c_cv = pu.get_equivalent_hex("tab:red", alpha=0.5)
        c_full = pu.get_equivalent_hex("tab:red", alpha=1)

        # Retrieve the key of the final stage to mark the selection metric
        stage_keys = list(stage_dfs.keys())
        last_stage_key = stage_keys[-1] if stage_keys else None

        orig_df = stage_dfs.get("Original")
        if orig_df is not None:
            orig_rsd = self.corr.extract_qc_rsd_series(orig_df)
            box_data.append(orig_rsd.values)
            positions.append(1.0)
            box_colors.append(c_base)
            box_styles.append("-")
            tick_pos.append(1.0)
            tick_labels.append("Before\ncorrection")
            medians_text.append(f"Before correction: {orig_rsd.median() * 100:.2f}%")

        current_x = 2.6  # Adjusted starting position for wider spacing
        for stage_name, df in stage_dfs.items():
            if stage_name == "Original":
                continue

            clean_name = stage_name.replace("\n", " ")
            has_cv = stage_name in stage_oof_dfs
            full_rsd = self.corr.extract_qc_rsd_series(df)
            is_last = stage_name == last_stage_key

            if has_cv:
                cv_rsd = self.corr.extract_qc_rsd_series(stage_oof_dfs[stage_name])
                box_data.extend([cv_rsd.values, full_rsd.values])

                # Widened symmetrical offset for fatter boxes (0.28 vs 0.22)
                positions.extend([current_x - 0.28, current_x + 0.28])

                box_colors.extend([c_cv, c_full])
                box_styles.extend(["--", "-"])

                # Append asterisk strictly to the CV metric if it's the final stage
                prefix = "* " if is_last else ""
                medians_text.append(
                    f"{prefix}{clean_name} (OOF): {cv_rsd.median() * 100:.2f}%"
                )
                medians_text.append(
                    f"{clean_name} (Full): {full_rsd.median() * 100:.2f}%"
                )
            else:
                box_data.append(full_rsd.values)
                positions.append(current_x)
                box_colors.append(c_full)
                box_styles.append("-")

                # Append asterisk strictly to the global model metric.
                prefix = "* " if is_last else ""
                medians_text.append(
                    f"{prefix}{clean_name} (Global): {full_rsd.median() * 100:.2f}%"
                )

            tick_pos.append(current_x)
            formatted_label = stage_name.replace(" ", "\n")
            tick_labels.append(formatted_label)
            current_x += 1.6  # Adjusted step for wider boxes

        if ax is None:
            try:
                import patchworklib as pw
            except ImportError:
                raise ImportError("patchworklib is required for this plot.")
            pw.clear()
            fig_width = max(4.0, len(stage_dfs) * 1.2 + 2)
            current_ax = pw.Brick(figsize=(fig_width, 4.0), label="rsd_box")
        else:
            current_ax = ax

        # Boxplot rendered with increased width (0.50 up from 0.35)
        bp = current_ax.boxplot(
            box_data,
            positions=positions,
            widths=0.50,
            patch_artist=True,
            showfliers=False,
        )

        for i in range(len(box_data)):
            bp["boxes"][i].set_facecolor(box_colors[i])
            bp["boxes"][i].set_edgecolor("k")
            bp["boxes"][i].set_linewidth(1.0)
            bp["boxes"][i].set_linestyle(box_styles[i])

            bp["medians"][i].set_color("k")
            bp["medians"][i].set_linewidth(1.5)
            bp["medians"][i].set_linestyle(box_styles[i])

            for j in range(2):
                idx = i * 2 + j
                bp["whiskers"][idx].set_color("k")
                bp["whiskers"][idx].set_linewidth(1.0)
                bp["whiskers"][idx].set_linestyle(box_styles[i])

                bp["caps"][idx].set_color("k")
                bp["caps"][idx].set_linewidth(1.0)
                bp["caps"][idx].set_linestyle(box_styles[i])

        current_ax.set_xticks(tick_pos)
        current_ax.set_xticklabels(tick_labels)

        annot_text = "Median QC RSD:\n" + "\n".join(medians_text)
        current_ax.text(
            0.96,
            0.98,
            annot_text,
            transform=current_ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            clip_on=False,
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor="white", edgecolor="none", alpha=0.6
            ),
        )

        self._apply_standard_format(current_ax, ylabel="QC RSD (%)", append_stage=False)
        pu.change_axis_format(current_ax, "percentage", "y")

        return current_ax

    def plot_correction_dashboard(
        self, results_store: dict[str, dict[str, Any]], best_method: str
    ) -> object | None:
        """Combine AUTO RSD evaluation plots and a shared legend."""
        try:
            import patchworklib as pw
        except ImportError:
            raise ImportError("patchworklib is required for this plot.")

        pw.clear()
        bricks = {}

        # Ensure row sums are exactly 10.2 for perfectly aligned layouts
        # Row 1: 3.0 + 3.0 + 3.0 + 1.2 (Legend) = 10.2
        # Row 2: 3.4 + 3.4 + 3.4 = 10.2
        width_map = {
            "SERRF": 3.0,
            "RUV-III": 3.0,
            "WaveICA 2.0": 3.0,
            "QC-RLSC": 3.4,
            "QC-RFSC": 3.4,
            "QC-SVR": 3.4,
        }

        for method, res in results_store.items():
            stage_dfs = res["stage_dfs"]
            stage_oof_dfs = res.get("stage_oof_dfs", {})

            fig_width = width_map.get(method, max(3.0, len(stage_dfs) * 1.0))
            safe_label = re.sub(r"[^A-Za-z0-9_]+", "_", f"rsd_box_{method}")

            b = pw.Brick(figsize=(fig_width, 4.0), label=safe_label)

            self.plot_corr_rsd(
                stage_dfs=stage_dfs,
                stage_oof_dfs=stage_oof_dfs,
                ax=b,
                show_legend=False,
            )

            method_label = _format_correction_method_label(method)
            # Formatted to prefix the asterisk cleanly (e.g., "* SERRF")
            title = f"* {method_label}" if method == best_method else method_label
            b.set_title(title)
            bricks[method] = b

        # Dynamically attach explicit handles legend to the 1.0-width slot
        leg_brick = pw.Brick(figsize=(1.2, 4.0), label="shared_legend")
        self.plot_rsd_standalone_legend(ax=leg_brick, show_cv=True)

        r1_keys = ["SERRF", "RUV-III", "WaveICA 2.0"]
        r2_keys = ["QC-RLSC", "QC-RFSC", "QC-SVR"]

        row1_bricks = [bricks[k] for k in r1_keys if k in bricks]
        row2_bricks = [bricks[k] for k in r2_keys if k in bricks]

        if not row1_bricks and not row2_bricks:
            return None

        row1 = None
        for b in row1_bricks:
            row1 = b if row1 is None else row1 | b
        row1 = leg_brick if row1 is None else row1 | leg_brick

        row2 = None
        for b in row2_bricks:
            row2 = b if row2 is None else row2 | b

        if row1 is not None and row2 is not None:
            grid_pw = row1 / row2
        else:
            grid_pw = row1 if row1 is not None else row2

        return grid_pw

    def _plot_standalone_is_legend(
        self,
        ax: plt.Axes,
        sample_type: str,
        batch: str,
        qc_label: str,
        actual_label: str,
        has_baseline: bool,
    ) -> plt.Axes:
        """Render a standalone multi-group legend for IS scatters."""
        import matplotlib.lines as mlines

        ax.axis("off")
        legend_handles = []
        legend_labels = []
        group_titles = [sample_type, batch]

        # Group 1: Sample Type
        legend_handles.append(mlines.Line2D([], [], color="none", label=sample_type))
        legend_labels.append(sample_type)

        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color="tab:red",
                marker="o",
                linestyle="none",
                markersize=6,
                markeredgecolor="k",
                markeredgewidth=0.5,
                label=qc_label,
            )
        )
        legend_labels.append(qc_label)

        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color="tab:gray",
                marker="o",
                linestyle="none",
                markersize=6,
                markeredgecolor="k",
                markeredgewidth=0.5,
                label=actual_label,
            )
        )
        legend_labels.append(actual_label)

        # Group 2: Batch (Reusing BaseVisualizer properties)
        legend_handles.append(mlines.Line2D([], [], color="none", label=batch))
        legend_labels.append(batch)

        for b_val in getattr(self, "all_batches", []):
            m_style = getattr(self, "style_map", {}).get(b_val, "o")
            legend_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color="tab:gray",
                    marker=m_style,
                    linestyle="none",
                    markersize=6,
                    markeredgecolor="k",
                    markeredgewidth=0.5,
                    label=str(b_val),
                )
            )
            legend_labels.append(str(b_val))

        # Group 3: Model Baseline (Rendered only if prediction exists)
        if has_baseline:
            group_titles.append("Model")
            legend_handles.append(mlines.Line2D([], [], color="none", label="Model"))
            legend_labels.append("Model")
            legend_handles.append(
                mlines.Line2D(
                    [], [], color="k", ls="-", lw=1.5, label="Fitted Baseline"
                )
            )
            legend_labels.append("Fitted Baseline")

        # =====================================================================
        # [CRITICAL FIX]: Must initialize the standard matplotlib legend FIRST
        # before passing it to the multi-legend layout formatter engine.
        # =====================================================================
        ax.legend(legend_handles, legend_labels)

        self._format_multi_legends(
            ax=ax,
            group_titles=group_titles,
            loc="upper left",
            start_bbox=(0.0, 0.95),
            row_gap=0.04,
            layout_cols=1,
            column_gap=0.1,
            sublegend_cols=1,
        )

        # Prevent Patchworklib from discarding figure-level legends
        if hasattr(ax.figure, "legends"):
            for leg in list(ax.figure.legends):
                ax.add_artist(leg)
            ax.figure.legends.clear()

        return ax

    def _get_is_shared_ylim(
        self,
        stage_dfs: dict[str, core_classes.MetaboInt],
        pred_df: core_classes.MetaboInt | None,
        feat: str,
        boundary: str,
    ) -> tuple[float, float] | None:
        """Calculate one y-axis range for one internal standard across stages."""
        y_values: list[float] = []
        boundary_helper = core_classes.MetaboInt()

        for df in stage_dfs.values():
            try:
                plot_data = df.int_order_info(feat_type="IS").reset_index()
            except Exception:
                continue

            if feat not in plot_data.columns:
                continue

            feature_values = pd.to_numeric(plot_data[feat], errors="coerce")
            finite_values = feature_values[np.isfinite(feature_values)]
            if finite_values.empty:
                continue

            y_values.extend(finite_values.astype(float).tolist())

            try:
                boundaries = boundary_helper.calculate_boundaries(
                    finite_values, boundary
                )
            except Exception:
                boundaries = ()
            y_values.extend(
                float(value) for value in boundaries if np.isfinite(float(value))
            )

        if pred_df is not None:
            try:
                pred_info = pred_df.int_order_info(feat_type="IS").reset_index()
            except Exception:
                pred_info = pd.DataFrame()

            if feat in pred_info.columns:
                pred_values = pd.to_numeric(pred_info[feat], errors="coerce")
                finite_pred = pred_values[np.isfinite(pred_values)]
                y_values.extend(finite_pred.astype(float).tolist())

        if not y_values:
            return None

        finite_y = np.asarray(y_values, dtype=float)
        finite_y = finite_y[np.isfinite(finite_y)]
        if finite_y.size == 0:
            return None

        y_min = float(np.min(finite_y))
        y_max = float(np.max(finite_y))
        if np.isclose(y_min, y_max):
            y_pad = max(abs(y_min) * 0.05, 1.0)
        else:
            y_pad = (y_max - y_min) * 0.08
        return y_min - y_pad, y_max + y_pad

    @staticmethod
    def _get_is_shared_yticks(ylim: tuple[float, float] | None) -> np.ndarray | None:
        """Resolve one set of y ticks shared by all IS scatter stages."""
        if ylim is None:
            return None

        locator = mticker.MaxNLocator(nbins=4, min_n_ticks=3, steps=[1, 2, 2.5, 5, 10])
        ticks = locator.tick_values(ylim[0], ylim[1])
        ticks = ticks[np.isfinite(ticks)]
        ticks = ticks[(ticks >= ylim[0]) & (ticks <= ylim[1])]

        if ticks.size < 3:
            ticks = np.linspace(ylim[0], ylim[1], num=4)

        return ticks

    @staticmethod
    def _apply_is_shared_y_axis(
        ax: plt.Axes,
        ylim: tuple[float, float] | None,
        yticks: np.ndarray | None,
    ) -> None:
        """Apply shared y limits, ticks, formatter, and tick styling."""
        if ylim is not None:
            ax.set_ylim(ylim)
        if yticks is not None:
            ax.yaxis.set_major_locator(mticker.FixedLocator(yticks))

        pu.change_axis_format(ax, "scientific notation", "y")
        pu.change_fontsize(ax, axis=pu.DEFAULT_FORMAT_AXIS)
        pu.change_weight(ax, axis=pu.DEFAULT_FORMAT_AXIS)
        offset_text = ax.yaxis.get_offset_text()
        offset_text.set_fontsize(pu.DEFAULT_AXIS_TICK_FONTSIZE)
        offset_text.set_weight(pu.DEFAULT_AXIS_TICK_WEIGHT)

    def plot_is_int_order_scatter(
        self,
        stage_dfs: dict[str, core_classes.MetaboInt],
        pred_df: core_classes.MetaboInt | None,
        valid: list[str],
        sample_type: str,
        batch: str,
        inject_order: str,
        qc_label: str,
        actual_label: str,
        boundary: str,
    ) -> Iterator[tuple[object, object]]:
        """Dynamically assemble IS scatters using a data-driven 2/3+1 grid.

        Yields figures iteratively to prevent Matplotlib memory leaks and
        registry collisions during sequential batch saving.
        """
        try:
            import patchworklib as pw
            import seaborn as sns
        except ImportError:
            return

        if not valid:
            return

        has_baseline = pred_df is not None

        for feat in valid:
            pw.clear()
            bricks = []
            shared_ylim = self._get_is_shared_ylim(
                stage_dfs=stage_dfs,
                pred_df=pred_df,
                feat=feat,
                boundary=boundary,
            )
            shared_yticks = self._get_is_shared_yticks(shared_ylim)

            for stage_name, df in stage_dfs.items():
                brick = pw.Brick(figsize=(6.5, 2.0))

                # Directly reuse existing base plotter for individual panels
                self.plot_single_is_scatter(
                    df=df,
                    feat=feat,
                    sample_type=sample_type,
                    batch=batch,
                    inject_order=inject_order,
                    qc_label=qc_label,
                    actual_label=actual_label,
                    ylabel=stage_name,
                    boundary=boundary,
                    ax=brick,
                    ylim=shared_ylim,
                    yticks=shared_yticks,
                )

                # Overlay prediction lines strictly for the Original stage
                if stage_name == "Original" and has_baseline:
                    pred_info = pred_df.int_order_info(feat_type="IS").reset_index()

                    for batch_id in pred_info[batch].unique():
                        b_pred = pred_info[pred_info[batch] == batch_id]
                        sns.lineplot(
                            data=b_pred,
                            x=inject_order,
                            y=feat,
                            color="k",
                            linestyle="-",
                            ax=brick,
                            zorder=3,
                        )
                    if shared_ylim is not None:
                        self._apply_is_shared_y_axis(
                            ax=brick, ylim=shared_ylim, yticks=shared_yticks
                        )

                # Strip internal standard legends to favor the global brick
                if brick.get_legend():
                    brick.get_legend().remove()

                bricks.append(brick)

            # Assemble left column iteratively via patchworklib
            if not bricks:
                continue

            left_col = bricks[0]
            for b in bricks[1:]:
                left_col = left_col / b

            # Assemble right column (Standalone Legend Brick)
            leg_h = len(bricks) * 2.0
            leg_brick = pw.Brick(figsize=(2.5, leg_h))

            self._plot_standalone_is_legend(
                ax=leg_brick,
                sample_type=sample_type,
                batch=batch,
                qc_label=qc_label,
                actual_label=actual_label,
                has_baseline=has_baseline,
            )

            # Yield immediately to allow saving before the next iteration
            yield feat, left_col | leg_brick

    def plot_single_is_scatter(
        self,
        df: core_classes.MetaboInt,
        feat: str,
        sample_type: str,
        batch: str,
        inject_order: str,
        qc_label: str,
        actual_label: str,
        ylabel: str,
        boundary: str,
        ax: plt.Axes | None = None,
        ylim: tuple[float, float] | None = None,
        yticks: np.ndarray | None = None,
    ) -> object:
        """Plot a single scatter panel with calculated boundaries."""
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(7.5, 2.5))
        else:
            current_ax = ax
            fig = current_ax.figure

        plot_data = df.int_order_info(feat_type="IS").reset_index()
        plot_data[sample_type] = pd.Categorical(
            plot_data[sample_type], categories=[actual_label, qc_label], ordered=True
        )
        plot_data = plot_data.sort_values(sample_type)

        sns.scatterplot(
            data=plot_data,
            x=inject_order,
            y=feat,
            hue=sample_type,
            style=batch,
            s=40,
            edgecolor="k",
            palette=self.pal,
            hue_order=[qc_label, actual_label],
            markers=self.style_map,
            style_order=self.all_batches,
            ax=current_ax,
        )

        solid_line, lower_limit, upper_limit = (
            core_classes.MetaboInt().calculate_boundaries(plot_data[feat], boundary)
        )
        for y, linestyle in zip(
            [solid_line, lower_limit, upper_limit], ["-", "--", "--"]
        ):
            current_ax.axhline(y, color="k", linestyle=linestyle)

        self._apply_is_shared_y_axis(
            ax=current_ax,
            ylim=ylim,
            yticks=yticks,
        )

        # Enable append_stage=True and feed the precise pipeline stage attribute
        self._apply_standard_format(
            current_ax,
            title=feat,
            xlabel=inject_order,
            ylabel=ylabel,
            append_stage=True,
            custom_stage=df.attrs.get("pipeline_stage", ""),
        )
        self._apply_is_shared_y_axis(
            ax=current_ax,
            ylim=ylim,
            yticks=yticks,
        )
        return fig

    def plot_pred_baseline_is(
        self,
        raw: core_classes.MetaboInt,
        pred: core_classes.MetaboInt | None,
        valid: list[str],
        sample_type: str,
        batch: str,
        inject_order: str,
        qc_label: str,
        actual_label: str,
        method: str = "QC-RLSC",
    ) -> object | None:
        """Assemble IS fitted-baseline panels with a single shared legend."""
        try:
            import patchworklib as pw
        except ImportError:
            return None

        if not valid:
            return None

        pw.clear()
        plot_bricks = []
        panel_cols = 1 if len(valid) == 1 else 2
        panel_size = (6.5, 2.0)

        pred_info = None
        global_model_methods = {"SERRF", "RUV-III", "WAVEICA 2.0"}
        if pred is not None and method.upper() not in global_model_methods:
            pred_info = pred.int_order_info(feat_type="IS").reset_index()

        for n, feat in enumerate(valid):
            ax = pw.Brick(figsize=panel_size, label=f"pred_base_is_{n}")
            plot_data = raw.int_order_info(feat_type="IS").reset_index()

            plot_data[sample_type] = pd.Categorical(
                plot_data[sample_type],
                categories=[actual_label, qc_label],
                ordered=True,
            )
            plot_data = plot_data.sort_values(sample_type)

            sns.scatterplot(
                data=plot_data,
                x=inject_order,
                y=feat,
                hue=sample_type,
                style=batch,
                s=40,
                edgecolor="k",
                palette=self.pal,
                hue_order=[qc_label, actual_label],
                markers=self.style_map,
                style_order=self.all_batches,
                ax=ax,
            )

            if pred_info is not None and feat in pred_info.columns:
                for batch_id in pred_info[batch].unique():
                    sns.lineplot(
                        data=pred_info[pred_info[batch] == batch_id],
                        x=inject_order,
                        y=feat,
                        color="k",
                        ax=ax,
                    )
            self._apply_standard_format(
                ax,
                title=feat,
                xlabel=inject_order,
                ylabel="Intensity",
                append_stage=False,
            )
            pu.change_axis_format(ax, "scientific notation", "y")
            pu.change_fontsize(ax, axis="y")
            pu.change_weight(ax, axis="y")
            offset_text = ax.yaxis.get_offset_text()
            offset_text.set_fontsize(pu.DEFAULT_AXIS_TICK_FONTSIZE)
            offset_text.set_weight(pu.DEFAULT_AXIS_TICK_WEIGHT)

            if ax.get_legend():
                ax.get_legend().remove()
            plot_bricks.append(ax)

        row_bricks = []
        for row_start in range(0, len(plot_bricks), panel_cols):
            row_items = plot_bricks[row_start : row_start + panel_cols]
            if panel_cols == 2 and len(row_items) == 1:
                spacer = pw.Brick(figsize=panel_size, label=f"pred_base_is_spacer_{n}")
                spacer.axis("off")
                row_items.append(spacer)

            row = row_items[0]
            for item in row_items[1:]:
                row = row | item
            row_bricks.append(row)

        plot_grid = row_bricks[0]
        for row in row_bricks[1:]:
            plot_grid = plot_grid / row

        legend_height = max(panel_size[1], len(row_bricks) * panel_size[1])
        leg_brick = pw.Brick(figsize=(2.5, legend_height), label="pred_base_is_legend")
        self._plot_standalone_is_legend(
            ax=leg_brick,
            sample_type=sample_type,
            batch=batch,
            qc_label=qc_label,
            actual_label=actual_label,
            has_baseline=pred_info is not None,
        )

        return plot_grid | leg_brick
