import os
import re
import math
import copy
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed
from loguru import logger
from numba import njit, prange
from typing import Tuple, Any, Callable, Union, Dict, Optional

from . import io_utils as iu
from . import plot_utils as pu
from . import core_classes
from . import visualizer_classes

# ==============================================================================
# Cross-Validation Engine for Robust Drift Correction to Prevent Overfitting
# ==============================================================================
def fit_predict_intra_batch_safely(
    base_model: Union[Any, Callable],
    x_qc: np.ndarray,
    y_qc: np.ndarray,
    x_all: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 123
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
        return (np.clip(pred_all_full, a_min=1e-6, a_max=None), 
                np.clip(pred_qc_oof, a_min=1e-6, a_max=None))

    kf = KFold(n_splits=safe_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(x_qc):
        pred_qc_oof[test_idx] = _run_model(
            x_train=x_qc[train_idx], y_train=y_qc[train_idx], x_test=x_qc[test_idx]
        )
        
    pred_all_full = _run_model(x_train=x_qc, y_train=y_qc, x_test=x_all)
    
    return (np.clip(pred_all_full, a_min=1e-6, a_max=None), 
            np.clip(pred_qc_oof, a_min=1e-6, a_max=None))

# ==============================================================================
# Numba JIT Engines for Fast Robust QC-RLSC 
# ==============================================================================
@njit(fastmath=True)
def _tricube_kernel(x: float) -> float:
    abs_x = abs(x)
    if abs_x >= 1.0: return 0.0
    return (1.0 - abs_x**3)**3

@njit(fastmath=True)
def _bisquare_weight(res: float, s: float) -> float:
    if s <= 1e-9: return 1.0 if abs(res) <= 1e-9 else 0.0
    v = res / (6.0 * s)
    if abs(v) >= 1.0: return 0.0
    return (1.0 - v*v)**2

@njit(fastmath=True)
def _numba_loess_1d_core(
    x: np.ndarray, y: np.ndarray, x_pred: np.ndarray, 
    loess_frac: float, delta: np.ndarray
) -> np.ndarray:
    n = len(x)
    m = len(x_pred)
    y_pred = np.zeros(m)
    
    k = int(math.ceil(n * loess_frac))
    if k < 2: k = 2
    if k > n: k = n

    for i in range(m):
        x0 = x_pred[i]
        diffs = np.abs(x - x0)
        sorted_diffs = np.sort(diffs)
        h = sorted_diffs[k-1]
        
        if h <= 0.0: h = 1e-9
            
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
    x: np.ndarray, y: np.ndarray, x_pred: np.ndarray, 
    loess_frac: float, max_iter: int
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
    data: np.ndarray, qc_mask: np.ndarray, injection_orders: np.ndarray, 
    loess_frac: float, max_iter: int
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

    def __init__(self, method: str, **kwargs: Any) -> None:
        """
        Initialize purely with mathematical hyperparameters.
        No domain knowledge (like sample types or columns) is permitted here.
        """
        self.method = method.upper()
        self.params = kwargs

    def _build_correction_pipeline(self) -> Union[Any, Callable]:
        """Construct the inner pipeline with forced single-thread policy."""
        if self.method in ("RF", "RANDOM FOREST", "QC-RFSC"):
            return RandomForestRegressor(
                n_estimators=self.params.get("rf_n_tree", 200), 
                random_state=self.params.get("global_seed", 123),
            )
        elif self.method in ("SVR", "QC-SVR", "QC-SVRC"):
            # SVR is inherently single-threaded in scikit-learn
            base_svr = make_pipeline(
                StandardScaler(), 
                SVR(
                    kernel=self.params.get("svr_kernel", "rbf"), 
                    C=self.params.get("svr_c", 10), 
                    gamma=self.params.get("svr_gamma", 1.0)
                )
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
        order_array: np.ndarray
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Execute core mathematical fitting using pure arrays and masks."""
        stages_output = {}
        logger.info(
            "Phase 1: Executing Intra-batch drift correction with "
            f"{self.method}...")
        
        # Force initialization to float64 to accept fractional predictions
        pred_df = intensity_df.copy().astype(float)
        oof_pred_df = intensity_df.copy().astype(float)
        unique_batches = np.unique(batch_array)
        
        # Restore full core utilization for lightweight threading
        n_jobs_conf = self.params.get("n_jobs", -1)
        
        # Intra-batch processing
        for batch_id in unique_batches:
            b_mask = (batch_array == batch_id)
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
                    data=b_data.values, qc_mask=b_qc_mask, 
                    injection_orders=b_orders, loess_frac=loess_frac, 
                    max_iter=max_iter
                )
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
                        fold_pred = _numba_batch_qc_rlsc(
                            data=b_data.values, qc_mask=train_qc_mask, 
                            injection_orders=b_orders, loess_frac=loess_frac, 
                            max_iter=max_iter
                        )
                        test_qcs = qc_indices[test_idx]
                        oof_matrix[:, test_qcs] = fold_pred[:, test_qcs]
                oof_pred_df.loc[:, b_mask] = oof_matrix
            else:
                feat_idx = b_data.index
                mat_vals = b_data.values
                
                tasks = [delayed(self._fit_predict_feature)(
                    feat_idx[i], mat_vals[i, :], b_qc_mask, b_orders
                ) for i in range(len(feat_idx))]
                
                # [FIX]: Use patched joblib context manager
                with iu.tqdm_joblib_env(total=len(tasks), desc=f"SC [{batch_id}]"):
                    results = Parallel(
                        n_jobs=n_jobs_conf, backend="loky")(tasks)
                
                for f_idx, p_full, p_oof in results:
                    pred_df.loc[f_idx, b_mask] = p_full
                    oof_pred_df.loc[f_idx, b_mask] = p_oof

        # Mathematical division with broadcasted QC means
        pred_df[pred_df <= 0] = np.nan
        oof_pred_df[oof_pred_df <= 0] = np.nan
        
        qc_intensity = intensity_df.loc[:, qc_mask]
        batch_qc_means = qc_intensity.T.groupby(batch_array[qc_mask]).mean().T
        
        base_bc = pd.DataFrame(
            index=intensity_df.index, columns=intensity_df.columns
        )
        for b_id in unique_batches:
            b_mask = (batch_array == b_id)
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
                b_mask = (batch_array == b_id)
                inter_full.loc[:, b_mask] = inter_full.loc[:, b_mask].multiply(
                    global_mean / bt_qc_mean[b_id], axis=0
                )
                inter_oof.loc[:, b_mask] = inter_oof.loc[:, b_mask].multiply(
                    global_mean_oof / bt_qc_mean_oof[b_id], axis=0
                )
            stages_output["Inter-batch corrected"] = (inter_full, inter_oof)

        return stages_output

    def _fit_predict_feature(
        self, feat_idx: Any, raw_vals: np.ndarray, 
        qc_mask: np.ndarray, inject_order_array: np.ndarray
    ) -> Tuple[Any, np.ndarray, np.ndarray]:
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
                base_model=model, x_qc=qc_x[valid], y_qc=qc_y[valid],
                x_all=x_all, cv_folds=cv_folds, random_state=seed
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
        self, n_estimators: int = 100, cv_folds: int = 5, 
        n_corr_features: int = 10, random_state: int = 123, n_jobs: int = -1
    ) -> None:
        self.n_estimators = n_estimators
        self.cv_folds = cv_folds
        self.n_corr_features = n_corr_features
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _prepare_base_features(self, batch_array: np.ndarray, order_array: np.ndarray) -> np.ndarray:
        x_df = pd.DataFrame({"Order": order_array, "Batch": batch_array})
        x_df["Order"] = pd.to_numeric(x_df["Order"], errors="coerce")
        x_encoded = pd.get_dummies(x_df, columns=["Batch"], drop_first=False)
        return np.nan_to_num(x_encoded.values, nan=0.0)

    def _process_single_feature(
        self, feat_idx, y_mat, x_base, is_qc, top_idx_row
    ):
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
        kf = KFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        y_pred_qc = np.zeros(len(y_qc_valid))
        
        rf_oof = RandomForestRegressor(
            n_estimators=self.n_estimators, 
            random_state=self.random_state, 
            n_jobs=1
        )
        for train_idx, test_idx in kf.split(x_qc_valid):
            rf_oof.fit(x_qc_valid[train_idx], y_qc_valid[train_idx])
            y_pred_qc[test_idx] = rf_oof.predict(x_qc_valid[test_idx])

        # 4. Predict Expected Baselines for ALL Samples
        rf_full = RandomForestRegressor(
            n_estimators=self.n_estimators, 
            random_state=self.random_state, 
            n_jobs=1
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
        self, intensity_df: pd.DataFrame, batch_array: np.ndarray, 
        qc_mask: np.ndarray, order_array: np.ndarray, 
        corr_mat: Optional[np.ndarray] = None
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        
        logger.info("Initializing High-Performance Hybrid SERRF Corrector...")
        y_mat = intensity_df.T.values 
        x_base = self._prepare_base_features(batch_array, order_array)
        n_features = y_mat.shape[1]
        
        if sum(qc_mask) < self.cv_folds:
            raise ValueError("Insufficient QCs for configured CV.")

        # Extract top features globally to avoid memory leak in parallel workers
        top_indices = None
        if self.n_corr_features > 0 and corr_mat is not None:
            # Prevent feature from selecting itself as highly correlated
            np.fill_diagonal(corr_mat, -1.0)
            top_indices = np.argsort(
                corr_mat, axis=1
            )[:, -self.n_corr_features:]

        y_corrected = np.zeros_like(y_mat)
        y_corrected_oof = np.zeros_like(y_mat)
        
        actual_cores = (os.cpu_count() or 1) if self.n_jobs == -1 else self.n_jobs
        safe_n_jobs = max(1, int(actual_cores / 2))
        
        # [FIX]: Use patched joblib context manager for SERRF
        with iu.tqdm_joblib_env(total=n_features, desc="SERRF"):
            results = Parallel(n_jobs=safe_n_jobs, backend="loky")(
                delayed(self._process_single_feature)(
                    feat_idx, y_mat, x_base, qc_mask, 
                    None if top_indices is None else top_indices[feat_idx]
                ) for feat_idx in range(n_features)
            )

        for res in results:
            feat_idx, res_full, res_oof = res
            y_corrected[:, feat_idx] = res_full
            y_corrected_oof[:, feat_idx] = res_oof

        res_df_full = pd.DataFrame(
            y_corrected.T, index=intensity_df.index, 
            columns=intensity_df.columns
        )
        res_df_oof = pd.DataFrame(
            y_corrected_oof.T, index=intensity_df.index, 
            columns=intensity_df.columns
        )
        return {"Global SERRF": (res_df_full, res_df_oof)}

# ==============================================================================
# Engine 3: RUVCorrector
# ==============================================================================
class RUVCorrector:
    """Pure mathematical engine for RUV-III correction via SVD projection."""

    def __init__(self, k: int = 3) -> None:
        self.k = k

    def fit_transform(
        self, intensity_df: pd.DataFrame, qc_mask: np.ndarray, 
        control_features: pd.Index
    ) -> Dict[str, Tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
        
        logger.info(f"Executing Global RUV-III (k={self.k})...")
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
        
        return {"Global RUV": (res_df_full, None)}

# ==============================================================================
# Main Dispatcher Class: MetaboIntCorrector
# ==============================================================================
class MetaboIntCorrector(core_classes.MetaboInt):
    """
    Quality control-based signal drift correction dispatcher.
    
    Orchestrates the dynamic execution routing between RegressionCorrector,
    SERRFCorrector, and RUVCorrector. Extracts domain-specific metadata
    into pure mathematical arrays to preserve engine purity. Manages file 
    exports, dual-mode RSD tracking, and downstream visualization diagnostics.
    """

    _metadata = ["attrs"]

    def __init__(
        self,
        *args: Any,
        pipeline_params: Optional[Dict[str, Any]] = None,
        base_est: Optional[str] = None,
        loess_frac: Optional[float] = None,
        rf_n_tree: Optional[int] = None,
        serrf_n_tree: Optional[int] = None,
        serrf_corr_features: Optional[int] = None,
        svr_kernel: Optional[str] = None,
        svr_c: Optional[Union[float, int]] = None,
        svr_gamma: Optional[Union[str, float]] = None,
        ruv_k: Optional[int] = None,
        cv_folds: Optional[int] = None,
        n_jobs: Optional[int] = None,
        **kwargs: Any
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
            "svr_kernel": "rbf",
            "svr_c": 10,
            "svr_gamma": 1.0,
            "cv_folds": 3,
            "ruv_k": 3,
            "n_jobs": getattr(iu, "__max_threading__", -1),
            "global_seed": 123
        }

        # 2. Extract pipeline configuration layer from TOML
        if pipeline_params and "MetaboIntCorrector" in pipeline_params:
            sc_configs.update(pipeline_params["MetaboIntCorrector"])

        # 3. Explicit functional kwargs override TOML configurations
        local_args = locals()
        explicit_params = [
            "base_est", "loess_frac", "rf_n_tree", "serrf_n_tree",
            "serrf_corr_features", "svr_kernel", "svr_c", "svr_gamma",
            "cv_folds", "ruv_k", "n_jobs"
        ]
        for param in explicit_params:
            if local_args[param] is not None:
                sc_configs[param] = local_args[param]

        # 4. Integrate unified properties into internal attributes dictionary
        self.attrs.update(sc_configs)

    @property
    def _constructor(self):
        """Override constructor to return MetaboIntCorrector."""
        return MetaboIntCorrector

    def __finalize__(self, other, method=None, **kwargs):
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
    def extract_qc_rsd_series(df_obj):
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
    def calculate_median_qc_rsd(df_obj):
        """Calculates the scalar median RSD of QC samples."""
        rsd_series = MetaboIntCorrector.extract_qc_rsd_series(df_obj)
        if rsd_series.empty:
            return float("nan")
        return float(rsd_series.median())

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

    def _evaluate_all_methods(
        self, methods_to_run: list, batch_array: np.ndarray,
        qc_mask: np.ndarray, order_array: np.ndarray,
        batch_col: str, sample_type_col: str, qc_label: str
    ) -> Dict[str, Any]:
        """Execute core computation and metrics extraction for all methods."""
        results_store = {}
        for method in methods_to_run:
            logger.info(f"--- Evaluating Method: {method} ---")
            
            # 1. Route to specific engine
            if method == "SERRF":
                corr_mat = self._prepare_serrf_correlation_matrix()
                engine = SERRFCorrector(
                    n_estimators=self.attrs.get("serrf_n_tree", 100),
                    cv_folds=self.attrs.get("cv_folds", 5),
                    n_corr_features=self.attrs.get("serrf_corr_features", 10),
                    random_state=self.attrs.get("global_seed", 123),
                    n_jobs=self.attrs.get("n_jobs", -1)
                )
                stages_output = engine.fit_transform(
                    intensity_df=self, batch_array=batch_array, 
                    qc_mask=qc_mask, order_array=order_array, corr_mat=corr_mat
                )
            elif method in ("RUV", "RUV-III"):
                ctrl_features = self._prepare_ruv_control_features()
                engine = RUVCorrector(k=self.attrs.get("ruv_k", 3))
                stages_output = engine.fit_transform(
                    intensity_df=self, qc_mask=qc_mask, 
                    control_features=ctrl_features
                )
            else:
                engine = RegressionCorrector(method=method, **self.attrs)
                stages_output = engine.fit_transform(
                    intensity_df=self, batch_array=batch_array, 
                    qc_mask=qc_mask, order_array=order_array
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
                final_df.attrs["pipeline_stage"] = clean_name
                final_df.attrs["qc_rsd_baseline"] = raw_rsd
                
                curr_full_rsd = MetaboIntCorrector.calculate_median_qc_rsd(
                    final_df
                )
                rsd_hist_full[clean_name] = curr_full_rsd
                final_df.attrs["qc_rsd_current_full"] = curr_full_rsd
                
                if oof_df is not None:
                    oof_wrap = self._constructor(oof_df).__finalize__(self)
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
                "SERRF", "RUV", "RUV-III"
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
                "final_rsd_oof": final_oof
            }
            
            log_rsd = final_full if method in ("RUV", "RUV-III") else final_oof
            if log_rsd is not None:
                logger.info(f"{method} Eval QC RSD: {log_rsd * 100:.2f}%")

        return results_store
    
    def _find_best_correction(self, results_store: Dict[str, Any]) -> str:
        """
        Identify the optimal correction method based on QC RSD.
        
        RUV methods evaluate using the full dataset RSD. All other 
        methods rely on the Out-Of-Fold (OOF) cross-validated RSD to 
        prevent overfitting.
        """
        best_method = ""
        min_rsd = float("inf")
        
        for method, res in results_store.items():
            if method in ("RUV", "RUV-III"):
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
        
        qc_mask = (self.columns.get_level_values(sample_type_col) == qc_label)
        batch_array = self.columns.get_level_values(batch_col).values
        order_array = self.columns.get_level_values(inject_order_col).values
        
        req_method = self.attrs.get("base_est", "QC-RLSC").upper()
        self.attrs["is_auto_mode"] = (req_method == "AUTO")
        
        if req_method == "AUTO":
            methods_to_run = [
                "QC-RLSC", "QC-RFSC", "QC-SVR", "SERRF", "RUV"
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
            qc_label=qc_label
        )

        # ---------------------------------------------------------------------
        # 2. Selection Phase
        # ---------------------------------------------------------------------
        best_method = self._find_best_correction(results_store)
        
        if req_method == "AUTO":
            if best_method in ("RUV", "RUV-III"):
                best_rsd = results_store[best_method]["final_rsd_full"]
            else:
                best_rsd = results_store[best_method]["final_rsd_oof"]
                
            logger.success(
                f"Auto selection: {best_method} is optimal "
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
        
        # Auto mode: Generate the 2x3 comprehensive grid
        if req_method == "AUTO" and len(results_store) > 1:
            logger.info("Assembling evaluation grid for all methods...")
            grid_obj = vis.plot_corr_rsd_grid(results_store, best_method)
            
            if grid_obj is not None:
                vis.save_and_show_pw(
                    pw_obj=grid_obj, 
                    width="60%", 
                    file_path=os.path.join(
                        output_dir, "QC_RSD_Evaluation_Grid.svg"
                    )
                )

        # Standard behavior: Generate the specific plot for the chosen method
        logger.info(f"Generating specific RSD plot for: {best_method}")
        m_res = results_store[best_method]
        fig_rsd = vis.plot_corr_rsd(
            stage_dfs=m_res["stage_dfs"], 
            stage_oof_dfs=m_res.get("stage_oof_dfs", {})
        )
        vis.save_and_show_pw(
            pw_obj=fig_rsd, width="30%",
            file_path=os.path.join(output_dir, f"QC_RSD_{best_method}.svg")
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
                file_name = "Global_SERRF.csv"
            elif best_method in ("RUV", "RUV-III"):
                file_name = "Global_RUV.csv"
            else:
                prefix = clean_name.replace(" corrected", "")
                prefix = prefix.replace(" ", "_")
                file_name = f"{prefix}_{best_method}.csv"
                
            df.to_csv(os.path.join(output_dir, file_name))

        if best_pred_df is not None:
            best_pred_df.to_csv(
                os.path.join(output_dir, f"QC_Fit_Base_{best_method}.csv")
            )

        pipe_params = self.attrs.get("pipeline_parameters", {})
        bound_type = pipe_params.get("MetaboInt", {}).get("boundary", "IQR")
        
        if len(self.valid_is) > 0:
            logger.info(f"Generating IS plots for {best_method}...")
            
            is_dir = os.path.join(output_dir, "Internal_Standard_Scatters")
            iu._check_dir_exists(is_dir, handle="makedirs")
            
            # Consume the generator: Create -> Save -> Clear iteratively
            for feat, fig in vis.plot_is_int_order_scatter(
                self.stage_dfs, best_pred_df, self.valid_is, sample_type_col, 
                batch_col, inject_order_col, qc_label, actual_label, bound_type
            ):
                safe_feat = re.sub(r"[^a-zA-Z0-9]", "_", feat)
                save_path = os.path.join(
                    is_dir, f"IS_Scatter_{safe_feat}_{best_method}.svg"
                )
                vis.save_and_show_pw(
                    pw_obj=fig, file_path=save_path, show_plot=False)
            
            if best_method not in ("SERRF", "RUV", "RUV-III") and (
                best_pred_df is not None):
                fig_pred = vis.plot_pred_baseline_is(
                    self, best_pred_df, self.valid_is, sample_type_col, 
                    batch_col, inject_order_col, qc_label, actual_label, 
                    method=best_method
                )
                vis.save_and_close_fig(
                    fig_pred, os.path.join(
                        output_dir, f"Pred_Base_IS_{best_method}"
                    )
                )
            else:
                logger.info(
                    f"Bypassing IS baseline prediction for {best_method}.")

        logger.success(f"Signal drift correction ({best_method}) completed.")
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
        method = self.attrs.get("base_est", "Unknown").upper()

        metrics = {
            "correction_status": stage,
            "is_auto_mode": self.attrs.get("is_auto_mode", False),
            "overall_performance": {
                "median_qc_rsd_baseline": rsd_base,
                "median_qc_rsd_current_oof": rsd_curr_oof,
                "median_qc_rsd_current_full": rsd_curr_full,
                "relative_noise_reduction_oof": None,
                "relative_noise_reduction_full": None
            },
            "stages_executed": []
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
                    stage_params["loess_frac"] = self.attrs.get("loess_frac")
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
                elif alg_identifier in ("RUV", "RUV-III"):
                    stage_params["ruv_k"] = self.attrs.get("ruv_k")
                    
                if alg_identifier not in ("RUV", "RUV-III"):
                    stage_params["cv_folds"] = self.attrs.get("cv_folds")
                    
            metrics["stages_executed"].append({
                "stage_name": stage_name,
                "algorithm": alg_identifier,
                "parameters": stage_params,
                "stage_qc_rsd_oof": hist_oof.get(stage_name),
                "stage_qc_rsd_full": hist_full.get(stage_name)
            })
            
        return metrics
    
# ==============================================================================
# Main Visualizer Class: MetaboVisualizerCorrector
# ==============================================================================
class MetaboVisualizerCorrector(visualizer_classes.BaseMetaboVisualizer):
    """Visualization suite matching original alpha output styles."""

    def __init__(self, corr_obj):
        """Initialize with a computed MetaboIntCorrector object."""
        super().__init__(metabo_obj=corr_obj)
        self.corr = corr_obj

    # =========================================================================
    # Evaluation & Diagnostic Plotters
    # =========================================================================
    def plot_rsd_standalone_legend(
        self, 
        ax=None, 
        show_cv: bool = True,
        bbox_to_anchor: tuple = (0.1, 0.5)
    ):
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
            Patch(
                facecolor=c_base, edgecolor="k", linewidth=1.0,
                label="Baseline"
            )
        ]
        if show_cv:
            legend_elements.append(
                Patch(
                    facecolor=c_cv, edgecolor="k", linewidth=1.0,
                    linestyle="--", label="OOF Model"
                )
            )
        legend_elements.append(
            Patch(
                facecolor=c_full, edgecolor="k", linewidth=1.0,
                label="Full Model"
            )
        )

        # Step 1: Bind the explicit handles to a temporary legend object
        current_ax.legend(handles=legend_elements)
        
        # Step 2: Pass to the updated robust base formatter for styling
        self._format_single_legend(
            ax=current_ax,
            title="Correction Mode",
            loc="center left",
            bbox_to_anchor=bbox_to_anchor
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
        stage_dfs: dict,
        stage_oof_dfs: dict,
        ax=None,
        show_legend: bool = True
    ):
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
            medians_text.append(
                f"Before correction: {orig_rsd.median() * 100:.2f}%"
            )

        current_x = 2.6  # Adjusted starting position for wider spacing
        for stage_name, df in stage_dfs.items():
            if stage_name == "Original":
                continue
                
            clean_name = stage_name.replace("\n", " ")
            has_cv = stage_name in stage_oof_dfs
            full_rsd = self.corr.extract_qc_rsd_series(df)
            is_last = (stage_name == last_stage_key)
            
            if has_cv:
                cv_rsd = self.corr.extract_qc_rsd_series(
                    stage_oof_dfs[stage_name]
                )
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
                
                # Append asterisk strictly to the Full metric (e.g., RUV-III)
                prefix = "* " if is_last else ""
                medians_text.append(
                    f"{prefix}{clean_name} (Full): {full_rsd.median() * 100:.2f}%"
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
            box_data, positions=positions, widths=0.50,
            patch_artist=True, showfliers=False
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
            0.96, 0.98, annot_text, transform=current_ax.transAxes,
            fontsize=10, verticalalignment="top",
            horizontalalignment="right", clip_on=False,
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor="white",
                edgecolor="none", alpha=0.6))

        self._apply_standard_format(
            current_ax, ylabel="QC RSD (%)", append_stage=False
        )
        pu.change_axis_format(current_ax, "percentage", "y")
        
        return current_ax
    
    def plot_corr_rsd_grid(
        self, results_store: dict, best_method: str
    ):
        """Combine 5 RSD evaluation plots and a shared legend into 2x3 grid."""
        try:
            import patchworklib as pw
        except ImportError:
            raise ImportError("patchworklib is required for this plot.")

        pw.clear()
        bricks = {}

        # Ensure row sums are exactly 10.0 for perfectly aligned layouts
        # Row 1: 4.5 + 4.5 + 1.0 (Legend) = 10.0
        # Row 2: 4.5 + 2.9 + 2.6 = 10.0
        width_map = {
            "QC-RLSC": 4.2,
            "QC-RFSC": 4.2,
            "QC-SVR": 4.2,
            "SERRF": 3.1,
            "RUV": 2.7
        }

        for method, res in results_store.items():
            stage_dfs = res["stage_dfs"]
            stage_oof_dfs = res.get("stage_oof_dfs", {})

            fig_width = width_map.get(method, max(3.0, len(stage_dfs) * 1.0))
            safe_label = f"rsd_box_{method.replace('-', '_')}"
            
            b = pw.Brick(figsize=(fig_width, 4.0), label=safe_label)
            
            self.plot_corr_rsd(
                stage_dfs=stage_dfs,
                stage_oof_dfs=stage_oof_dfs,
                ax=b,
                show_legend=False
            )
            
            # Formatted to prefix the asterisk cleanly (e.g., "* SERRF")
            title = f"* {method}" if method == best_method else method
            b.set_title(title)
            bricks[method] = b

        # Dynamically attach explicit handles legend to the 1.0-width slot
        leg_brick = pw.Brick(figsize=(1.6, 4.0), label="shared_legend")
        self.plot_rsd_standalone_legend(ax=leg_brick, show_cv=True)

        r1_keys = ["QC-RLSC", "QC-RFSC"]
        r2_keys = ["QC-SVR", "SERRF", "RUV"]
        
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
        self, ax, sample_type, batch, qc_label, actual_label, has_baseline
    ):
        """Render a standalone multi-group legend for IS scatters."""
        import matplotlib.lines as mlines
        
        ax.axis("off")
        legend_handles = []
        legend_labels = []
        group_titles = [sample_type, batch]
        
        # Group 1: Sample Type
        legend_handles.append(
            mlines.Line2D([], [], color="none", label=sample_type)
        )
        legend_labels.append(sample_type)
        
        legend_handles.append(
            mlines.Line2D(
                [], [], color="tab:red", marker="o", linestyle="none", 
                markersize=6, markeredgecolor="k", markeredgewidth=0.5, 
                label=qc_label
            )
        )
        legend_labels.append(qc_label)
        
        legend_handles.append(
            mlines.Line2D(
                [], [], color="tab:gray", marker="o", linestyle="none", 
                markersize=6, markeredgecolor="k", markeredgewidth=0.5, 
                label=actual_label
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
                    [], [], color="tab:gray", marker=m_style, 
                    linestyle="none", markersize=6, markeredgecolor="k", 
                    markeredgewidth=0.5, label=str(b_val)
                )
            )
            legend_labels.append(str(b_val))
            
        # Group 3: Model Baseline (Rendered only if prediction exists)
        if has_baseline:
            group_titles.append("Model")
            legend_handles.append(
                mlines.Line2D([], [], color="none", label="Model")
            )
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
            ax=ax, group_titles=group_titles, loc="upper left", 
            start_bbox=(0.0, 0.95), group_pad=0.04, ncols=1, col_pad=0.1
        )
        
        # Prevent Patchworklib from discarding figure-level legends
        if hasattr(ax.figure, "legends"):
            for leg in list(ax.figure.legends):
                ax.add_artist(leg)
            ax.figure.legends.clear()
            
        return ax
    
    def plot_is_int_order_scatter(
        self, stage_dfs: dict, pred_df, valid, sample_type, batch,
        inject_order, qc_label, actual_label, boundary
    ):
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
            
            for stage_name, df in stage_dfs.items():
                brick = pw.Brick(figsize=(6.5, 2.0))
                
                # Directly reuse existing base plotter for individual panels
                self.plot_single_is_scatter(
                    df=df, feat=feat, sample_type=sample_type, batch=batch, 
                    inject_order=inject_order, qc_label=qc_label, 
                    actual_label=actual_label, ylabel=stage_name, 
                    boundary=boundary, ax=brick
                )
                
                # Overlay prediction lines strictly for the Original stage
                if stage_name == "Original" and has_baseline:
                    pred_info = pred_df.int_order_info(
                        feat_type="IS"
                    ).reset_index()
                    
                    for batch_id in pred_info[batch].unique():
                        b_pred = pred_info[pred_info[batch] == batch_id]
                        sns.lineplot(
                            data=b_pred, x=inject_order, y=feat, 
                            color="k", linestyle="-", ax=brick, zorder=3
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
                ax=leg_brick, sample_type=sample_type, batch=batch, 
                qc_label=qc_label, actual_label=actual_label, 
                has_baseline=has_baseline
            )
            
            # Yield immediately to allow saving before the next iteration
            yield feat, left_col | leg_brick

    def plot_single_is_scatter(
        self, df, feat, sample_type, batch, inject_order, qc_label,
        actual_label, ylabel, boundary, ax=None
    ):
        """Plot a single scatter panel with calculated boundaries."""
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(7.5, 2.5))
        else:
            current_ax = ax
            fig = current_ax.figure
            
        plot_data = df.int_order_info(feat_type="IS").reset_index()
        plot_data[sample_type] = pd.Categorical(
            plot_data[sample_type], categories=[actual_label, qc_label],
            ordered=True
        )
        plot_data = plot_data.sort_values(sample_type)
        
        sns.scatterplot(
            data=plot_data, x=inject_order, y=feat, hue=sample_type,
            style=batch, s=40, edgecolor="k", palette=self.pal,
            hue_order=[qc_label, actual_label], markers=self.style_map, 
            style_order=self.all_batches, ax=current_ax
        )
        
        solid_line, lower_limit, upper_limit = core_classes.MetaboInt(
            ).calculate_boundaries(plot_data[feat], boundary)
        for y, linestyle in zip(
            [solid_line, lower_limit, upper_limit], ["-", "--", "--"]):
            current_ax.axhline(y, color="k", linestyle=linestyle)
            
        # Enable append_stage=True and feed the precise pipeline stage attribute
        self._apply_standard_format(
            current_ax, 
            title=feat,
            xlabel=inject_order, 
            ylabel=ylabel, 
            append_stage=True,
            custom_stage=df.attrs.get("pipeline_stage", "")
        )
        pu.change_axis_format(current_ax, "scientific notation", "y")
        return fig

    def plot_pred_baseline_is(
        self, raw, pred, valid, sample_type, batch, inject_order, qc_label,
        actual_label, method="QC-RLSC"
    ):
        """Reconstruct original multi-panel baseline overlay grid."""
        num_cols = 2
        num_rows = int(np.ceil(len(valid) / num_cols))
        fig = plt.figure(
            figsize=(7.5 * num_cols, 2.5 * num_rows), layout="constrained")
        
        for n, feat in enumerate(valid):
            ax = plt.subplot(num_rows, num_cols, n + 1)
            plot_data = raw.int_order_info(feat_type="IS").reset_index()
            
            plot_data[sample_type] = pd.Categorical(
                plot_data[sample_type], categories=[actual_label, qc_label],
                ordered=True
            )
            plot_data = plot_data.sort_values(sample_type)
            
            sns.scatterplot(
                data=plot_data, x=inject_order, y=feat, hue=sample_type,
                style=batch, s=40,  edgecolor="k", palette=self.pal,
                hue_order=[qc_label, actual_label], 
                markers=self.style_map, style_order=self.all_batches, ax=ax
            )
            
            # [Fix]: Only overlay baseline if it exists (i.e., not SERRF)
            if pred is not None and method.upper() != "SERRF":
                pred_info = pred.int_order_info(feat_type="IS").reset_index()
                for batch_id in pred_info[batch].unique():
                    sns.lineplot(
                        data=pred_info[pred_info[batch] == batch_id],
                        x=inject_order, y=feat, color="k", ax=ax)
            self._apply_standard_format(
                ax, xlabel=inject_order, ylabel=feat, append_stage=False)
            
            if n == len(valid) - 1:
                self._format_multi_legends(
                    ax=ax, group_titles=[sample_type, batch])
            elif ax.get_legend():
                ax.legend().remove()
        return fig