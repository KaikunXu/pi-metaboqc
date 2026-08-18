"""Implement QC-anchored regression correction kernels.

``RegressionCorrector`` fits QC-RLSC, QC-SVR, or QC-RFSC models from numerical
arrays and returns intra- and inter-batch correction stages. Domain metadata,
candidate selection, file export, and plotting are intentionally excluded.
"""

import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from ...constants import DEFAULT_RANDOM_SEED
from ...runtime import joblib_execution_context, joblib_progress
from .algorithms import (
    CorrectionModel,
    _numba_batch_qc_rlsc,
    _numba_batch_qc_rlsc_per_feature_spans,
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
                random_state=self.params.get(
                    "global_seed", DEFAULT_RANDOM_SEED
                ),
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
                seed = self.params.get("global_seed", DEFAULT_RANDOM_SEED)
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
                with joblib_execution_context(joblib_backend):
                    with joblib_progress(
                        total=len(feat_idx), desc=f"SC [{batch_id}]"
                    ):
                        results = Parallel(
                            n_jobs=n_jobs_conf,
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
        seed = self.params.get("global_seed", DEFAULT_RANDOM_SEED)

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
