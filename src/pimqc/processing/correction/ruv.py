"""Implement RUV-III unwanted-variation removal for intensity matrices.

``RUVCorrector`` estimates unwanted factors from QC samples and control
features, applies the SVD-based projection, and returns corrected numerical
data. It does not depend on configuration, export, or visualization.
"""

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class RUVCorrector:
    """Pure mathematical engine for RUV-III correction via SVD projection."""

    def __init__(self, k: int = 3) -> None:
        """Initialize the unwanted-factor model.

        Args:
            k: Number of unwanted latent factors to remove.
        """
        self.k = k

    def fit_transform(
        self,
        intensity_df: pd.DataFrame,
        qc_mask: np.ndarray,
        control_features: pd.Index,
        blank_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
        """Fit RUV-III on eligible samples and transform the full matrix.

        Args:
            intensity_df: Feature-by-sample intensity matrix.
            qc_mask: Boolean mask identifying pooled-QC samples.
            control_features: Features used to estimate unwanted variation.
            blank_mask: Optional mask excluding blanks from model fitting.

        Returns:
            Mapping containing the corrected matrix and optional OOF result.

        Raises:
            ValueError: If controls or eligible fitting samples are missing.
        """

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
