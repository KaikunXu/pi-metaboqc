"""Implement the native Python WaveICA 2.0 correction workflow.

The module contains wavelet decomposition, stICA/Jade-style component analysis,
injection-order association filtering, and matrix reconstruction. Pipeline
selection, filesystem output, and diagnostic rendering are handled elsewhere.
"""

import math
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer

from ...constants import DEFAULT_RANDOM_SEED


class WaveICA2Corrector:
    """
    Native Python WaveICA 2.0 correction using order-associated stICA removal.

    This implementation follows the public WaveICA_2.0 R workflow: samples are
    ordered by injection order, and each feature is decomposed with periodic
    Haar MODWT,
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
        random_state: int = DEFAULT_RANDOM_SEED,
    ) -> None:
        """Initialize WaveICA decomposition and selection settings.

        Args:
            n_components: Maximum independent components per wavelet scale.
            cutoff: Minimum order-association threshold for component removal.
            n_levels: Optional explicit number of wavelet levels.
            spline_knots: Knots used to model injection-order trends.
            max_iter: Maximum iterations for component estimation.
            random_state: Seed for deterministic factor initialization.
        """
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
