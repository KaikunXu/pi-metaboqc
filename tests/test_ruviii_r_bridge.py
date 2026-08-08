# tests/test_ruviii_r_bridge.py
"""
Script purpose: Compare the Python RUV-III corrector with an R implementation.

RUV-III is deterministic linear algebra in the project implementation, so this
bridge test checks numerical equivalence against an R reference written with the
same matrix operations.
"""

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from pimqc.processing.correction import RUVCorrector


def _make_ruviii_matrix() -> tuple[pd.DataFrame, np.ndarray]:
    """Create a synthetic intensity matrix with order and batch drift."""
    rng = np.random.default_rng(20260611)
    n_features, n_samples = 36, 30
    order = np.arange(1, n_samples + 1, dtype=float)
    batch = np.array(["B1"] * 15 + ["B2"] * 15)
    qc_mask = np.array([(i % 3) == 0 for i in range(n_samples)])

    base = rng.lognormal(mean=8.0, sigma=0.35, size=n_features)
    order_scaled = (order - order.mean()) / np.ptp(order)
    drift = 1.0 + 0.45 * order_scaled + 0.12 * np.sin(order / 3.5)
    batch_factor = np.where(batch == "B2", 1.15, 1.0)
    feature_weight = rng.uniform(0.75, 1.25, size=n_features)

    arr = base[:, None] * (1.0 + feature_weight[:, None] * (drift[None, :] - 1.0))
    arr *= batch_factor[None, :]
    arr *= rng.normal(1.0, 0.015, size=arr.shape)
    arr = np.clip(arr, a_min=1e-6, a_max=None)

    index = [f"Met_{i:02d}" for i in range(n_features)]
    columns = [f"S{i:02d}" for i in range(n_samples)]
    return pd.DataFrame(arr, index=index, columns=columns), qc_mask


def run_r_ruviii(
    df_input: pd.DataFrame,
    qc_mask: np.ndarray,
    control_features: pd.Index,
    k: int = 2,
) -> pd.DataFrame:
    """Execute an R implementation matching the Python RUV-III engine."""
    r_script = """
    function(df, qc_mask, control_features, k) {
        mat <- as.matrix(df)
        storage.mode(mat) <- "double"
        y_raw <- t(mat)
        n_samples <- nrow(y_raw)

        y_safe <- pmax(y_raw, 0)
        y <- log1p(y_safe)
        zero_mask <- apply(y_safe == 0, 2, all)
        nan_mask <- is.na(y)
        if (any(nan_mask)) {
            col_medians <- apply(y, 2, function(x) median(x, na.rm=TRUE))
            col_medians[is.na(col_medians)] <- 0
            idx <- which(nan_mask, arr.ind=TRUE)
            y[idx] <- col_medians[idx[, 2]]
        }

        group_ids <- integer(n_samples)
        uid <- 1L
        for (i in seq_len(n_samples)) {
            if (qc_mask[i]) {
                group_ids[i] <- 0L
            } else {
                group_ids[i] <- uid
                uid <- uid + 1L
            }
        }
        groups <- sort(unique(group_ids))
        m <- matrix(0, nrow=n_samples, ncol=length(groups))
        for (i in seq_len(n_samples)) {
            m[i, match(group_ids[i], groups)] <- 1
        }

        group_sizes <- t(m) %*% m
        group_means <- solve(group_sizes, t(m) %*% y)
        y0 <- y - m %*% group_means

        feature_names <- rownames(mat)
        ctl_mask <- feature_names %in% control_features & !zero_mask
        y0_ctl <- y0[, ctl_mask, drop=FALSE]
        sv <- svd(y0_ctl)
        safe_k <- min(as.integer(k), nrow(y0_ctl), ncol(y0_ctl))
        alpha_ctl <- t(sv$v[, seq_len(safe_k), drop=FALSE])

        w <- y[, ctl_mask, drop=FALSE] %*% t(alpha_ctl)
        w_means <- solve(group_sizes, t(m) %*% w)
        w0 <- w - m %*% w_means
        alpha_full <- qr.solve(w0, y0)

        correction <- w %*% alpha_full
        correction <- sweep(correction, 2, colMeans(correction), "-")
        correction[, zero_mask] <- 0

        y_corr_log <- y - correction
        y_corrected <- expm1(y_corr_log)
        y_corrected[y_corrected < 1e-6] <- 1e-6
        y_corrected[, zero_mask] <- 1e-6
        if (any(nan_mask)) {
            y_corrected[nan_mask] <- NA
        }

        res <- t(y_corrected)
        dimnames(res) <- dimnames(mat)
        return(as.data.frame(res))
    }
    """
    r_ruv_func = ro.r(r_script)
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df_result = r_ruv_func(
            df_input,
            ro.BoolVector([bool(x) for x in qc_mask]),
            ro.StrVector(list(control_features)),
            k,
        )
    r_df_result.index = df_input.index
    r_df_result.columns = df_input.columns
    return r_df_result


def test_ruviii_matches_r_linear_algebra_reference() -> None:
    """Check that Python RUV-III matches the equivalent R implementation."""
    df_input, qc_mask = _make_ruviii_matrix()
    control_features = pd.Index(df_input.index[:12])

    py_df = RUVCorrector(k=2).fit_transform(
        df_input, qc_mask=qc_mask, control_features=control_features
    )["RUV-III"][0]
    r_df = run_r_ruviii(
        df_input, qc_mask=qc_mask, control_features=control_features, k=2
    )

    np.testing.assert_allclose(py_df.to_numpy(), r_df.to_numpy(), rtol=1e-8, atol=1e-8)
