# tests/test_serrf_r_bridge.py
"""
Script purpose: Compare the Python SERRF corrector with an R randomForest path.

SERRF uses different random-forest engines in Python and R, so this bridge test
checks the shared correction contract rather than requiring identical values:
preserved shape, finite output, improved QC reproducibility, and high corrected
matrix concordance.
"""

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from scipy.stats import spearmanr

from pimqc.correction import SERRFCorrector


def _require_r_package(package: str) -> None:
    """Skip the bridge test when an optional R package is unavailable."""
    r_require = ro.r("function(pkg) requireNamespace(pkg, quietly=TRUE)")
    available = bool(r_require(package)[0])
    if not available:
        pytest.skip(f"R package '{package}' is not installed.")


def _make_serrf_matrix() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
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
    return pd.DataFrame(arr, index=index, columns=columns), batch, qc_mask, order


def _median_qc_rsd(df: pd.DataFrame, qc_mask: np.ndarray) -> float:
    """Calculate median feature RSD across QC samples."""
    qc_df = df.iloc[:, qc_mask]
    means = qc_df.mean(axis=1)
    rsd = qc_df.std(axis=1, ddof=1).div(means.replace(0.0, np.nan))
    return float(np.nanmedian(rsd.to_numpy()))


def run_r_serrf(
    df_input: pd.DataFrame,
    batch: np.ndarray,
    qc_mask: np.ndarray,
    order: np.ndarray,
    n_trees: int = 120,
) -> pd.DataFrame:
    """Execute a compact R randomForest implementation of SERRF correction."""
    _require_r_package("randomForest")
    r_script = """
    function(df, batch, qc_mask, order, n_trees) {
        suppressWarnings(suppressPackageStartupMessages(library(randomForest)))
        set.seed(123)
        mat <- as.matrix(df)
        storage.mode(mat) <- "double"
        y_mat <- t(mat)
        x_df <- data.frame(Order=as.numeric(order), Batch=factor(batch))
        x_mat <- model.matrix(~ Order + Batch - 1, data=x_df)
        corrected <- matrix(NA_real_, nrow=nrow(y_mat), ncol=ncol(y_mat))

        for (j in seq_len(ncol(y_mat))) {
            y_all <- y_mat[, j]
            y_qc <- y_all[qc_mask]
            valid_qc <- !is.na(y_qc) & y_qc > 0
            if (sum(valid_qc) < 3) {
                corrected[, j] <- y_all
                next
            }
            fit <- randomForest(
                x=x_mat[qc_mask, , drop=FALSE][valid_qc, , drop=FALSE],
                y=y_qc[valid_qc],
                ntree=as.integer(n_trees),
                mtry=ncol(x_mat),
                nodesize=1
            )
            pred <- predict(fit, x_mat)
            pred[pred <= 0] <- 1e-6
            qc_median <- median(y_qc[valid_qc], na.rm=TRUE)
            corrected[, j] <- (y_all / pred) * qc_median
        }

        res <- t(corrected)
        dimnames(res) <- dimnames(mat)
        return(as.data.frame(res))
    }
    """
    r_serrf_func = ro.r(r_script)
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df_result = r_serrf_func(
            df_input,
            ro.StrVector(list(batch)),
            ro.BoolVector([bool(x) for x in qc_mask]),
            ro.FloatVector([float(x) for x in order]),
            n_trees,
        )
    r_df_result.index = df_input.index
    r_df_result.columns = df_input.columns
    return r_df_result


def test_serrf_matches_r_randomforest_correction_contract() -> None:
    """Check that Python SERRF and R randomForest SERRF improve QC RSD."""
    df_input, batch, qc_mask, order = _make_serrf_matrix()
    raw_rsd = _median_qc_rsd(df_input, qc_mask)

    py_df = SERRFCorrector(
        n_estimators=120,
        cv_folds=5,
        n_corr_features=0,
        random_state=123,
        n_jobs=1,
    ).fit_transform(
        intensity_df=df_input,
        batch_array=batch,
        qc_mask=qc_mask,
        order_array=order,
        corr_mat=None,
    )["SERRF"][0]
    r_df = run_r_serrf(df_input, batch, qc_mask, order, n_trees=120)

    assert py_df.shape == df_input.shape
    assert r_df.shape == df_input.shape
    assert np.isfinite(py_df.to_numpy()).all()
    assert np.isfinite(r_df.to_numpy()).all()

    py_rsd = _median_qc_rsd(py_df, qc_mask)
    r_rsd = _median_qc_rsd(r_df, qc_mask)
    assert py_rsd < raw_rsd
    assert r_rsd < raw_rsd

    corr = spearmanr(py_df.to_numpy().ravel(), r_df.to_numpy().ravel()).correlation
    assert corr is not None and corr > 0.90
