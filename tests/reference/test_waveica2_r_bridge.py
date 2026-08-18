"""
Script purpose: Compare Python WaveICA2 correction with an R reference.

This bridge test evaluates the embedded upstream WaveICA 2.0 reference code,
then checks that the native Python WaveICA2Corrector remains close to that
implementation while reducing injection-order structure. Embedding the two
reference functions keeps the test self-contained without installing the
unreleased GitHub R package or managing separate source files.
"""

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from scipy.stats import spearmanr

from pimqc.constants import DEFAULT_RANDOM_SEED
from pimqc.processing.correction import WaveICA2Corrector

from .helpers import median_qc_rsd, require_r_package

# Adapted from the executable portions of:
# https://github.com/dengkuistat/WaveICA_2.0
#
# MIT License
# Copyright (c) 2021 Kui Deng
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# The upstream source attributes ``unbiased_stICA`` to Emilie Renard (2013),
# based in part on the spatiotemporal separation work of Theis et al. and the
# Jacobi joint-diagonalization method of Cardoso and Souloumiac. Those credits
# are retained here even though the non-executable historical notes were
# removed from the embedded test fixture.

WAVEICA2_REFERENCE_R = r"""
unbiased_stICA <- function(X, k=10, alpha) {
  library(JADE)
  library(corpcor)

  jadeCummulantMatrices <- function(X) {
    n <- nrow(X)
    t <- ncol(X)
    M <- array(0, c(n, n, n*(n+1)/2))
    scale <- matrix(1, n, 1)/t
    R <- cov(t(X))
    k <- 1
    for (p in 1:n) {
      C <- ((scale %*% (X[p,]*X[p,]))*X) %*% t(X)
      E <- matrix(0, n, n)
      E[p, p] <- 1
      M[, , k] <- C - R %*% E %*% R - sum(diag(E %*% R)) * R -
        R %*% t(E) %*% R
      k <- k + 1
      if (p > 1) {
        for (q in 1:(p-1)) {
          C <- ((scale %*% (X[p,]*X[q,]))*X) %*% t(X) * sqrt(2)
          E <- matrix(0, n, n)
          E[p, q] <- 1/sqrt(2)
          E[q, p] <- E[p, q]
          M[, , k] <- C - R %*% E %*% R - sum(diag(E %*% R)) * R -
            R %*% t(E) %*% R
          k <- k + 1
        }
      }
    }
    return(M)
  }

  p <- nrow(X)
  n <- ncol(X)
  dimmin <- min(n, p)
  if (dimmin < k) {
    k <- dimmin
  }
  if (alpha < 0 | alpha > 1) {
    stop("alpha not in [0 1]")
  }

  Xc <- X - matrix(rep(colMeans(X, dims=1), p), nrow=p, byrow=TRUE)
  Xc <- Xc - matrix(rep(rowMeans(Xc, dims=1), n), nrow=p)
  udv <- svd(Xc, k, k)
  D <- diag(udv$d[1:k])
  if (k == 1) {
    D <- udv$d[1]
  }
  U <- udv$u
  V <- udv$v

  nummat <- k*(k+1)/2
  M <- array(0, c(k, k, 2*nummat))
  Bt <- D^(1-alpha) %*% t(V)
  if (alpha == 1) {
    Bt <- t(V)
  }
  At <- D^alpha %*% t(U)
  if (alpha == 0) {
    At <- t(U)
  }
  M[, , 1:nummat] <- jadeCummulantMatrices(Bt)
  M[, , (nummat+1):(2*nummat)] <- jadeCummulantMatrices(At)
  M[, , 1:nummat] <- alpha*M[, , 1:nummat] /
    mean(sqrt(apply(M[, , 1:nummat]^2, 3, sum)))
  M[, , (nummat+1):(2*nummat)] <-
    (1-alpha)*M[, , (nummat+1):(2*nummat)] /
    mean(sqrt(apply(M[, , (nummat+1):(2*nummat)]^2, 3, sum)))

  Worth <- rjd(M, eps=1e-06, maxiter=1000)
  Wo <- t(Worth$V)
  A0 <- U %*% D^alpha %*% solve(Wo)
  B0 <- V %*% D^(1-alpha) %*% t(Wo)
  if (alpha == 1) {
    B0 <- V %*% t(Wo)
  }
  if (alpha == 0) {
    A0 <- U %*% solve(Wo)
  }

  meanCol <- matrix(colMeans(X, dims=1), ncol=1)
  meanRows <- matrix(rowMeans(X, dims=1), ncol=1)
  meanB <- pseudoinverse(A0) %*% meanRows
  meanA <- pseudoinverse(B0) %*% meanCol
  Bfin <- B0 + matrix(rep(meanB, n), nrow=n, byrow=TRUE)
  Afin <- A0 + matrix(rep(meanA, p), nrow=p, byrow=TRUE)
  return(list(A=Afin, B=Bfin, W=Wo))
}

WaveICA_2.0 <- function(data, wf="haar", Injection_Order, alpha, Cutoff, K) {
  library(waveslim)
  library(parallel)
  library(ica)
  library(mgcv)

  level <- floor(log(nrow(data), 2))
  if (is.null(colnames(data))) {
    stop("data must have colnames")
  }
  coef <- list()
  for (k in 1:(level+1)) {
    coef[[k]] <- matrix(NA, nrow(data), ncol(data))
  }
  for (j in 1:ncol(data)) {
    data_temp <- data[, j]
    x_modwt <- modwt(data_temp, wf=wf, n.levels=level)
    for (k in 1:(level+1)) {
      coef[[k]][, j] <- x_modwt[[k]]
    }
  }

  index <- level + 1
  data_wave_ICA <- list()
  for (i in 1:index) {
    data_coef <- coef[[i]]
    data_coef_ICA <- unbiased_stICA(X=t(data_coef), k=K, alpha=alpha)
    B <- as.data.frame(data_coef_ICA$B)
    A <- data_coef_ICA$A
    corr <- mclapply(B, function(x) {
      corr_summary <- summary(gam(x ~ s(Injection_Order)))
      return(corr_summary$r.sq)
    })
    label <- which(unlist(corr) >= Cutoff)
    B_new <- B[, label, drop=FALSE]
    A_new <- A[, label, drop=FALSE]
    Xn <- data_coef - t(A_new %*% t(B_new))
    data_wave_ICA[[i]] <- Xn
  }

  index <- ncol(data)
  index1 <- length(data_wave_ICA)
  data_coef <- matrix(NA, nrow(data_wave_ICA[[1]]), index1)
  data_wave <- matrix(
    NA,
    nrow(data_wave_ICA[[1]]),
    ncol(data_wave_ICA[[1]])
  )
  for (i in 1:index) {
    for (j in 1:index1) {
      data_coef[, j] <- data_wave_ICA[[j]][, i]
    }
    data_temp <- data[, i]
    data_coef <- as.data.frame(data_coef)
    colnames(data_coef) <- c(
      paste("d", 1:(index1-1), sep=""),
      paste("s", index1-1, sep="")
    )
    y <- as.list(data_coef)
    attributes(y)$class <- "modwt"
    attributes(y)$wavelet <- wf
    attributes(y)$boundary <- "periodic"
    data_wave[, i] <- imodwt(y) + mean(data_temp)
  }
  rownames(data_wave) <- rownames(data)
  colnames(data_wave) <- colnames(data)
  return(list(data_wave=data_wave))
}
"""


def _make_waveica_matrix() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Create a matrix with QC samples and a smooth injection-order artifact."""
    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
    n_features, n_samples = 14, 32
    order = np.arange(1, n_samples + 1, dtype=float)
    qc_mask = np.array([(idx % 4) == 0 for idx in range(n_samples)])
    order_scaled = (order - order.mean()) / np.ptp(order)
    artifact = 0.32 * order_scaled + 0.10 * np.sin(order / 4.0)
    artifact += np.where(qc_mask, 0.07 * np.cos(order / 5.0), 0.0)

    base = rng.lognormal(mean=8.0, sigma=0.25, size=n_features)
    weights = rng.uniform(0.7, 1.3, size=n_features)
    arr = base[:, None] * (1.0 + weights[:, None] * artifact[None, :])
    arr *= rng.normal(1.0, 0.01, size=arr.shape)
    arr = np.clip(arr, a_min=1e-6, a_max=None)

    shuffle_idx = rng.permutation(n_samples)
    arr = arr[:, shuffle_idx]
    order = order[shuffle_idx]
    qc_mask = qc_mask[shuffle_idx]

    index = [f"Met_{idx:02d}" for idx in range(n_features)]
    columns = [f"S{idx:02d}" for idx in range(n_samples)]
    return pd.DataFrame(arr, index=index, columns=columns), order, qc_mask


def _median_abs_order_corr(df: pd.DataFrame, order: np.ndarray) -> float:
    """Summarize absolute feature-wise Spearman association with order."""
    vals = []
    for _, row in df.iterrows():
        corr = spearmanr(row.to_numpy(dtype=float), order).correlation
        if corr is not None and np.isfinite(corr):
            vals.append(abs(float(corr)))
    return float(np.nanmedian(vals))


def run_r_original_waveica2(
    df_input: pd.DataFrame,
    order: np.ndarray,
    n_components: int = 4,
    cutoff: float = 0.10,
) -> pd.DataFrame:
    """Execute the official WaveICA 2.0 R implementation through rpy2."""
    for package in ("waveslim", "ica", "JADE", "corpcor", "mgcv"):
        require_r_package(package)

    ro.r("options(mc.cores=1)")
    ro.r(WAVEICA2_REFERENCE_R)
    r_waveica_func = ro.globalenv["WaveICA_2.0"]

    sort_idx = np.argsort(order, kind="mergesort")
    inverse_idx = np.argsort(sort_idx)
    sorted_order = order[sort_idx]
    sorted_sample_by_peak = df_input.iloc[:, sort_idx].T

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_result = r_waveica_func(
            sorted_sample_by_peak,
            "haar",
            ro.FloatVector([float(x) for x in sorted_order]),
            0.0,
            float(cutoff),
            int(n_components),
        )

    result_items = dict(zip(list(r_result.names()), list(r_result)))
    r_arr_sorted = np.asarray(result_items["data_wave"], dtype=float)
    r_arr = r_arr_sorted[inverse_idx, :]
    return pd.DataFrame(r_arr.T, index=df_input.index, columns=df_input.columns)


def test_waveica2_matches_original_r_reference() -> None:
    """Check that Python WaveICA2 is close to the original R algorithm."""
    df_input, order, qc_mask = _make_waveica_matrix()
    raw_corr = _median_abs_order_corr(df_input, order)
    raw_qc_rsd = median_qc_rsd(df_input, qc_mask)

    py_df = WaveICA2Corrector(
        n_components=4,
        cutoff=0.10,
        n_levels=None,
        spline_knots=5,
        max_iter=1000,
        random_state=DEFAULT_RANDOM_SEED,
    ).fit_transform(df_input, order_array=order)["WaveICA 2.0"][0]
    r_df = run_r_original_waveica2(
        df_input,
        order=order,
        n_components=4,
        cutoff=0.10,
    )

    assert py_df.shape == df_input.shape
    assert r_df.shape == df_input.shape
    assert np.isfinite(py_df.to_numpy()).all()
    assert np.isfinite(r_df.to_numpy()).all()
    assert float(py_df.min().min()) > 0.0
    assert float(r_df.min().min()) > 0.0

    py_corr = _median_abs_order_corr(py_df, order)
    py_qc_rsd = median_qc_rsd(py_df, qc_mask)
    assert py_corr < raw_corr
    assert py_qc_rsd < raw_qc_rsd

    py_flat = py_df.to_numpy(dtype=float).ravel()
    r_flat = r_df.to_numpy(dtype=float).ravel()
    matrix_corr = spearmanr(py_flat, r_flat).correlation
    relative_mae = float(
        np.mean(np.abs(py_flat - r_flat)) / np.mean(np.abs(r_flat))
    )

    assert matrix_corr is not None and matrix_corr > 0.995
    assert relative_mae < 0.03
