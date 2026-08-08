"""Bridge checks for pi-metaboqc QC-RLSC and Metanorm rLOESS.

Metanorm rLOESS is intentionally not treated as an exact reference for robust
QC-RLSC: it fits all sample types by default, chooses span by GCV, uses a
quadratic LOESS fit, and applies a per-batch correction. These tests first
check the shared fixed-span robust LOESS core against R, then compare the full
workflows on bundled project data. The all-feature comparison uses
``QConly=TRUE`` so both workflows learn their drift baselines from QCs.
"""

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from scipy.stats import spearmanr, wilcoxon

from pimqc.processing.correction import RegressionCorrector, _numba_loess_robust
from pimqc.dataset.builder import build_dataset


def _require_r_package(package: str) -> None:
    """Skip only when the optional system-R reference package is unavailable."""
    r_require = ro.r("function(pkg) requireNamespace(pkg, quietly=TRUE)")
    if not bool(r_require(package)[0]):
        pytest.skip(f"System R package '{package}' is not installed.")


def _r_fixed_span_robust_loess(
    x: np.ndarray,
    y: np.ndarray,
    x_pred: np.ndarray,
    span: float,
    iterations: int,
) -> np.ndarray:
    """Fit the R LOESS counterpart of the Numba fixed-span robust core."""
    fit_r_loess = ro.r(
        """
        function(x, y, x_pred, span, iterations) {
            fit <- stats::loess(
                y ~ x,
                span = span,
                degree = 1,
                family = "symmetric",
                control = stats::loess.control(iterations = iterations)
            )
            as.numeric(predict(fit, newdata = data.frame(x = x_pred)))
        }
        """
    )
    return np.asarray(
        fit_r_loess(
            ro.FloatVector(x.tolist()),
            ro.FloatVector(y.tolist()),
            ro.FloatVector(x_pred.tolist()),
            float(span),
            int(iterations),
        ),
        dtype=float,
    )


def _run_metanorm_rloess(
    intensity_df: pd.DataFrame,
    order: np.ndarray,
    batch: np.ndarray,
    sample_type: np.ndarray,
    *,
    qc_only: bool,
) -> pd.DataFrame:
    """Run Metanorm rLOESS and preserve per-feature fitting failures as NaN."""
    _require_r_package("metanorm")
    run_rloess = ro.r(
        """
        function(mat, order, batch, sample_type, qc_only) {
            suppressPackageStartupMessages(library(metanorm))
            batch <- as.factor(as.character(batch))
            result <- lapply(seq_len(nrow(mat)), function(i) {
                tryCatch(
                    metanorm::metanormWorker(
                        raw = unname(mat[i, ]),
                        order = as.numeric(order),
                        keepScale = TRUE,
                        QConly = qc_only,
                        QCcheck = FALSE,
                        QCcheckp = 0.1,
                        changepoints = FALSE,
                        type = as.character(sample_type),
                        batch = batch,
                        batchwise = TRUE,
                        weights = rep(1, ncol(mat)),
                        model = "rLOESS",
                        k = min(ncol(mat) * 0.9, 10),
                        cv = "GCV",
                        plotdir = NULL,
                        plottype = "pdf",
                        i = i
                    ),
                    error = function(e) rep(NA_real_, ncol(mat))
                )
            })
            do.call(rbind, result)
        }
        """
    )
    values = intensity_df.to_numpy(dtype=float, copy=True)
    r_matrix = ro.r["matrix"](
        ro.FloatVector(values.ravel(order="F")),
        nrow=values.shape[0],
        ncol=values.shape[1],
    )
    with localconverter(ro.default_converter + numpy2ri.converter):
        r_result = run_rloess(
            r_matrix,
            ro.FloatVector(order.tolist()),
            ro.StrVector(batch.tolist()),
            ro.StrVector(sample_type.tolist()),
            bool(qc_only),
        )
        result = np.asarray(r_result, dtype=float)
    return pd.DataFrame(result, index=intensity_df.index, columns=intensity_df.columns)


def _qc_rsd_series(
    intensity_df: pd.DataFrame, qc_mask: np.ndarray
) -> pd.Series:
    """Return per-feature QC RSD for features with at least two valid QCs."""
    qc_df = intensity_df.loc[:, qc_mask]
    qc_df = qc_df.replace([np.inf, -np.inf], np.nan)
    counts = qc_df.notna().sum(axis=1)
    rsd = qc_df.std(axis=1, ddof=1).div(qc_df.mean(axis=1).replace(0.0, np.nan))
    return rsd.where(counts >= 2)


def _median_qc_rsd(intensity_df: pd.DataFrame, qc_mask: np.ndarray) -> float:
    """Return the median of the valid per-feature QC RSD values."""
    return float(np.nanmedian(_qc_rsd_series(intensity_df, qc_mask)))


def _qc_rsd_comparison_table(
    raw: pd.DataFrame,
    corrected_workflows: dict[str, pd.DataFrame],
    qc_mask: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize paired feature-level QC-RSD outcomes for several workflows."""
    rsd = pd.DataFrame({"Raw": _qc_rsd_series(raw, qc_mask)})
    for label, corrected in corrected_workflows.items():
        rsd[label] = _qc_rsd_series(corrected, qc_mask)
    paired = rsd.dropna()
    if paired.empty:
        raise AssertionError("No features have finite QC-RSD for all workflows.")

    rows: list[dict[str, float | int | str]] = []
    for method in rsd.columns:
        values = paired[method] * 100.0
        improvement = (1.0 - paired[method].div(paired["Raw"])) * 100.0
        rows.append(
            {
                "Workflow": method,
                "Features evaluated": int(values.size),
                "Median QC-RSD (%)": float(values.median()),
                "IQR QC-RSD (%)": float(values.quantile(0.75) - values.quantile(0.25)),
                "Mean QC-RSD (%)": float(values.mean()),
                "P90 QC-RSD (%)": float(values.quantile(0.90)),
                "QC-RSD <15% (%)": float((values < 15.0).mean() * 100.0),
                "QC-RSD <30% (%)": float((values < 30.0).mean() * 100.0),
                "Median change from raw (%)": float(improvement.median()),
                "Features improved from raw (%)": float((improvement > 0.0).mean() * 100.0),
            }
        )

    pair_rows: list[dict[str, float | int | str]] = []
    method_names = list(corrected_workflows)
    for left, right in (
        (method_names[0], method_names[1]),
        (method_names[0], method_names[2]),
        (method_names[1], method_names[2]),
    ):
        left_values = paired[left].to_numpy(dtype=float)
        right_values = paired[right].to_numpy(dtype=float)
        signed_rank = wilcoxon(left_values, right_values, alternative="two-sided")
        difference = (left_values - right_values) * 100.0
        pair_rows.append(
            {
                "Left workflow": left,
                "Right workflow": right,
                "Features evaluated": int(len(paired)),
                "Left lower QC-RSD": int((left_values < right_values).sum()),
                "Right lower QC-RSD": int((right_values < left_values).sum()),
                "Ties": int(
                    np.isclose(left_values, right_values, rtol=1e-12, atol=1e-12).sum()
                ),
                "Median left-minus-right (pp)": float(np.median(difference)),
                "Mean left-minus-right (pp)": float(np.mean(difference)),
                "Wilcoxon paired p-value": float(signed_rank.pvalue),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def test_numba_fixed_span_robust_loess_matches_r_loess_core() -> None:
    """The Numba core should closely follow R LOESS under matching settings."""
    x = np.arange(1.0, 25.0)
    y = 500.0 + 11.0 * x + 45.0 * np.sin(x / 3.0)
    y[11] += 500.0
    span = 0.5

    py_pred = _numba_loess_robust(x, y, x, span, max_iter=3)
    r_pred = _r_fixed_span_robust_loess(x, y, x, span, iterations=3)
    valid = np.isfinite(py_pred) & np.isfinite(r_pred)

    assert valid.sum() >= 20
    agreement = spearmanr(py_pred[valid], r_pred[valid]).correlation
    assert agreement is not None and agreement > 0.99


def test_metanorm_rloess_bridge_on_bundled_project_data(
    real_project_data: tuple[pd.DataFrame, pd.DataFrame, dict[str, object]],
) -> None:
    """Compare workflow-level behavior on package data without claiming equality."""
    meta_df, intensity_df, pipeline_params = real_project_data
    metabo_obj = build_dataset(
        meta_info=meta_df,
        int_df=intensity_df,
        pipeline_params=pipeline_params,
    )
    finite_counts = metabo_obj.notna().sum(axis=1)
    subset = metabo_obj.loc[finite_counts.nlargest(8).index].astype(float)

    batch_name = metabo_obj.attrs["batch"]
    sample_type_name = metabo_obj.attrs["sample_type"]
    order_name = metabo_obj.attrs["inject_order"]
    qc_label = metabo_obj.attrs["sample_dict"]["QC sample"]
    batch = metabo_obj.columns.get_level_values(batch_name).to_numpy(dtype=str)
    sample_type = metabo_obj.columns.get_level_values(sample_type_name).to_numpy(
        dtype=str
    )
    order = metabo_obj.columns.get_level_values(order_name).to_numpy(dtype=float)
    qc_mask = sample_type == qc_label

    py_stages = RegressionCorrector(
        "QC-RLSC",
        loess_span=0.5,
        robust=True,
        robust_iterations=3,
        cv_folds=3,
    ).fit_transform(
        intensity_df=subset,
        batch_array=batch,
        qc_mask=qc_mask,
        order_array=order,
    )
    py_corrected = list(py_stages.values())[-1][0]
    # Metanorm's additive ``keepScale=TRUE`` workflow is intended for a
    # log-scale matrix. Transform its result back to raw intensity units for
    # QC-RSD and matrix-level comparison with pi-metaboqc.
    r_log_corrected = _run_metanorm_rloess(
        np.log2(subset), order, batch, sample_type, qc_only=False
    )
    with np.errstate(invalid="ignore"):
        r_corrected = np.exp2(r_log_corrected)

    assert py_corrected.shape == subset.shape == r_corrected.shape
    assert np.isfinite(py_corrected.to_numpy(dtype=float)).all()
    assert np.isfinite(r_corrected.to_numpy(dtype=float)).all()

    raw_rsd = _median_qc_rsd(subset, qc_mask)
    py_rsd = _median_qc_rsd(py_corrected, qc_mask)
    r_rsd = _median_qc_rsd(r_corrected, qc_mask)
    assert np.isfinite([raw_rsd, py_rsd, r_rsd]).all()

    # The workflows are deliberately not expected to be pointwise equivalent.
    agreement = spearmanr(
        py_corrected.to_numpy(dtype=float).ravel(),
        r_corrected.to_numpy(dtype=float).ravel(),
    ).correlation
    assert agreement is not None and np.isfinite(agreement)
    assert not np.allclose(
        py_corrected.to_numpy(dtype=float),
        r_corrected.to_numpy(dtype=float),
        rtol=1e-3,
        atol=1e-8,
    )


def test_metanorm_qc_only_rloess_all_feature_qc_rsd_comparison(
    real_project_data: tuple[pd.DataFrame, pd.DataFrame, dict[str, object]],
) -> None:
    """Report paired QC-RSD outcomes for all eligible bundled-data features.

    This is deliberately an evaluative bridge test rather than an equality
    test. Metanorm selects a span per feature and batch with quadratic robust
    LOESS, while pi-metaboqc uses fixed-span local-linear Tukey reweighting and
    then aligns QC means across batches.
    """
    meta_df, intensity_df, pipeline_params = real_project_data
    metabo_obj = build_dataset(
        meta_info=meta_df,
        int_df=intensity_df,
        pipeline_params=pipeline_params,
    )
    data = metabo_obj.astype(float)

    batch_name = metabo_obj.attrs["batch"]
    sample_type_name = metabo_obj.attrs["sample_type"]
    order_name = metabo_obj.attrs["inject_order"]
    qc_label = metabo_obj.attrs["sample_dict"]["QC sample"]
    batch = metabo_obj.columns.get_level_values(batch_name).to_numpy(dtype=str)
    sample_type = metabo_obj.columns.get_level_values(sample_type_name).to_numpy(
        dtype=str
    )
    order = metabo_obj.columns.get_level_values(order_name).to_numpy(dtype=float)
    qc_mask = sample_type == qc_label

    standard_stages = RegressionCorrector(
        "QC-RLSC",
        loess_span=0.5,
        robust=False,
        robust_iterations=3,
        cv_folds=3,
    ).fit_transform(
        intensity_df=data,
        batch_array=batch,
        qc_mask=qc_mask,
        order_array=order,
    )
    py_standard = list(standard_stages.values())[-1][0]

    robust_stages = RegressionCorrector(
        "QC-RLSC",
        loess_span=0.5,
        robust=True,
        robust_iterations=3,
        cv_folds=3,
    ).fit_transform(
        intensity_df=data,
        batch_array=batch,
        qc_mask=qc_mask,
        order_array=order,
    )
    py_robust = list(robust_stages.values())[-1][0]

    # rLOESS's additive keepScale mode is applied on log2 intensities. Convert
    # non-positive measurements to missing before transformation because R's
    # prediction path cannot accept the resulting negative infinities.
    r_log_input = np.log2(data.where(data > 0))
    r_log_corrected = _run_metanorm_rloess(
        r_log_input, order, batch, sample_type, qc_only=True
    )
    with np.errstate(invalid="ignore"):
        r_corrected = pd.DataFrame(
            np.exp2(r_log_corrected.to_numpy(dtype=float)),
            index=r_log_corrected.index,
            columns=r_log_corrected.columns,
        )

    summary, paired = _qc_rsd_comparison_table(
        data,
        {
            "Numba QC-RLSC (robust=False)": py_standard,
            "Numba robust QC-RLSC": py_robust,
            "Metanorm rLOESS (QConly=TRUE)": r_corrected,
        },
        qc_mask,
    )
    coverage = pd.DataFrame(
        {
            "Workflow": [
                "Raw",
                "Numba QC-RLSC (robust=False)",
                "Numba robust QC-RLSC",
                "Metanorm rLOESS (QConly=TRUE)",
            ],
            "Finite feature QC-RSD": [
                int(_qc_rsd_series(data, qc_mask).notna().sum()),
                int(_qc_rsd_series(py_standard, qc_mask).notna().sum()),
                int(_qc_rsd_series(py_robust, qc_mask).notna().sum()),
                int(_qc_rsd_series(r_corrected, qc_mask).notna().sum()),
            ],
        }
    )
    coverage["Coverage of bundled features (%)"] = (
        coverage["Finite feature QC-RSD"] / len(data) * 100.0
    )
    print(
        "\nAll-feature QC-RSD comparison "
        f"(bundled features={len(data)})"
    )
    print(coverage.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nPaired three-workflow comparisons")
    print(paired.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    assert len(summary) == 4
    assert len(paired) == 3
    assert int(paired["Features evaluated"].min()) >= 100
    assert np.isfinite(summary.select_dtypes(include=[np.number]).to_numpy()).all()
