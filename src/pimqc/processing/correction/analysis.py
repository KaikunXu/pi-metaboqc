"""Signal-correction orchestration, candidate evaluation, and selection.

MetaboIntCorrector prepares QC, batch, and injection-order inputs; applies a
configured correction method or evaluates AUTO candidates; and records metrics
for QC precision and sample-structure preservation. It writes corrected stage
matrices and audit artifacts while delegating numerical kernels and plotting.
"""

import warnings
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from ...config import resolve_stage_config
from ...constants import DEFAULT_RANDOM_SEED
from ...core import model
from ...runtime import log_execution_time
from ...runtime.progress import MAX_WORKERS
from ...statistics import metrics as su
from ...statistics import sample_structure as structure_stats
from ...statistics import selection as selection_utils
from ..stage import StageResult
from .algorithms import (
    _format_correction_method_label,
    _normalize_correction_method,
    _parse_correction_candidate,
)
from .regression import RegressionCorrector
from .runner import CorrectionStageRunner
from .ruv import RUVCorrector
from .serrf import SERRFCorrector
from .waveica import WaveICA2Corrector

FitPredictCallable = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
CorrectionModel = (
    RandomForestRegressor
    | TransformedTargetRegressor
    | Pipeline
    | FitPredictCallable
)


# =============================================================================
# Signal-Correction Processor
# =============================================================================
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
    _RUNTIME_CONFIG_KEYS = frozenset(
        {
            "base_est",
            "loess_span",
            "loess_degree",
            "rlsc_span_selection",
            "rlsc_span_grid",
            "rlsc_min_qc",
            "rlsc_robust",
            "rlsc_robust_iterations",
            "rf_n_tree",
            "serrf_n_tree",
            "serrf_corr_features",
            "serrf_backend",
            "serrf_batch_size",
            "svr_kernel",
            "svr_c",
            "svr_gamma",
            "ruv_k",
            "waveica_components",
            "waveica_cutoff",
            "waveica_levels",
            "waveica_spline_knots",
            "waveica_max_iter",
            "regression_backend",
            "regression_batch_size",
            "cv_folds",
            "n_jobs",
            "global_seed",
        }
    )

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
                "n_jobs": MAX_WORKERS,
                "global_seed": DEFAULT_RANDOM_SEED,
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

        # Integrate unified properties into internal attributes dictionary
        self.attrs.update(sc_configs)

    @property
    def _constructor(self) -> type["MetaboIntCorrector"]:
        """Override constructor to return MetaboIntCorrector."""
        return MetaboIntCorrector

    # =========================================================================
    # Domain Preprocessing and Statistical Methods
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

            # Route to specific engine
            if method == "SERRF":
                corr_mat = self._prepare_serrf_correlation_matrix()
                engine = SERRFCorrector(
                    n_estimators=self.attrs.get("serrf_n_tree", 100),
                    cv_folds=self.attrs.get("cv_folds", 5),
                    n_corr_features=self.attrs.get("serrf_corr_features", 10),
                    random_state=self.attrs.get(
                        "global_seed", DEFAULT_RANDOM_SEED
                    ),
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
                    random_state=self.attrs.get(
                        "global_seed", DEFAULT_RANDOM_SEED
                    ),
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

            # Extract DataFrames and calculate RSD tracking
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

            # Algebraically Reverse-Engineer Smooth Fit Baseline Matrix
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

            # Extract final RSD to evaluate performance
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
            structure_metrics = (
                structure_stats.calc_sample_structure_preservation(
                    raw_obj=self,
                    transformed_obj=final_corrected_df,
                    max_features=5000,
                    seed=int(
                        self.attrs.get("global_seed", DEFAULT_RANDOM_SEED)
                    ),
                )
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
    def transform_correction(self) -> StageResult[dict[str, model.MetaboInt]]:
        """Evaluate correction candidates and return the selected stages."""
        # Cast the internal matrix to float for safe in-place regression
        # updates.
        self._update_inplace(self.astype(float))

        self.attrs["pipeline_stage"] = "Original"
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
        is_auto = req_method == "AUTO"
        requested_method = req_method

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
        # Computation & Evaluation Phase
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
        # Selection Phase
        # ---------------------------------------------------------------------
        selected_label = self._select_best_correction_method(results_store)
        selected_result = results_store[selected_label]
        selected_method = selected_result.get(
            "method",
            _normalize_correction_method(selected_label),
        )

        if is_auto:
            selected_rsd = self._get_correction_eval_rsd(
                method=selected_label, result=selected_result
            )
            selected_score = su.finite_or_nan(
                selected_result.get("auto_score")
            )

            logger.success(
                "Auto selection: "
                f"{_format_correction_method_label(selected_label)} is "
                f"optimal (score = {selected_score:.3f}, "
                f"Eval QC RSD = {selected_rsd * 100:.2f}%)."
            )
            # Update metric tracker to reflect dynamically chosen algorithm
            self.attrs["base_est"] = selected_method
            self.attrs["correction_method_label"] = selected_label
            selected_params = selected_result.get("candidate_params", {})
            if selected_method == "QC-RLSC":
                self.attrs["rlsc_robust"] = selected_params.get(
                    "robust", self.attrs.get("rlsc_robust", True)
                )
                self.attrs["rlsc_robust_iterations"] = selected_params.get(
                    "robust_iterations",
                    self.attrs.get("rlsc_robust_iterations", 3),
                )

            # Ensure the propagated DataFrames carry the resolved name
            for df in selected_result["stage_dfs"].values():
                df.attrs["base_est"] = selected_method
                df.attrs["correction_method_label"] = selected_label

        selection = {
            "requested_method": requested_method,
            "selected_method": selected_method,
            "selected_label": selected_label,
            "is_auto": is_auto,
        }
        self.attrs["selection"] = selection
        self.attrs["is_auto"] = is_auto

        # Keep candidate stage matrices local; only selected outputs cross the
        # StageResult boundary and no ad-hoc DataFrame attribute is created.
        stage_dfs = selected_result["stage_dfs"]
        for df in stage_dfs.values():
            df.attrs["selection"] = dict(selection)
            df.attrs["is_auto"] = is_auto
        selected_stages = {
            name: stage
            for name, stage in stage_dfs.items()
            if name != "Original"
        }
        final_stage = list(selected_stages.values())[-1]
        return StageResult(
            data=selected_stages,
            metrics=final_stage.correction_metrics,
            candidates=results_store,
            metadata={
                "requested_method": requested_method,
                "selected_method": selected_method,
                "selected_label": selected_label,
                "selected_pred_df": selected_result["pred_df"],
                "is_auto": is_auto,
                "sample_type_col": sample_type_col,
                "batch_col": batch_col,
                "inject_order_col": inject_order_col,
                "qc_label": qc_label,
                "actual_label": actual_label,
            },
        )

    @log_execution_time
    def run_signal_correction(
        self,
        output_dir: str | None = None,
        **runtime_overrides: object,
    ) -> StageResult[dict[str, model.MetaboInt]]:
        """Return the structured signal-correction stage result.

        Named keyword settings such as ``base_est``, ``loess_span``,
        ``rlsc_robust``, or ``serrf_n_tree`` take precedence over the pipeline
        configuration and module defaults for this processor instance.
        """
        return CorrectionStageRunner(
            self,
            output_dir,
            runtime_overrides=runtime_overrides,
            allowed_override_keys=self._RUNTIME_CONFIG_KEYS,
        ).run()

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
            "selection": self.attrs.get(
                "selection",
                {
                    "requested_method": method,
                    "selected_method": method,
                    "selected_label": self.attrs.get(
                        "correction_method_label", method
                    ),
                    "is_auto": self.attrs.get("is_auto", False),
                },
            ),
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
