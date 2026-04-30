# src/pimqc/imputation.py
"""
Missing value imputation module with biological-aware evaluation.
"""

import os
import copy
import numpy as np
import pandas as pd
from functools import cached_property

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import KNNImputer
from loguru import logger

from . import io_utils as iu
from . import stat_utils as su
from . import plot_utils as pu
from . import core_classes
from . import visualizer_classes


class MetaboIntImputer(core_classes.MetaboInt):
    """Missing value imputation engine with hybrid stratified evaluation."""

    _metadata = ["attrs"]

    def __init__(
        self,
        *args,
        pipeline_params=None,
        method="probabilistic",
        knn_neighbors=5,
        **kwargs
    ):
        """Initialize MetaboIntImputer with parameters and metadata.

        Args:
            *args: Variable length arguments passed to pandas DataFrame.
            pipeline_params: Global configuration dictionary.
            method: Default imputation algorithm to be used.
            knn_neighbors: Number of neighbors for the KNN algorithm.
            **kwargs: Keyword arguments for the DataFrame constructor.
        """
        super().__init__(*args, pipeline_params=pipeline_params, **kwargs)

        configs = {
            "method": method,
            "knn_neighbors": knn_neighbors,
        }

        if pipeline_params and "MetaboIntImputer" in pipeline_params:
            configs.update(pipeline_params["MetaboIntImputer"])

        self.attrs.update(configs)

    @property
    def _constructor(self):
        """Return the class constructor for stable subclassing."""
        return MetaboIntImputer

    def __finalize__(self, other, method=None, **kwargs):
        """Ensure custom metadata (attrs) is preserved safely."""
        try:
            super().__finalize__(other, method=method, **kwargs)
        except ValueError:
            pass  # Bypass array comparison ValueError during concat
            
        if method == "concat" and hasattr(other, "objs"):
            for obj in other.objs:
                if hasattr(obj, "attrs") and obj.attrs:
                    self.attrs = copy.deepcopy(obj.attrs)
                    break
        elif hasattr(other, "attrs"):
            self.attrs = copy.deepcopy(other.attrs)
            
        return self


    # ====================================================================
    # Statistical Similarity Metrics
    # ====================================================================
    def calculate_imputation_similarity(self, imputed_obj):
        """Compute JSD metrics before and after imputation for QC/Samples."""
        
        log_before = su._extract_log2_target(self)
        log_after = su._extract_log2_target(imputed_obj)
        
        results = {"QC": {}, "Sample": {}}
        
        qc_cols = self._qc.columns
        sam_cols = self._actual_sample.columns
        
        for grp, cols in [("QC", qc_cols), ("Sample", sam_cols)]:
            if cols.empty: 
                continue
            d1 = log_before[cols].values.flatten()
            d2 = log_after[cols].values.flatten()
            
            results[grp]["Before vs Imputation"] = su.calc_jsd_similarity(d1, d2)
            
        return results

    # ====================================================================
    # Core Algorithms (Log2 Space)
    # ====================================================================
    @staticmethod
    def impute_by_constant(df_log, fraction = 1.0, imp_mode = "row"):
        """Imputes missing values using a constant LOD heuristic.

        Args:
            df: The dataset (typically log-transformed).
            fraction: The heuristic multiplier (e.g., 0.5 for half-minimum).
            imp_mode: "row" (feature-wise), "column" (sample-wise), or "global".
            is_log2: If True, executes fractional math in linear space safely.

        Returns:
            Dataframe with constant imputation applied.
        """
        if imp_mode in ("row","row-wise","row min"):
            raw_mins = df_log.min(axis=1)
        elif imp_mode in ("column","column-wise","column min"):
            raw_mins = df_log.min(axis=0)
        else: # elif imp_mode in ("global", "global min"):
            raw_mins = df_log.min().min()

        linear_mins = np.exp2(raw_mins) - 1.0
        target_mins = np.log2((linear_mins * fraction) + 1.0)

        # 3. Broadcast the computed minimums to fill NaNs
        if imp_mode in ("row", "row-wise", "row min"):
            return df_log.apply(lambda x: x.fillna(target_mins[x.name]), axis=1)
        else:
            return df_log.fillna(target_mins)

    @staticmethod
    def impute_by_knn(df_log, n_neighbors):
        """Impute missing values using K-Nearest Neighbors algorithm."""
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
        # Transpose since KNNImputer works on (samples x features)
        arr_imp = imputer.fit_transform(df_log.T).T
        return pd.DataFrame(
            arr_imp, index=df_log.index, columns=df_log.columns
        )

    @staticmethod
    def impute_by_prob(df_log, global_seed = 123):
        """Impute using a normal distribution to simulate values below LOD.
        
        This method adopts a left-shifted Gaussian distribution (Perseus style)
        without hard clipping, preserving the natural variance of the unobserved
        low-abundance tail.
        """
        rng = np.random.default_rng(global_seed)
        res_df = df_log.copy()
        for col in res_df.columns:
            s = res_df[col]
            if s.isna().sum() == 0:
                continue
                
            valid = s.dropna()
            m, sd = valid.mean(), valid.std()
            
            # Shift the distribution leftward to simulate the missing tail
            # standard parameters: shift by -1.8 std, width of 0.3 std
            shift_mean = m - 1.8 * sd
            shift_std = max(0.3 * sd, 0.01)
            
            # Draw random values simulating noise below the detection limit
            drawn = rng.normal(
                loc=shift_mean, scale=shift_std, size=s.isna().sum())
            
            # [CRITICAL FIX]: Prevent negative intensities in linear space
            # Log2 values must be >= 0 so that exp2(x) - 1.0 >= 0
            drawn = np.clip(drawn, a_min=0.0, a_max=None)
            res_df.loc[s.isna(), col] = drawn
            
        return res_df

    # ====================================================================
    # Evaluation Logic (Hybrid Masking & Stratified NRMSE)
    # ====================================================================

    # @staticmethod
    # def generate_abundance_mask(df_log, mask_ratio, noise_factor=1.0):
    #     """Generate an abundance-dependent mask for MNAR simulation."""
    #     np.random.seed(42)
    #     shape = df_log.shape
    #     valid_mask = ~df_log.isna()
    #     target_nas = int(valid_mask.values.sum() * mask_ratio)
    
    #     if target_nas == 0:
    #         return pd.DataFrame(
    #             False, index=df_log.index, columns=df_log.columns
    #         )
    
    #     feat_meds = df_log.median(axis=1).fillna(0)
    #     log_meds = np.log2(feat_meds + 1.0)
    #     max_v = log_meds.max() if log_meds.max() > 0 else 1.0
    #     rel_abd = log_meds / max_v
    
    #     # Base probability weight is inversely proportional to abundance
    #     weight_mat = np.tile((1.0 - rel_abd).values[:, None], (1, shape[1]))
    #     final_score = weight_mat + np.random.uniform(0, noise_factor, shape)
    #     final_score[~valid_mask.values] = -1.0
    
    #     cutoff = np.sort(final_score.flatten())[-target_nas]
    #     mask_arr = (final_score >= cutoff) & valid_mask.values
    
    #     return pd.DataFrame(
    #         mask_arr, index=df_log.index, columns=df_log.columns
    #     )

    @staticmethod
    def generate_gmm_noise_mask(
        df_log: pd.DataFrame, 
        mask_ratio: float, 
        noise_factor: float = 0.7,
        global_seed: int = 123
    ) -> pd.DataFrame:
        """Generate a MNAR mask using GMM probability scoring with added noise.
        
        This function uses GMM to determine the baseline probability of a 
        feature being instrument noise/low abundance. It then adds random noise 
        to this probability before ranking, simulating the probabilistic nature 
        of instrument detection limits (where some very low signals might be 
        detected, and some moderate signals might be missed).
        """
        from sklearn.mixture import GaussianMixture
        rng = np.random.default_rng(global_seed)
        
        shape = df_log.shape

        # Extract valid data (exclude pre-existing NaNs to avoid GMM errors)
        valid_mask = ~df_log.isna().values
        valid_data = df_log.values[valid_mask].reshape(-1, 1)
        
        target_nas = int(valid_mask.sum() * mask_ratio)
        
        if target_nas == 0:
            return pd.DataFrame(
                False, index=df_log.index, columns=df_log.columns)
            
        # Fit GMM to resolve the bimodal distribution (noise vs. true signal)
        gmm = GaussianMixture(n_components=2, random_state=global_seed)
        gmm.fit(valid_data)
        lower_cluster_idx = np.argmin(gmm.means_)
        
        # The base_prob values typically range from 0.0 to 1.0
        base_prob = gmm.predict_proba(valid_data)[:, lower_cluster_idx]
        
        # Introduce randomness: Generate uniform noise.
        # The generated noise strictly ranges from 0 to noise_factor
        noise = rng.uniform(0, noise_factor, size=base_prob.shape)
        
        # Calculate final score = baseline probability + uniform noise
        final_score = base_prob + noise
        
        # Truncate based on the final score to guarantee the exact missing ratio
        cutoff_score = np.sort(final_score)[-target_nas]
        
        mask_arr = np.zeros(shape, dtype=bool)
        mask_arr[valid_mask] = final_score >= cutoff_score
        
        return pd.DataFrame(
            mask_arr, index=df_log.index, columns=df_log.columns
        )

    @staticmethod
    def compute_stratified_nrmse(df_true, df_imp, mask_df, lod_q=0.25):
        """Calculate NRMSE stratified by low and high abundance regions."""
        feat_meds = df_true.median(axis=1).fillna(0)
        lod_val = feat_meds.quantile(lod_q)
        
        t_all = df_true.values[mask_df.values]
        p_all = df_imp.values[mask_df.values]
        med_all = np.tile(
            feat_meds.values[:, None], (1, df_true.shape[1])
        )[mask_df.values]
        
        low_m, hi_m = (med_all <= lod_val), (med_all > lod_val)
        
        def _get_nrmse(t, p):
            if len(t) < 2 or (np.max(t) - np.min(t)) < 1e-9:
                return np.nan
            rmse = np.sqrt(np.mean((t - p)**2))
            return float(rmse / (np.max(t) - np.min(t)))
            
        # 1. Compile the metrics dictionary
        metrics = {
            "NRMSE_Total": _get_nrmse(t_all, p_all),
            "NRMSE_Low": _get_nrmse(t_all[low_m], p_all[low_m]),
            "NRMSE_High": _get_nrmse(t_all[hi_m], p_all[hi_m]),
            "Count_Low": int(np.sum(low_m)),
            "Count_High": int(np.sum(hi_m)),
            "Threshold": float(lod_val)
        }
        
        # 2. Return exactly 3 objects to match the unpacking logic
        return metrics, t_all, p_all

    def run_benchmark_simulation(
        self,
        df_log: pd.DataFrame,
        idx_mar: pd.Index,
        target_cols: pd.Index,
        method: str,
        ratio: float = 0.05,
        global_seed: int = 123
    ) -> tuple:
        """Runs MNAR/MAR mask simulation strictly on MAR features."""
        # 1. Isolate the MAR subset for benchmarking
        mar_data = df_log.loc[idx_mar, target_cols].astype(float)
        
        # 2. Generate the boolean mask (True where values should be removed)
        mask = self.generate_gmm_noise_mask(
            mar_data, ratio, global_seed=global_seed)
        
        # 3. Apply the mask to create the simulated missing dataset
        masked_df = mar_data.copy()
        masked_df[mask] = np.nan
        
        # 4. Execute imputation on the masked subset
        if method in ("Probabilistic", "probabilistic", "Prob", "prob"):
            imp_res = self.impute_by_prob(masked_df, global_seed=global_seed)
        elif method in ("knn", "KNN"):
            k_val = self.attrs.get("knn_neighbors", 5)
            imp_res = self.impute_by_knn(masked_df, k_val)
        else:
            # First attempt row-wise median, fallback to 0.0 if row is entirely NaN
            imp_res = masked_df.apply(
                lambda x: x.fillna(x.median()), axis=1
            ).fillna(0.0)
            
        # 5. Compute fidelity strictly on the artificially masked locations
        eval_met, t_vals, p_vals = self.compute_stratified_nrmse(
            mar_data, imp_res, mask
        )
        return eval_met, t_vals, p_vals
    
    # ====================================================================
    # Execution & Auto-Selection
    # ====================================================================
    def select_best_algorithm(
        self,
        df_log: pd.DataFrame,
        idx_mar: pd.Index,
        target_cols: pd.Index,
        ratio: float = 0.05,
        global_seed: int =123
    ) -> tuple:
        """Autonomously selects the best algorithm using MAR-only subset."""
        candidates = ["knn", "probabilistic","median"]
        best_method = "knn"
        best_nrmse = float("inf")
        cache = {}

        for cand in candidates:
            logger.info(f'Simulating "{cand}" on MAR subset...')
            emet, tv, pv = self.run_benchmark_simulation(
                df_log, idx_mar, target_cols, cand, ratio, global_seed
            )
            cache[cand] = (emet, tv, pv)
            
            # Prioritize low-abundance preservation
            nrmse_low = emet.get("NRMSE_Low", float("inf")) 
            if nrmse_low < best_nrmse:
                best_nrmse = nrmse_low
                best_method = cand

        logger.info(f"Optimal MAR algorithm selected: {best_method}")
        return best_method, cache

    @iu._exe_time
    def execute_imputation(
        self,
        mar_method: str = None,
        mnar_method: str = None,
        mnar_fraction: float = None,
        knn_neighbors: int = None,
        sim_ratio: float = None,
        output_dir: str = None
    ) -> pd.DataFrame:
        """Executes hybrid imputation and exports complete visualizations.

        Args:
            mar_method: Strategy for MAR features ("auto", "knn", "prob").
            mnar_method: LOD calculation axis ("row-wise", "column-wise", "global").
            mnar_fraction: Multiplier for minimum value (default 0.5).
            knn_neighbors: Number of neighbors for KNN algorithm.
            sim_ratio: Masking ratio for autonomous benchmarking.
            output_dir: Path to export diagnostic results and plots.

        Returns:
            A new MetaboInt DataFrame with all missing values resolved.
        """
        # ====================================================================
        # 1. Parameter Extraction & Priority Fallback
        # ====================================================================
        _mnar = mnar_method or self.attrs.get("mnar_method", "row-wise")
        _frac = (
            mnar_fraction if mnar_fraction is not None 
            else self.attrs.get("mnar_fraction", 0.5))
        
        _mar = mar_method or self.attrs.get("mar_method", "auto")
        _knn_k = (
            knn_neighbors if knn_neighbors is not None 
            else self.attrs.get("knn_neighbors", 5))
        _ratio = (
            sim_ratio if sim_ratio is not None 
            else self.attrs.get("sim_mask_ratio", 0.05))

        # Extract the global random seed securely from the metadata passport
        _seed = self.attrs.get("global_seed", 123)

        target_cols = self.columns.difference(self._blank.columns)

        logger.info(
            f"Hybrid Imputation Engine Initialized. "
            f"MAR: {_mar} | MNAR: {_mnar} (LOD={_frac}x)"
            f" | KNN: {_knn_k} | Sim: {_ratio}"
        )

        # Cast to float and transform to log2 space safely
        df_log = np.log2(self.astype(float).replace({0: np.nan}) + 1.0)
        
        # ====================================================================
        # 2. ROUTE A: MNAR (Missing Not At Random) -> Localized LOD
        # ====================================================================
        idx_mnar = pd.Index(self.attrs.get("idx_mnar", [])).intersection(
            df_log.index)
        if len(idx_mnar) > 0:
            logger.info(f"Applying {_mnar}-wise LOD to {len(idx_mnar)} MNAR.")
            mnar_imp = MetaboIntImputer.impute_by_constant(
                df_log=df_log.loc[idx_mnar, target_cols],
                fraction=_frac,
                imp_mode=_mnar,
            )
            df_log.loc[idx_mnar, target_cols] = mnar_imp
        else:
            logger.info("MNAR index empty. Bypassing LOD imputation.")

        # ====================================================================
        # 3. ROUTE B: MAR (Missing At Random) -> ML Simulation & Impute
        # ====================================================================
        idx_mar = pd.Index(self.attrs.get("idx_mar", [])).intersection(
            df_log.index)
        cache, eval_met, t_vals, p_vals = {}, {}, [], []
        is_auto = (_mar in ("auto", "Auto", "Best", "best"))
        
        if len(idx_mar) > 0:
            if is_auto:
                # Pass seed to autonomous selector
                _mar, cache = self.select_best_algorithm(
                    df_log, idx_mar, target_cols, _ratio, _seed)
                eval_met, t_vals, p_vals = cache[_mar]
            else:
                # Pass seed to direct simulation
                eval_met, t_vals, p_vals = self.run_benchmark_simulation(
                    df_log, idx_mar, target_cols, _mar, _ratio, _seed)
                cache = {_mar: (eval_met, t_vals, p_vals)}
                
            logger.info(f'Executing "{_mar}" on {len(idx_mar)} MAR features.')
            mar_slice = df_log.loc[idx_mar, target_cols]
            
            if _mar in ("Probabilistic", "probabilistic", "Prob", "prob"):
                # Ensure final execution also uses the global seed
                mar_imp = MetaboIntImputer.impute_by_prob(
                    mar_slice, global_seed=_seed)
            elif _mar in ("knn", "KNN"):
                mar_imp = MetaboIntImputer.impute_by_knn(mar_slice, _knn_k)
            else:
                mar_imp = mar_slice.apply(
                    lambda x: x.fillna(x.median()), axis=1)
                
            df_log.loc[idx_mar, target_cols] = mar_imp

        # ====================================================================
        # 4. FINALIZATION: Matrix Reconstruction & Passport Update
        # ====================================================================
        final_log = pd.concat(
            [df_log[target_cols], df_log[self._blank.columns]], axis=1
        )[self.columns]
        
        # Convert back to linear space
        res_val = np.exp2(final_log) - 1.0
        imputed_obj = self._constructor(res_val).__finalize__(self)
        
        # Update the "Data Passport"
        imputed_obj.attrs["pipeline_stage"] = "Imputation"
        imputed_obj.attrs["imputation_status"] = "Completed"
        imputed_obj.attrs["selected_mar_method"] = _mar
        imputed_obj.attrs["mar_requested"] = mar_method or self.attrs.get(
            "mar_method", "auto"
        )
        imputed_obj.attrs["mnar_method"] = _mnar
        imputed_obj.attrs["mnar_fraction"] = _frac
        
        # Extract and store metrics for ALL evaluated candidates
        cand_mets = {}
        for m_name, (m_eval, _, _) in cache.items():
            cand_mets[m_name] = {
                "nrmse_low": m_eval.get("NRMSE_Low", float("nan")),
                "nrmse_high": m_eval.get("NRMSE_High", float("nan")),
                "nrmse_total": m_eval.get("NRMSE_Total", float("nan")),
            }
        imputed_obj.attrs["candidate_metrics"] = cand_mets

        # ====================================================================
        # 5. EXPORT & VISUALIZATIONS (Fully Restored)
        # ====================================================================
        sim_metrics = self.calculate_imputation_similarity(imputed_obj)
        imputed_obj.attrs["kde_similarity_metrics"] = sim_metrics

        if output_dir:
            iu._check_dir_exists(output_dir, handle="makedirs")
            imputed_obj.to_csv(
                os.path.join(output_dir, f"Imputed_Data_{_mar}.csv")
            )
            
            vis = MetaboVisualizerImputer(self, imputed_obj)
            
            vis.save_and_close_fig(
                vis.plot_imputed_kde_overlay(metrics=sim_metrics),
                os.path.join(output_dir, f"Impute_KDE")
            )
            
            # Ensure MAR metrics exist before plotting simulation charts
            if len(idx_mar) > 0 and cache:
                if is_auto:
                    vis.save_and_show_pw(
                        vis.plot_multi_nrmse_scatters(cache),
                        os.path.join(
                            output_dir, f"Imputer_Candidates"
                        )
                    )
                
                vis.save_and_show_pw(
                    vis.plot_imputation_summary_grid(
                        t_vals, p_vals, eval_met, _mar
                    ),
                    os.path.join(
                        output_dir, f"Impute_Summary_{_mar}"
                    )
                )

        return imputed_obj

    @cached_property
    def imputation_metrics(self):
        """Extracts key parameters and performance metrics from imputation.

        Returns:
            dict: A structured dictionary of imputation metadata.
        """
        # 1. Strategy extraction
        mar_req = self.attrs.get("mar_requested", "auto")
        mar_sel = self.attrs.get("selected_mar_method", "Unknown")
        mnar_meth = self.attrs.get("mnar_method", "row")
        mnar_frac = self.attrs.get("mnar_fraction", 0.5)

        # 2. Performance and status extraction
        status = self.attrs.get("imputation_status", "Pending")

        # Helper to safely round floats while bypassing NaNs
        def _safe_round(val):
            if pd.isna(val):
                return float("nan")
            return round(float(val), 4)

        # Process metrics for all evaluated candidates dynamically
        raw_mets = self.attrs.get("candidate_metrics", {})
        perf_dict = {}
        for cand, mets in raw_mets.items():
            perf_dict[cand] = {
                "nrmse_low": _safe_round(mets.get("nrmse_low")),
                "nrmse_high": _safe_round(mets.get("nrmse_high")),
                "nrmse_total": _safe_round(mets.get("nrmse_total")),
            }

        # 3. Feature count extraction
        raw_idx_mar = pd.Index(self.attrs.get("idx_mar", []))
        raw_idx_mnar = pd.Index(self.attrs.get("idx_mnar", []))
        
        idx_mar = raw_idx_mar.intersection(self.index)
        idx_mnar = raw_idx_mnar.intersection(self.index)

        # [REFACTORED]: Extract KDE JSD metrics from the data passport
        kde_jsd = self.attrs.get("kde_similarity_metrics", {})
        
        metrics = {
            "imputation_status": status,
            "strategies": {
                "mar_method_requested": mar_req,
                "mar_method_selected": mar_sel,
                "mnar_method": mnar_meth,
                "mnar_fraction": float(mnar_frac),
            },
            "performance": perf_dict,
            "feature_distribution": {
                # "mar":idx_mar.tolist(),
                "mar_count": len(idx_mar),
                # "mnar":idx_mnar.tolist(),
                "mnar_count": len(idx_mnar),
            },
            # [REFACTORED]: Added JSD metrics to the final output dictionary
            "distribution_similarity": kde_jsd,
        }
        
        return metrics

class MetaboVisualizerImputer(visualizer_classes.BaseMetaboVisualizer):
    """Visualization suite for evaluating imputation accuracy."""

    def __init__(self, raw_obj, imp_obj):
        """Initialize the visualizer and cast datasets to float safely."""
        super().__init__(metabo_obj=imp_obj)
        # Casting to float directly prevents NumPy log2 ufunc type errors
        self.raw_obj = raw_obj.astype(float).replace({0: np.nan})
        self.imp_obj = imp_obj.astype(float)

    def _prepare_kde_data(self):
        """Prepare long-form DataFrame for KDE plotting."""
        dfs = []
        raw_log = np.log2(self.raw_obj + 1.0)
        imp_log = np.log2(self.imp_obj + 1.0)
        
        qc_cols = self.imp_obj._qc.columns
        sample_cols = self.imp_obj._actual_sample.columns
        
        for grp, cols in [("QC", qc_cols), ("Sample", sample_cols)]:
            if cols.empty:
                continue
                
            obs = raw_log[cols].values.flatten()
            obs = obs[~np.isnan(obs)]
            if len(obs) > 0:
                dfs.append(pd.DataFrame({
                    "Log2_Intensity": obs, "Group": grp, "Type": "Observed"
                }))
                
            imp = imp_log[cols].values.flatten()
            imp = imp[~np.isnan(imp)]
            if len(imp) > 0:
                dfs.append(pd.DataFrame({
                    "Log2_Intensity": imp, "Group": grp, "Type": "Imputed"
                }))
                
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def plot_imputed_kde_overlay(self, metrics=None, ax_qc=None, ax_sample=None):
        """Plot KDE overlay, split into distinct QC and Sample subplots.
        
        Args:
            ax_qc: Optional matplotlib axis for the QC plot.
            ax_sample: Optional matplotlib axis for the Sample plot.
            
        Returns:
            fig if no axes are provided, else (ax_qc, ax_sample).
        """
        df_plot = self._prepare_kde_data()
        return_fig = False
        
        if df_plot.empty:
            if ax_qc is None or ax_sample is None:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "No valid data to plot.", ha="center")
                return fig
            return ax_qc, ax_sample

        # Create a 1x2 figure if no external axes are passed
        if ax_qc is None or ax_sample is None:
            fig, (ax_qc, ax_sample) = plt.subplots(1, 2, figsize=(10, 4))
            return_fig = True

        #  Plot the KDE Curves
        colors = {"Observed": "tab:gray", "Imputed": "tab:red"}
        for grp, ax in [("QC", ax_qc), ("Sample", ax_sample)]:
            subset_grp = df_plot[df_plot["Group"] == grp]
            
            if subset_grp.empty:
                ax.text(0.5, 0.5, f"No {grp} data available.", ha="center")
                self._apply_standard_format(
                    ax=ax, title=f"Density Overlay ({grp})"
                )
                continue
                
            for t in ["Observed", "Imputed"]:
                subset = subset_grp[subset_grp["Type"] == t]
                if not subset.empty:
                    sns.kdeplot(
                        data=subset, x="Log2_Intensity", 
                        color=colors.get(t, "black"), ax=ax, 
                        linewidth=2, alpha=0.8, label=t
                    )

        # Annotate with Metrics (JSD Only)
        for ax, name in [(ax_qc, "QC"), (ax_sample, "Sample")]:
            if metrics and name in metrics:
                # Construct text: Header + comparison
                lines = ["Jensen-Shannon Divergence"]
                
                for pair, vals in metrics[name].items():
                    jsd_val = vals.get("jsd", vals) if isinstance(
                        vals, dict) else vals
                    lines.append(f"{pair}: {jsd_val:.3f}")
                
                m_str = "\n".join(lines)
                
                # Render text box in the upper right corner
                ax.text(
                    0.96, 0.02, m_str, transform=ax.transAxes, 
                    fontsize=10, verticalalignment="bottom", 
                    horizontalalignment="right", clip_on=False,
                    bbox=dict(
                        boxstyle="round,pad=0.4", facecolor="white", 
                        edgecolor="none", alpha=0.6))
                
            self._apply_standard_format(
                ax=ax, title=f"Density Overlay ({name})",
                xlabel="Log2 Intensity", ylabel="Density"
            )
            
            # Place legend inside the best location and format it
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="best")
                self._format_single_legend(ax=ax, title="Data Type")
                
        if return_fig:
            plt.tight_layout()
            return fig
            
        return ax_qc, ax_sample

    def plot_nrmse_scatter(
        self, true_vals, pred_vals, metrics, method_name="", 
        axis_lims=None, ax=None
    ):
        """Plot hexbin scatter of true vs imputed values from mask test."""
        
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(5, 4))
        else:
            current_ax = ax
            fig = current_ax.figure
            
        # Handle shared limits and binning extents for grid consistency
        if axis_lims is not None:
            ax_min, ax_max = axis_lims
            extent = (ax_min, ax_max, ax_min, ax_max)
            lim_min, lim_max = ax_min, ax_max
        else:
            extent = None
            lim_min = min(true_vals.min(), pred_vals.min())
            lim_max = max(true_vals.max(), pred_vals.max())

        color_map = pu.custom_linear_cmap(
            color_list=["white", "tab:red"], n_colors=256, cmin=0.1, cmax=1.0)

        hb = current_ax.hexbin(
            x=true_vals, y=pred_vals, gridsize=40, extent=extent,
            cmap=color_map,
            mincnt=1, # bins="log"
        )
        
        current_ax.plot(
            [lim_min, lim_max], [lim_min, lim_max], 
            color="tab:gray", linestyle="--", linewidth=1.0, zorder=3
        )

        threshold = metrics.get("Threshold")
        if threshold is not None:
            current_ax.axvline(
                x=threshold, color="tab:gray", linestyle="--", linewidth=1.0,
                alpha=0.8)
            current_ax.axhline(
                y=threshold, color="tab:gray", linestyle="--", linewidth=1.0,
                alpha=0.8)

        # Format the stratified metrics dictionary into the text block
        annot_text = (
            f"NRMSE_Total: {metrics['NRMSE_Total']:.4f}\n"
            f"NRMSE_Low:   {metrics['NRMSE_Low']:.4f}\n"
            f"NRMSE_High:  {metrics['NRMSE_High']:.4f}"
        )

        current_ax.text(
            0.96, 0.02, annot_text, transform=current_ax.transAxes,
            fontsize=10, verticalalignment="bottom", 
            horizontalalignment="right",
            clip_on=False,
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor="white", 
                edgecolor="none", alpha=0.6))
        
        title_str = "Masked Simulation"
        if method_name:
            title_str += f" ({method_name.title()})"
            
        self._apply_standard_format(
            ax=current_ax, title=title_str,
            xlabel="True Intensity (Log2)", ylabel="Imputed Intensity (Log2)"
        )
        
        # Enforce exact axes clipping if global limits are provided
        if axis_lims is not None:
            current_ax.set_xlim(ax_min, ax_max)
            current_ax.set_ylim(ax_min, ax_max)
        
        cb = fig.colorbar(hb, ax=current_ax)
        cb.set_label("Log10(Count)")
        
        if ax is None:
            return fig
        return current_ax

    def plot_multi_nrmse_scatters(self, results_dict):
        """Plot a 2x2 grid of NRMSE scatter plots for all candidate methods."""
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping multi grid.")
            return None
            
        # Calculate global bounds across all candidate methods with 5% padding
        g_min, g_max = float("inf"), float("-inf")
        best_method = None
        best_nrmse = float("inf")
        
        for m, (met, t, p) in results_dict.items():
            g_min = min(g_min, t.min(), p.min())
            g_max = max(g_max, t.max(), p.max())
            # Identify the optimal method based on NRMSE_Low
            if met["NRMSE_Low"] < best_nrmse:
                best_nrmse = met["NRMSE_Low"]
                best_method = m
                
        margin = (g_max - g_min) * 0.05
        shared_lims = (g_min - margin, g_max + margin)

        pw.clear()
        bricks = []
        for m, (met, t, p) in results_dict.items():
            ax = pw.Brick(figsize=(4, 4), label=f"nrmse_{m}")
            # Add an asterisk to the title of the optimal method
            display_name = f"*{m}" if m == best_method else m
            
            self.plot_nrmse_scatter(
                t, p, met, method_name=display_name, axis_lims=shared_lims, ax=ax
            )
            bricks.append(ax)
            
        while len(bricks) % 3 != 0 or len(bricks) == 0:
            e = pw.Brick(figsize=(4, 4), label=f"empty_{len(bricks)}")
            e.axis("off")
            bricks.append(e)
            
        rows = []
        for i in range(0, len(bricks), 3):
            row = bricks[i] | bricks[i+1] | bricks[i+2]
            rows.append(row)
            
        final_grid = rows[0]
        for row in rows[1:]:
            final_grid = final_grid / row
            
        return final_grid


    def plot_imputation_summary_grid(self, t, p, met, method):
        """Combine NRMSE scatter and split KDE density subplots into a grid."""
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping summary grid.")
            return None
            
        pw.clear()
        
        ax1 = pw.Brick(figsize=(4, 4), label="NRMSE")
        self.plot_nrmse_scatter(t, p, met, method, ax=ax1)
        
        ax_qc = pw.Brick(figsize=(4, 4), label="KDE_QC")
        ax_sample = pw.Brick(figsize=(4, 4), label="KDE_Sample")
        
        # Populate the split KDE plots using the provided patchwork bricks
        sim_metrics = self.imp_obj.attrs.get("kde_similarity_metrics")
        self.plot_imputed_kde_overlay(
            ax_qc=ax_qc, ax_sample=ax_sample, metrics=sim_metrics)
        
        # Assemble 1x3 horizontal layout
        return ax1 | ax_qc | ax_sample