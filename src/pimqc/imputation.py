# src/pimqc/imputation.py
"""
Purpose of script: Missing value imputation for MetaboInt.
"""

import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

from pca import pca
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, List, Optional, Tuple, Union
from loguru import logger

from . import core_classes
from . import io_utils as iu
from . import visualizer_classes
from . import plot_utils as pu


class MetaboIntImputer(core_classes.MetaboInt):
    """Missing value imputation engine for metabolomics data.

    Supports KNN, Random Forest (MissForest), Bayesian Ridge, 
    Half-minimum, Minimum, and Probabilistic approaches.
    """

    _metadata: List[str] = ["attrs"]

    def __init__(
        self,
        *args: Any,
        pipeline_params: Optional[Dict[str, Any]] = None,
        method: str = "probabilistic",
        knn_neighbors: int = 5,
        halfmin_fraction: float = 0.5,
        **kwargs: Any
    ) -> None:
        """Initialize MetaboIntImputer with pipeline parameter loading."""
        super().__init__(*args, pipeline_params=pipeline_params, **kwargs)

        imp_configs: Dict[str, Any] = {
            "method": method,
            "knn_neighbors": knn_neighbors,
            "halfmin_fraction": halfmin_fraction
        }

        if pipeline_params and "MetaboIntImputer" in pipeline_params:
            imp_configs.update(pipeline_params["MetaboIntImputer"])

        self.attrs.update(imp_configs)

    @property
    def _constructor(self) -> type:
        return MetaboIntImputer

    def __finalize__(
        self, other: Any, method: Optional[str] = None, **kwargs: Any
    ) -> "MetaboIntImputer":
        super().__finalize__(other, method=method, **kwargs)
        if hasattr(other, "attrs"):
            self.attrs = copy.deepcopy(other.attrs)
        return self

    # ====================================================================
    # Imputation Core Algorithms (Operating in Log2 Space)
    # ====================================================================

    def _impute_constant(
        self, df_log: pd.DataFrame, fraction: float = 1.0
    ) -> pd.DataFrame:
        """Impute missing values using a fraction of the minimum valid value."""
        global_min = df_log.min().min()
        fill_val = global_min * fraction
        return df_log.fillna(fill_val)

    def _impute_knn(
        self, df_log: pd.DataFrame, n_neighbors: int = 5
    ) -> pd.DataFrame:
        """Impute missing values using K-Nearest Neighbors."""
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
        arr_imp = imputer.fit_transform(df_log.T).T
        return pd.DataFrame(
            arr_imp, index=df_log.index, columns=df_log.columns
        )

    def _impute_iterative(
        self, df_log: pd.DataFrame, estimator: str = "bayesian"
    ) -> pd.DataFrame:
        """Impute missing values using iterative predictive models."""
        if estimator == "rf":
            est = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            est = None

        imputer = IterativeImputer(
            estimator=est, max_iter=10, random_state=42, 
            initial_strategy="mean"
        )
        arr_imp = imputer.fit_transform(df_log.T).T
        return pd.DataFrame(
            arr_imp, index=df_log.index, columns=df_log.columns
        )

    def _impute_probabilistic(self, df_log: pd.DataFrame) -> pd.DataFrame:
        """Impute using normal distribution bounded by LOD (Probabilistic)."""
        res_df = df_log.copy()
        for col in res_df.columns:
            s = res_df[col]
            n_missing = s.isna().sum()
            if n_missing == 0:
                continue
                
            valid_vals = s.dropna()
            mean_val = valid_vals.mean()
            std_val = valid_vals.std()
            lod = valid_vals.min()
            
            shift_mean = mean_val - 1.8 * std_val
            shift_std = max(0.3 * std_val, 0.01)
            
            drawn = np.random.normal(
                loc=shift_mean, scale=shift_std, size=n_missing
            )
            drawn = np.clip(drawn, a_min=None, a_max=lod)
            res_df.loc[s.isna(), col] = drawn
            
        return res_df

    # ====================================================================
    # Evaluation Metrics
    # ====================================================================

    def simulate_evaluation(
        self, method: str = "knn", mask_ratio: float = 0.05, 
        return_arrays: bool = True
    ) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
        """Calculate NRMSE by artificially masking valid target data."""
        df_log = np.log2(self.replace({0: np.nan}) + 1.0)
        
        blank_cols = self._blank.columns
        target_cols = self.columns.difference(blank_cols)
        df_target = df_log[target_cols]
        
        if df_target.empty:
            raise ValueError("No target samples available for evaluation.")
            
        valid_mask = ~df_target.isna()
        np.random.seed(42)
        mask_prob = np.random.rand(*df_target.shape)
        artificial_nas = valid_mask & (mask_prob < mask_ratio)
        df_masked = df_target.mask(artificial_nas)
        
        if method == "knn":
            k = self.attrs.get("knn_neighbors", 5)
            df_imp = self._impute_knn(df_masked, n_neighbors=k)
        elif method == "halfmin":
            frac = self.attrs.get("halfmin_fraction", 0.5)
            df_imp = self._impute_constant(df_masked, fraction=frac)
        elif method == "min":
            df_imp = self._impute_constant(df_masked, fraction=1.0)
        elif method == "probabilistic":
            df_imp = self._impute_probabilistic(df_masked)
        else:
            raise ValueError(f"Unknown imputation method: {method}")
            
        true_vals = df_target.values[artificial_nas]
        pred_vals = df_imp.values[artificial_nas]
        
        rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
        nrmse = rmse / (np.max(true_vals) - np.min(true_vals))
        
        if return_arrays:
            return float(nrmse), true_vals, pred_vals
        return float(nrmse)

    def _auto_select_method(self, mask_ratio: float = 0.05) -> str:
        """Automatically evaluate methods and select the one with lowest NRMSE."""
        # [RESTORED]: Skipping 'missforest', keeping 'min'
        candidates = ["probabilistic", "knn", "halfmin", "min"] 
        best_method = "probabilistic"
        best_nrmse = float("inf")
        
        logger.info("Running 'auto' mode: Benchmarking imputation algorithms...")
        
        for m in candidates:
            try:
                score = self.simulate_evaluation(
                    method=m, mask_ratio=mask_ratio, return_arrays=False
                )
                score_val = float(score)  # type: ignore
                logger.info(f"  - Algorithm [{m.ljust(13)}]: NRMSE = {score_val:.4f}")
                
                if score_val < best_nrmse:
                    best_nrmse = score_val
                    best_method = m
            except Exception as e:
                logger.warning(f"  - Algorithm [{m}] failed: {e}")
                
        logger.success(
            f"Auto-selection complete. Optimal algorithm: '{best_method}'"
        )
        return best_method

    # ====================================================================
    # Core Execution Logic
    # ====================================================================

    @iu._exe_time
    def execute_nrmse_simulation(
        self, method: Optional[str] = None, mask_ratio: float = 0.05, 
        output_dir: Optional[str] = None
    ) -> float:
        """Execute NRMSE simulation, log results, and optionally save plots."""
        act_method = method or self.attrs.get("method", "auto")
        
        if act_method.lower() == "auto":
            act_method = self._auto_select_method(mask_ratio=mask_ratio)
            
        logger.info(
            f"Executing NRMSE simulation for '{act_method}' "
            f"(Mask ratio: {mask_ratio})..."
        )
        
        sim_nrmse, t_vals, p_vals = self.simulate_evaluation(
            method=act_method, mask_ratio=mask_ratio, return_arrays=True
        )
        assert isinstance(sim_nrmse, float)
        
        if output_dir:
            iu._check_dir_exists(dir_path=output_dir, handle="makedirs")
            mode = self.attrs.get("mode", "POS")
            
            vis = MetaboVisualizerImp(raw_obj=self, imp_obj=self)
            fig_nrmse = vis.plot_simulated_nrmse_scatter(
                true_vals=t_vals, pred_vals=p_vals, nrmse=sim_nrmse
            )
            
            file_name = f"Imp_Sim_NRMSE_{act_method.title()}_{mode}.pdf"
            file_path = os.path.join(output_dir, file_name)
            
            fig_nrmse.savefig(file_path, bbox_inches="tight")
            logger.info(f"Simulation scatter plot saved to: {file_path}")
            
        return sim_nrmse

    @iu._exe_time
    def execute_imputation(
        self, method: Optional[str] = None, output_dir: Optional[str] = None
    ) -> "MetaboIntImputer":
        """Execute missing value imputation exclusively on target samples."""
        act_method = method or self.attrs.get("method", "auto")
        
        if act_method.lower() == "auto":
            act_method = self._auto_select_method()
            
        logger.info(f"Executing '{act_method}' missing value imputation...")
        
        df_raw = self.copy().replace({0: np.nan})
        df_log = np.log2(df_raw + 1.0)
        
        blank_cols = self._blank.columns
        target_cols = self.columns.difference(blank_cols)
        
        df_target = df_log[target_cols]
        df_blank = df_log[blank_cols]
        
        if not blank_cols.empty:
            logger.info(
                f"Bypassed {len(blank_cols)} Blank samples from imputation."
            )
        
        if act_method == "probabilistic":
            df_imp_target = self._impute_probabilistic(df_target)
        elif act_method == "knn":
            k = self.attrs.get("knn_neighbors", 5)
            df_imp_target = self._impute_knn(df_target, n_neighbors=k)
        elif act_method == "halfmin":
            df_imp_target = self._impute_constant(df_target, fraction=0.5)
        elif act_method == "min":
            df_imp_target = self._impute_constant(df_target, fraction=1.0)
        # [RESTORED]: Kept 'missforest' commented out
        # elif act_method in ("missforest", "rf"):
        #     df_imp_target = self._impute_iterative(df_target, estimator="rf")
        else:
            raise ValueError(f"Unknown imputation method: {act_method}")

        df_imp_log = pd.concat([df_imp_target, df_blank], axis=1)
        df_imp_log = df_imp_log[self.columns] 

        raw_final = np.exp2(df_imp_log) - 1.0
        imputed_df = self._constructor(raw_final).__finalize__(self)

        if output_dir:
            iu._check_dir_exists(dir_path=output_dir, handle="makedirs")
            mode = self.attrs.get("mode", "POS")
            
            file_name = f"Imputed_Data_{act_method.title()}_{mode}.csv"
            imputed_df.to_csv(
                os.path.join(output_dir, file_name), na_rep="NA", 
                encoding="utf-8-sig"
            )
            
            vis = MetaboVisualizerImp(raw_obj=self, imp_obj=imputed_df)
            
            fig_kde = vis.plot_observed_vs_imputed_kde()
            if fig_kde:
                fig_kde.savefig(
                    os.path.join(output_dir, f"Imp_Obs_vs_KDE_{mode}.pdf"),
                    bbox_inches="tight"
                )
            
            fig_pca = vis.plot_pre_post_pca()
            if fig_pca:
                fig_pca.savefig(
                    os.path.join(output_dir, f"Imp_Pre_Post_PCA_{mode}.pdf"),
                    bbox_inches="tight"
                )

        return imputed_df


class MetaboVisualizerImp(visualizer_classes.BaseMetaboVisualizer):
    """Visualization suite for missing value imputation evaluation."""

    # Unify legend visual parameters across all plotting methods in this class

    def __init__(
        self, raw_obj: "MetaboIntImputer", imp_obj: "MetaboIntImputer"
    ) -> None:
        """Initialize with both raw (with NaNs) and imputed datasets."""
        super().__init__(metabo_obj=imp_obj)
        self.raw_obj = raw_obj.replace({0: np.nan})
        self.imp_obj = imp_obj

    def _get_group_mapping(self) -> Dict[str, str]:
        """Map columns to their biological groups using base class properties."""
        mapping = {}
        for col in self.imp_obj.columns:
            if col in self.imp_obj._blank.columns:
                mapping[col] = "Blank"
            elif col in self.imp_obj._qc.columns:
                mapping[col] = "QC"
            elif hasattr(self.imp_obj, "_is") and \
                 col in getattr(self.imp_obj, "_is").columns:
                mapping[col] = "IS"
            else:
                mapping[col] = "Sample"
        return mapping

    def plot_simulated_nrmse_scatter(
        self, true_vals: np.ndarray, pred_vals: np.ndarray, nrmse: float
    ) -> plt.Figure:
        """Plot hexbin scatter of true vs imputed values from masking test."""
        fig, ax = plt.subplots(figsize=(6, 5))
        
        hb = ax.hexbin(
            x=true_vals, y=pred_vals, gridsize=40, 
            cmap=pu.custom_linear_cmap(
                color_list=["white", "tab:red"], n_colors=100),
            mincnt=1, bins="log"
        )
        
        lim_min = min(true_vals.min(), pred_vals.min())
        lim_max = max(true_vals.max(), pred_vals.max())
        ax.plot(
            [lim_min, lim_max], [lim_min, lim_max], 
            color="tab:gray", linestyle="--", linewidth=1.0, zorder=3
        )
        
        textstr = f"NRMSE = {nrmse:.4f}"
        props = dict(
            boxstyle="round,pad=0.4", facecolor="white", 
            edgecolor="black", alpha=0.9
        )
        ax.text(
            0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", bbox=props
        )
        
        self._apply_standard_format(
            ax=ax, title="Masked Simulation: True vs Imputed",
            xlabel="True Intensity (Log2)", ylabel="Imputed Intensity (Log2)"
        )
        
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Log10(Count)")
        
        plt.tight_layout()
        plt.close(fig)
        return fig

    def plot_observed_vs_imputed_kde(self) -> plt.Figure:
        """Plot grouped KDE overlay of observed vs. imputed values."""
        raw_log = np.log2(self.raw_obj + 1.0)
        imp_log = np.log2(self.imp_obj + 1.0)
        missing_mask = self.raw_obj.isna()
        
        grp_map = self._get_group_mapping()
        plot_data: List[pd.DataFrame] = []
        
        for col in self.raw_obj.columns:
            grp = grp_map[col]
            if grp == "Blank":
                continue
                
            obs_vals = raw_log.loc[~missing_mask[col], col].values
            imp_vals = imp_log.loc[missing_mask[col], col].values
            
            if len(obs_vals) > 0:
                df_obs = pd.DataFrame({"Log2_Intensity": obs_vals})
                df_obs["Type"] = "Observed"
                df_obs["Group"] = grp
                plot_data.append(df_obs)
                
            if len(imp_vals) > 0:
                df_imp = pd.DataFrame({"Log2_Intensity": imp_vals})
                df_imp["Type"] = "Imputed"
                df_imp["Group"] = grp
                plot_data.append(df_imp)
                
        if not plot_data:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No valid target data to plot.", ha="center")
            return fig
            
        df_plot = pd.concat(plot_data, ignore_index=True)
        unique_groups = df_plot["Group"].unique()
        
        fig, axes = plt.subplots(
            nrows=1, ncols=len(unique_groups), 
            figsize=(7 * len(unique_groups), 5), sharey=True
        )
        
        if len(unique_groups) == 1:
            axes = [axes]
            
        import matplotlib.patches as mpatches
        
        for ax, grp in zip(axes, unique_groups):
            grp_data = df_plot[df_plot["Group"] == grp]
            sns.kdeplot(
                data=grp_data, x="Log2_Intensity", hue="Type", 
                fill=True, ax=ax, common_norm=False,
                palette={"Observed": "tab:blue", "Imputed": "tab:red"},
                alpha=0.4
            )
            self._apply_standard_format(
                ax=ax, title=f"Density: {grp} Only",
                xlabel="Log2 Intensity", ylabel="Density"
            )
            
            # [FIXED]: Seaborn's kdeplot (fill=True) generates PolyCollections that 
            # ax.get_legend_handles_labels() cannot reliably catch. 
            # We explicitly reconstruct the legend patches for absolute stability.
            if ax.get_legend():
                ax.get_legend().remove()
                
            present_types = grp_data["Type"].unique()
            handles = []
            
            # Dynamically generate legend elements based on what is actually plotted
            if "Observed" in present_types:
                handles.append(mpatches.Patch(
                    color="tab:blue", alpha=0.4, label="Observed"
                ))
            if "Imputed" in present_types:
                handles.append(mpatches.Patch(
                    color="tab:red", alpha=0.4, label="Imputed"
                ))
                
            if handles:
                ax.legend(
                    handles=handles, title="Data Type", 
                    loc="upper right", **self.LEGEND_KWARGS
                )
            
        plt.tight_layout()
        plt.close(fig)
        return fig

    def plot_pre_post_pca(self) -> plt.Figure:
        """Plot global PCA structures before and after imputation."""
        st_col = self.attrs.get("sample_type", "Sample Type")
        bt_col = self.attrs.get("batch", "Batch")
        sample_dict = self.attrs.get("sample_dict", {})
        qc_lbl = sample_dict.get("QC sample", "QC sample")
        act_lbl = sample_dict.get("Actual sample", "Actual sample")

        valid_mask = self.imp_obj.columns.get_level_values(
            st_col
        ).isin([act_lbl, qc_lbl])
        
        raw_filt = self.raw_obj.loc[:, valid_mask].copy()
        imp_filt = self.imp_obj.loc[:, valid_mask].copy()

        def _get_pca_coords(df_in: pd.DataFrame) -> pd.DataFrame:
            df_trans = df_in.transpose()
            df_log = np.log10(df_trans.replace({0: np.nan}))
            df_imputed = df_log.fillna(df_log.min().min())
            
            scaler = StandardScaler()
            scaled_feat = scaler.fit_transform(df_imputed)
            
            import logging
            logging.getLogger('pca').setLevel(logging.CRITICAL)
            with iu.HiddenPrints():
                # Force verbose=0 to suppress internal print statements
                model = pca(n_components=2, verbose=0)
                res = model.fit_transform(scaled_feat)

            pca_scatter =  pd.DataFrame(
                res["PC"], columns=["PC1", "PC2"], index=df_in.index
            )
            pca_var = pd.Series(
                res["variance_ratio"], index=pca_scatter.columns
            ).rename("Variance")
            return pca_scatter,pca_var

        pca_raw, pca_var_raw = _get_pca_coords(raw_filt)
        pca_imp, pca_var_imp = _get_pca_coords(imp_filt)

        # pca_raw[st_col] = pca_raw[st_col].astype("category")
        # pca_raw[bt_col] = pca_raw[bt_col].astype("category")
        # pca_raw = pca_raw.sort_values(by=st_col, ascending=False)
        
        # pca_imp[st_col] = pca_imp[st_col].astype("category")
        # pca_imp[bt_col] = pca_imp[bt_col].astype("category")
        # pca_imp = pca_imp.sort_values(by=st_col, ascending=False)

        pal_dict = {qc_lbl: "tab:red", act_lbl: "tab:gray"}

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
        titles = ["Pre-Imputation PCA", "Post-Imputation PCA"]

        for ax, df, var, title in zip(
            axes, [pca_raw, pca_imp], [pca_var_raw, pca_var_imp], titles):
            sns.scatterplot(
                data=df, x="PC1", y="PC2", hue=st_col, style=bt_col,
                s=50, edgecolor="k", palette=pal_dict, linewidth=0.5,
                ax=ax, hue_order=[qc_lbl, act_lbl], 
                style_order=self.all_batches, markers=self.style_map,
            )

            for group in [qc_lbl, act_lbl]:
                sub_df = df[df[st_col] == group]
                if len(sub_df) > 2:
                    pu.confidence_ellipse(
                        x=sub_df["PC1"], y=sub_df["PC2"], ax=ax, n_std=3.0, 
                        alpha=0.1, facecolor=pal_dict[group], 
                        edgecolor=pal_dict[group]
                    )

            self._apply_standard_format(
                ax=ax, title=title, 
                xlabel=f'{"PC1"} ({100 * var.loc["PC1"]:.1f}%)', 
                ylabel=f'{"PC2"} ({100 * var.loc["PC2"]:.1f}%)'
            )
            
            if ax == axes[1]:
                self._format_complex_legend(fig=fig, ax=ax)
            elif ax.get_legend():
                ax.legend().remove()
            ax.autoscale()
        # plt.tight_layout()
        plt.close(fig)
        return fig