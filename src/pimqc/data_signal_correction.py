"""
Purpose of script: Execute Quality control-based signal correction.
"""

import os
import warnings
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d

from . import io_utils as iu
from . import plot_utils as pu
from . import core_classes
from . import visualizer_classes

class _LoessSmoother(BaseEstimator, RegressorMixin):
    """Loess corrector implementation following scikit-learn API."""

    def __init__(self, span: float = 0.3) -> None:
        """Initialize the LOESS smoother.

        Args:
            span: The fraction of data used when estimating each y-value.
                Must be between 0 and 1. Defaults to 0.3.
        """
        self.span = span
        self.interpolator_ = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_LoessSmoother":
        """Fit the loess algorithm to the training data.

        Args:
            x: Injection order of QC samples.
            y: Intensity of QC samples.

        Returns:
            The fitted estimator object.
        """
        x_arr = np.asarray(a=x)
        y_arr = np.asarray(a=y)
        try:
            y_fit = lowess(
                endog=y_arr, 
                exog=x_arr, 
                frac=self.span, 
                it=3, 
                missing="drop", 
                is_sorted=True, 
                return_sorted=False
            )
            self.interpolator_ = interp1d(
                x=x_arr, 
                y=y_fit, 
                bounds_error=False
            )
        except Exception as e:
            logger.error(f"Loess fitting failed: {e}")
            
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict intensity based on injection order.

        Args:
            x: Injection order of samples to be predicted.

        Returns:
            Predicted intensities. NaNs if the model was not fitted.
        """
        try:
            if self.interpolator_ is None:
                raise NotFittedError("Loess smoother is not fitted yet.")
            return self.interpolator_(x)
        except NotFittedError:
            return np.full(shape=x.shape, fill_value=np.nan)


class MetaboIntSC(core_classes.MetaboInt):
    """Quality control-based signal correction for metabolomics data.

    This class handles baseline estimation (LOESS, RF, or SVR),
    intra-batch drift correction, and inter-batch alignment.
    """

    _metadata: List[str] = ["attrs"]

    def __init__(
        self,
        *args: Any,
        pipeline_params: Optional[Dict[str, Any]] = None,
        mode: str = "POS",
        batch: str = "Batch",
        inject_order: str = "Inject Order",
        internal_standard: Optional[Union[List[str], str]] = None,
        base_est: str = "QC-RLSC",
        span: float = 0.3,
        n_tree: int = 500,
        random_state: int = 12345,
        svr_kernel: str = "rbf",
        svr_c: float = 1.0,
        svr_gamma: Union[str, float] = "scale",
        **kwargs: Any
    ) -> None:
        """Initialize the MetaboIntSC object.

        Args:
            *args: Arguments passed to pandas DataFrame.
            pipeline_params: Global configuration dictionary.
            mode: MS polarity mode ("POS" or "NEG").
            batch: Column name representing analytical batch.
            inject_order: Column name representing injection sequence.
            internal_standard: List of internal standards.
            base_est: Estimator type ("QC-RLSC", "QC-RFSC", or "QC-SVR").
            span: Smoothing parameter for LOESS.
            n_tree: Number of estimators for Random Forest.
            random_state: Random seed for model reproducibility.
            svr_kernel: Kernel type to be used in SVR algorithm.
            svr_c: Regularization parameter for SVR.
            svr_gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
            **kwargs: Extra arguments passed to pandas DataFrame.
        """
        super().__init__(*args, pipeline_params=pipeline_params, **kwargs)

        sc_configs: Dict[str, Any] = {
            "mode": mode,
            "batch": batch,
            "inject_order": inject_order,
            "base_est": base_est,
            "span": span,
            "n_tree": n_tree,
            "random_state": random_state,
            "svr_kernel": svr_kernel,
            "svr_c": svr_c,
            "svr_gamma": svr_gamma,
            "internal_standard": self._to_list(internal_standard)
        }

        if pipeline_params and "MetaboIntSC" in pipeline_params:
            sc_configs.update(pipeline_params["MetaboIntSC"])

        self.attrs.update(sc_configs)

    @property
    def _constructor(self) -> type:
        """Override constructor to return MetaboIntSC."""
        return MetaboIntSC

    def __finalize__(
        self, other: Any, method: Optional[str] = None, **kwargs: Any
    ) -> "MetaboIntSC":
        """Explicitly preserve custom attributes during pandas operations."""
        super().__finalize__(other, method=method, **kwargs)
        if hasattr(other, "attrs"):
            self.attrs = __import__("copy").deepcopy(other.attrs)
        return self

    @cached_property
    def _injection_dict(self) -> Dict[str, pd.Series]:
        """Injection order grouped by batch."""
        io_col = self.attrs["inject_order"]
        bt_col = self.attrs["batch"]
        io_series = self.columns.to_frame().loc[:, io_col].astype(dtype=int)
        return dict(list(io_series.groupby(by=bt_col)))

    @cached_property
    def _matrix_dict(self) -> Dict[str, pd.DataFrame]:
        """Intensity matrix transposed and grouped by batch."""
        return dict(list(self.transpose().groupby(by=self.attrs["batch"])))

    @cached_property
    def _int_base(self) -> pd.DataFrame:
        """Features' basic mean intensity of QC samples by batch.

        Returns:
            pd.DataFrame: Matrix with shape (features * batches).
        """
        bt_col = self.attrs["batch"]
        return self._qc.transpose().groupby(by=bt_col).mean().transpose()

    @cached_property
    def _int_base_bc(self) -> "MetaboIntSC":
        """Broadcast _int_base to all samples by batch.

        Returns:
            MetaboIntSC: Broadcasted baseline matrix (features * samples).
        """
        int_base = self._int_base
        bt_col = self.attrs["batch"]
        int_base_bc = pd.DataFrame([])
        
        batch_labels = self.columns.get_level_values(level=bt_col).unique()
        
        for batch in batch_labels:
            mask = self.columns.get_level_values(level=bt_col) == batch
            int_batch = self.loc[:, mask]
            n_batch = int_batch.shape[1]
            fac = pd.concat(
                objs=[int_base.loc[:, batch]] * n_batch, 
                axis=1
            )
            int_base_bc = pd.concat(objs=[int_base_bc, fac], axis=1)
            
        int_base_bc.columns = self.columns
        res_obj = MetaboIntSC(data=int_base_bc)
        res_obj.attrs.update(self.attrs)
        return res_obj

    @cached_property
    def _fit_data(self) -> Dict[str, List[Any]]:
        """Yield chunks of X and Y data for estimators based on batch."""
        def _merge_dicts(dicts: List[Dict], strict: bool = False) -> Dict:
            if not dicts:
                return {}
            elif not strict and len(dicts) == 1:
                return dicts[0]
            
            merged = {}
            for item in dicts:
                for k, v in item.items():
                    if k in merged:
                        merged[k].append(v)
                    else:
                        merged[k] = [v]
            return merged

        return _merge_dicts(dicts=[
            self._qc._injection_dict,
            self._qc._matrix_dict,
            self._injection_dict,
            self._matrix_dict
        ])

    def _loess_fit(
        self, x_train: pd.Series, y_train: pd.Series,
        x_process: pd.Series, span: float = 0.3
    ) -> np.ndarray:
        """Baseline prediction based on _LoessSmoother."""
        try:
            loess = _LoessSmoother(span=span)
            loess.fit(x=x_train, y=y_train)
            return loess.predict(x=x_process)
        except Exception as e:
            logger.debug(f"LOESS fit failed for a feature: {e}")
            return np.full(shape=len(x_process), fill_value=np.nan)

    def _rf_fit(
        self, x_train: pd.Series, y_train: pd.Series,
        x_process: pd.Series, n_tree: int = 500, random_state: int = 12345
    ) -> np.ndarray:
        """Baseline prediction based on Random Forest regression."""
        try:
            valid = ~x_train.isna() & ~y_train.isna()
            
            if valid.sum() > 0:
                x_val = x_train[valid].values.reshape(-1, 1)
                y_val = y_train[valid].values
                
                rf_reg = RandomForestRegressor(
                    n_estimators=n_tree, 
                    random_state=random_state
                )
                rf_reg.fit(X=x_val, y=y_val)
                
                x_proc_val = x_process.values.reshape(-1, 1)
                return rf_reg.predict(X=x_proc_val)
                
            return np.full(shape=len(x_process), fill_value=np.nan)
        except Exception as e:
            logger.debug(f"RF fit failed for a feature: {e}")
            return np.full(shape=len(x_process), fill_value=np.nan)

    def _svr_fit(
        self, x_train: pd.Series, y_train: pd.Series,
        x_process: pd.Series, kernel: str = "rbf",
        c_val: float = 1.0, gamma_val: Union[str, float] = "scale"
    ) -> np.ndarray:
        """Baseline prediction based on Support Vector Regression."""
        try:
            valid = ~x_train.isna() & ~y_train.isna()
            
            if valid.sum() > 0:
                x_val = x_train[valid].values.reshape(-1, 1)
                y_val = y_train[valid].values
                
                # Standardize data to ensure SVR convergence efficiency
                svr_pipe = make_pipeline(
                    StandardScaler(),
                    SVR(kernel=kernel, C=c_val, gamma=gamma_val)
                )
                svr_pipe.fit(X=x_val, y=y_val)
                
                x_proc_val = x_process.values.reshape(-1, 1)
                return svr_pipe.predict(X=x_proc_val)
                
            return np.full(shape=len(x_process), fill_value=np.nan)
        except Exception as e:
            logger.debug(f"SVR fit failed for a feature: {e}")
            return np.full(shape=len(x_process), fill_value=np.nan)

    def _process_matrix_fit_sb(
        self, batch_data: List[Any], batch: str,
        base_est: str = "QC-RLSC", span: float = 0.3,
        n_tree: int = 500, random_state: int = 12345,
        svr_kernel: str = "rbf", svr_c: float = 1.0, 
        svr_gamma: Union[str, float] = "scale"
    ) -> pd.DataFrame:
        """Estimate the intensity baseline for a single batch."""
        x_tr, y_tr, x_pr, y_pr = batch_data
        
        try:
            logger.info(f"Analyzing {batch} using {base_est}...")
            
            if base_est in ("LOESS", "LOWESS", "QC-RLSC"):
                y_fit = y_tr.apply(
                    func=lambda x: self._loess_fit(
                        x_train=x_tr, 
                        y_train=x, 
                        x_process=x_pr, 
                        span=span
                    ), 
                    axis=0
                )
            elif base_est in ("RF", "Random forest", "QC-RFSC"):
                y_fit = y_tr.apply(
                    func=lambda x: self._rf_fit(
                        x_train=x_tr, 
                        y_train=x, 
                        x_process=x_pr, 
                        n_tree=n_tree, 
                        random_state=random_state
                    ), 
                    axis=0
                )
            elif base_est in ("SVR", "QC-SVR"):
                y_fit = y_tr.apply(
                    func=lambda x: self._svr_fit(
                        x_train=x_tr, 
                        y_train=x, 
                        x_process=x_pr, 
                        kernel=svr_kernel, 
                        c_val=svr_c, 
                        gamma_val=svr_gamma
                    ), 
                    axis=0
                )
            else:
                raise ValueError(f"Unknown estimator: {base_est}")
                
            y_fit.index = y_pr.index
            y_fit.columns = y_pr.columns
            return y_fit.transpose()
            
        except Exception as e:
            logger.error(f"Critical failure in batch '{batch}': {e}.")
            empty_df = pd.DataFrame(
                data=np.nan, 
                index=y_pr.columns, 
                columns=y_pr.index
            )
            return empty_df

    @cached_property
    def process_matrix_fit(self) -> pd.DataFrame:
        """Parallel calculation of predicted baseline for each batch."""
        n_jobs = min(iu.__max_threading__, len(self._fit_data))
        logger.info(f"Using {n_jobs} threads for parallel SC analysis.")
        
        # Dispatch only relevant slices to reduce IPC overhead
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(function=self._process_matrix_fit_sb)(
                batch_data=self._fit_data[batch_id],
                batch=batch_id,
                base_est=self.attrs["base_est"],
                span=self.attrs["span"],
                n_tree=self.attrs["n_tree"],
                random_state=self.attrs["random_state"],
                svr_kernel=self.attrs["svr_kernel"],
                svr_c=self.attrs["svr_c"],
                svr_gamma=self.attrs["svr_gamma"]
            ) for batch_id in self._fit_data.keys()
        )
        
        res_df = pd.concat(objs=results, axis=1)
        res_df[res_df <= 0] = np.nan
        return res_df

    @cached_property
    def intra_batch_corr(self) -> "MetaboIntSC":
        """Perform intra-batch correction via base / fit ratio."""
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=RuntimeWarning)
            corr_matrix = self._int_base_bc * (self / self.process_matrix_fit)
            
        res_obj = MetaboIntSC(data=corr_matrix)
        res_obj.attrs = self.attrs
        return res_obj

    @cached_property
    def inter_batch_corr(self) -> "MetaboIntSC":
        """Correct the mean of QCs in each batch to a global common mean."""
        bt_col = self.attrs["batch"]
        intra_df = self.intra_batch_corr
        
        bt_qc_mean = intra_df.transpose().groupby(by=bt_col).mean()
        global_qc_mean = intra_df.mean(axis=1)
        corr_factor = (global_qc_mean / bt_qc_mean).transpose()
        
        inter_df = intra_df.transpose().groupby(
            by=bt_col, group_keys=False
        ).apply(
            func=lambda x: x * corr_factor[x.name]
        ).transpose()
        
        res_obj = MetaboIntSC(data=inter_df)
        res_obj.attrs = self.attrs
        return res_obj

    @cached_property
    def _rsd(self) -> pd.Series:
        """Calculate Relative Standard Deviation (RSD)."""
        return (self.std(axis=1, ddof=1) / self.mean(axis=1)).rename("RSD")

    @iu._exe_time
    def execute_sc(self, output_dir: str) -> Tuple["MetaboIntSC", "MetaboIntSC"]:
        """Execute Signal Correction workflow and save outputs to disk.

        Args:
            output_dir: Target directory path for saving results.

        Returns:
            Tuple containing the intra-batch and inter-batch corrected 
            MetaboIntSC objects.
        """
        iu._check_dir_exists(dir_path=output_dir, handle="makedirs")

        est = self.attrs["base_est"]
        mode = self.attrs["mode"]

        self.process_matrix_fit.to_csv(
            path_or_buf=os.path.join(
                output_dir, f"QC_Fit_Baseline_{est}_{mode}.csv"
            ),
            na_rep="NA", 
            encoding="utf-8-sig"
        )
        self.intra_batch_corr.to_csv(
            path_or_buf=os.path.join(
                output_dir, f"Intra_Batch_Corrected_{est}_{mode}.csv"
            ),
            na_rep="NA", 
            encoding="utf-8-sig"
        )
        self.inter_batch_corr.to_csv(
            path_or_buf=os.path.join(
                output_dir, f"Inter_Batch_Corrected_{est}_{mode}.csv"
            ),
            na_rep="NA", 
            encoding="utf-8-sig"
        )

        vis = MetaboVisualizerSC(sc_obj=self)
        
        rsd_df, rsd_fig = vis.plot_corr_rsd()
        rsd_fig.savefig(
            fname=os.path.join(
                output_dir, f"QC_RSD_Boxplot_{est}_{mode}.pdf"
            ),
            bbox_inches="tight", 
            dpi=300
        )

        int_figs = vis.plot_is_int_order_scatter()
        for feat, fig in int_figs.items():
            fig.savefig(
                fname=os.path.join(
                    output_dir, f"Scatter_{feat}_{mode}.pdf"
                ),
                bbox_inches="tight", 
                dpi=300
            )

        if len(self.valid_is) != 0:
            pred_fig = vis.plot_pred_baseline_is()
            pred_fig.savefig(
                fname=os.path.join(
                    output_dir, f"Pred_Base_IS_{est}_{mode}.pdf"
                ),
                bbox_inches="tight", 
                dpi=300
            )

        return self.intra_batch_corr, self.inter_batch_corr


class MetaboVisualizerSC(visualizer_classes.BaseMetaboVisualizer):
    """Visualization suite for signal correction."""

    def __init__(self, sc_obj: "MetaboIntSC") -> None:
        super().__init__(metabo_obj=sc_obj)
        self.sc = sc_obj

    def plot_corr_rsd(self) -> Tuple[pd.DataFrame, Figure]:
        """Plot RSD boxplots across different correction stages."""
        rsd_df = pd.concat(
            objs=[
                self.sc._qc._rsd, self.sc.intra_batch_corr._qc._rsd,
                self.sc.inter_batch_corr._qc._rsd], 
            keys=["Original", "Intra-batch\ncorrected", "Inter-batch\ncorrected"], 
            names=["Stage"], axis=1
        ).unstack().rename("RSD").reset_index()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        cmap = pu.extract_linear_cmap(
            cmap=pu.custom_linear_cmap(
                color_list=["white", "tab:red"], n_colors=3),
            vmin=0.0, vmax=1.0)
        
        sns.boxplot(
            data=rsd_df, x="Stage", y="RSD", hue="Stage", width=0.6,
            showfliers=False, dodge=False, palette=cmap, ax=ax
        )
        
        self._apply_standard_format(
            ax=ax, ylabel="RSD of QC samples (%)", tick_fontsize=12)
        pu.change_axis_format(ax=ax, axis_format="percentage", axis="y")
        pu.change_axis_rotation(ax=ax, rotation=0, axis="x")
        plt.close(fig=fig)
        return rsd_df, fig

    def plot_pred_baseline_is(self) -> Optional[Figure]:
        """Plot scatter charts of IS and overlay predicted baseline lines."""
        if not self.sc.valid_is: return None

        plot_data = self.sc.int_order_info(feat_type="IS").reset_index()
        plot_data[self.st_col] = plot_data[self.st_col].astype(dtype="category")
        plot_data[self.bat_col] = plot_data[self.bat_col].astype(dtype="category")
        plot_data = plot_data.sort_values(by=self.st_col, ascending=False)
        
        pred_info = self.sc.process_matrix_fit.int_order_info(feat_type="IS")
        batch_set = pred_info.index.get_level_values(level=self.bat_col).unique()

        ncols = 2
        nrows = int(np.ceil(a=len(self.sc.valid_is) / ncols))
        fig = plt.figure(figsize=(7.5 * ncols, 3 * nrows), layout="constrained")
        
        for n, feat in enumerate(iterable=self.sc.valid_is):
            ax = plt.subplot(nrows, ncols, n + 1)
            
            sns.scatterplot(
                ax=ax, data=plot_data, x=self.io_col, y=feat, s=40, 
                edgecolor="k", linewidth=0.5, style=self.bat_col,
                palette=self.pal, hue=self.st_col, hue_order=[self.qc_lbl, self.act_lbl]
            )
            
            for batch in batch_set:
                sub_info = pred_info.loc[
                    pred_info.index.get_level_values(level=self.bat_col) == batch]
                sns.lineplot(
                    data=sub_info, x=self.io_col, y=feat, linewidth=1.5, 
                    linestyle="solid", color="k", label=None, ax=ax
                )

            self._apply_standard_format(
                ax=ax, xlabel=self.io_col, ylabel=feat, tick_fontsize=11,
                title_fontsize=13)
            pu.change_axis_format(ax=ax, axis_format="scientific notation", axis="y")

            if n == len(self.sc.valid_is) - 1:
                self._format_complex_legend(fig=fig, ax=ax)
            elif ax.get_legend():
                ax.legend().remove()
                            
        plt.suptitle(t="Predicted Baseline of IS", fontsize=14, weight="bold")
        plt.close(fig=fig)
        return fig

    def plot_is_int_order_scatter(self) -> Dict[str, Figure]:
        """Plot scatter charts comparing IS before and after correction."""
        data_dict = {
            "Raw Intensity": self.sc.int_order_info(feat_type="IS"),
            "After Intra-batch \nCorrected": self.sc.intra_batch_corr.int_order_info(feat_type="IS"),
            "After Inter-batch \nCorrected": self.sc.inter_batch_corr.int_order_info(feat_type="IS")
        }

        fig_dict = {}
        for feat in self.sc.valid_is:
            fig = plt.figure(figsize=(7.5, 9), layout="constrained")
            for n, data_type in enumerate(data_dict.keys()):
                plot_data = data_dict[data_type].reset_index().copy()
                plot_data[self.st_col] = plot_data[self.st_col].astype(dtype="category")
                plot_data[self.bat_col] = plot_data[self.bat_col].astype(dtype="category")
                plot_data = plot_data.sort_values(by=self.st_col, ascending=False)
                
                ax = plt.subplot(3, 1, n + 1)
                sns.scatterplot(
                    ax=ax, data=plot_data, x=self.io_col, y=feat, s=40, 
                    edgecolor="k", linewidth=0.5, style=self.bat_col,
                    palette=self.pal, hue=self.st_col, hue_order=[self.qc_lbl, self.act_lbl]
                )

                self._apply_standard_format(
                    ax=ax, xlabel=self.io_col, ylabel=data_type, tick_fontsize=11,
                    title_fontsize=13)
                pu.change_axis_format(ax=ax, axis_format="scientific notation", axis="y")

                if n == len(data_dict) - 1:
                    self._format_complex_legend(fig=fig, ax=ax)
                elif ax.get_legend():
                    ax.legend().remove()
                            
            plt.suptitle(t=f"Scatter of {feat}", fontsize=14, weight="bold")
            plt.close(fig=fig)
            fig_dict[feat] = fig
            
        return fig_dict