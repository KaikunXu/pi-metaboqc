"""Module for missing value imputation in metabolomics data."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.special import expit

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # Required import
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.decomposition import PCA
from typing import Dict, Any


class MetaboIntImputer:
    """Missing value imputation engine for MetaboInt objects.
    
    Attributes:
        metabo_obj: The input MetaboInt dataset (pandas DataFrame subclass).
        params: Pipeline configuration parameters.
        idx_mar: Index of features classified as Missing At Random.
        idx_mnar: Index of features classified as Missing Not At Random.
    """

    def __init__(
        self, metabo_obj: pd.DataFrame, pipeline_params: Dict[str, Any]
    ) -> None:
        """Initialize the imputation engine."""
        self.metabo_obj = metabo_obj
        self.params = pipeline_params
        self.idx_mar = pd.Index([])
        self.idx_mnar = pd.Index([])
        self._classify_mechanisms()

    def _classify_mechanisms(self) -> None:
        """Classify missing values as MAR or MNAR based on QC stability."""
        qc_col = self.params.get("MetaboInt", {}).get("sample_type", "Type")
        qc_lbl = self.params.get("MetaboInt", {}).get("qc_label", "QC")
        
        qc_mask = self.metabo_obj.columns.get_level_values(qc_col) == qc_lbl
        if not qc_mask.any():
            self.idx_mnar = self.metabo_obj.index
            return

        df_qc = self.metabo_obj.loc[:, qc_mask]
        df_act = self.metabo_obj.loc[:, ~qc_mask]
        
        qc_na_rate = df_qc.isna().mean(axis=1)
        self.idx_mar = qc_na_rate[qc_na_rate <= 0.2].index
        self.idx_mnar = df_act.index.difference(self.idx_mar)

    def execute_imputation(self, method: str = "probabilistic") -> pd.DataFrame:
        """Execute the specified imputation method in log2 space.
        
        Args:
            method: 'probabilistic', 'knn', 'min', 'halfmin', 'mice', 
                    or 'missforest'.
            
        Returns:
            pd.DataFrame: Imputed dataset retaining original attributes.
        """
        df_raw = self.metabo_obj.copy()
        df_log = np.log2(df_raw + 1.0)
        
        if method == "probabilistic":
            df_imp_log = self._impute_probabilistic(df_log)
        elif method == "knn":
            df_imp_log = self._impute_knn(df_log)
        elif method == "min":
            df_imp_log = self._impute_constant(df_log, fraction=1.0)
        elif method == "halfmin":
            df_imp_log = self._impute_constant(df_log, fraction=0.5)
        elif method == "mice":
            df_imp_log = self._impute_iterative(df_log, estimator="bayesian")
        elif method == "missforest":
            df_imp_log = self._impute_iterative(df_log, estimator="rf")
        else:
            raise ValueError(f"Unknown imputation method: {method}")

        df_final = np.exp2(df_imp_log) - 1.0
        return df_final.__finalize__(self.metabo_obj)

    def _impute_knn(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform KNN imputation."""
        imputer = KNNImputer(n_neighbors=5, weights="distance")
        arr_imp = imputer.fit_transform(df.T).T
        return pd.DataFrame(arr_imp, index=df.index, columns=df.columns)

    def _impute_iterative(
        self, df: pd.DataFrame, estimator: str
    ) -> pd.DataFrame:
        """Perform MICE imputation using Bayesian Ridge or Random Forest."""
        if estimator == "rf":
            est_model = RandomForestRegressor(
                n_estimators=50, random_state=42, n_jobs=-1
            )
        else:
            est_model = BayesianRidge()

        imputer = IterativeImputer(
            estimator=est_model, max_iter=10, random_state=42
        )
        arr_imp = imputer.fit_transform(df.T).T
        return pd.DataFrame(arr_imp, index=df.index, columns=df.columns)

    def _impute_constant(
        self, df: pd.DataFrame, fraction: float
    ) -> pd.DataFrame:
        """Perform constant small value imputation."""
        min_val = df.min().min() * fraction
        return df.fillna(min_val)

    def _impute_probabilistic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform probabilistic weighted imputation blending MAR and MNAR."""
        df_mar = self._impute_knn(df)
        df_mnar = self._impute_constant(df, fraction=0.5)

        means = df.mean(axis=1)
        mean_std = (means - means.mean()) / means.std()
        p_mnar_series = expit(-mean_std)
        
        p_mnar_df = pd.DataFrame(
            {col: p_mnar_series for col in df.columns}, index=df.index
        )
        
        df_fused = (p_mnar_df * df_mnar) + ((1.0 - p_mnar_df) * df_mar)
        df_final = df.copy()
        mask_na = df.isna()
        df_final[mask_na] = df_fused[mask_na]
        return df_final

    @staticmethod
    def calculate_nrmse(df_true: pd.DataFrame, df_imp: pd.DataFrame) -> float:
        """Calculate Normalized Root Mean Square Error."""
        mse = ((df_true - df_imp) ** 2).mean().mean()
        var_true = df_true.values.var()
        return np.sqrt(mse / var_true)


class MetaboVisualizerImputer:
    """Visualization suite for imputation evaluation."""

    def __init__(self, df_raw: pd.DataFrame, df_imp: pd.DataFrame) -> None:
        """Initialize with raw and imputed dataframes."""
        self.df_raw = np.log2(df_raw + 1.0)
        self.df_imp = np.log2(df_imp + 1.0)

    def plot_kde_overlay(self) -> plt.Figure:
        """Plot Kernel Density Estimation overlay of distributions."""
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.kdeplot(self.df_raw.values.flatten(), label="Raw", ax=ax)
        sns.kdeplot(self.df_imp.values.flatten(), label="Imputed", ax=ax)
        ax.set_title("Distribution Preservation (Log Scale)")
        ax.legend()
        plt.tight_layout()
        plt.close(fig)
        return fig

    def _add_confidence_ellipse(
        self, x: np.ndarray, y: np.ndarray, ax: plt.Axes, 
        n_std: float = 2.0, **kwargs: Any
    ) -> None:
        """Add a covariance confidence ellipse to an axes."""
        if x.size == 0 or y.size == 0:
            return
        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse(
            (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
            facecolor="none", **kwargs
        )
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)

    def plot_pca_trajectory(self, qc_mask: np.ndarray) -> plt.Figure:
        """Plot PCA observing QC clustering before and after imputation."""
        pca_raw = PCA(n_components=2).fit_transform(self.df_raw.fillna(0).T)
        pca_imp = PCA(n_components=2).fit_transform(self.df_imp.T)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        ax1.scatter(
            pca_raw[~qc_mask, 0], pca_raw[~qc_mask, 1], 
            alpha=0.5, color="lightgray", label="Samples"
        )
        ax1.scatter(
            pca_raw[qc_mask, 0], pca_raw[qc_mask, 1], 
            alpha=0.8, color="tab:blue", label="QC"
        )
        self._add_confidence_ellipse(
            pca_raw[qc_mask, 0], pca_raw[qc_mask, 1], ax1, edgecolor="tab:blue"
        )
        
        ax2.scatter(
            pca_imp[~qc_mask, 0], pca_imp[~qc_mask, 1], 
            alpha=0.5, color="lightgray"
        )
        ax2.scatter(
            pca_imp[qc_mask, 0], pca_imp[qc_mask, 1], 
            alpha=0.8, color="tab:orange", label="QC Imputed"
        )
        self._add_confidence_ellipse(
            pca_imp[qc_mask, 0], pca_imp[qc_mask, 1], ax2, 
            edgecolor="tab:orange"
        )
        
        ax1.autoscale()
        ax2.autoscale()
        ax1.set_title("PCA Before Imputation (0-filled)")
        ax2.set_title("PCA After Imputation")
        ax1.legend()
        ax2.legend()
        plt.tight_layout()
        plt.close(fig)
        return fig

    def plot_variance_comparison(self) -> plt.Figure:
        """Plot feature-wise variance comparison scatter plot."""
        var_raw = self.df_raw.var(axis=1)
        var_imp = self.df_imp.var(axis=1)
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(var_raw, var_imp, alpha=0.3, s=10)
        
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
        ]
        ax.plot(lims, lims, "k--", alpha=0.75, zorder=0)
        ax.set_xlabel("Original Variance")
        ax.set_ylabel("Imputed Variance")
        ax.set_title("Feature-wise Variance Preservation")
        plt.tight_layout()
        plt.close(fig)
        return fig