# src/pimqc/pca_utils.py
"""
Purpose of script: Core PCA engine and statistical outlier utility module.

Provides a robust, state-aware Principal Component Analysis (PCA) workflow
for metabolomics data. It handles transient data transformations, executes
multivariate dimensionality reduction, and computes exact statistical
boundaries (Hotelling's T2 and SPE/DModX) to flag distinct outlier classes.
Additionally, it supplies multivariate diagnostic metrics (dispersion,
silhouette, centrality shift) for downstream quality assessment.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import f, chi2
from . import stat_utils as su

class PCAEngine:
    """Core engine for PCA-based metabolomics data analysis."""

    def __init__(
        self, n_components=2, alpha=0.05, od_method="box", global_seed=123
    ):
        """
        Initialize the PCA computational engine.

        Args:
            n_components: Number of principal components to extract.
            alpha: Statistical significance level for outlier detection.
            od_method: Approximation method for OD limit ('box' or 'jm').
            global_seed: Random seed for reproducibility.
        """
        self.n_components = n_components
        self.alpha = alpha
        self.od_method = od_method
        self.global_seed = global_seed

    @staticmethod
    def extract_features(
        metabo_obj, sample_type, sample_name, actual_label, qc_label,
        scaling_method="None"
    ):
        """Prepare feature matrix and labels from a MetaboInt object.
        
        Applies JIT feature scaling and transient log transformations based
        on the current sample subset to prevent data leakage.
        """
        
        # 1. Isolate target samples (Features as rows, Samples as columns)
        sample_types = metabo_obj.columns.get_level_values(sample_type)
        valid_sample_mask = sample_types.isin([actual_label, qc_label])
        subset_df = metabo_obj.loc[:, valid_sample_mask].astype(float)
        
        # ====================================================================
        # State-Aware Transient Log Transformation
        # ====================================================================
        is_logged = metabo_obj.attrs.get("is_logged", False)
        norm_method = str(metabo_obj.attrs.get("norm_method", "None")).upper()
        
        # Apply transient log2 for variance stabilization if not done upstream
        if not is_logged and norm_method not in ["VSN", "QUANTILE"]:
            subset_df = su.robust_log2_transform(subset_df)
            
        # ====================================================================
        # Just-In-Time (JIT) Feature Scaling
        # ====================================================================
        # Executed BEFORE transposition to properly scale feature rows
        subset_df = su.apply_feature_scaling(
            df=subset_df, method=scaling_method
        )
        
        # 2. Transpose to (Samples x Features) required by scikit-learn PCA
        feature_dataframe = subset_df.transpose()
        
        # 3. Missing Value (NaN) Handling for PCA
        if feature_dataframe.isna().any().any():
            # Fill NaNs with the median of each feature
            feature_dataframe = feature_dataframe.apply(
                lambda column: column.fillna(column.median()), axis=0
            )
            # If a feature is completely NaN, fill with 0 to prevent failure
            if feature_dataframe.isna().any().any():
                feature_dataframe = feature_dataframe.fillna(0)

        # 4. Extract labels and format return
        labels = feature_dataframe.index.to_frame().reset_index(drop=True)
        feature_columns = list(
            set(feature_dataframe.index.names) - {sample_name}
        )
        features = feature_dataframe.reset_index(
            feature_columns, drop=True
        )
        
        return features, labels

    def run_pca_workflow(self, features):
        """Execute PCA, calculate metrics, and flag statistical outliers."""
        
        # Note: Input 'features' is already JIT-scaled and log-transformed.
        # No internal StandardScaler is needed here anymore.
        model = PCA(
            n_components=self.n_components, random_state=self.global_seed
        )
        scores = model.fit_transform(features)
        
        metrics, sd_limit, od_limit = self._compute_exact_limits(
            model, scores, features
        )
        
        # Categorize spatial distributions into specific outlier domains
        metrics["Category"] = "Normal"
        cond_strong = (metrics["SD"] > sd_limit) & (metrics["OD"] <= od_limit)
        cond_orthogonal = (metrics["SD"] <= sd_limit) & (
            metrics["OD"] > od_limit
        )
        cond_extreme = (metrics["SD"] > sd_limit) & (metrics["OD"] > od_limit)

        metrics.loc[cond_strong, "Category"] = "Strong Outlier"
        metrics.loc[cond_orthogonal, "Category"] = "Orthogonal Outlier"
        metrics.loc[cond_extreme, "Category"] = "Extreme Outlier"

        # Embed explicit boolean flags for statistical threshold breaches
        metrics["is_sd_outlier"] = metrics["SD"] > sd_limit
        metrics["is_od_outlier"] = metrics["OD"] > od_limit
        
        return {
            "scores": scores,
            "variance": model.explained_variance_ratio_,
            "metrics": metrics,
            "sd_limit": sd_limit,
            "od_limit": od_limit,
            "model": model
        }

    def _compute_exact_limits(self, model, scores, scaled_array):
        """Calculate Hotelling T2 and DModX (SPE) with exact limits."""
        from scipy.stats import norm

        n_samples, _ = scores.shape
        loadings = model.components_
        
        # Score Distance (SD) via Hotelling's T2 logic
        variances = np.var(scores, axis=0, ddof=1)
        variances[variances == 0] = 1e-10
        sd_values = np.sum((scores ** 2) / variances, axis=1)
        
        # Orthogonal Distance (OD) via squared prediction error
        x_predicted = np.dot(scores, loadings)
        residuals = scaled_array - x_predicted
        spe_values = np.sum(residuals ** 2, axis=1)

        # Statistical limit for SD based on F-distribution
        f_critical_value = f.ppf(
            1 - self.alpha, self.n_components, n_samples - self.n_components
        )
        sd_limit = float((self.n_components * (n_samples - 1) / (
            n_samples - self.n_components)) * f_critical_value)

        # Calculate OD limit based on selected statistical approximation
        if self.od_method == "box":
            spe_mean = np.mean(spe_values)
            spe_variance = np.var(spe_values, ddof=1)
            
            if spe_variance > 1e-10:
                g_factor = spe_variance / (2 * spe_mean)
                h_factor = (2 * spe_mean ** 2) / spe_variance
                od_limit = float(
                    g_factor * chi2.ppf(1 - self.alpha, df=h_factor)
                )
            else:
                od_limit = float(spe_mean * 1.05)
                
        elif self.od_method == "jm":
            # Utilize inner product (N x N) to avoid OOM in LC-MS datasets
            # Non-zero eigenvalues of E*E.T are identical to E.T*E
            residual_covariance = np.cov(residuals, rowvar=True)
            eigenvalues = np.linalg.eigvalsh(residual_covariance)
            eigenvalues = eigenvalues[eigenvalues > 1e-9]
            
            t1 = np.sum(eigenvalues)
            t2 = np.sum(eigenvalues ** 2)
            t3 = np.sum(eigenvalues ** 3)
            
            # Jackson-Mudholkar approximation logic
            h0 = 1.0 - (2.0 * t1 * t3) / (3.0 * (t2 ** 2))
            critical_alpha = norm.ppf(1 - self.alpha)
            
            term1 = (critical_alpha * np.sqrt(2.0 * t2 * (h0 ** 2))) / t1
            term2 = (t2 * h0 * (h0 - 1.0)) / (t1 ** 2)
            od_limit = float(t1 * ((term1 + 1.0 + term2) ** (1.0 / h0)))
            
        else:
            raise ValueError("Parameter od_method must be 'box' or 'jm'.")

        return pd.DataFrame({"SD": sd_values, "OD": spe_values}), sd_limit, od_limit

    @staticmethod
    def calc_relative_dispersion(coords, types, qc_label, actual_label):
        """Compute the relative variance of QC versus actual samples."""
        qc_indices = np.where(types == qc_label)[0]
        actual_indices = np.where(types == actual_label)[0]

        if len(qc_indices) < 2 or len(actual_indices) < 2:
            return np.nan

        # Calculate grouped variance using Bessel's correction
        qc_variance = np.var(coords[qc_indices, 0], ddof=1) + np.var(
            coords[qc_indices, 1], ddof=1
        )
        actual_variance = np.var(
            coords[actual_indices, 0], ddof=1
        ) + np.var(coords[actual_indices, 1], ddof=1)

        return float(
            qc_variance / actual_variance
        ) if actual_variance > 1e-9 else np.nan

    @staticmethod
    def calc_qc_batch_silhouette(coords, types, batches, qc_label):
        """Compute silhouette score for QC samples across batches."""
        valid_sample_mask = (types == qc_label)
        if valid_sample_mask.sum() < 3 or len(
            np.unique(batches[valid_sample_mask])
        ) < 2:
            return np.nan
        return float(
            silhouette_score(coords[valid_sample_mask], batches[valid_sample_mask])
        )

    @staticmethod
    def calc_qc_centrality_shift(coords, types, qc_label, actual_label):
        """
        Calculate relative distance between QC and Actual Sample centroids.
        """
        qc_indices = np.where(types == qc_label)[0]
        actual_indices = np.where(types == actual_label)[0]
        
        if len(qc_indices) == 0 or len(actual_indices) == 0:
            return {
                "absolute_distance": np.nan, 
                "actual_sample_dispersion": np.nan, 
                "relative_shift": np.nan
            }
        
        qc_coordinates = coords[qc_indices]
        actual_coordinates = coords[actual_indices]
        
        # Calculate centroids and absolute Euclidean distance
        qc_centroid = np.mean(qc_coordinates, axis=0)
        actual_centroid = np.mean(actual_coordinates, axis=0)
        absolute_distance = np.linalg.norm(qc_centroid - actual_centroid)
        
        # Calculate dispersion of Actual Samples
        actual_distances = np.linalg.norm(
            actual_coordinates - actual_centroid, axis=1
        )
        actual_sample_dispersion = np.mean(actual_distances)
        
        if actual_sample_dispersion > 0:
            relative_shift = absolute_distance / actual_sample_dispersion
        else:
            relative_shift = np.nan
        
        return {
            "absolute_distance": float(absolute_distance),
            "actual_sample_dispersion": float(actual_sample_dispersion),
            "relative_shift": float(relative_shift)
        }


if __name__ == "__main__":
    # =========================================================================
    # Standalone Testing & Usage Example
    # =========================================================================
    import warnings
    warnings.filterwarnings("ignore")

    print("--- Testing PCAEngine with Mock Data ---")

    # 1. Generate Mock MultiIndex Data (Features as rows, Samples as columns)
    np.random.seed(42)
    col_arrays = [
        ["Sample", "Sample", "Sample", "QC", "QC", "QC"],
        ["Batch1", "Batch1", "Batch2", "Batch1", "Batch1", "Batch2"],
        ["S1", "S2", "S3", "Q1", "Q2", "Q3"]
    ]
    col_tuples = list(zip(*col_arrays))
    multi_columns = pd.MultiIndex.from_tuples(
        col_tuples, names=["Sample Type", "Batch", "Sample Name"]
    )
    
    # 100 features, 6 samples (log-normal distribution to mimic MS data)
    mock_data = np.random.lognormal(mean=2.0, sigma=0.5, size=(100, 6))
    mock_df = pd.DataFrame(mock_data, columns=multi_columns)
    mock_df.index = [f"Feature_{i}" for i in range(100)]
    
    # 2. Initialize Engine
    engine = PCAEngine(
        n_components=2, alpha=0.05, od_method="box", global_seed=123
    )

    # 3. Extract Features (Handling transformations automatically)
    features, labels = engine.extract_features(
        df=mock_df, 
        sample_type="Sample Type", 
        sample_name="Sample Name", 
        actual_label="Sample", 
        qc_label="QC"
    )
    print(f"\n[1] Extracted Features Shape: {features.shape} (Samples x Features)")

    # 4. Run Core Workflow
    results = engine.run_pca_workflow(features=features)
    
    print("\n[2] PCA Explained Variance:")
    print(f"    PC1: {results['variance'][0]*100:.2f}%")
    print(f"    PC2: {results['variance'][1]*100:.2f}%")
    
    print(f"\n[3] Statistical Limits (Alpha={engine.alpha}):")
    print(f"    SD Limit (Hotelling's T2): {results['sd_limit']:.2f}")
    print(f"    OD Limit (SPE / DModX): {results['od_limit']:.2f}")

    print("\n[4] Outlier Detection Metrics (Head):")
    # Attach sample names for readable output
    metrics_preview = results["metrics"].copy()
    metrics_preview.index = labels["Sample Name"]
    print(metrics_preview[["SD", "OD", "Category"]].head())
    
    print("\n--- Test Completed Successfully ---")