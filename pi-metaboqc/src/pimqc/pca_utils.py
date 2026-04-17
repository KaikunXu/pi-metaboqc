# src/pimqc/pca_utils.py

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import f, chi2


class PCAEngine:
    """Core engine for PCA-based metabolomics data analysis."""

    def __init__(self, n_components=2, alpha=0.05, od_method="box"):
        """
        Initialize the PCA computational engine.

        Args:
            n_components: Number of principal components to extract.
            alpha: Statistical significance level for outlier detection.
            od_method: Approximation method for OD limit ('box' or 'jm').
        """
        self.n_comps = n_components
        self.alpha = alpha
        self.od_method = od_method

    @staticmethod
    def extract_features(df, st_col, sn_col, act_lbl, qc_lbl):
        """Prepare feature matrix and labels from a MetaboInt dataframe."""
        # Isolate target samples based on provided labels
        sample_types = df.columns.get_level_values(st_col)
        valid_mask = sample_types.isin([act_lbl, qc_lbl])
        
        # Transpose to (samples, features) and ensure float type
        feat_df = df.loc[:, valid_mask].transpose().astype(float)

        # Apply log transformation with half-minimum imputation
        feat_df = feat_df.replace({0: np.nan})
        feat_df = feat_df.fillna(feat_df.min().min() / 2)
        feat_df = np.log(feat_df)

        labels = feat_df.index.to_frame().reset_index(drop=True)
        feat_cols = list(set(feat_df.index.names) - {sn_col})
        features = feat_df.reset_index(feat_cols, drop=True)
        
        return features, labels

    def run_pca_workflow(self, features):
        """Execute PCA, calculate metrics, and flag statistical outliers."""
        scaler = StandardScaler()
        scaled_arr = scaler.fit_transform(features)
        
        model = PCA(n_components=self.n_comps)
        scores = model.fit_transform(scaled_arr)
        
        metrics, sd_lim, od_lim = self._compute_exact_limits(
            model, scores, scaled_arr
        )
        
        # Categorize spatial distributions into specific outlier domains
        metrics["Category"] = "Normal"
        c_str = (metrics["SD"] > sd_lim) & (metrics["OD"] <= od_lim)
        c_ort = (metrics["SD"] <= sd_lim) & (metrics["OD"] > od_lim)
        c_ext = (metrics["SD"] > sd_lim) & (metrics["OD"] > od_lim)

        metrics.loc[c_str, "Category"] = "Strong Outlier"
        metrics.loc[c_ort, "Category"] = "Orthogonal Outlier"
        metrics.loc[c_ext, "Category"] = "Extreme Outlier"

        # Embed explicit boolean flags for statistical threshold breaches
        metrics["is_sd_outlier"] = metrics["SD"] > sd_lim
        metrics["is_od_outlier"] = metrics["OD"] > od_lim
        
        return {
            "scores": scores,
            "variance": model.explained_variance_ratio_,
            "metrics": metrics,
            "sd_limit": sd_lim,
            "od_limit": od_lim,
            "model": model
        }

    def _compute_exact_limits(self, model, scores, scaled_x):
        """Calculate Hotelling T2 and DModX (SPE) with exact limits."""
        from scipy.stats import norm

        n_samples, _ = scores.shape
        loadings = model.components_
        
        # Score Distance (SD) via Hotelling's T2 logic
        variances = np.var(scores, axis=0, ddof=1)
        variances[variances == 0] = 1e-10
        sd_vals = np.sum((scores ** 2) / variances, axis=1)
        
        # Orthogonal Distance (OD) via squared prediction error
        x_pred = np.dot(scores, loadings)
        residuals = scaled_x - x_pred
        spe_vals = np.sum(residuals ** 2, axis=1)

        # Statistical limit for SD based on F-distribution
        f_crit = f.ppf(
            1 - self.alpha, self.n_comps, n_samples - self.n_comps
        )
        sd_lim = float((self.n_comps * (n_samples - 1) / (
            n_samples - self.n_comps)) * f_crit)

        # Calculate OD limit based on selected statistical approximation
        if self.od_method == "box":
            spe_mean = np.mean(spe_vals)
            spe_var = np.var(spe_vals, ddof=1)
            
            if spe_var > 1e-10:
                g = spe_var / (2 * spe_mean)
                h = (2 * spe_mean ** 2) / spe_var
                od_lim = float(g * chi2.ppf(1 - self.alpha, df=h))
            else:
                od_lim = float(spe_mean * 1.05)
                
        elif self.od_method == "jm":
            # Utilize inner product (N x N) to avoid OOM in LC-MS datasets
            # Non-zero eigenvalues of E*E.T are identical to E.T*E
            res_cov = np.cov(residuals, rowvar=True)
            evals = np.linalg.eigvalsh(res_cov)
            evals = evals[evals > 1e-9]
            
            t1 = np.sum(evals)
            t2 = np.sum(evals ** 2)
            t3 = np.sum(evals ** 3)
            
            # Jackson-Mudholkar approximation logic
            h0 = 1.0 - (2.0 * t1 * t3) / (3.0 * (t2 ** 2))
            c_alpha = norm.ppf(1 - self.alpha)
            
            term1 = (c_alpha * np.sqrt(2.0 * t2 * (h0 ** 2))) / t1
            term2 = (t2 * h0 * (h0 - 1.0)) / (t1 ** 2)
            od_lim = float(t1 * ((term1 + 1.0 + term2) ** (1.0 / h0)))
            
        else:
            raise ValueError("Parameter od_method must be 'box' or 'jm'.")

        return pd.DataFrame({"SD": sd_vals, "OD": spe_vals}), sd_lim, od_lim

    @staticmethod
    def calc_relative_dispersion(coords, types, qc_lbl, act_lbl):
        """Compute the relative variance of QC versus actual samples."""
        qc_idx = np.where(types == qc_lbl)[0]
        act_idx = np.where(types == act_lbl)[0]

        if len(qc_idx) < 2 or len(act_idx) < 2:
            return np.nan

        # Calculate grouped variance using Bessel's correction
        qc_v = np.var(coords[qc_idx, 0], ddof=1) + np.var(
            coords[qc_idx, 1], ddof=1)
        act_v = np.var(coords[act_idx, 0], ddof=1) + np.var(
            coords[act_idx, 1], ddof=1)

        return float(qc_v / act_v) if act_v > 1e-9 else np.nan

    @staticmethod
    def calc_qc_batch_silhouette(coords, types, batches, qc_lbl):
        """Compute silhouette score for QC samples across batches."""
        mask = (types == qc_lbl)
        if mask.sum() < 3 or len(np.unique(batches[mask])) < 2:
            return np.nan
        return float(silhouette_score(coords[mask], batches[mask]))

    @staticmethod
    def calc_qc_centrality_shift(coords, types, qc_lbl, act_lbl):
        """
        Calculate the relative distance between QC and Actual Sample centroids.
        """
        qc_idx = np.where(types == qc_lbl)[0]
        act_idx = np.where(types == act_lbl)[0]
        
        if len(qc_idx) == 0 or len(act_idx) == 0:
            return {
                "abs_dist": np.nan, "as_dispersion": np.nan, "rel_shift": np.nan
            }
        
        qc_coords = coords[qc_idx]
        act_coords = coords[act_idx]
        
        # Calculate centroids and absolute Euclidean distance
        qc_center = np.mean(qc_coords, axis=0)
        act_center = np.mean(act_coords, axis=0)
        abs_dist = np.linalg.norm(qc_center - act_center)
        
        # Calculate dispersion of Actual Samples (average distance to centroid)
        act_dist = np.linalg.norm(act_coords - act_center, axis=1)
        as_dispersion = np.mean(act_dist)
        
        rel_shift = abs_dist / as_dispersion if as_dispersion > 0 else np.nan
        
        return {
            "abs_dist": float(abs_dist),
            "as_dispersion": float(as_dispersion),
            "rel_shift": float(rel_shift)
        }