"""Module for invalid feature and sample filtering in metabolomics."""

import pandas as pd
import numpy as np
from typing import Dict, Any


class MetaboIntFLTR:
    """Filtering engine for metabolomics datasets."""

    def __init__(
        self, metabo_obj: pd.DataFrame, pipeline_params: Dict[str, Any]
    ) -> None:
        """Initialize the filtering engine."""
        self.metabo_obj = metabo_obj
        self.params = pipeline_params

    def execute_mv_fltr(self) -> pd.DataFrame:
        """Execute Stage-1 missing value filter with robust group validation.
        
        Features pass if they meet the tolerance in AT LEAST ONE valid 
        biological group. Technical groups (QC, Blank, IS) are strictly 
        excluded from the biological override logic.
        
        Returns:
            pd.DataFrame: Missing value filtered dataset.
        """
        grp_col = self.params.get("MetaboInt", {}).get("bio_group", "Group")
        qc_col = self.params.get("MetaboInt", {}).get("sample_type", "Type")
        qc_lbl = self.params.get("MetaboInt", {}).get("qc_label", "QC")
        mv_tol = self.params.get("Filter", {}).get("mv_tol", 0.8)
        
        valid_bio_groups = []
        
        if grp_col in self.metabo_obj.columns.names:
            raw_groups = self.metabo_obj.columns.get_level_values(
                grp_col
            ).unique()
            
            invalid_strs = {
                "unknown", "na", "n/a", "nan", "none", "null", "", "unassigned",
                "blank", "blk", "is", "internal standard", "internal_standard",
                "solvent", "wash", "sst", "system suitability", "pool"
            }
            
            for g in raw_groups:
                if pd.isna(g):
                    continue
                g_str = str(g).strip().lower()
                if g_str in invalid_strs or g_str == str(qc_lbl).lower():
                    continue
                valid_bio_groups.append(g)

        if valid_bio_groups:
            na_rate = self.metabo_obj.isna().groupby(
                level=grp_col, axis=1
            ).mean()
            pass_mask = (na_rate[valid_bio_groups] <= mv_tol).any(axis=1)
        else:
            if qc_col in self.metabo_obj.columns.names:
                qc_mask = (
                    self.metabo_obj.columns.get_level_values(qc_col) == qc_lbl
                )
            else:
                qc_mask = np.array([False] * self.metabo_obj.shape[1])
                
            if qc_mask.any():
                df_qc = self.metabo_obj.loc[:, qc_mask]
                pass_mask = df_qc.isna().mean(axis=1) <= mv_tol
            else:
                pass_mask = self.metabo_obj.isna().mean(axis=1) <= mv_tol
                
        valid_idx = self.metabo_obj.index[pass_mask]
        df_final = self.metabo_obj.loc[valid_idx]
        return df_final.__finalize__(self.metabo_obj)

    def execute_quality_fltr(
        self, idx_mar: pd.Index, idx_mnar: pd.Index
    ) -> pd.DataFrame:
        """Execute Stage-2 quality filter bypassing MNAR features.
        
        Args:
            idx_mar: Features classified as Missing At Random.
            idx_mnar: Features classified as Missing Not At Random.
            
        Returns:
            pd.DataFrame: Quality filtered dataset.
        """
        qc_col = self.params.get("MetaboInt", {}).get("sample_type", "Type")
        qc_lbl = self.params.get("MetaboInt", {}).get("qc_label", "QC")
        rsd_tol = self.params.get("Filter", {}).get("rsd_tol", 0.3)
        
        qc_mask = self.metabo_obj.columns.get_level_values(qc_col) == qc_lbl
        
        if not qc_mask.any():
            return self.metabo_obj.copy().__finalize__(self.metabo_obj)
            
        df_qc = self.metabo_obj.loc[:, qc_mask]
        
        valid_mar_idx = idx_mar.intersection(self.metabo_obj.index)
        df_qc_mar = df_qc.loc[valid_mar_idx]
        
        if not df_qc_mar.empty:
            rsd_mar = df_qc_mar.std(axis=1, ddof=1) / df_qc_mar.mean(axis=1)
            pass_rsd_mar = rsd_mar[rsd_mar <= rsd_tol].index
        else:
            pass_rsd_mar = pd.Index([])
        
        valid_idx = pass_rsd_mar.union(idx_mnar)
        valid_idx = valid_idx.intersection(self.metabo_obj.index)
        
        df_final = self.metabo_obj.loc[valid_idx]
        return df_final.__finalize__(self.metabo_obj)


class MetaboVisualizerFLTR(BaseMetaboVisualizer):
    """Visualization suite for filtering results."""

    def plot_filtering_summary(
        self, 
        before_count: int, 
        after_stage1: int, 
        after_stage2: int
    ) -> plt.Figure:
        """Plot a waterfall or bar chart showing feature attrition."""
        fig, ax = plt.subplots(figsize=(8, 5))
        counts = [before_count, after_stage1, after_stage2]
        labels = ['Raw', 'Post-MV (Stage 1)', 'Post-Quality (Stage 2)']
        
        sns.barplot(x=labels, y=counts, ax=ax, palette="viridis")
        ax.set_title("Feature Filtering Summary")
        ax.set_ylabel("Number of Features")
        
        # Add labels on top of bars
        for i, v in enumerate(counts):
            ax.text(i, v + (max(counts)*0.02), str(v), ha='center')
            
        plt.tight_layout()
        plt.close(fig)
        return fig

    def plot_rsd_distribution_with_exemptions(
        self, 
        rsd_values: pd.Series, 
        idx_mnar: pd.Index,
        rsd_tol: float = 0.3
    ) -> plt.Figure:
        """Plot RSD distribution highlighting exempted MNAR features."""
        fig, ax = plt.subplots(figsize=(7, 5))
        
        # Separate RSDs into MAR (subject to filter) and MNAR (exempted)
        rsd_mar = rsd_values.drop(idx_mnar, errors='ignore')
        rsd_mnar = rsd_values.loc[rsd_values.index.intersection(idx_mnar)]
        
        sns.histplot(rsd_mar, color="skyblue", label="MAR Features", ax=ax, kde=True)
        sns.histplot(rsd_mnar, color="salmon", label="MNAR (Exempted)", ax=ax, kde=True)
        
        ax.axvline(rsd_tol, color='red', linestyle='--', label=f'Threshold ({rsd_tol})')
        ax.set_title("RSD Distribution & Imputation-Based Exemptions")
        ax.set_xlabel("Relative Standard Deviation (RSD)")
        ax.legend()
        
        plt.tight_layout()
        plt.close(fig)
        return fig