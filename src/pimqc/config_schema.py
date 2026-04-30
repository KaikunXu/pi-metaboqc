# src/pimqc/config_schema.py
"""
Purpose of script: Global Configuration Schema and Data Validation Engine.

This module defines the strict Pydantic V2 models for the pi-metaboqc pipeline. 
It acts as the impenetrable "Gatekeeper" for all user-provided configurations 
(TOML/JSON), providing three core guarantees before parameters enter the 
computational engine:
    1. Auto-fill Defaults: 
        Seamlessly populates missing configuration keys or blocks.
    2. Boundary Enforcement: 
        Prevents mathematically invalid inputs (e.g., negative tolerances).
    3. Type Coercion: Ensures 
        strict type safety for downstream pandas and scikit-learn operations.
"""

from typing import List, Dict, Union, Literal, Optional
from pydantic import BaseModel, Field

class MetaboIntConfig(BaseModel):
    """Core Dataset Construction Schema"""
    mode: Literal[
        "POS", "Pos", "Positve", "ESI+",
        "NEG", "Neg", "Negative", "ESI-"] = "ESI+"
    sample_name: str = "Sample Name"
    sample_type: str = "Sample Type"
    bio_group: str = "Bio Group"
    group_order: Optional[List[str]] = Field(default_factory=list)
    batch: str = "Batch"
    inject_order: str = "Inject Order"
    boundary: Literal["IQR", "sigma"] = "IQR"
    global_seed: int = Field(default=123, ge=0)
    internal_standard: List[str] = Field(default_factory=list)
    outlier_id_feats: List[str] = Field(default_factory=list)
    resort_inject_order: Union[Literal["auto", "none"], bool] = "auto"
    sample_dict: Dict[str, str] = Field(
        default={
            "Actual sample": "Sample",
            "Blank sample": "Blank",
            "QC sample": "QC"
        }
    )

class AssessorConfig(BaseModel):
    """Quality Assessment Schema"""
    corr_method: Literal["spearman", "pearson"] = "spearman"
    mask: bool = True
    stat_outlier: Literal["HT2", "SPE-DModX", "both"] = "both"

class FilterConfig(BaseModel):
    """Filtering Thresholds Schema"""
    # Use Field(ge=0.0, le=1.0) to strictly enforce percentage boundaries
    mv_global_tol: float = Field(default=0.7, ge=0.0, le=1.0)
    mv_qc_tol: float = Field(default=0.8, ge=0.0, le=1.0)
    mv_group_tol: float = Field(default=0.5, ge=0.0, le=1.0)
    rsd_qc_tol: float = Field(default=0.3, ge=0.0, le=1.0)
    blank_qc_ratio: float = Field(default=0.2, ge=0.0)
    sample_mv_tol: float = Field(default=0.5, ge=0.0, le=1.0)
    mnar_group_mv_tol: float = Field(default=0.8, ge=0.0, le=1.0)
    mnar_qc_mv_tol: float = Field(default=0.2, ge=0.0, le=1.0)
    mnar_intensity_percentile: float = Field(default=0.1, ge=0.0, le=1.0)

class CorrectorConfig(BaseModel):
    """Signal Drift Correction Schema"""
    base_est: Literal["QC-SVR", "QC-RFSC", "loess", "QC-RLSC", "RF"] = "QC-SVR"
    span: float = Field(default=0.3, gt=0.0, le=1.0)
    n_tree: int = Field(default=500, gt=0)
    svr_kernel: Literal["rbf", "linear", "poly"] = "rbf"
    svr_c: float = Field(default=100.0, gt=0.0)
    svr_gamma: Union[Literal["scale", "auto"], float] = 1.0

class NormalizerConfig(BaseModel):
    """Normalization Schema"""
    sample_wise_norm: Literal["PQN", "TIC", "Median", "None"] = "PQN"
    feature_wise_norm: Literal[
        "VSN", "Auto-scaling", "Pareto-scaling", "None"] = "VSN"
    quantile_norm: bool = False

class ImputerConfig(BaseModel):
    """Missing Value Imputation Schema"""
    mnar_method: Literal["row-wise", "column-wise", "global"] = "row-wise"
    mnar_fraction: float = Field(default=0.5, gt=0.0)
    mar_method: Literal["auto", "probabilistic", "knn", "median"] = "auto"
    knn_neighbors: int = Field(default=5, gt=0)
    sim_mask_ratio: float = Field(default=0.05, gt=0.0, lt=1.0)

class PipelineConfig(BaseModel):
    """Master Pipeline Configuration Root"""
    MetaboInt: MetaboIntConfig = Field(
        default_factory=MetaboIntConfig)
    MetaboIntAssessor: AssessorConfig = Field(
        default_factory=AssessorConfig)
    MetaboIntFilter: FilterConfig = Field(
        default_factory=FilterConfig)
    MetaboIntCorrector: CorrectorConfig = Field(
        default_factory=CorrectorConfig)
    MetaboIntNormalizer: NormalizerConfig = Field(
        default_factory=NormalizerConfig)
    MetaboIntImputer: ImputerConfig = Field(
        default_factory=ImputerConfig)