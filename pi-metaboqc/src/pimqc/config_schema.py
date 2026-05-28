# src/pimqc/config_schema.py
"""Global Configuration Schema and Data Validation Engine.

This module defines the strict Pydantic V2 models for the pi-metaboqc.
It acts as the gatekeeper for user configurations (TOML/JSON).
"""
from typing import List, Dict, Union, Literal, Optional
from pydantic import BaseModel, Field, model_validator, field_validator


class MetaboIntConfig(BaseModel):
    """Core Dataset Construction Schema."""
    mode: Literal[
        "POS", "NEG", "Pos", "Neg", "Positive", "Negative", "ESI+", "ESI-",
        "None", "N/A"] = "ESI+"
    sample_name: str = "Sample Name"
    sample_type: str = "Sample Type"
    bio_group: str = "Bio Group"
    group_order: Optional[List[str]] = Field(default_factory=list)
    batch: str = "Batch"
    inject_order: str = "Inject Order"
    boundary: Literal["IQR", "sigma"] = "IQR"
    global_seed: int = Field(default=123, ge=0)
    internal_standard: List[str] = Field(default_factory=list)
    outlier_ref_feat: List[str] = Field(default_factory=list)
    resort_inject_order: Union[Literal["Auto", "None"], bool] = "Auto"
    sample_dict: Dict[str, str] = Field(
        default={
            "Actual sample": "Sample",
            "Blank sample": "Blank",
            "QC sample": "QC"
        }
    )


class AssessorConfig(BaseModel):
    """Quality Assessment Schema.

    Includes scaling parameters applied locally during assessment.
    """
    corr_method: Literal["Spearman", "Pearson"] = "Spearman"
    scaling_method: Literal[
        "Auto-scaling", "Pareto-scaling", "None"
    ] = "Auto-scaling"
    
    # 1. Update fields with the correct default values (0.75 for IS, 0.5 for ORF)
    is_outlier_threshold: Union[float, int] = Field(
        default=0.75, 
        description=(
            "IS outlier cutoff. Float (0.0-1.0) for ratio, "
            "Int (>=1) for absolute count."))
    orf_outlier_threshold: Union[float, int] = Field(
        default=0.5, 
        description=(
            "ORF outlier cutoff. Float (0.0-1.0) for ratio, "
            "Int (>=1) for absolute count."))

    # 2. Strict validation matching your architectural design
    @field_validator("is_outlier_threshold", "orf_outlier_threshold")
    @classmethod
    def validate_thresholds(cls, v: Union[float, int]) -> Union[float, int]:
        """Enforces the dual-mode threshold logic for outliers.
        
        - Floats must represent a valid ratio (0.0 to 1.0 inclusive).
        - Integers must represent a valid absolute count (>= 1).
        """
        # Guard against Python's base behavior where bool is a subclass of int
        if isinstance(v, bool):
            raise ValueError("Threshold cannot be a boolean.")
        if isinstance(v, float):
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    f"Invalid float threshold '{v}'. "
                    "When provided as a float (ratio), "
                    "it must be between 0.0 and 1.0.")
        elif isinstance(v, int):
            if v < 1:
                raise ValueError(
                    f"Invalid integer threshold '{v}'. "
                    "When provided as an integer (absolute count), "
                    "it must be >= 1.")
        else:
            raise ValueError(
                f"Unsupported type {type(v)} for threshold. "
                "Must be either float or integer.")
            
        return v


class FilterConfig(BaseModel):
    """Filtering Thresholds Schema for samples and features."""
    # Sample-level filtering
    sample_mv_tol: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Feature-level filtering (missing values)
    mv_global_tol: float = Field(default=0.7, ge=0.0, le=1.0)
    mv_qc_tol: float = Field(default=0.3, ge=0.0, le=1.0)
    mv_group_tol: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # MNAR rescue thresholds
    mnar_group_mv_tol: float = Field(default=0.8, ge=0.0, le=1.0)
    mnar_qc_mv_tol: float = Field(default=0.2, ge=0.0, le=1.0)
    mnar_intensity_pct: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # Feature-level filtering (quality-based)
    blank_qc_ratio_tol: float = Field(default=0.2, ge=0.0)
    qc_rsd_tol: float = Field(default=0.3, ge=0.0, le=1.0)


class CorrectorConfig(BaseModel):
    """Signal Drift Correction Schema."""
    base_est: Literal[
        "QC-SVR", "QC-RFSC", "QC-RLSC", "SERRF", "RUV", "Auto"] = "Auto"
    loess_frac: float = Field(default=0.3, gt=0.0, le=1.0)
    rf_n_tree: int = Field(default=500, gt=0, description="Trees for QC-RFSC")
    serrf_n_tree: int = Field(default=100, gt=0, description="Trees for SERRF")
    serrf_corr_features: int = Field(
        default=10, ge=0, description="Correlated features for SERRF")
    svr_kernel: Literal["rbf", "linear", "poly"] = "rbf"
    svr_c: float = Field(default=500.0, gt=0.0)
    svr_gamma: Union[Literal["scale", "auto"], float] = 1.0
    ruv_k: int = Field(
        default=5, gt=0, description="K-factors to remove for RUV-III")
    cv_folds: int = Field(
        default=5, 
        ge=2, 
        description="Number of folds for Out-Of-Fold (OOF) cross-validation."
    )


class NormalizerConfig(BaseModel):
    """Configuration for global normalization and log transform."""
    norm_method: Literal[
        "PQN", "MDFC", "TIC", "Median", "VSN", "Quantile", "None"
    ] = "PQN"
    robust_log: bool = False

    @model_validator(mode="after")
    def validate_normalization_logic(self) -> "NormalizerConfig":
        """Enforces exclusivity between VSN and manual robust log."""
        method = self.norm_method.upper()
        is_log = self.robust_log

        if method == "VSN" and is_log:
            raise ValueError(
                "Incompatible parameters: VSN internally applies a "
                "generalized log. 'robust_log' must be set to False."
            )
        return self


class ImputerConfig(BaseModel):
    """Missing Value Imputation Schema."""
    mnar_method: Literal["Row-wise", "Column-wise", "Global", "QRILC"] = "QRILC"
    mnar_fraction: float = Field(default=0.5, gt=0.0)
    mar_method: Literal["Auto", "MinProb", "KNN", "LLS", "Median"] = "Auto"
    knn_neighbors: int = Field(default=5, gt=0)
    lls_neighbors: int = Field(default=15, gt=0)
    sim_mask_ratio: float = Field(default=0.05, gt=0.0, lt=1.0)


class PipelineConfig(BaseModel):
    """Master Pipeline Configuration Root mapping to TOML sections."""
    MetaboInt: MetaboIntConfig = Field(default_factory=MetaboIntConfig)
    MetaboIntAssessor: AssessorConfig = Field(default_factory=AssessorConfig)
    MetaboIntFilter: FilterConfig = Field(default_factory=FilterConfig)
    MetaboIntCorrector: CorrectorConfig = Field(default_factory=CorrectorConfig)
    MetaboIntNormalizer: NormalizerConfig = Field(
        default_factory=NormalizerConfig)
    MetaboIntImputer: ImputerConfig = Field(default_factory=ImputerConfig)