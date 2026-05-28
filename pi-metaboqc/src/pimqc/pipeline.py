# src/pimqc/pipeline.py
"""
Main execution orchestration pipeline for pi-metaboqc.

This module provides a unified, automated runner to execute the comprehensive 
metabolomics data quality control (QC) workflow. It sequentially orchestrates 
data ingestion, multi-tier feature filtering, signal drift and batch effect 
correction, missing value imputation, and global data normalization. 

Designed for high reproducibility and robust data provenance, the pipeline 
strictly enforces standardized architectural logic and naming conventions, 
ensuring a seamless transition from raw matrices to consolidated reports.
"""

import os
import pandas as pd
from loguru import logger

from . import io_utils as iu
from . import report_utils as ru
from .dataset_builder import build_dataset
from .assessment import MetaboIntAssessor
from .filtering import MetaboIntFilter
from .correction import MetaboIntCorrector
from .imputation import MetaboIntImputer
from .normalization import MetaboIntNormalizer

@iu._exe_time
def run_pipeline(
    meta_df: pd.DataFrame,
    int_df: pd.DataFrame,
    params: dict,
    output_dir: str,
) -> pd.DataFrame:
    """Run the complete pi-metaboqc pipeline.

    Executes data construction, sequential quality assessments, sample filtering
    and dual-stage feature filtering, signal correction, imputation, and 
    normalization.

    Args:
        meta_df (pd.DataFrame): The metadata pandas DataFrame.
        int_df (pd.DataFrame): The intensity pandas DataFrame.
        params (dict): The pipeline parameters dict. 
        output_dir (str): Root directory for outputting results.
    """
    # ========================================================================
    # Step 00: Environment Initialization
    # **Purpose & Function**: 
        # Strictly validate and mount the output directory structure before 
        # initiating computationally intensive tasks.
    # **Method Deconstruction**: 
        # Ensures I/O safety through interceptor mechanisms. This guarantees 
        # that diagnostic charts, intermediate CSV matrices, and final audit 
        # reports have valid physical storage paths throughout the pipeline's 
        # lifecycle, preventing unexpected crashes (OOM or I/O Errors) caused 
        # by missing paths.
    # ========================================================================
    logger.info("Step 00: Environment Initialization...")
    iu._check_dir_exists(dir_path=output_dir, handle="makedirs")
    logger.info(f"Pipeline workspace securely mounted at: {output_dir}")

    # ========================================================================
    # Step 01: Dataset Construction
    # **Purpose & Function**: 
        # Instantiate fragmented raw peak tables and metadata into a unified 
        # `MetaboInt` object, establishing a Single Source of Truth (SSOT).
    # **Method Deconstruction**: 
        # Executes precise coordinate alignment between sample identifiers and 
        # feature intensities. This phase implicitly registers sample types 
        # (e.g., QC, Actual samples, Blanks) and batch information, laying a 
        # structured foundation for topological classification and batch effect 
        # correction.
    # ========================================================================
    logger.info("Step 01: Dataset Construction...")
    step1_dir = os.path.join(output_dir, "01_Raw_Data")
    raw_data = build_dataset(
        meta_info=meta_df,
        int_df=int_df,
        pipeline_params=params,
        output_dir=step1_dir
    )
    is_multi_batch_flag = raw_data.attrs["is_multi_batch"]

    # ========================================================================
    #QA-Step 01: Quality Assessment of Raw Data
    # **Evaluation Logic**: 
        # A comprehensive "health check" of the raw data prior to any 
        # algorithmic intervention. This baseline diagnostic anchors the initial 
        # degree of inter-batch offset, injection-order-dependent signal drift, 
        # and the topological distribution of global missing values. It serves 
        # as the baseline for evaluating the benefits of subsequent 
        # preprocessing.
    # ========================================================================
    logger.info("QA-Step 01: Quality Assessment of Raw Data...")
    qa_step1_dir = os.path.join(output_dir, "QA_01_Raw_Data")
    qa_raw_engine = MetaboIntAssessor(data=raw_data, pipeline_params=params)
    qa_raw_engine.execute_assessment(output_dir=qa_step1_dir)

    # ========================================================================
    ## Step 02: High-Missing Value Feature Filtering
    # **Purpose & Function**: 
        # Systematically eliminate highly degraded samples and redundant 
        # features with excessive missingness, while classifying the missing 
        # mechanisms for surviving features to guide downstream imputation.
    # **Method Deconstruction (`filtering.py`)**: 
        # 1. **Sample-Level Filtering**: Evaluates missing rates strictly on 
        # QC and Actual samples, removing samples with critical missingness.
        # 2. **Feature-Level Classification (MAR vs. MNAR)**:
            # 2.1 Introduces dynamic topological routing to strictly classify 
            # missing features into MAR and MNAR.
            # 2.2 Biological Group Rescue: If a feature is robustly expressed 
            # in a specific biological group, it is classified as MNAR and 
            # retained despite a high global missing rate (preventing the 
            # accidental deletion of group-specific biomarkers).
            # 2.3 QC Rescue: Evaluates low-abundance features using a 
            # dual-threshold approach (abundance limits combined with QC 
            # missing rates).
    # ========================================================================
    logger.info("Step 02: High-Missing Value Feature Filtering...")
    step2_dir = os.path.join(output_dir, "02_MV_Filtered")
    fltr_mv_engine = MetaboIntFilter(data=raw_data, pipeline_params=params)
    mv_filter_data = fltr_mv_engine.execute_mv_filtering(output_dir=step2_dir)

    # ========================================================================
    # QA-Step 02: Quality Assessment of High-MV Filtered Data
    # **Evaluation Logic**: 
        # Verifies that the global abundance distribution of the feature matrix 
        # remains undistorted after structural missing data is cleaned. Ensures 
        # that specific sparsity issues are resolved without disrupting the 
        # intrinsic biological variance between groups.
    # ========================================================================
    logger.info("QA-Step 02: Quality Assessment of High-MV Filtered Data...")
    qa_step2_dir = os.path.join(output_dir,  "QA_02_MV_Filtered")
    qa_mv_filter_engine = MetaboIntAssessor(
        data=mv_filter_data, pipeline_params=params)
    qa_mv_filter_engine.execute_assessment(output_dir=qa_step2_dir)

    # ========================================================================
    # Step 03: Signal Drift & Batch Effect Correction
    # **Purpose & Function**: 
        # Effectively attenuate (rather than absolutely eliminate) intra-batch 
        # signal drift and systematic inter-batch variations caused by 
        # instrument fluctuations and column aging.
    # **Method Deconstruction (`correction.py`)**: 
        # Supports two advanced paradigms, both embedding strict 
        # anti-overfitting mechanisms (K-Fold Out-of-Fold Cross-Validation):
            # 1. Classical Multi-Stage Fitting (QC-RLSC / SVR / RFSC):
                # 1.1 Intra-batch: Uses QC samples as anchor points to fit a 
                # non-linear drift baseline via LOESS, or SVR/Random Forest.
                # 1.2 Inter-batch: Executes global QC median alignment based on 
                # the intra-batch corrected matrix.
            # 2. **Unified Machine Learning Network (Global SERRF)**:
                # Employs Hybrid Temporal-Correlation Random Forest Regression 
                # (SERRF). Computes a Spearman rank correlation network across 
                # features in parallel, combined with injection order, to model 
                # and suppress technical noise in a single step.
    # ========================================================================
    logger.info("Step 03: Signal Drift & Batch Effect Correction...")
    step3_dir = os.path.join(output_dir, "03_Corrected_Data")
    sc_engine = MetaboIntCorrector(data=mv_filter_data, pipeline_params=params)
    corrected_stages = sc_engine.execute_signal_correction(output_dir=step3_dir)
    final_corr_data = list(corrected_stages.values())[-1]
    
    # ========================================================================
    # QA-Step 03: Quality Assessment of Signal Corrected Data
    # **Evaluation Logic**: 
        # Verifies that technical error signatures are systematically 
        # suppressed, ensuring subsequent multivariate clustering (e.g., PCA) 
        # is driven by intrinsic biological traits rather than analytical 
        # artifacts.
    # ========================================================================
    qa_step3_dir = os.path.join(output_dir, "QA_03_Signal_Corrected_Data")
    qa_engines_dict = {}
    for stage_name, stage_data in corrected_stages.items():
        logger.info(f"Executing Quality Assessment (QA) for: {stage_name}")
        qa_corr_engine = MetaboIntAssessor(data=stage_data, pipeline_params=params) 
        qa_corr_engine.execute_assessment(
            output_dir=os.path.join(qa_step3_dir, stage_name))
        qa_engines_dict[stage_name] = qa_corr_engine

    # ========================================================================
    # Step 04: Low-Quality Feature Filtering
    # **Purpose & Function**: 
        # Deeply purifies the feature matrix from the dual dimensions of 
        # technical reproducibility and biological relevance, permanently 
        # removing noise features contaminated or highly sensitive to 
        # instrument fluctuations.
    # **Method Deconstruction (`filtering.py`)**: 
        # 1. Blank-to-QC Ratio Check: Compares the mean abundance in Blank 
        # samples to QC samples. Features exceeding the ratio threshold are 
        # removed to strip solvent background noise and column bleed.
        # 2. QC RSD Control: Features with exceptionally poor reproducibility 
        # (High RSD in QC samples) are discarded. *(Note: Low-abundance features 
        # previously marked as MNAR are automatically exempted to prevent the 
        # misidentification of trace metabolites).*
    # ========================================================================
    logger.info("Step 04: Low-Quality Feature Filtering...")
    step4_dir = os.path.join(output_dir, "04_Quality_Filtered")
    fltr_low_quality_engine = MetaboIntFilter(
        data=final_corr_data, pipeline_params=params)
    low_quality_filter_data = fltr_low_quality_engine.execute_quality_filtering(
        output_dir=step4_dir)

    # ========================================================================
    # QA-Step 04: Quality Assessment of Quality Filtered Data
    # **Evaluation Logic**: 
        # The final "health check" prior to imputation. Confirms that surviving 
        # features represent high-fidelity biological signals and that low 
        # Signal-to-Noise Ratio (SNR) dimensions have been safely truncated.
    # ========================================================================
    logger.info(
        "QA-Step 04: Quality Assessment on Low-Quality Feature Filtered Data...")
    qa_step4_dir = os.path.join(output_dir, "QA_04_Quality_Filtered")
    qa_low_quality_filter_engine = MetaboIntAssessor(
        data=low_quality_filter_data, pipeline_params=params)
    qa_low_quality_filter_engine.execute_assessment(output_dir=qa_step4_dir)

    # ========================================================================
    # Step 05: Missing Value Imputation
    # **Purpose & Function**: 
        # Reconstructs matrix completeness using missing-mechanism-aware hybrid 
        # strategies based on topological classifications (MAR/MNAR).
    # **Method Deconstruction (`imputation.py`)**: 
        # To prevent technical variance leakage between biological groups, 
        # imputation is strictly isolated.
        # 1. **MAR (Random Missing)**: Managed via local estimations such as 
        # KNN, median, or probabilistic distributions. Includes an autonomous 
        # simulated masking and stratified NRMSE evaluation mechanism to 
        # dynamically select the optimal algorithm.
        # 2. **MNAR (Systematic Missing)**: Supports QRILC or LOD-based 
        # fractional constant imputation. QRILC estimates underlying normal 
        # distributions using robust estimators (Median/MAD) and draws randomly 
        # from the truncated left tail, safely preserving natural variance in 
        # low-abundance regions.
    # ========================================================================
    logger.info("Step 05: Missing Value Imputation...")
    step5_dir = os.path.join(output_dir, "05_Imputation")
    imp_engine = MetaboIntImputer(
        data=low_quality_filter_data, pipeline_params=params)
    imputed_data = imp_engine.execute_imputation(output_dir=step5_dir)

    # ========================================================================
    # QA-Step 05: Quality Assessment of Imputed Data
    # **Evaluation Logic**: 
        # Quantifies data distribution shifts pre- and post-imputation. Core 
        # metrics include Jensen-Shannon Divergence (JSD). Validates that the 
        # imputation strategy seamlessly blends into the original data 
        # distribution (KDE Overlay) without introducing artificial data cliffs 
        # or singular value clusters.
    # ========================================================================
    logger.info("QA-Step 05: Quality Assessment of Imputated Data...")
    qa_step5_dir = os.path.join(output_dir, "QA_05_Imputed_Data")
    qa_imp_engine = MetaboIntAssessor(
        data=imputed_data, pipeline_params=params)
    qa_imp_engine.execute_assessment(output_dir=qa_step5_dir)

    # ========================================================================
    # Step 06: Data Normalization
    # **Purpose & Function**: 
        # Calibrates sample-wise systematic shifts and stabilizes 
        # heteroscedasticity across dynamic abundance ranges.
    # **Method Deconstruction (`normalization.py`)**: 
        # 1. **Linear Scaling Strategies**: TIC, Median, or PQN (based on 
        # reference spectrum offset correction). Can be followed by a robust 
        # Log2 transformation.
        # 2. **Distribution Alignment & Variance Stabilization**:
            # Quantile Normalization: Forces all samples to share an identical 
            # global distribution.
            # VSN: Uses a Numba-compiled L-BFGS-B optimizer to solve the maximum 
            # likelihood function, applying a generalized logarithm (glog) 
            # transformation to decouple the dependency between variance and 
            # mean, achieving optimal balance between noise suppression and 
            # biological signal preservation.
    # ========================================================================
    logger.info("Step 06: Data Normalization...")
    step6_dir = os.path.join(output_dir, "06_Normalized_Data")
    norm_engine = MetaboIntNormalizer(imputed_data, pipeline_params=params)
    normalized_data = norm_engine.execute_normalization(
        output_dir=step6_dir)

    # ========================================================================
    # QA-Step 06: Quality Assessment of Normalized Data
    # **Evaluation Logic**: 
        # Comprehensively confirms the smoothing of sample-level systematic 
        # biases via Relative Log Expression (RLE) boxplots, MA-plots 
        # (monitoring abundance-dependent bias via MAD and Spearman correlation
        # ), and eCDF distribution alignments.
    # ========================================================================
    logger.info("QA-Step 06: Quality Assessment of Normalized Data...")
    qa_step6_dir = os.path.join(output_dir, "QA_06_Norm_Data")
    qa_norm_engine = MetaboIntAssessor(
        data=normalized_data, pipeline_params=params)
    qa_norm_engine.execute_assessment(output_dir=qa_step6_dir)

    # ========================================================================
    # Step 07: Sequential Audit Report Compilation
    # **Purpose & Function**: 
        # Synthesizes all quality control metrics, interception logs, and 
        # visual assets generated throughout the pipeline lifecycle into a 
        # highly readable, structured analytical report.
    # **Method Deconstruction**: 
        # Extracts intermediate evaluation metrics stored within object 
        # metrics pools. Consolidates SVG vector graphics and diagnostic tables 
        # from each step, rendering them into professional Markdown and PDF 
        # comprehensive reports via the `weasyprint` engine, guaranteeing 
        # absolute data provenance and traceability.
    # ========================================================================
    logger.info("Step 07: Sequential Audit Report Compilation...")

    report_dir = "07_Report_Summary"
    # 1. Process Visual Assets 
    visual_rep = ru.VisualAssetReporter(base_dir=output_dir)
    visual_rep.compile_assessor_report(
        report_folder=report_dir, is_multi_batch=is_multi_batch_flag
    )

    # 2. Dynamic QA Mapping
    qa_corr_metrics = {}
    stage_key_map = {
        "Intra-batch corrected": "intra_batch_correction",
        "Inter-batch corrected": "inter_batch_correction",
        "Global SERRF": "global_correction"
    }

    for stage_name, engine in qa_engines_dict.items():
        safe_key = stage_key_map.get(stage_name, "unknown_correction")
        qa_corr_metrics[safe_key] = engine.assessment_metrics

    # 3. Assemble Final Metrics
    pipeline_metrics_objs = {
        "raw_dataset": raw_data.dataset_metrics,
        "high_mv_feature_filtering": mv_filter_data.mv_filtering_metrics,
        "signal_correction": final_corr_data.correction_metrics, 
        "low_quality_feature_filtering": 
            low_quality_filter_data.quality_filtering_metrics,
        "missing_value_imputation": imputed_data.imputation_metrics,
        "normalization": normalized_data.normalization_metrics
    }

    qa_metrics_objs = {
        "raw_dataset": qa_raw_engine.assessment_metrics,
        "high_mv_feature_filtering": qa_mv_filter_engine.assessment_metrics,
        **qa_corr_metrics, 
        "low_quality_feature_filtering":
            qa_low_quality_filter_engine.assessment_metrics, 
        "missing_value_imputation": qa_imp_engine.assessment_metrics,
        "normalization": qa_norm_engine.assessment_metrics
    }

    # 4. Initialize reporter and generate ONE markdown file
    print(f"Initializing narrative reporter at workspace: {output_dir}")
    md_reporter = ru.NarrativeStatsReporter(base_dir=output_dir)

    md_reporter.generate_markdown(
        pipeline_metrics=pipeline_metrics_objs, 
        qa_metrics=qa_metrics_objs,
        report_folder=report_dir
    )

    success = md_reporter.export_report(pdf_engine="weasyprint")

    if success:
        logger.success("PI-METABOQC PIPELINE COMPLETED SUCCESSFULLY.")