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
    # Step-00: Environment Initialization
    # The pipeline guarantees the integrity of the output directory structure 
    # before commencing computationally intensive tasks.
    # ========================================================================
    logger.info("Step 00: Environment Initialization...")
    iu._check_dir_exists(dir_path=output_dir, handle="makedirs")
    logger.info(f"Pipeline workspace securely mounted at: {output_dir}")

    # ========================================================================
    # Step-01: Dataset Construction
    # The pipeline initializes the workflow by transforming fragmented raw 
    # peak tables and metadata into a standardized MetaboInt object. This 
    # phase ensures precise coordinate alignment between sample identifiers 
    # and feature intensities, establishing a robust structural foundation.
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
    # QA Step-01: Raw Data Quality Assessment
    # A comprehensive diagnostic audit of the raw dataset is executed prior 
    # to any corrective interventions. This baseline evaluation identifies 
    # initial batch offsets, injection-order-dependent signal attenuation, 
    # and global missing value distribution topologies.
    # ========================================================================
    logger.info("QA-Step 01: Quality Assessment of Raw Data...")
    qa_step1_dir = os.path.join(output_dir, "QA_01_Raw_Data")
    qa_raw_engine = MetaboIntAssessor(data=raw_data, pipeline_params=params)
    qa_raw_engine.execute_assessment(output_dir=qa_step1_dir)

    # ========================================================================
    # Step-02: High-Missing Value Feature Filtering
    # Features exhibiting excessive missingness are systematically removed. 
    # A topology-aware diagnostic is applied to classify missing data into 
    # MAR and MNAR populations, which directly guides the algorithmic 
    # selection for subsequent filtering and imputation.
    # ========================================================================
    logger.info("Step 02: High-Missing Value Feature Filtering...")
    step2_dir = os.path.join(output_dir, "02_MV_Filtered")
    fltr_mv_engine = MetaboIntFilter(data=raw_data, pipeline_params=params)
    mv_filter_data = fltr_mv_engine.execute_mv_filtering(output_dir=step2_dir)

    # ========================================================================
    # QA Step-02: Data Assessment of High MV Features Filtered Data
    # The structural integrity of the feature set is verified following the 
    # initial missing value filtration. This assessment ensures that global data
    # distributions remain undisturbed and specific sparsity issues have been 
    # adequately resolved.
    # ========================================================================
    logger.info("QA-Step 02: Quality Assessment of High-MV Filtered Data...")
    qa_step2_dir = os.path.join(output_dir,  "QA_02_MV_Filtered")
    qa_mv_filter_engine = MetaboIntAssessor(
        data=mv_filter_data, pipeline_params=params)
    qa_mv_filter_engine.execute_assessment(output_dir=qa_step2_dir)

    # ========================================================================
    # Step-03: Signal Drift and Batch Effect Correction
    # Systemic technical variations are mitigated. The algorithm 
    # stabilizes signal trajectories utilizing QC-anchored regression 
    # models (QC-SVR, QC-RLSC or QC-RFSC), thereby extracting and preserving 
    # true biological variance from instrument-induced noise.
    # ========================================================================
    logger.info("Step 03 & 04: Signal Drift & Batch Effect Correction...")
    step3_4_dir = os.path.join(output_dir, "03_04_Corrected_Data")
    sc_engine = MetaboIntCorrector(data=mv_filter_data, pipeline_params=params)
    intra_sc_data, inter_sc_data = sc_engine.execute_signal_correction(
        output_dir=step3_4_dir)

    # ========================================================================
    # QA Step-03: Intra-Batch Signal Correction Data Assessment
    # High-resolution visual audits are conducted to validate the 
    # elimination of within-batch drift. Statistical metrics confirm 
    # whether individual signal trajectories have been successfully 
    # normalized to a stable, horizontal baseline.
    # ========================================================================
    logger.info(
        "QA-Step 03: Quality Assessment of Intra-batch Corrected Data...")
    qa_step3_dir = os.path.join(output_dir, "QA_03_Intra_Corrected_Data")
    qa_intra_engine = MetaboIntAssessor(
        data=intra_sc_data, pipeline_params=params) 
    qa_intra_engine.execute_assessment(output_dir=qa_step3_dir)

    # ========================================================================
    # QA Step-04: Inter-Batch Signal Correction Data Assessment
    # The alignment of intensity scales across multiple independent 
    # acquisition batches is analyzed. This step ensures that the dataset 
    # is systemically unified and artificial inter-batch discrepancies 
    # have been effectively mitigated.
    # ========================================================================
    logger.info(
        "QA-Step 04: Quality Assessment of Inter-batch Corrected Data...")
    qa_step4_dir = os.path.join(output_dir, "QA_04_Inter_Corrected_Data")
    qa_inter_engine = MetaboIntAssessor(
        data=inter_sc_data, pipeline_params=params)
    qa_inter_engine.execute_assessment(output_dir=qa_step4_dir)

    # ========================================================================
    # Step-05: Low-Quality Feature Filtering
    # The feature matrix is further refined based on technical 
    # reproducibility and biological relevance. Features heavily 
    # contaminated by analytical blanks or exhibiting poor precision across 
    # QC replicates are permanently discarded.
    # ========================================================================
    logger.info("Step 05: Low-Quality Feature Filtering...")
    step5_dir = os.path.join(output_dir, "05_Quality_Filtered")
    fltr_low_quality_engine = MetaboIntFilter(
        data=inter_sc_data, pipeline_params=params)
    low_quality_filter_data = fltr_low_quality_engine.execute_quality_filtering(
        output_dir=step5_dir)

    # ========================================================================
    # QA Step-05: Quality Assessment of Low-quality features Filtered Data
    # A pre-imputation health check is performed on the refined dataset. 
    # This evaluation confirms that the surviving features represent high-
    # fidelity biological signals, ensuring the matrix is optimally prepared.
    # ========================================================================
    logger.info(
        "QA-Step 05: Quality Assessment of Low-quality features Filtered "
        "Data...")
    qa_step5_dir = os.path.join(output_dir, "QA_05_Quality_Filtered")
    qa_low_quality_filter_engine = MetaboIntAssessor(
        data=low_quality_filter_data, pipeline_params=params)
    qa_low_quality_filter_engine.execute_assessment(output_dir=qa_step5_dir)

    # ========================================================================
    # Step-06: Missing Value Imputation
    # An autonomous multi-algorithm benchmarking simulation is executed for MAR. 
    # The optimal algorithm is programmatically selected based on its ability 
    # to reconstruct established distributions while minimizing bias in the 
    # original variance structure.
    # Execute min-value constant imputation for MNAR.
    # ========================================================================
    logger.info("Step 06: Missing Value Imputation...")
    step6_dir = os.path.join(output_dir, "06_Imputation")
    imp_engine = MetaboIntImputer(
        data=low_quality_filter_data, pipeline_params=params)
    imputed_data = imp_engine.execute_imputation(output_dir=step6_dir)

    # ========================================================================
    # QA Step-06: Post-Imputation Data Assessment
    # The statistical fidelity of the imputed values is rigorously evaluated. 
    # This step evaluates the extent to which synthetic data points might 
    # introduce artificial clustering or distort biological correlations.
    # ========================================================================
    logger.info("QA-Step 06: Quality Assessment of Imputed Data...")
    qa_step6_dir = os.path.join(output_dir, "QA_06_Imputed_Data")
    qa_imp_engine = MetaboIntAssessor(data=imputed_data, pipeline_params=params)
    qa_imp_engine.execute_assessment(output_dir=qa_step6_dir)

    # ========================================================================
    # Step-07: Normalization
    # The dataset undergoes final quantitative refinement. Sample-wise 
    # normalization is applied to compensate for volume variations, 
    # followed by feature-wise scaling to stabilize variance.
    # ========================================================================
    logger.info("Step 07: Data Normalization...")
    step7_dir = os.path.join(output_dir, "07_Normalized_Data")
    norm_engine = MetaboIntNormalizer(imputed_data, pipeline_params=params)
    normalized_data = norm_engine.execute_normalization(
        output_dir=step7_dir)

    # ========================================================================
    # QA Step-07: Normalization Data Assessment
    # The efficacy of the normalization is validated. This assessment confirms 
    # that global intensity levels across individual samples are properly 
    # aligned, effectively reducing systematic biases.
    # ========================================================================
    logger.info("QA-Step 07: Quality Assessment of Normalized Data...")
    qa_step7_dir = os.path.join(output_dir, "QA_07_Sample_Norm_Data")
    qa_norm_engine = MetaboIntAssessor(
        data=normalized_data, pipeline_params=params)
    qa_norm_engine.execute_assessment(output_dir=qa_step7_dir)

    # ========================================================================
    # Step 8: Step-Sequential Narrative Report
    # The workflow concludes by synthesizing all extracted metadata and visual 
    # assets. Diagnostic grids are compiled, and critical decision-making 
    # metrics are extracted to autonomously generate a comprehensive, 
    # human-readable Markdown/PDF/HTML report.
    # ========================================================================
    logger.info("Final: Compiling Sequential Audit Report...")
    report_dir = "08_Report_Summary"

    # 1. Process Visual Assets
    visual_rep = ru.VisualAssetReporter(
        base_dir=output_dir
    )
    visual_rep.compile_assessor_report(
        report_folder=report_dir, is_multi_batch=is_multi_batch_flag)

    # 2. Extract Metadata & Render Markdown (Sequential Mapping)
    # 2.1 Define the consolidated object pool using sub-step keys
    pipeline_metrics_objs = {
        "raw_dataset": raw_data.dataset_metrics,
        "high_mv_feature_filtering": mv_filter_data.mv_filtering_metrics,
        "intra_signal_correction": intra_sc_data.correction_metrics,
        "inter_signal_correction": inter_sc_data.correction_metrics,
        "low_quality_feature_filtering": 
            low_quality_filter_data.quality_filtering_metrics,
        "missing_value_imputation": imputed_data.imputation_metrics,
        "normalization": normalized_data.normalization_metrics
    }

    qa_metrics_objs = {
        "raw_dataset": qa_raw_engine.assessment_metrics,
        "high_mv_feature_filtering": qa_mv_filter_engine.assessment_metrics,
        "intra_signal_correction": qa_intra_engine.assessment_metrics,
        "inter_signal_correction": qa_inter_engine.assessment_metrics,
        "low_quality_feature_filtering": 
            qa_low_quality_filter_engine.assessment_metrics, 
        "missing_value_imputation": qa_imp_engine.assessment_metrics,
        "normalization": qa_norm_engine.assessment_metrics
    }

    # 1.2 Initialize reporter and generate ONE markdown file
    logger.info(f"Initializing narrative reporter at workspace: {output_dir}")
    md_reporter = ru.NarrativeStatsReporter(base_dir=output_dir)

    md_reporter.generate_markdown(
        pipeline_metrics=pipeline_metrics_objs, 
        qa_metrics=qa_metrics_objs,
        report_folder=report_dir
    )

    success = md_reporter.export_report(pdf_engine="weasyprint")

    if success:
        logger.success("PI-METABOQC PIPELINE COMPLETED SUCCESSFULLY.")