"""End-to-end pipeline orchestration from input tables to report artifacts.

run_pipeline constructs the dataset, executes traceable assessment checkpoints,
runs missing-value filtering, correction, low-quality filtering, imputation,
and normalization, then assembles reports. It carries stage metrics, matrices,
and audit paths forward so each decision remains inspectable in final outputs.
"""

import os
import pandas as pd
from loguru import logger

from .io import utils as iu
from .reporting import utils as ru
from .dataset.builder import build_dataset
from .processing.assessment import MetaboIntAssessor
from .processing.filtering import MetaboIntFilter
from .processing.correction import MetaboIntCorrector
from .processing.imputation import MetaboIntImputer
from .processing.normalization import MetaboIntNormalizer


@iu._exe_time
def run_pipeline(
    meta_df: pd.DataFrame,
    int_df: pd.DataFrame,
    params: dict,
    output_dir: str,
) -> None:
    """Run the complete pi-metaboqc pipeline.

    Executes dataset construction, QA checkpoints, sample filtering, two-stage
    feature filtering, signal correction, missingness-aware imputation,
    normalization, and report generation. All intermediate matrices, tracking
    tables, visual diagnostics, and final reports are written to `output_dir`.

    Args:
        meta_df (pd.DataFrame): The metadata pandas DataFrame.
        int_df (pd.DataFrame): The intensity pandas DataFrame.
        params (dict): The pipeline parameters dict.
        output_dir (str): Root directory for outputting results.
    """
    # ========================================================================
    # Step 00: Environment Initialization
    # **Purpose & Function**:
    # Creates the root output directory before any stage writes artifacts.
    # Each processing and QA stage then owns a dedicated subdirectory, keeping
    # matrices, metrics, and figures traceable to their point of generation.
    # ========================================================================
    logger.info("Step 00: Environment Initialization...")
    iu._check_dir_exists(dir_path=output_dir, handle="makedirs")
    logger.info(f"Pipeline workspace securely mounted at: {output_dir}")

    # ========================================================================
    # Step 01: Dataset Construction
    # **Purpose & Function**:
    # Validates metadata and peak-table consistency, resolves duplicated
    # features/samples and injection-order issues, and aligns intensity columns
    # to metadata. It builds the MultiIndex-backed `MetaboInt` source object
    # with sample types, batches, injection order, and optional bio-groups.
    # Explicit zero intensities are converted to missing values, and the raw
    # matrix plus acquisition overview are exported.
    # ========================================================================
    logger.info("Step 01: Dataset Construction...")
    step1_dir = os.path.join(output_dir, "01_Raw_Data")
    raw_data = build_dataset(
        meta_info=meta_df,
        int_df=int_df,
        pipeline_params=params,
        output_dir=step1_dir,
    )
    is_multi_batch_flag = raw_data.attrs["is_multi_batch"]

    # ========================================================================
    # QA-Step 01: Quality Assessment of Raw Data
    # **Evaluation Logic**:
    # Establishes the unprocessed baseline using QC and batch QC correlations,
    # PCA, QC RSD distributions, acquisition-order diagnostics, and internal-
    # standard/outlier-reference-feature checks. These metrics provide the
    # comparison point for all later transformations.
    # ========================================================================
    logger.info("QA-Step 01: Quality Assessment of Raw Data...")
    qa_step1_dir = os.path.join(output_dir, "QA_01_Raw_Data")
    qa_raw_engine = MetaboIntAssessor(data=raw_data, pipeline_params=params)
    qa_raw_engine.execute_assessment(output_dir=qa_step1_dir)

    # ========================================================================
    # Step 02: High-Missing Value Feature Filtering
    # **Purpose & Function**:
    # First removes only QC and biological samples exceeding the sample-level
    # missingness tolerance; non-target sample types are retained. It then
    # calculates global, QC, and per-bio-group missingness to label features as
    # MAR, MNAR, or dropped. Biological-group rescue retains group-specific
    # sparse signals, while QC rescue retains plausible low-abundance features
    # according to QC missingness and a low-intensity percentile criterion.
    # The resulting labels and attrition tables drive later filtering and
    # imputation.
    # ========================================================================
    logger.info("Step 02: High-Missing Value Feature Filtering...")
    step2_dir = os.path.join(output_dir, "02_MV_Filtered")
    fltr_mv_engine = MetaboIntFilter(data=raw_data, pipeline_params=params)
    mv_filter_data = fltr_mv_engine.execute_mv_filtering(output_dir=step2_dir)

    # ========================================================================
    # QA-Step 02: Quality Assessment of High-MV Filtered Data
    # **Evaluation Logic**:
    # Re-runs the standard QA suite after sample removal and feature routing.
    # It shows the impact of structural sparsity filtering on QC consistency,
    # sample structure, missingness, and reference-feature behaviour.
    # ========================================================================
    logger.info("QA-Step 02: Quality Assessment of High-MV Filtered Data...")
    qa_step2_dir = os.path.join(output_dir, "QA_02_MV_Filtered")
    qa_mv_filter_engine = MetaboIntAssessor(
        data=mv_filter_data, pipeline_params=params
    )
    qa_mv_filter_engine.execute_assessment(output_dir=qa_step2_dir)

    # ========================================================================
    # Step 03: Signal Drift & Batch Effect Correction
    # **Purpose & Function**:
    # Reduces injection-order drift and batch effects with a selected
    # QC-anchored or global strategy. QC-RLSC (including robust and optional
    # GCV span selection), QC-SVR, and QC-RFSC fit feature-wise drift from QC
    # samples; multi-batch QC-anchored runs then apply QC-median alignment.
    # SERRF borrows correlated features in a random-forest model, RUV-III
    # removes estimated unwanted factors, and WaveICA 2.0 removes injection-
    # order-associated independent components. In `Auto` mode, SERRF, RUV-III,
    # WaveICA 2.0, standard/robust QC-RLSC, and QC-SVR are compared and the
    # selected candidate alone is propagated. Blanks are excluded from fitting
    # where applicable and receive frozen-model corrections in the output.
    # ========================================================================
    logger.info("Step 03: Signal Drift & Batch Effect Correction...")
    step3_dir = os.path.join(output_dir, "03_Corrected_Data")
    sc_engine = MetaboIntCorrector(data=mv_filter_data, pipeline_params=params)
    corrected_stages = sc_engine.execute_signal_correction(output_dir=step3_dir)
    final_corr_data = list(corrected_stages.values())[-1]

    # ========================================================================
    # QA-Step 03: Quality Assessment of Signal Corrected Data
    # **Evaluation Logic**:
    # Assesses every returned correction stage, rather than only the final
    # matrix. The same QC correlation, RSD, PCA, acquisition-order, and
    # reference-feature diagnostics quantify drift reduction and batch
    # alignment while exposing possible distortion of sample structure.
    # ========================================================================
    qa_step3_dir = os.path.join(output_dir, "QA_03_Corrected_Data")
    qa_engines_dict = {}
    for stage_name, stage_data in corrected_stages.items():
        logger.info(f"Executing Quality Assessment (QA) for: {stage_name}")
        qa_corr_engine = MetaboIntAssessor(
            data=stage_data, pipeline_params=params
        )
        qa_corr_engine.execute_assessment(
            output_dir=os.path.join(qa_step3_dir, stage_name)
        )
        qa_engines_dict[stage_name] = qa_corr_engine

    # ========================================================================
    # Step 04: Low-Quality Feature Filtering
    # **Purpose & Function**:
    # Removes features dominated by blank background (mean Blank/QC ratio
    # above its tolerance) and technically irreproducible MAR features (QC RSD
    # above its tolerance). Features previously routed to MNAR remain exempt
    # from the QC-RSD screen so trace-level, left-censored signals are not
    # discarded solely for expected low-abundance variability.
    # ========================================================================
    logger.info("Step 04: Low-Quality Feature Filtering...")
    step4_dir = os.path.join(output_dir, "04_Quality_Filtered")
    fltr_low_quality_engine = MetaboIntFilter(
        data=final_corr_data, pipeline_params=params
    )
    low_quality_filter_data = fltr_low_quality_engine.execute_quality_filtering(
        output_dir=step4_dir
    )

    # ========================================================================
    # QA-Step 04: Quality Assessment of Quality Filtered Data
    # **Evaluation Logic**:
    # Confirms the reproducibility and sample/QC structure of the feature set
    # passed to imputation, with the retained-feature and removal-reason
    # diagnostics documenting the blank and RSD decisions.
    # ========================================================================
    logger.info(
        "QA-Step 04: Quality Assessment on Low-Quality Feature Filtered Data..."
    )
    qa_step4_dir = os.path.join(output_dir, "QA_04_Quality_Filtered")
    qa_low_quality_filter_engine = MetaboIntAssessor(
        data=low_quality_filter_data, pipeline_params=params
    )
    qa_low_quality_filter_engine.execute_assessment(output_dir=qa_step4_dir)

    # ========================================================================
    # Step 05: Missing Value Imputation
    # **Purpose & Function**:
    # Applies separate, label-aware routes to the non-blank matrix. MNAR
    # features use QRILC or row-, column-, or global LOD-fraction constants;
    # MAR features use KNN, LLS, BPCA, MinProb, Median, or an `Auto` choice.
    # Auto benchmarks KNN, LLS, BPCA, MinProb, and Median by stratified masking
    # and reconstruction metrics before applying the selected method. Blank
    # columns are not imputed, and the stage is safely skipped when no target
    # values are missing.
    # ========================================================================
    logger.info("Step 05: Missing Value Imputation...")
    step5_dir = os.path.join(output_dir, "05_Imputation")
    imp_engine = MetaboIntImputer(
        data=low_quality_filter_data, pipeline_params=params
    )
    imputed_data = imp_engine.execute_imputation(output_dir=step5_dir)

    # ========================================================================
    # QA-Step 05: Quality Assessment of Imputed Data
    # **Evaluation Logic**:
    # Examines the completed matrix for distributional and structural shifts.
    # The imputation passport records the selected MAR method and candidate
    # benchmark results; QA compares observed, completed, and imputed-only
    # values using distribution distances alongside the standard diagnostics.
    # ========================================================================
    logger.info("QA-Step 05: Quality Assessment of Imputated Data...")
    qa_step5_dir = os.path.join(output_dir, "QA_05_Imputed_Data")
    qa_imp_engine = MetaboIntAssessor(data=imputed_data, pipeline_params=params)
    qa_imp_engine.execute_assessment(output_dir=qa_step5_dir)

    # ========================================================================
    # Step 06: Data Normalization
    # **Purpose & Function**:
    # Removes sample-wise scale effects and stabilizes intensity variance using
    # robust-log-only, TIC, Median, PQN, MDFC, Quantile, or VSN processing.
    # TIC/Median/PQN/MDFC are followed by robust Log2; Quantile aligns robust-
    # logged distributions; VSN applies its intrinsic generalized log. `Auto`
    # scores fixed candidates with QC RLE, QC variance/structure, and sample-
    # structure guardrails, then selects a conservative valid method. Blank
    # samples are permanently excluded from the normalized output.
    # ========================================================================
    logger.info("Step 06: Data Normalization...")
    step6_dir = os.path.join(output_dir, "06_Normalized_Data")
    norm_engine = MetaboIntNormalizer(imputed_data, pipeline_params=params)
    normalized_data = norm_engine.execute_normalization(output_dir=step6_dir)

    # ========================================================================
    # QA-Step 06: Quality Assessment of Normalized Data
    # **Evaluation Logic**:
    # Verifies the selected normalization through RLE, MA, eCDF, PCA, and QC
    # consistency diagnostics. The Auto summary, when applicable, preserves
    # candidate scores and the rationale for the selected method.
    # ========================================================================
    logger.info("QA-Step 06: Quality Assessment of Normalized Data...")
    qa_step6_dir = os.path.join(output_dir, "QA_06_Norm_Data")
    qa_norm_engine = MetaboIntAssessor(
        data=normalized_data, pipeline_params=params
    )
    qa_norm_engine.execute_assessment(output_dir=qa_step6_dir)

    # ========================================================================
    # Step 07: Sequential Audit Report Compilation
    # **Purpose & Function**:
    # Consolidates stage metrics, QA metrics, tracking tables, and visual
    # assets into the report workspace. The reporter renders comprehensive and
    # brief Markdown reports and exports the configured PDF representation,
    # preserving the evidence needed to audit each pipeline decision.
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
        "SERRF": "global_correction",
        "RUV-III": "global_correction",
        "WaveICA 2.0": "global_correction",
    }

    for stage_name, engine in qa_engines_dict.items():
        safe_key = stage_key_map.get(stage_name, "unknown_correction")
        qa_corr_metrics[safe_key] = engine.assessment_metrics

    # 3. Assemble Final Metrics
    pipeline_metrics_objs = {
        "raw_dataset": raw_data.dataset_metrics,
        "high_mv_feature_filtering": mv_filter_data.mv_filtering_metrics,
        "signal_correction": final_corr_data.correction_metrics,
        "low_quality_feature_filtering": (
            low_quality_filter_data.quality_filtering_metrics
        ),
        "missing_value_imputation": imputed_data.imputation_metrics,
        "normalization": normalized_data.normalization_metrics,
    }

    qa_metrics_objs = {
        "raw_dataset": qa_raw_engine.assessment_metrics,
        "high_mv_feature_filtering": qa_mv_filter_engine.assessment_metrics,
        **qa_corr_metrics,
        "low_quality_feature_filtering": (
            qa_low_quality_filter_engine.assessment_metrics
        ),
        "missing_value_imputation": qa_imp_engine.assessment_metrics,
        "normalization": qa_norm_engine.assessment_metrics,
    }

    # 4. Initialize reporter and generate ONE markdown file
    print(f"Initializing narrative reporter at workspace: {output_dir}")
    md_reporter = ru.NarrativeStatsReporter(base_dir=output_dir)

    md_reporter.generate_markdown(
        pipeline_metrics=pipeline_metrics_objs,
        qa_metrics=qa_metrics_objs,
        report_folder=report_dir,
    )

    success = md_reporter.export_report(pdf_engine="weasyprint")

    if success:
        logger.success("PI-METABOQC PIPELINE COMPLETED SUCCESSFULLY.")
