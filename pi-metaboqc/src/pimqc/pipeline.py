# src/pimqc/pipeline.py
"""
Main execution pipeline for pi-metaboqc.

This module provides a unified runner to execute the entire metabolomics
data quality control workflow sequentially, strictly following the 
quickstart tutorial logic and ensuring each stage is properly assessed.
"""

import os
import pandas as pd
from loguru import logger

from . import io_utils as iu
from .dataset_builder import build_dataset
from .assessment import MetaboIntAssessor
from .correction import MetaboIntCorrector
from .filtering import MetaboIntFilter
from .imputation import MetaboIntImputer
from .normalization import MetaboIntNormalizer


def run_pipeline(
    meta_df: pd.DataFrame,
    int_df: pd.DataFrame,
    params_path: str,
    output_dir: str,
    compress_output: bool = True
) -> pd.DataFrame:
    """Run the complete pi-metaboqc pipeline.

    Executes data construction, multiple assessment stages, multi-stage 
    filtering, signal correction, imputation, and dual-stage normalization.

    Args:
        meta_df: The metadata pandas DataFrame.
        int_df: The intensity pandas DataFrame.
        params_path: Path to the pipeline parameters JSON file.
        output_dir: Root directory for outputting results.
        compress_output: Whether to zip the final results.

    Returns:
        The final fully normalized MetaboInt dataset object.
    """
    logger.info("Starting pi-metaboqc pipeline execution.")
    iu._check_dir_exists(dir_path=output_dir, handle="makedirs")

    # Load global pipeline parameters from JSON (Single Source of Truth)
    params = iu._load_json_file(input_file_path=params_path)
    meta_params = params.get("MetaboInt", {})

    # ====================================================================
    # Step 1: Dataset Construction
    # ====================================================================
    step1_dir = os.path.join(output_dir, "01_Raw_Data")
    logger.info("Step 1: Building standardized dataset.")
    
    raw_data = build_dataset(
        meta_info=meta_df,
        int_df=int_df,
        pipeline_params=params,
        mode=meta_params.get("mode", "POS"),
        batch=meta_params.get("batch", "Batch"),
        sample_type=meta_params.get("sample_type", "Sample Type"),
        bio_group=meta_params.get("bio_group", "Bio Group"),
        sample_name=meta_params.get("sample_name", "Sample Name"),
        inject_order=meta_params.get("inject_order", "Inject Order"),
        output_dir=step1_dir
    )

    # ====================================================================
    # Step 2: Quality Assessment (Raw Data)
    # ====================================================================
    step2_dir = os.path.join(output_dir, "02_QA_Raw_Data")
    logger.info("Step 2: Conducting QA on Raw Data...")
    
    qa_raw = MetaboIntAssessor(raw_data, pipeline_params=params)
    qa_raw.execute_assessment(output_dir=step2_dir)

    # ====================================================================
    # Step 3: High Missing Value (MV) Feature Filter (Stage 1)
    # ====================================================================
    step3_dir = os.path.join(output_dir, "03_Filtered_Stage1")
    logger.info("Step 3: Filtering high MV features & MNAR Classification...")
    
    # Identify missing mechanisms and filter high MV features
    metabo_fltr_stg1 = MetaboIntFilter(raw_data, pipeline_params=params)
    filtered_st1 = metabo_fltr_stg1.execute_mv_filtering(output_dir=step3_dir)

    # ====================================================================
    # Step 4: Signal Drift & Batch Effect Correction
    # ====================================================================
    step4_dir = os.path.join(output_dir, "04_Corrected_Data")
    logger.info("Step 4: Performing signal drift correction (SVR/LOESS)...")
    
    metabo_sc = MetaboIntCorrector(filtered_st1, pipeline_params=params)
    intra_data, inter_data = metabo_sc.execute_signal_correction(
        output_dir=step4_dir)

    # ====================================================================
    # Step 5 & 6: QA on Corrected Data (Intra & Inter)
    # ====================================================================
    step5_dir = os.path.join(output_dir, "05_QA_Intra_Corrected")
    logger.info("Step 5: Conducting QA on Intra-batch Corrected Data...")
    qa_intra = MetaboIntAssessor(intra_data, pipeline_params=params)
    qa_intra.execute_assessment(output_dir=step5_dir)

    step6_dir = os.path.join(output_dir, "06_QA_Inter_Corrected")
    logger.info("Step 6: Conducting QA on Inter-batch Corrected Data...")
    qa_inter = MetaboIntAssessor(inter_data, pipeline_params=params)
    qa_inter.execute_assessment(output_dir=step6_dir)

    # ====================================================================
    # Step 7: Low Quality Feature Filter (Stage 2: Blank & RSD)
    # ====================================================================
    step7_dir = os.path.join(output_dir, "07_Filtered_Quality")
    logger.info("Step 7: Filtering low quality features (Stage 2)...")
    
    metabo_fltr_stg2 = MetaboIntFilter(inter_data, pipeline_params=params)
    # Use classified indices from Stage 1 to protect MNAR features
    quality_filtered_data = metabo_fltr_stg2.execute_quality_filtering(
        idx_mar=filtered_st1.attrs.get("idx_mar"),
        idx_mnar=filtered_st1.attrs.get("idx_mnar"),
        output_dir=step7_dir
    )

    # ====================================================================
    # Step 8: Quality Assessment (Post-Filtering)
    # ====================================================================
    step8_dir = os.path.join(output_dir, "08_QA_Filtered_Data")
    logger.info("Step 8: Conducting QA on Post-Filtered Data...")
    
    qa_filtered = MetaboIntAssessor(
        quality_filtered_data, pipeline_params=params
    )
    qa_filtered.execute_assessment(output_dir=step8_dir)

    # ====================================================================
    # Step 9: Missing Value Imputation
    # ====================================================================
    step9_dir = os.path.join(output_dir, "09_Imputed_Data")
    logger.info("Step 9: Executing simulation-guided imputation...")
    
    metabo_imp = MetaboIntImputer(
        quality_filtered_data, pipeline_params=params
    )
    imputed_data = metabo_imp.execute_imputation(output_dir=step9_dir) 

    # ====================================================================
    # Step 10: Quality Assessment (Post-Imputation)
    # ====================================================================
    step10_dir = os.path.join(output_dir, "10_QA_Imputed_Data")
    logger.info("Step 10: Conducting QA on Imputed Data...")
    
    qa_imputed = MetaboIntAssessor(imputed_data, pipeline_params=params)
    qa_imputed.execute_assessment(output_dir=step10_dir)

    # ====================================================================
    # Step 11: Data Normalization (Column & Row)
    # ====================================================================
    step11_dir = os.path.join(output_dir, "11_Normalized_Data")
    logger.info("Step 11: Executing data normalization and scaling...")
    
    metabo_norm = MetaboIntNormalizer(imputed_data, pipeline_params=params)
    col_data, normalized_data = metabo_norm.execute_normalization(
        output_dir=step11_dir)

    # ====================================================================
    # Step 12: QA on Column-Normalized Data
    # ====================================================================
    step12_dir = os.path.join(output_dir, "12_QA_Column_Normalized")
    logger.info("Step 12: Conducting QA on Column-Normalized Data...")
    
    qa_col = MetaboIntAssessor(col_data, pipeline_params=params)
    qa_col.execute_assessment(output_dir=step12_dir)

    # ====================================================================
    # Step 13: QA on Fully Normalized Data (Final Assessment)
    # ====================================================================
    step13_dir = os.path.join(output_dir, "13_QA_Final_Normalized")
    logger.info("Step 13: Conducting final QA on Fully Normalized Data...")
    
    qa_final = MetaboIntAssessor(normalized_data, pipeline_params=params)
    qa_final.execute_assessment(output_dir=step13_dir)

    # ====================================================================
    # Final Step: Output Archiving
    # ====================================================================
    if compress_output:
        logger.info("Finalizing: Compressing pipeline outputs...")
        zip_path = os.path.join(
            os.path.dirname(output_dir), 
            f"{os.path.basename(output_dir)}_Archived.zip"
        )
        iu._zip_folder(source_folder=output_dir, output_path=zip_path)

    logger.success("PI-METABOQC PIPELINE COMPLETED SUCCESSFULLY.")
    
    return normalized_data