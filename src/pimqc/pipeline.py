# src/pimqc/pipeline.py
"""
Purpose of script: Main execution pipeline for pi-metaboqc.
"""

import os
import sys
import pandas as pd
from datetime import datetime
from typing import Optional
from loguru import logger

from . import io_utils as iu
from .dataset_builder import build_dataset
from .data_quality_assessment import MetaboIntQA
from .invalid_feature_sample_filtering import MetaboIntFLTR
from .data_signal_correction import MetaboIntSC
from .normalization import MetaboIntNorm


@iu._exe_time
def run_pipeline(
    meta_info_path: str,
    int_df_path: str,
    output_dir: str,
    params_path: Optional[str] = None,
    time_suffix: bool = True,
    compress_output: bool = True,
) -> None:
    """Execute the complete metabolomics data quality control pipeline."""
    
    # ---------------------------------------------------------
    # Step 0: Preset paths, configure logger, and load parameters
    # ---------------------------------------------------------
    
    if time_suffix:
        output_dir = f"{output_dir}_{datetime.now().strftime('%Y%m%d')}"
    iu._check_dir_exists(dir_path=output_dir, handle="makedirs")

    for hid in set(logger._core.handlers.keys()):
        logger.remove(handler_id=hid)
        
    logger.add(sink=sys.stdout, level="INFO", backtrace=True, colorize=True)
    logger.add(
        sink=os.path.join(output_dir, "logging_pi_metaboqc.log"),
        format="".join([
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name} | ",
            "{function}: {line} - {message}"
        ]),
        encoding="utf-8-sig",
        enqueue=True, colorize=True, backtrace=True, mode="w", level="TRACE"
    )

    logger.info("=" * 60)
    logger.info("🚀 INITIALIZING PI-METABOQC PIPELINE")
    logger.info(f"Output directory set to:\n\t{output_dir}")
    logger.info("=" * 60)

    if params_path is None:
        params_path = os.path.join(
            os.path.dirname(__file__), 
            "data", 
            "pipeline_parameters.json"
        )
        logger.info(f"Using default pipeline parameters: {params_path}")
    for file_path in [meta_info_path, int_df_path, params_path]:
        iu._check_file_exists(file_path=file_path)

    if params_path.endswith(".json"):
        pipeline_params = iu._load_json_file(input_file_path=params_path)
    elif params_path.endswith((".yaml", ".yml")):
        pipeline_params = iu._load_yaml_file(input_file_path=params_path)
    else:
        logger.critical("Parameter file must be either .json or .yaml format.")
        raise ValueError("Parameter file must be either .json or .yaml format.")
        
    logger.success(f"Parameter file successfully loaded: {os.path.basename(params_path)}.")

    meta_info = pd.read_csv(filepath_or_buffer=meta_info_path, header=0)
    int_df = pd.read_csv(filepath_or_buffer=int_df_path, header=0)

    # ---------------------------------------------------------
    # Step 1: Build Standardized Dataset
    # ---------------------------------------------------------
    logger.info("▶ Step 1: Building standardized MetaboInt dataset...")
    step1_dir = os.path.join(output_dir, "01_Raw_Data")
    
    raw_data = build_dataset(
        meta_info=meta_info,
        int_df=int_df,
        pipeline_params=pipeline_params,
        mode=pipeline_params.get(
            "MetaboInt", {}).get("mode", "POS"),
        batch=pipeline_params.get(
            "MetaboInt", {}).get("batch", "Batch"),
        sample_type=pipeline_params.get(
            "MetaboInt", {}).get("sample_type", "Sample Type"),
        bio_group=pipeline_params.get(
            "MetaboInt", {}).get("bio_group", "Bio Group"),
        sample_name=pipeline_params.get(
            "MetaboInt", {}).get("sample_name", "Sample Name"),
        inject_order=pipeline_params.get(
            "MetaboInt", {}).get("inject_order", "Inject Order"),
        output_dir=step1_dir
    )

    # ---------------------------------------------------------
    # Step 2: Quality Assessment (Raw Data)
    # ---------------------------------------------------------
    logger.info("▶ Step 2: Conducting QA on Raw Data...")
    step2_dir = os.path.join(output_dir, "02_QA_Raw_Data")
    
    qa_raw = MetaboIntQA(raw_data, pipeline_params=pipeline_params)
    qa_raw.execute_qa(output_dir=step2_dir)

    # ---------------------------------------------------------
    # Step 3: Invalid Feature and Sample Filtering
    # ---------------------------------------------------------
    logger.info("▶ Step 3: Filtering invalid features and outlier samples...")
    step3_dir = os.path.join(output_dir, "03_Filtered_Data")
    
    metabo_fltr = MetaboIntFLTR(raw_data, pipeline_params=pipeline_params)
    filtered_data = metabo_fltr.execute_mv_fltr(output_dir=step3_dir)

    # ---------------------------------------------------------
    # Step 4: Quality Assessment (After Filtering)
    # ---------------------------------------------------------
    logger.info("▶ Step 4: Conducting QA on Filtered Data...")
    step4_dir = os.path.join(output_dir, "04_QA_Filtered_Data")
    
    qa_filtered = MetaboIntQA(filtered_data, pipeline_params=pipeline_params)
    qa_filtered.execute_qa(output_dir=step4_dir)

    # ---------------------------------------------------------
    # Step 5: Signal Drift & Batch Effect Correction
    # ---------------------------------------------------------
    logger.info("▶ Step 5: Performing signal drift and batch effect correction...")
    step5_dir = os.path.join(output_dir, "05_Corrected_Data")
    
    metabo_sc = MetaboIntSC(filtered_data, pipeline_params=pipeline_params)
    intra_data, inter_data = metabo_sc.execute_sc(output_dir=step5_dir)

    # ---------------------------------------------------------
    # Step 6: Quality Assessment (After Correction)
    # ---------------------------------------------------------
    logger.info("▶ Step 6: Conducting QA on Corrected Data...")
    step6_dir = os.path.join(output_dir, "06_QA_Corrected_Data")
    
    # [BUG FIXED] Use inter_data instead of undefined corrected_data
    qa_corrected = MetaboIntQA(inter_data, pipeline_params=pipeline_params)
    qa_corrected.execute_qa(output_dir=step6_dir)

    # ---------------------------------------------------------
    # Step 7: Quality Filtering (Stage 2 Filter: RSD & Blank)
    # ---------------------------------------------------------
    logger.info("▶ Step 7: Stage-2 Filtering (QC RSD & Blank Ratio)...")
    step7_dir = os.path.join(output_dir, "07_Filtered_Quality")
    
    metabo_qual_fltr = MetaboIntFLTR(inter_data, pipeline_params=pipeline_params)
    quality_filtered_data = metabo_qual_fltr.execute_quality_fltr(output_dir=step7_dir)

    # ---------------------------------------------------------
    # Step 8: Data Normalization
    # ---------------------------------------------------------
    logger.info("▶ Step 8: Executing data normalization (Sample & Feature dimensions)...")
    step8_dir = os.path.join(output_dir, "08_Normalized_Data")
    
    metabo_norm = MetaboIntNorm(quality_filtered_data, pipeline_params=pipeline_params)
    normalized_data = metabo_norm.execute_norm(output_dir=step8_dir)

    # ---------------------------------------------------------
    # Step 9: Quality Assessment (Final / After Normalization)
    # ---------------------------------------------------------
    logger.info("▶ Step 9: Conducting final QA on Normalized Data...")
    step9_dir = os.path.join(output_dir, "09_QA_Normalized_Data")
    
    qa_normalized = MetaboIntQA(normalized_data, pipeline_params=pipeline_params)
    qa_normalized.execute_qa(output_dir=step9_dir)

    # ---------------------------------------------------------
    # Step 10: Output Compression
    # ---------------------------------------------------------
    if compress_output:
        logger.info("▶ Step 10: Compressing pipeline outputs into a ZIP archive...")
        zip_path = os.path.join(
            os.path.dirname(output_dir), 
            f"{os.path.basename(output_dir)}_Archived.zip"
        )
        iu._zip_folder(source_folder=output_dir, output_path=zip_path)

    logger.success("✅ PI-METABOQC PIPELINE COMPLETED SUCCESSFULLY.")
    logger.info("=" * 60)

if __name__=="__main__":
    pass