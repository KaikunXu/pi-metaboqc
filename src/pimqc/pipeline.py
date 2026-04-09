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
from .assessment import MetaboIntAssessor
from .filtering import MetaboIntFilter
from .correction import MetaboIntCorrector
from .normalization import MetaboIntNormalizer
from .imputation import MetaboIntImputer


@iu._exe_time
def run_pipeline(
    meta_info_path: str,
    int_df_path: str,
    output_dir: str,
    params_path: Optional[str] = None,
    time_suffix: bool = True,
    compress_output: bool = True,
) -> None:
    """Execute the complete metabolomics data quality control pipeline.

    Args:
        meta_info_path: File path to the sample metadata CSV.
        int_df_path: File path to the raw intensity matrix CSV.
        output_dir: Base directory path for saving pipeline results.
        params_path: File path to the JSON/YAML configuration file.
        time_suffix: Whether to append a timestamp to the output directory.
        compress_output: Whether to zip the final output folder.
    """
    
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
    logger.info("INITIALIZING PI-METABOQC PIPELINE")
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
        logger.critical("Parameter file must be either .json or .yaml.")
        raise ValueError("Parameter file must be either .json or .yaml.")
        
    logger.success(f"Parameter file loaded: {os.path.basename(params_path)}.")

    meta_info = pd.read_csv(filepath_or_buffer=meta_info_path, header=0)
    int_df = pd.read_csv(filepath_or_buffer=int_df_path, header=0)

    # ---------------------------------------------------------
    # Step 1: Build Standardized Dataset
    # ---------------------------------------------------------
    logger.info("Step 1: Building standardized MetaboInt dataset...")
    step1_dir = os.path.join(output_dir, "01_Raw_Data")
    
    raw_data = build_dataset(
        meta_info=meta_info,
        int_df=int_df,
        pipeline_params=pipeline_params,
        mode=pipeline_params.get("MetaboInt", {}).get("mode", "POS"),
        batch=pipeline_params.get("MetaboInt", {}).get("batch", "Batch"),
        sample_type=pipeline_params.get(
            "MetaboInt", {}).get("sample_type", "Sample Type"
        ),
        bio_group=pipeline_params.get(
            "MetaboInt", {}).get("bio_group", "Bio Group"
        ),
        sample_name=pipeline_params.get(
            "MetaboInt", {}).get("sample_name", "Sample Name"
        ),
        inject_order=pipeline_params.get(
            "MetaboInt", {}).get("inject_order", "Inject Order"
        ),
        output_dir=step1_dir
    )

    # ---------------------------------------------------------
    # Step 2: Quality Assessment (Raw Data)
    # ---------------------------------------------------------
    logger.info("Step 2: Conducting QA on Raw Data...")
    step2_dir = os.path.join(output_dir, "02_QA_Raw_Data")
    
    qa_raw = MetaboIntAssessor(raw_data, pipeline_params=pipeline_params)
    qa_raw.execute_qa(output_dir=step2_dir)

    # ---------------------------------------------------------
    # Step 3: High Missing Value (MV) Feature Filter (Stage 1)
    # ---------------------------------------------------------
    logger.info("Step 3: Filtering high MV features (Stage 1)...")
    
    metabo_fltr_stg1 = MetaboIntFilter(raw_data, pipeline_params=pipeline_params)
    filtered_data = metabo_fltr_stg1.execute_mv_fltr()

    # ---------------------------------------------------------
    # Step 4: Signal Drift & Batch Effect Correction
    # ---------------------------------------------------------
    logger.info("Step 4: Performing signal drift correction...")
    step4_dir = os.path.join(output_dir, "04_Corrected_Data")
    
    metabo_sc = MetaboIntCorrector(filtered_data, pipeline_params=pipeline_params)
    intra_data, inter_data = metabo_sc.execute_sc(output_dir=step4_dir)

    # ---------------------------------------------------------
    # Step 5: QA (Intra-batch Corrected Data)
    # ---------------------------------------------------------
    logger.info("Step 5: Conducting QA on Intra-batch Corrected Data...")
    step5_dir = os.path.join(output_dir, "05_QA_Intra_Corrected")
    
    qa_intra = MetaboIntAssessor(intra_data, pipeline_params=pipeline_params)
    qa_intra.execute_qa(output_dir=step5_dir)

    # ---------------------------------------------------------
    # Step 6: QA (Inter-batch Corrected Data)
    # ---------------------------------------------------------
    logger.info("Step 6: Conducting QA on Inter-batch Corrected Data...")
    step6_dir = os.path.join(output_dir, "06_QA_Inter_Corrected")
    
    qa_inter = MetaboIntAssessor(inter_data, pipeline_params=pipeline_params)
    qa_inter.execute_qa(output_dir=step6_dir)

    # ---------------------------------------------------------
    # Step 7: Low Quality Feature Filter (Stage 2: Blank & RSD)
    # ---------------------------------------------------------
    logger.info("Step 7: Filtering low quality features (Stage 2)...")
    step7_dir = os.path.join(output_dir, "07_Filtered_Quality")
    
    metabo_fltr_stg2 = MetaboIntFilter(
        inter_data, pipeline_params=pipeline_params
    )
    
    # Placeholder indices since imputation occurs later
    idx_mar = inter_data.index
    idx_mnar = pd.Index([])
    
    quality_filtered_data = metabo_fltr_stg2.execute_quality_fltr(
        idx_mar=idx_mar, idx_mnar=idx_mnar
    )

    # ---------------------------------------------------------
    # Step 8: Quality Assessment (Post-Filtering)
    # ---------------------------------------------------------
    logger.info("Step 8: Conducting QA on Post-Filtered Data...")
    step8_dir = os.path.join(output_dir, "08_QA_Filtered_Data")
    
    qa_filtered = MetaboIntAssessor(
        quality_filtered_data, pipeline_params=pipeline_params
    )
    qa_filtered.execute_qa(output_dir=step8_dir)

    # ---------------------------------------------------------
    # Step 9: Imputation
    # ---------------------------------------------------------
    logger.info("Step 9: Executing missing value imputation...")
    
    metabo_imp = MetaboIntImputer(
        quality_filtered_data, pipeline_params=pipeline_params
    )
    imputed_data = metabo_imp.execute_probabilistic_imputation()

    # ---------------------------------------------------------
    # Step 10: Quality Assessment (Post-Imputation)
    # ---------------------------------------------------------
    logger.info("Step 10: Conducting QA on Imputed Data...")
    step10_dir = os.path.join(output_dir, "10_QA_Imputed_Data")
    
    qa_imputed = MetaboIntAssessor(imputed_data, pipeline_params=pipeline_params)
    qa_imputed.execute_qa(output_dir=step10_dir)

    # ---------------------------------------------------------
    # Step 11: Data Normalization
    # ---------------------------------------------------------
    logger.info("Step 11: Executing data normalization and scaling...")
    step11_dir = os.path.join(output_dir, "11_Normalized_Data")
    
    metabo_norm = MetaboIntNormalizer(imputed_data, pipeline_params=pipeline_params)
    normalized_data = metabo_norm.execute_norm(output_dir=step11_dir)

    # ---------------------------------------------------------
    # Step 12: Quality Assessment (Final)
    # ---------------------------------------------------------
    logger.info("Step 12: Conducting final QA on Normalized Data...")
    step12_dir = os.path.join(output_dir, "12_QA_Normalized_Data")
    
    qa_final = MetaboIntAssessor(normalized_data, pipeline_params=pipeline_params)
    qa_final.execute_qa(output_dir=step12_dir)

    # ---------------------------------------------------------
    # Step 13: Output Compression
    # ---------------------------------------------------------
    if compress_output:
        logger.info("Step 13: Compressing pipeline outputs...")
        zip_path = os.path.join(
            os.path.dirname(output_dir), 
            f"{os.path.basename(output_dir)}_Archived.zip"
        )
        iu._zip_folder(source_folder=output_dir, output_path=zip_path)

    logger.success("PI-METABOQC PIPELINE COMPLETED SUCCESSFULLY.")
    logger.info("=" * 60)

if __name__ == "__main__":
    pass