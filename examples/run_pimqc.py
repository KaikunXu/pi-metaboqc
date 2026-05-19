#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Orchestration CLI for pi-metaboqc Pipeline.

This script provides a standardized command-line interface (CLI) to 
execute the full metabolomics quality control workflow in a non-interactive,
headless terminal environment. 

Usage:
    python run_pimqc.py [-m META] [-i INTENSITY] [-c CONFIG] [-o OUTDIR] [-q]
    
    # 1. Run with default demo data
    python run_pimqc.py

    # 2. Run with custom clinical data
    python run_pimqc.py -m /path/to/meta.csv \\
                        -i /path/to/intensity.csv \\
                        -c /path/to/config.toml \\
                        -o /path/to/output_dir
                        
    # 3. Run in silent mode (For background processing)
    python run_pimqc.py -q
"""

import os
import sys
import argparse
import pandas as pd
from loguru import logger

# =============================================================================
# PRE-FLIGHT CONFIGURATION & BACKEND INITIALIZATION
# -----------------------------------------------------------------------------
# Explicitly set the Matplotlib backend to 'Agg' (Anti-Grain Geometry).
# This is critical for headless server-side execution to prevent GUI 
# dependencies and X11 display-related crashes during automated plotting.
# =============================================================================
import matplotlib as mpl
mpl.use("Agg")

# Import internal modules after backend initialization
import pimqc
import pimqc.io_utils as iu
from pimqc.pipeline import run_pipeline



def parse_arguments():
    """
    Constructs and parses the command-line arguments for the CLI.
    
    Returns:
        argparse.Namespace: An object containing parsed arguments.
    """
    # Resolve relative base directories for default path construction
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "src", "pimqc", "data")
    default_out = os.path.join(script_dir, "tutorial_output")

    parser = argparse.ArgumentParser(
        description=(
            "pi-metaboqc CLI: Automated Metabolomics QC Pipeline. "
            "Optimized for headless deployment and batch processing."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-m", "--meta",
        type=str,
        default=os.path.join(data_dir, "project_meta.csv"),
        help="Path to the metadata CSV file."
    )
    parser.add_argument(
        "-i", "--intensity",
        type=str,
        default=os.path.join(data_dir, "project_intensity.csv"),
        help="Path to the intensity matrix CSV file."
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=os.path.join(data_dir, "pipeline_parameters.toml"),
        help="Path to the configuration TOML file."
    )
    parser.add_argument(
        "-o", "--outdir",
        type=str,
        default=default_out,
        help="Directory for exporting analytical results and reports."
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Enable silent mode: suppress all console log outputs."
    )

    return parser.parse_args()


def main():
    """
    Main execution function orchestrating data ingestion and pipeline flow.
    """
    # 1. Parse configuration arguments
    args = parse_arguments()
    logger.info(f"pimqc.__version__:{pimqc.__version__}")

    # 2. Setup Silent Mode (just show error logger)
    if args.quiet:
        pimqc.init(check_hardware=False, log_level="ERROR", show_progress=False)
    else:
        pimqc.init(check_hardware=True, log_level="INFO", show_progress=True)
        
    # 3. Path Normalization (The 'clean_xx' convention)
    # -------------------------------------------------------------------------
    # We transform all input paths into clean, absolute physical paths.
    # This eliminates redundant '..' segments and ensures elegant logging.
    clean_meta = os.path.abspath(args.meta)
    clean_intensity = os.path.abspath(args.intensity)
    clean_config = os.path.abspath(args.config)
    clean_outdir = os.path.abspath(args.outdir)

    # 4. Professional Execution Header
    logger.info("=" * 79)
    logger.info("🚀 Starting pi-metaboqc Automated Quality Control Pipeline")
    logger.info("-" * 79)
    logger.info(f">>> Metadata   : {clean_meta}")
    logger.info(f">>> Intensity  : {clean_intensity}")
    logger.info(f">>> Config     : {clean_config}")
    logger.info(f">>> Output Dir : {clean_outdir}")
    logger.info("=" * 79)

    # 5. Workspace Initialization
    try:
        # Safely create or verify the target output directory mount
        iu._check_dir_exists(dir_path=clean_outdir, handle="makedirs")
    except Exception as e:
        logger.error(f"Failed to initialize output workspace: {e}")
        sys.exit(1)

    # 6. Data & Parameter Ingestion
    try:
        logger.info("Ingesting configuration and raw data matrices...")
        
        # Load analytical parameters via io_utils
        params = iu.load_pipeline_config(config_path=clean_config)
        
        # Ingest metadata (Sample info, Batch, Injection Order)
        meta_df = pd.read_csv(clean_meta, header=[0])
        
        # Ingest the feature intensity peak table (Feature IDs as index)
        int_df = pd.read_csv(clean_intensity, index_col=[0], header=[0])
        
    except FileNotFoundError as fnf_err:
        logger.error(f"File not found: {fnf_err}")

        sys.exit(1)
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        sys.exit(1)

    # 7. Pipeline Orchestration
    try:
        logger.info("Triggering algorithmic core. Processing...")
        
        # Execute the unified runner
            # Data Builder -> MV-related Filtering -> Correction 
            # -> Quality-related Filtering -> Imputation -> Normalization
            # -> Report Generation
            
        run_pipeline(
            meta_df=meta_df,
            int_df=int_df,
            params=params,
            output_dir=clean_outdir
        )
        
        logger.success("=" * 79)
        logger.success("Execution successful! Audit reports generated successfully.")
        logger.success(f"   --> {clean_outdir}")
        logger.success("=" * 79)
        
    except Exception as e:
        logger.error(f"Pipeline runtime exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Entry point for CLI execution
    main()