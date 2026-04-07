
#src/pimqc/dataset_builder.py
"""
Purpose of script: 
    Build a standardized MetaboInt object from metadata and intensity matrices.
"""

import os
import pandas as pd
from typing import Optional, Dict, Any

from . import io_utils as iu
from .core_classes import MetaboInt


@iu._exe_time
def build_dataset(
    meta_info: pd.DataFrame,
    int_df: pd.DataFrame,
    pipeline_params: Optional[Dict[str, Any]] = None,
    mode: str = "POS",
    batch: str = "Batch",
    sample_type: str = "Sample Type",
    bio_group: str = "Bio Group",
    sample_name: str = "Sample Name",
    inject_order: str = "Inject Order",
    output_dir: Optional[str] = None
) -> MetaboInt:
    """Merge metadata and intensity dataframes into a MetaboInt object.

    This function serves as the entry point of the pimqc pipeline. It verifies 
    the consistency between sample names, checks for completeness of required 
    metadata columns, constructs a MultiIndex pandas DataFrame, and finally 
    initializes and returns a MetaboInt core object for downstream analysis.

    Args:
        meta_info (pd.DataFrame): Project metadata dataframe. Each column 
            represents one property (e.g., Sample Name, Batch, Sample Type).
        int_df (pd.DataFrame): Raw intensity dataframe of metabolomics 
            (features * samples).
        pipeline_params (Optional[Dict[str, Any]], optional): 
            Global pipeline settings, parsed from JSON/YAML. Defaults to None.
        mode (str, optional): Polarity mode of MS ("POS" or "NEG"). 
            Defaults to "POS".
        batch (str, optional): Column name for analytical batch. 
            Defaults to "Batch".
        sample_type (str, optional): Column name for sample type. 
            Defaults to "Sample Type".
        bio_group (str, optional): Column name for biological group. 
            Defaults to "Bio Group".
        sample_name (str, optional): Column name for sample names. 
            Defaults to "Sample Name".
        inject_order (str, optional): Column name for injection order. 
            Defaults to "Inject Order".
        output_dir (Optional[str], optional): Directory to save the generated 
            raw intensity CSV. Defaults to None.

    Returns:
        MetaboInt: A multi-index standard MetaboInt object ready for 
            downstream analysis.

    Raises:
        AssertionError: If duplicate sample names exist in the intensity 
            dataframe.
        AssertionError: If sample names between metadata and intensity dataframe
            do not match (Jaccard Score != 1.0).
        AssertionError: If required metadata columns are missing.
    """
    # 1. Check duplicate sample names in the intensity dataframe.
    assert (
        int_df.columns.value_counts().max() <= 1
    ), "Duplicate sample name detected in the intensity dataframe."

    # 2. Check consistency of sample names between intensity dataframe and meta information.
    s1 = set(meta_info[sample_name])
    s2 = set(int_df.columns)
    jaccard_score = len(s1.intersection(s2)) / len(s1.union(s2))
    assert jaccard_score == 1.0, (
        f"Inconsistency of sample names between '{sample_name}' column "
        f"of meta info and column names of intensity info. "
        f"Jaccard-Score is {jaccard_score:.4f} (Must be 1.0)."
    )

    # 3. Check completion of given meta information.
    meta_info_dict = {
        "Batch Name": batch,
        "Sample Type": sample_type,
        "Sample Name": sample_name,
        "Inject Order": inject_order
    }
    # Only check 'Bio Group' if it is passed and not NA
    if pd.notna(bio_group) and bio_group in meta_info.columns:
        meta_info_dict["Bio Group"] = bio_group
        
    assert_dict = {
        k: v for k, v in meta_info_dict.items() if v not in meta_info.columns
    }
    assert len(assert_dict) == 0, f"Incomplete project meta info. Missing columns: {assert_dict}."

    # 4. Merge int_df and meta_info to construct the multi-index matrix.
    int_df = int_df.rename_axis(index=["Metabolite"], columns=[sample_name])
    column_df = int_df.columns.to_frame().reset_index(drop=True)
    column_df = pd.merge(
        left=column_df, right=meta_info, on=sample_name, how="left")
    
    column_order = (
        [batch, sample_type, bio_group, inject_order, sample_name]
        if ("Bio Group" in meta_info_dict.keys())
        else [batch, sample_type, inject_order, sample_name]
    )
    
    int_df.columns = pd.MultiIndex.from_frame(column_df.loc[:, column_order])
    
    # 5. Filter out samples that exist in intensity matrix but have null sample names (if any).
    int_df = int_df.loc[
        :, int_df.columns.get_level_values(level=sample_name).notnull()]
    
    # 6. Save intermediate raw data if output directory is provided.
    if output_dir:
        iu._check_dir_exists(dir_path=output_dir, handle="makedirs")
        output_path = os.path.join(
            output_dir, f"Metabolomics_Intensity_Raw_{mode}.csv")
        int_df.to_csv(
            path_or_buf=output_path, na_rep="NA", encoding="utf-8-sig")
        
    # 7. Instantiate and return the MetaboInt core object.
    metabo_obj = MetaboInt(
        int_df,
        pipeline_params=pipeline_params,
        mode=mode,
        sample_name=sample_name,
        sample_type=sample_type,
        bio_group=bio_group,
        batch=batch,
        inject_order=inject_order
    )
    
    return metabo_obj