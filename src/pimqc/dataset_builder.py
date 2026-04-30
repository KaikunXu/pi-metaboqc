# src/pimqc/dataset_builder.py
"""
Purpose of script: 
    Build a standardized MetaboInt object from metadata and intensity matrices.
"""

import os
import pandas as pd
from loguru import logger
from typing import Optional, Dict, Any, Union

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
    resort_inject_order: Optional[Union[str, bool]] = "auto",
    output_dir: Optional[str] = None
) -> MetaboInt:
    """Merge metadata and intensity dataframes into a MetaboInt object.

    This function serves as the entry point of the pi-metaboqc pipeline. It 
    verifies consistency, resolves duplicate feature names, manages injection 
    order continuity across batches, and returns a initialized MetaboInt object.

    Args:
        meta_info: Project metadata dataframe.
        int_df: Raw intensity dataframe (features * samples).
        pipeline_params: Global pipeline settings. Defaults to None.
        mode: Polarity mode of MS ("POS" or "NEG"). Defaults to "POS".
        batch: Column name for analytical batch. Defaults to "Batch".
        sample_type: Column name for sample type. Defaults to "Sample Type".
        bio_group: Column name for biological group. Defaults to "Bio Group".
        sample_name: Column name for sample names. Defaults to "Sample Name".
        inject_order: Column name for injection order.
        resort_inject_order: Strategy to re-number injection orders across
            multiple batches. Options: 'auto' (only resort if overlap detected),
            True/'force' (always force strict sequential continuity), 
            or None/False (disable resorting). Defaults to 'auto'.
        output_dir: Directory to save generated CSV. Defaults to None.

    Returns:
        MetaboInt: A multi-index standard MetaboInt object.
    """
    if isinstance(int_df.index, pd.RangeIndex):
        logger.warning(
            "Intensity dataframe has a RangeIndex. "
            f'Automatically setting the first column "{int_df.columns[0]}" '
            "as the feature index."
        )
        int_df = int_df.set_index(int_df.columns[0])
    
    int_df.index.name = "Metabolite"

    # 1. Feature correction: Resolve duplicate metabolite names (Row Indices)
    if int_df.index.duplicated().any():
        num_dups = int_df.index.duplicated().sum()
        dup_feats = int_df.index[int_df.index.duplicated()].unique().tolist()
        log_feats = dup_feats[:5] + ["..."] if len(dup_feats) > 5 else dup_feats
        
        logger.warning(
            f"Detected {num_dups} duplicate row indices (metabolite names) "
            f"such as: {log_feats}. Merging their intensities by summation to "
            f"ensure strict mathematical uniqueness."
        )
        int_df = int_df.groupby(level=0, sort=False).sum()

    # 2. Check duplicate sample names in the intensity dataframe.
    val_counts = int_df.columns.value_counts()
    if val_counts.max() > 1:
        dup_names = val_counts[val_counts > 1].index.tolist()
        dup_details = []
        
        for name in dup_names:
            # Find all column indices matching the duplicated name
            locs = [i for i, col in enumerate(int_df.columns) if col == name]
            dup_details.append(f'"{name}" (indices: {locs})')
            
        err_msg = (
            "Duplicate sample names detected in the intensity dataframe: "
            f'{", ".join(dup_details)}.'
        )
        raise AssertionError(err_msg)

    # 3. Check consistency of sample names between intensity dataframe and meta.
    meta_samples = set(meta_info[sample_name])
    int_samples = set(int_df.columns)
    
    if meta_samples != int_samples:
        only_in_meta = sorted(list(meta_samples - int_samples))
        only_in_int = sorted(list(int_samples - meta_samples))
        
        # Calculate Jaccard score for the error message
        intersection_size = len(meta_samples.intersection(int_samples))
        union_size = len(meta_samples.union(int_samples))
        j_score = intersection_size / union_size
        
        err_msg_parts = [
            f"Sample name inconsistency detected (Jaccard Score: {j_score:.4f})."
        ]
        
        if only_in_meta:
            log_meta = (
                only_in_meta[:5] + ["..."] 
                if len(only_in_meta) > 5 else only_in_meta)
            err_msg_parts.append(f"Samples only in Metadata: {log_meta}")
            
        if only_in_int:
            log_int = (
                only_in_int[:5] + ["..."] 
                if len(only_in_int) > 5 else only_in_int)
            err_msg_parts.append(f"Samples only in Intensity: {log_int}")
            
        raise AssertionError(" ".join(err_msg_parts))

    # 4. Check completion of given meta information.
    required_cols_map = {
        "Batch Name": batch,
        "Sample Type": sample_type,
        "Sample Name": sample_name,
        "Inject Order": inject_order
    }
    
    # Identify which expected column names are missing from meta_info
    missing_cols = [
        col_name for label, col_name in required_cols_map.items() 
        if col_name not in meta_info.columns
    ]
    
    if missing_cols:
        raise AssertionError(
            "Incomplete project metadata. The following required columns are "
            f"missing from the provided dataframe: {missing_cols}."
        )
        
    unique_batches = meta_info[batch].unique()
    is_multi_batch = len(unique_batches) > 1
    
    # 5. Injection Order Management: Resolve overlaps or enforce continuity
    if is_multi_batch and (
        inject_order in meta_info.columns) and resort_inject_order:
        meta_info[inject_order] = pd.to_numeric(
            meta_info[inject_order], errors="coerce"
        )
        ordered_batches = sorted(meta_info[batch].dropna().unique().tolist())
        
        is_overlap = False
        current_max = -float("inf")
        for b in ordered_batches:
            b_mask = meta_info[batch] == b
            if b_mask.sum() == 0:
                continue
            b_min = meta_info.loc[b_mask, inject_order].min()
            b_max = meta_info.loc[b_mask, inject_order].max()
            if b_min <= current_max:
                is_overlap = True
                break
            current_max = max(current_max, b_max)
            
        # Determine whether to trigger the sequence alignment algorithm
        trigger_resort = False
        if resort_inject_order == "auto" and is_overlap:
            trigger_resort = True
        elif str(resort_inject_order).lower() in ["force", "true", "always"]:
            trigger_resort = True
            
        if trigger_resort:
            logger.warning(
                f"Inject orders resort triggered (mode: {resort_inject_order})."
                " Re-numbering sequentially to ensure global continuity."
            )
            prev_max = None
            for b in ordered_batches:
                b_mask = meta_info[batch] == b
                if b_mask.sum() == 0:
                    continue
                
                b_min = meta_info.loc[b_mask, inject_order].min()
                if prev_max is not None:
                    exact_offset = prev_max - b_min + 1
                    meta_info.loc[b_mask, inject_order] += exact_offset
                    
                prev_max = meta_info.loc[b_mask, inject_order].max()

    # 6. Merge int_df and meta_info to construct the multi-index matrix.
    int_df = int_df.rename_axis(index=["Metabolite"], columns=[sample_name])
    column_df = int_df.columns.to_frame().reset_index(drop=True)
    column_df = pd.merge(
        left=column_df, right=meta_info, on=sample_name, how="left")
    
    # Check if bio_group is valid and exists to determine column hierarchy
    has_bio_group = pd.notna(bio_group) and bio_group in meta_info.columns
    column_order = (
        [batch, sample_type, bio_group, inject_order, sample_name]
        if has_bio_group
        else [batch, sample_type, inject_order, sample_name]
    )
    
    int_df.columns = pd.MultiIndex.from_frame(column_df.loc[:, column_order])
    
    # 7. Filter out samples that exist in intensity matrix but have null names.
    int_df = int_df.loc[
        :, int_df.columns.get_level_values(level=sample_name).notnull()]
    
    # 8. Save intermediate raw data if output directory is provided.
    if output_dir:
        iu._check_dir_exists(dir_path=output_dir, handle="makedirs")
        output_path = os.path.join(
            output_dir, f"Metabolomics_Intensity_Raw.csv")
        int_df.to_csv(
            path_or_buf=output_path, na_rep="NA", encoding="utf-8-sig")
        logger.info(f"MetaboInt object saved as: {output_path}")

    # 9. Instantiate and return the MetaboInt core object.
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
    
    metabo_obj.attrs["is_multi_batch"] = is_multi_batch
    metabo_obj.attrs["batch_list"] = unique_batches.tolist()

    logger.info(
        f"MetaboInt object built: {metabo_obj.shape[0]} metabolites, "
        f"{metabo_obj.shape[1]} samples.")

    return metabo_obj