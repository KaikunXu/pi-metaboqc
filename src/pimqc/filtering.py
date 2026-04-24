# src/pimqc/filtering.py
"""
Purpose of script: Invalid feature/sample filtering and visualization.
"""

import os
import copy
import numpy as np
import pandas as pd
from functools import cached_property

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from loguru import logger

from . import io_utils as iu
from . import plot_utils as pu
from . import core_classes
from . import visualizer_classes


class MetaboIntFilter(core_classes.MetaboInt):
    """Filtering engine subclassing MetaboInt for metabolomics datasets."""

    _metadata = ["stats", "params"]
    
    _INVALID_STRS = {
        "unknown", "na", "n/a", "nan", "none", "null", "",
        "unassigned", "blank", "blk", "is", "solvent", "wash",
        "sst", "pool", "invalid", "unvalid"
    }

    def __init__(self, data=None, pipeline_params=None, *args, **kwargs):
        """Initialize the filtering engine.

        Args:
            data: Input intensity data.
            pipeline_params: Global configuration dictionary.
            *args: Arguments passed to pandas DataFrame.
            **kwargs: Extra arguments passed to pandas DataFrame.
        """
        super().__init__(data=data, *args, **kwargs)

        if pipeline_params is not None:
            self.attrs["pipeline_parameters"] = pipeline_params

        self.params = self.attrs.get("pipeline_parameters", {})

        if "MetaboIntFilter" not in self.params:
            self.params["MetaboIntFilter"] = {}
        if "feature_counts" not in self.params["MetaboIntFilter"]:
            self.params["MetaboIntFilter"]["feature_counts"] = {}

        self.stats = {
            "mv_group_df": pd.DataFrame(),
            "mv_qc_series": pd.Series(dtype=float),
            "mv_global_series": pd.Series(dtype=float),
            "blank_mean": pd.Series(dtype=float),
            "qc_mean": pd.Series(dtype=float),
            "qc_rsd_all": pd.Series(dtype=float),
            "idx_mar": pd.Index([]),
            "idx_mnar": pd.Index([]),
            "idx_mnar_group": pd.Index([]),
            "idx_mnar_qc": pd.Index([])
        }

    @property
    def _constructor(self):
        """Override constructor to return MetaboIntFilter."""
        return MetaboIntFilter

    def __finalize__(self, other, method=None, **kwargs):
        """Deepcopy custom attributes during pandas operations."""
        self = super().__finalize__(other, method=method, **kwargs)
        for name in self._metadata:
            if hasattr(other, name):
                setattr(self, name, copy.deepcopy(getattr(other, name)))
        return self

    # =========================================================================
    # Internal Utility Methods
    # =========================================================================

    def _get_valid_bio_groups(self):
        """Extract valid biological group names from the column index."""
        meta_params = self.params.get("MetaboInt", {})
        grp_col = meta_params.get("bio_group", "Bio Group")
        sample_dict = meta_params.get("sample_dict", {})
        qc_lbl = sample_dict.get("QC sample", "QC")

        valid_bio_groups = []
        if grp_col in self.columns.names:
            raw_groups = self.columns.get_level_values(grp_col).unique()
            for g in raw_groups:
                if pd.isna(g):
                    continue
                g_str = str(g).strip().lower()
                if g_str in self._INVALID_STRS or g_str == str(qc_lbl).lower():
                    continue
                valid_bio_groups.append(g)

        return valid_bio_groups

    @iu._exe_time
    def classify_missing_types(self):
        """Classifies features into MAR, MNAR, and strictly dropped."""
        total_missing = self.isna().sum().sum()
        empty_idx = self.index[:0]
        if total_missing == 0:
            logger.warning(
                "No missing values detected. Defaulting all features to MAR."
            )
            return self.index, empty_idx, empty_idx
            
        try:
            meta_params = self.params.get("MetaboInt", {})
            grp_col = meta_params.get("bio_group", "Bio Group")
            qc_col = meta_params.get("sample_type", "Sample Type")
            qc_lbl = meta_params.get("sample_dict", {}).get("QC sample", "QC")
            
            fltr_params = self.params.get("MetaboIntFilter", {})
            
            # Extract critical topological filtering thresholds
            base_mv_tol = fltr_params.get("sample_mv_tol", 0.5) 
            grp_mv_tol = fltr_params.get("mnar_group_mv_tol", 0.8)
            qc_mv_tol = fltr_params.get("mnar_qc_mv_tol", 0.2)
            qc_int_pct = fltr_params.get("mnar_intensity_percentile", 0.1)

            idx_mnar_group = pd.Index([])
            idx_mnar_qc = pd.Index([])
            valid_bio_groups = self._get_valid_bio_groups()

            # ================================================================
            # 1. Identify MNAR candidates (Bio-group truncation)
            # ================================================================
            if valid_bio_groups:
                na_rate_grp = self.isna().T.groupby(level=grp_col).mean().T
                na_rate_valid = na_rate_grp[valid_bio_groups]
                
                cond_grp_mnar = (na_rate_valid >= grp_mv_tol).any(axis=1)
                idx_mnar_group = self.index[cond_grp_mnar]

            qc_mask = (
                self.columns.get_level_values(qc_col) == qc_lbl 
                if qc_col in self.columns.names 
                else np.zeros(self.shape[1], dtype=bool)
            )
            
            # ================================================================
            # 2. Identify MNAR candidates (QC low abundance validation)
            # ================================================================
            if qc_mask.any() and not self.index.empty:
                df_qc = self.loc[:, qc_mask]
                qc_na_rate = df_qc.isna().mean(axis=1)
                qc_median = df_qc.median(axis=1)
                int_threshold = qc_median.quantile(qc_int_pct)
                
                cond_qc_mv = qc_na_rate > qc_mv_tol
                cond_qc_int = qc_median <= int_threshold
                idx_mnar_qc = self.index[cond_qc_mv | cond_qc_int]

            # Consolidate confirmed MNAR features
            idx_mnar_all = idx_mnar_group.union(idx_mnar_qc)
            
            # ================================================================
            # 3. Identify MAR features (Must meet baseline health criteria)
            # ================================================================
            if valid_bio_groups:
                cond_healthy = (na_rate_valid <= base_mv_tol).any(axis=1)
            elif qc_mask.any():
                cond_healthy = self.stats["mv_qc_series"] <= base_mv_tol
            else:
                cond_healthy = self.stats["mv_global_series"] <= base_mv_tol

            idx_healthy = self.index[cond_healthy]
            
            # MAR features are healthy and mutually exclusive from MNAR
            idx_mar = idx_healthy.difference(idx_mnar_all)
            
            # ================================================================
            # 4. Identify invalid features to be dropped
            # ================================================================
            # Features failing both MAR health and MNAR truncation criteria
            retained_idx = idx_mar.union(idx_mnar_all)
            idx_dropped = self.index.difference(retained_idx)

            # Update engine state
            self.stats["idx_mar"] = idx_mar
            self.stats["idx_mnar"] = idx_mnar_all
            self.stats["idx_mnar_group"] = idx_mnar_group
            self.stats["idx_mnar_qc"] = idx_mnar_qc
            self.stats["idx_dropped_stage1"] = idx_dropped

            self.attrs["idx_mar"] = idx_mar.tolist()
            self.attrs["idx_mnar"] = idx_mnar_all.tolist()
            
            logger.info(
                f"Classification: {len(idx_mar)} MAR, {len(idx_mnar_all)} "
                f"MNAR, Dropping {len(idx_dropped)} invalid features."
            )
            
            return idx_mar, idx_mnar_all, idx_dropped

        except Exception as e:
            logger.error(f"Classification failed ({e}). Fallback to MAR.")
            return self.index, empty_idx, empty_idx


    # =========================================================================
    # Filtering Execution Stream
    # =========================================================================

    @iu._exe_time
    def execute_mv_filtering(self, output_dir: str = None) -> pd.DataFrame:
        """Executes Stage-1 missing value filter with defined topology.

        This engine orchestrates the missing value filtering process. It 
        pre-calculates global statistics for visualization, invokes the 
        topological classification engine (MAR/MNAR routing), physically 
        truncates invalid features, and exports diagnostic reports.
        """
        # 1. Parameter initialization and state tracking
        fc = self.params["MetaboIntFilter"]["feature_counts"]
        if "raw" not in fc:
            fc["raw"] = self.shape[0]

        meta_params = self.params.get("MetaboInt", {})
        grp_col = meta_params.get("bio_group", "Bio Group")
        group_order = meta_params.get("group_order", None)
        qc_col = meta_params.get("sample_type", "Sample Type")
        sample_dict = meta_params.get("sample_dict", {})
        qc_lbl = sample_dict.get("QC sample", "QC")
        
        valid_bio_groups = self._get_valid_bio_groups()

        # 2. Pre-calculate missing value rates for visualization
        qc_mask = (
            self.columns.get_level_values(qc_col) == qc_lbl 
            if qc_col in self.columns.names 
            else np.zeros(self.shape[1], dtype=bool)
        )

        if qc_mask.any():
            qc_na = self.loc[:, qc_mask].isna().mean(axis=1)
            self.stats["mv_qc_series"] = qc_na
            
        self.stats["mv_global_series"] = self.isna().mean(axis=1)
        
        if valid_bio_groups:
            filter_level = "Group"
            grp_na = self.isna().T.groupby(level=grp_col).mean().T
            self.stats["mv_group_df"] = grp_na[valid_bio_groups]
        elif qc_mask.any():
            filter_level = "QC"
        else:
            filter_level = "Global"

        logger.info(f"High missing value filter (Stage1): {filter_level}")

        # 3. Execute classification engine for topological routing
        idx_mar, idx_mnar, idx_dropped = self.classify_missing_types()
        
        # 4. Matrix truncation: Retain only verified MAR and MNAR
        retained_idx = idx_mar.union(idx_mnar)
        fc["post_stage1"] = len(retained_idx)
        df_final = self.loc[retained_idx].copy(deep=True)

        # 5. Export and Visualizations
        if output_dir:
            iu._check_dir_exists(output_dir, handle="makedirs")
            df_final.to_csv(
                os.path.join(output_dir, "MV_Filtered_Data.csv")
            )

            vis = MetaboVisualizerFilter(self)
            
            # Safely extract filtering parameters for plot thresholds
            fltr_params = self.params.get(
                "Filtering", 
                self.params.get("MetaboIntFilter", {})
            )

            plot_tasks = []
            if valid_bio_groups:
                g_tol = fltr_params.get("mnar_group_mv_tol", 0.8)
                fig_g = vis.plot_mv_group(
                    mv_df=self.stats["mv_group_df"], 
                    tol=g_tol,
                    group_order=group_order
                )
                plot_tasks.append((fig_g, "Group"))
                
            if qc_mask.any():
                q_tol = fltr_params.get("mnar_qc_mv_tol", 0.2)
                fig_q = vis.plot_mv_downgrade(
                    mv_series=self.stats["mv_qc_series"], 
                    level="QC",
                    tol=q_tol
                )
                plot_tasks.append((fig_q, "QC"))

            # Global baseline validation
            base_tol = fltr_params.get("sample_mv_tol", 0.5)
            fig_glob = vis.plot_mv_downgrade(
                mv_series=self.stats["mv_global_series"], 
                level="Global",
                tol=base_tol
            )
            plot_tasks.append((fig_glob, "Global"))

            # Render and export individual threshold diagnostic plots
            for fig_obj, lvl in plot_tasks:
                if fig_obj is not None:
                    vis.save_and_close_fig(
                        fig_obj,
                        os.path.join(output_dir, f"MV_Filter_{lvl}.pdf")
                    )

            # ================================================================
            # RESTORED: Summary and Classification Plots
            # ================================================================
            # 1. Missing Value Classification Distribution
            if hasattr(vis, "plot_missing_classification"):
                fig_class = vis.plot_missing_classification()
                if fig_class is not None:
                    vis.save_and_close_fig(
                        fig_class,
                        os.path.join(output_dir, "MV_Classification.pdf")
                    )

            # 2. Comprehensive Patchworklib Summary Grid
            # Fallback to check multiple naming conventions safely
            grid_method = getattr(
                vis, "plot_mv_filtering_summary_grid", 
                getattr(vis, "plot_quality_filtering_summary_grid", None)
            )
            
            if grid_method is not None:
                fig_grid = grid_method(filter_level=filter_level)
                if fig_grid is not None:
                    vis.save_and_show_pw(
                        fig_grid,
                        os.path.join(output_dir, "MV_Filtering_Summary.pdf")
                    )
        
        logger.success("High-MV features filtering completed.")
        return df_final

    @cached_property
    def mv_filtering_metrics(self):
        """Extracts metrics from the Stage-1 missing value filtering.

        Returns:
            Dict[str, Any]: A dictionary containing thresholds, counts, 
                retention stats, and explicit lists of dropped/retained 
                features for downstream audit trails.
        """
        # 1. Parameter extraction
        meta_params = self.params.get("MetaboInt", {})
        fltr_params = self.params.get("MetaboIntFilter", {})
        fc = fltr_params.get("feature_counts", {})
        
        # 2. Statelessly infer the active filtering level
        valid_groups = self._get_valid_bio_groups()
        qc_col = meta_params.get("sample_type", "Sample Type")
        qc_lbl = meta_params.get("sample_dict", {}).get("QC sample", "QC")
        
        has_qc = False
        if qc_col in self.columns.names:
            has_qc = (self.columns.get_level_values(qc_col) == qc_lbl).any()
            
        if valid_groups:
            fltr_level = "Group"
        elif has_qc:
            fltr_level = "QC"
        else:
            fltr_level = "Global"

        # 3. Extract indices from the Data Passport (stats)
        idx_mar = self.stats.get("idx_mar", pd.Index([]))
        idx_mnar = self.stats.get("idx_mnar", pd.Index([]))
        idx_mnar_grp = self.stats.get("idx_mnar_group", pd.Index([]))
        idx_mnar_qc = self.stats.get("idx_mnar_qc", pd.Index([]))

        dropped_idx = self.stats.get("idx_dropped_stage1", pd.Index([]))
        retained_idx = idx_mar.union(idx_mnar)

        # 4. Compute retention statistics safely
        raw_count = int(fc.get("raw", 0))
        retained_count = len(retained_idx)
        dropped_count = max(0, raw_count - retained_count)
        ret_rate = 0.0
        if raw_count > 0:
            ret_rate = round((retained_count / raw_count) * 100, 2)
        
        # 5. Compile structured JSON-serializable dictionary
        metrics = {
            "filtering_level": fltr_level,
            "thresholds": {
                "sample_mv_tol": fltr_params.get("sample_mv_tol", 0.5),
                "mnar_group_tol": fltr_params.get("mnar_group_mv_tol", 0.8),
                "mnar_qc_tol": fltr_params.get("mnar_qc_mv_tol", 0.2)
            },
            "missing_classification": {
                "mar_count": int(len(idx_mar)),
                "mnar_total": int(len(idx_mnar)),
                "mnar_group": int(len(idx_mnar_grp)),
                "mnar_qc": int(len(idx_mnar_qc))
            },
            "feature_retention": {
                "pre_mv_filter_count": raw_count,
                "after_mv_filter_count": retained_count,
                "dropped_count": dropped_count,
                "retention_rate_pct": ret_rate
            },
            # "feature_lists": {
            #     "dropped_features": list(dropped_idx),
            #     "mar_features": list(idx_mar),
            #     "mnar_features": list(idx_mnar)
            # }
        }
        
        return metrics

    @iu._exe_time
    def execute_quality_filtering(
        self, idx_mar=None, idx_mnar=None, output_dir=None):
        """Execute Stage-2 quality filter (Blank Ratio & QC RSD)."""
        if idx_mar is None:
            idx_mar = self.attrs.get("idx_mar")
        if idx_mnar is None:
            idx_mnar = self.attrs.get("idx_mnar")

        if idx_mar is None or idx_mnar is None:
            logger.warning(
                "MAR/MNAR indices not found in pipeline. Recomputing natively. "
                "Warning: If imputation was already applied, this will fail!"
            )
            idx_mar, idx_mnar = self.classify_missing_types()
        if not isinstance(idx_mar, pd.Index):
            idx_mar = pd.Index(idx_mar)
        if not isinstance(idx_mnar, pd.Index):
            idx_mnar = pd.Index(idx_mnar)
            
        self.stats["idx_mnar_group"] = self.attrs.get(
            "idx_mnar_group", pd.Index([])
        )
        self.stats["idx_mnar_qc"] = self.attrs.get(
            "idx_mnar_qc", pd.Index([])
        )
        self.stats["idx_mar"] = idx_mar
        self.stats["idx_mnar"] = idx_mnar

        fc = self.params["MetaboIntFilter"]["feature_counts"]
        meta_params = self.params.get("MetaboInt", {})
        
        qc_col = meta_params.get("sample_type", "Sample Type")
        sample_dict = meta_params.get("sample_dict", {})
        qc_lbl = sample_dict.get("QC sample", "QC")
        blk_lbl = sample_dict.get("Blank sample", "Blank")
        
        fltr_params = self.params.get("MetaboIntFilter", {})
        rsd_tol = fltr_params.get("rsd_qc_tol", 0.3)
        blk_tol = fltr_params.get("blank_qc_ratio", 0.2)

        qc_mask = self.columns.get_level_values(qc_col) == qc_lbl
        blk_mask = self.columns.get_level_values(qc_col) == blk_lbl
        current_idx = self.index
        logger.info(f"Features before filtering: {len(current_idx)}")

        # ==============================================================
        # 1. Blank Ratio Quality Check
        # ==============================================================
        if blk_mask.any() and qc_mask.any():
            logger.info("Low quality features filter (Stage2): Blank/QC check.")

            qc_m = self.loc[:, qc_mask].mean(axis=1)
            blk_m = self.loc[:, blk_mask].mean(axis=1)
            
            self.stats.update({"qc_mean": qc_m, "blank_mean": blk_m})
            max_ratio = blk_tol
            
            qc_safe = qc_m.replace(0, np.finfo(float).eps)
            pass_blk = blk_m[blk_m / qc_safe <= max_ratio].index

            next_idx = current_idx.intersection(pass_blk)
            self.stats["idx_dropped_blank"] = current_idx.difference(next_idx)
            current_idx = next_idx
            
            logger.info(f"Features after Blank/QC check: {len(current_idx)}")
        else:
            self.stats["idx_dropped_blank"] = pd.Index([])
        
        fc["post_stage2_blank"] = len(current_idx)
        # ==============================================================
        # 2. QC RSD Quality Check
        # ==============================================================
        if qc_mask.any():
            logger.info("Low quality features filter (Stage2): QC RSD check.")
            df_qc = self.loc[current_idx, qc_mask]
            std_qc = df_qc.std(axis=1, ddof=1)
            mean_qc = df_qc.mean(axis=1)
            
            self.stats["qc_rsd_all"] = std_qc / mean_qc
            
            pass_mar = self.stats["qc_rsd_all"].loc[
                idx_mar.intersection(current_idx)]
            final_idx = pass_mar[pass_mar <= rsd_tol].index.union(
                idx_mnar.intersection(current_idx))
            
            self.stats["idx_dropped_rsd"] = current_idx.difference(final_idx)
            
            logger.info(f"Features after QC RSD check: {len(final_idx)}")
        else:
            final_idx = current_idx
            self.stats["idx_dropped_rsd"] = pd.Index([])

        self.stats["idx_retained_stage2"] = final_idx
        fc["post_stage2_rsd"] = len(final_idx)
        df_final = self.loc[final_idx].copy()

        if output_dir:
            iu._check_dir_exists(dir_path=output_dir, handle="makedirs")
            mode = meta_params.get("mode", "POS")
            # ==============================================================
            df_final.attrs["pipeline_stage"] = "Low-quality features filtering"
            # ==============================================================
            csv_path = os.path.join(
                output_dir, f"Filtered_Data_Stage2_{mode}.csv"
            )
            df_final.to_csv(csv_path, encoding="utf-8-sig", na_rep="NA")
            logger.info(
                "Data after low-quality features filtering saved as: "
                f"{csv_path}")
            
            vis = MetaboVisualizerFilter(engine=df_final)

            try:
                fig_quality_fltr_summary = vis.plot_quality_filtering_summary_grid()
                if fig_quality_fltr_summary:
                    summary_path = os.path.join(
                        output_dir, f"Quality_Filtering_Grid_{mode}.pdf"
                    )
                    vis.save_and_show_pw(
                        pw_obj=fig_quality_fltr_summary,file_path=summary_path)
                    logger.info(
                        f"Low-quality features filter summary grid saved as: {summary_path}")
            except Exception as e:
                logger.error(f"Stage 2 summary grid generation failed: {e}")

            plot_tasks = [
                (vis.plot_qc_blank_scatter(), "Blank_Scatter"),
                (vis.plot_rsd_dist(idx_mnar=idx_mnar), "RSD_Dist"),
                (vis.plot_retained_count_steps(), "Retained_Count")
            ]
            
            for fig, name in plot_tasks:
                if fig:
                    pdf_path = os.path.join(
                        output_dir, f"Filter_Stage2_{name}_{mode}.pdf"
                    )
                    vis.save_and_close_fig(fig, pdf_path)

        logger.success("Low-quality features filtering completed.")
        return df_final

    @cached_property
    def quality_filtering_metrics(self) -> dict:
        """Extracts metrics from the Stage-2 low-quality feature filtering.

        Returns:
            Dict[str, Any]: A dictionary containing the blank/RSD thresholds,
                and fine-grained feature retention counts strictly categorized 
                by MAR and MNAR mechanisms.
        """
        fltr_params = self.params.get("MetaboIntFilter", {})
        fc = fltr_params.get("feature_counts", {})
        
        # Extract features dropped in each Stage 2 step from internal states
        idx_drop_blk = self.stats.get("idx_dropped_blank", pd.Index([]))
        idx_drop_rsd = self.stats.get("idx_dropped_rsd", pd.Index([]))
        
        # Extract classification origins inherited from Stage 1
        idx_mar = self.stats.get("idx_mar", pd.Index([]))
        idx_mnar = self.stats.get("idx_mnar", pd.Index([]))

        # --------------------------------------------------------------------
        # Accurately compute components dropped by Blank Check 
        # (Globally applies to both MAR and MNAR)
        # --------------------------------------------------------------------
        blk_drop_mar = len(idx_drop_blk.intersection(idx_mar))
        blk_drop_mnar = len(idx_drop_blk.intersection(idx_mnar))
        
        # --------------------------------------------------------------------
        # Accurately compute components dropped by RSD Check 
        # (Logically applies strictly to MAR only, MNAR exempted)
        # --------------------------------------------------------------------
        rsd_drop_mar = len(idx_drop_rsd.intersection(idx_mar))
        rsd_drop_mnar = len(idx_drop_rsd.intersection(idx_mnar))

        # --------------------------------------------------------------------
        # Calculate cascading retention counts (Waterfall logic)
        # --------------------------------------------------------------------
        pre_mar = len(idx_mar)
        pre_mnar = len(idx_mnar)
        
        post_blk_mar = pre_mar - blk_drop_mar
        post_blk_mnar = pre_mnar - blk_drop_mnar
        
        post_rsd_mar = post_blk_mar - rsd_drop_mar
        post_rsd_mnar = post_blk_mnar - rsd_drop_mnar

        metrics = {
            "thresholds": {
                "blank_ratio_tol": fltr_params.get("blank_qc_ratio", 0.2),
                "qc_rsd_tol": fltr_params.get("rsd_qc_tol", 0.3)
            },
            "feature_retention": {
                "pre_stage2": {
                    "total": fc.get("post_stage1", 0),
                    "mar_count": pre_mar,
                    "mnar_count": pre_mnar
                },
                "post_blank_check": {
                    "total": fc.get("post_stage2_blank", 0),
                    "mar_count": post_blk_mar,
                    "mnar_count": post_blk_mnar
                },
                "post_rsd_check": {
                    "total": fc.get("post_stage2_rsd", 0),
                    "mar_count": post_rsd_mar,
                    "mnar_count": post_rsd_mnar
                }
            },
            "filtering_breakdown": {
                "dropped_by_blank": {
                    "total": len(idx_drop_blk),
                    "mar_count": blk_drop_mar,
                    "mnar_count": blk_drop_mnar
                },
                "dropped_by_rsd": {
                    "total": len(idx_drop_rsd),
                    "mar_count": rsd_drop_mar,
                    "mnar_count": rsd_drop_mnar
                }
            }
        }
        return metrics


class MetaboVisualizerFilter(visualizer_classes.BaseMetaboVisualizer):
    """Visualization suite for metabolomics filtering results."""

    def __init__(self, engine):
        """Initialize with the filtering engine."""
        super().__init__(metabo_obj=engine)
        self.engine = engine
        self.fltr_params = engine.params.get("MetaboIntFilter", {})

    # =========================================================================
    # Diagnostic Plots
    # =========================================================================
    def plot_retained_count_steps(self, ax=None):
        """Plot feature attrition cascade stacked bar chart by MAR/MNAR."""
        import matplotlib.colors as mcolors
        
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(4.5, 4))
        else:
            current_ax = ax
            fig = current_ax.figure
            
        fc = self.fltr_params.get("feature_counts", {})
        stats = self.engine.stats
        
        # 1. Determine valid steps to plot based on pipeline stages
        step_keys = [
            "raw", "post_stage1", "post_stage2_blank", "post_stage2_rsd"
        ]
        step_labels = [
            "Raw\nData", "High-MV\nCheck", "QC/Blank\nCheck", "QC RSD\nCheck"
        ]
        
        valid_idx = [i for i, k in enumerate(step_keys) if k in fc]
        if not valid_idx:
            return fig if ax is None else current_ax
            
        labels = [step_labels[i] for i in valid_idx]
        
        # 2. Extract classification origins and Stage 2 drop records
        idx_mar = stats.get("idx_mar", pd.Index([]))
        idx_mnar = stats.get("idx_mnar", pd.Index([]))
        idx_drop_blk = stats.get("idx_dropped_blank", pd.Index([]))
        idx_drop_rsd = stats.get("idx_dropped_rsd", pd.Index([]))
        
        mar_base = len(idx_mar)
        mnar_base = len(idx_mnar)
        
        blk_drop_mar = len(idx_drop_blk.intersection(idx_mar))
        blk_drop_mnar = len(idx_drop_blk.intersection(idx_mnar))
        
        rsd_drop_mar = len(idx_drop_rsd.intersection(idx_mar))
        rsd_drop_mnar = len(idx_drop_rsd.intersection(idx_mnar))
        
        # 3. Build cascading arrays for all possible steps
        mar_all = np.array([
            mar_base,
            mar_base,
            mar_base - blk_drop_mar,
            mar_base - blk_drop_mar - rsd_drop_mar
        ])
        
        mnar_all = np.array([
            mnar_base,
            mnar_base,
            mnar_base - blk_drop_mnar,
            mnar_base - blk_drop_mnar - rsd_drop_mnar
        ])
        
        # Extract invalid features filtered out in Stage 1
        inv_base = max(0, fc.get("raw", 0) - (mar_base + mnar_base))
        inv_all = np.array([inv_base, 0, 0, 0])
        
        # 4. Generate data arrays and static category colors
        mar_counts = mar_all[valid_idx]
        mnar_counts = mnar_all[valid_idx]
        inv_counts = inv_all[valid_idx]
        
        # === COLOR UPDATE: Aligned with assessment logic ===
        c_mar = "tab:red"
        c_mnar = mcolors.to_rgba("tab:red", alpha=0.4)
        c_inv = "tab:gray"
        
        x = np.arange(len(labels))
        width = 0.55
        
        # 5. Plot Stacked Bars using category-specific static colors
        b1 = np.zeros(len(labels))
        current_ax.bar(
            x, mar_counts, bottom=b1, label="MAR", 
            color=c_mar, edgecolor="k", width=width
        )
        
        b2 = b1 + mar_counts
        current_ax.bar(
            x, mnar_counts, bottom=b2, label="MNAR", 
            color=c_mnar, edgecolor="k", width=width
        )
        
        b3 = b2 + mnar_counts
        if inv_base > 0:
            current_ax.bar(
                x, inv_counts, bottom=b3, label="Invalid", 
                color=c_inv, edgecolor="k", width=width
            )
            totals = b3 + inv_counts
        else:
            totals = b3
            
        current_ax.set_xticks(x)
        current_ax.set_xticklabels(labels)
        
        # === ANNOTATION UPDATE: Smart thresholding & Auto color ===
        pu.show_values_on_bars(
            axs=current_ax,
            value_format="{:.0f}",
            fontsize=8,
            stacked=True,
            skip_zero=True,
            threshold_pct=0.05
        )
        
        self._apply_standard_format(
            ax=current_ax, 
            title="Feature Retention Across Filtering Steps",
            xlabel="Filtering Steps", 
            ylabel="Feature Count"
        )
        
        # 7. Customize legend inside upper right 
        self._format_single_legend(
            ax=current_ax, title="Feature Type", loc="upper right",
            bbox_to_anchor=None
        )
        
        # Extend Y-axis limit to prevent label cutoff
        max_height = totals.max() if len(totals) > 0 else 1
        current_ax.set_ylim(0, max_height * 1.25)
        
        if ax is None:
            return fig
            
        return current_ax

    def plot_missing_classification(self, ax=None):
        """Plot the classification boundary for MAR and MNAR features."""
        idx_mnar_grp = self.engine.stats.get("idx_mnar_group")
        idx_mnar_qc = self.engine.stats.get("idx_mnar_qc")
        idx_mar = self.engine.stats.get("idx_mar")
        
        if idx_mar is None:
            return None if ax is None else ax
            
        meta_params = self.engine.params.get("MetaboInt", {})
        qc_col = meta_params.get("sample_type", "Sample Type")
        qc_lbl = meta_params.get("sample_dict", {}).get("QC sample", "QC")
        
        qc_mask = self.engine.columns.get_level_values(qc_col) == qc_lbl
        if not qc_mask.any():
            return None if ax is None else ax
            
        df_qc = self.engine.loc[:, qc_mask]
        
        qc_na_pct = df_qc.isna().mean(axis=1) * 100
        qc_median = np.log2(df_qc.median(axis=1).astype(float) + 1)
        
        types = pd.Series("MAR", index=self.engine.index)
        
        if idx_mnar_grp is not None and not idx_mnar_grp.empty:
            valid_grp = idx_mnar_grp.intersection(self.engine.index)
            if not valid_grp.empty:
                types.loc[valid_grp] = "MNAR (Group)"
                
        if idx_mnar_qc is not None and not idx_mnar_qc.empty:
            valid_qc = idx_mnar_qc.intersection(self.engine.index)
            if not valid_qc.empty:
                types.loc[valid_qc] = "MNAR (QC)"
            
        df_p = pd.DataFrame({
            "Missing_Rate": qc_na_pct,
            "Log2_Intensity": qc_median,
            "Feature Type": types
        })
        
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(5, 4))
        else:
            current_ax = ax
            fig = current_ax.figure
            
        pal = {
            "MAR": "tab:gray", 
            "MNAR (QC)": "tab:red", 
            "MNAR (Group)": "tab:blue"
        }
        all_types = ["MAR", "MNAR (Group)", "MNAR (QC)"]
        hue_order = [t for t in all_types if t in df_p["Feature Type"].unique()]
        
        sns.scatterplot(
            data=df_p, x="Log2_Intensity", y="Missing_Rate", hue="Feature Type",
            palette=pal, hue_order=hue_order,
            s=35, edgecolor="k", linewidth=0.5, alpha=0.75, ax=current_ax
        )
        
        self._apply_standard_format(
            ax=current_ax, title="Classification based on Missing Values",
            xlabel="Log2(Median QC Intensity + 1)", 
            ylabel="QC Missing Rate (%)"
        )
        
        self._format_single_legend(
            ax=current_ax, title="Feature Type", loc="upper left",
            bbox_to_anchor=(1.05, 1.0)
        )
        
        if ax is None:
            return fig
            
        return current_ax

    def plot_mv_group(self, mv_df, tol, group_order=None, ax=None):
        """Plot histograms of MV ratios across biological groups.

        Args:
            mv_df: DataFrame containing missing value ratios per group.
            tol: Tolerance threshold for missing values.
            group_order: Optional list specifying the rendering order of groups.
            ax: Optional matplotlib Axes object for aggregated plotting.

        Returns:
            A matplotlib Figure or Axes object containing the plots.
        """
        if mv_df.empty:
            return None if ax is None else ax

        # Reorder columns based on the provided categorical group_order
        if group_order is not None:
            valid_order = [g for g in group_order if g in mv_df.columns]
            missing = [g for g in mv_df.columns if g not in valid_order]
            final_order = valid_order + missing
            mv_df = mv_df[final_order]
        else:
            sorted_cols = np.sort(mv_df.columns)
            mv_df = mv_df[sorted_cols]

        mv_df = mv_df * 100
        tol_pct = tol * 100
        n_groups = mv_df.shape[1]

        ncols = min(n_groups, 3)
        nrows = int(np.ceil(n_groups / ncols))

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows), 
            sharey=True, layout="constrained"
        )
        axes_flat = [axes] if n_groups == 1 else axes.flatten()

        for idx, group_name in enumerate(mv_df.columns):
            current_ax = axes_flat[idx]
            
            df_p = pd.DataFrame({
                "MV": mv_df[group_name], 
                "Status": np.where(
                    mv_df[group_name] <= tol_pct, "Retained", "Filtered"
                )
            })
            
            sns.histplot(
                data=df_p, x="MV", hue="Status", 
                hue_order=["Retained", "Filtered"],
                palette={"Retained": "tab:gray", "Filtered": "tab:red"}, 
                multiple="stack", edgecolor="k", bins=np.arange(0, 105, 5), 
                ax=current_ax, legend=False
            )
            
            current_ax.axvline(
                x=tol_pct, color="k", linestyle="--", linewidth=1.5
            )
            
            ylabel = "Feature Count" if idx % ncols == 0 else ""
            self._apply_standard_format(
                ax=current_ax, title=f"Group: {group_name}", 
                xlabel="MV Ratio (%)", ylabel=ylabel
            )
            
            if idx == n_groups - 1:
                # [BUG FIX]: Bypassing the overwrite behavior of the base class
                h_ret = mpatches.Patch(
                    facecolor="tab:gray", edgecolor="k", label="Retained",
                    linestyle="-", linewidth=1.0)
                h_flt = mpatches.Patch(
                    facecolor="tab:red", edgecolor="k", label="Filtered",
                    linestyle="-", linewidth=1.0)
                h_thr = mlines.Line2D(
                    [], [], color="k", linestyle="--", 
                    label=f"Threshold ({tol_pct:.1f}%)"
                )
                current_ax.legend(
                    handles=[h_ret, h_flt, h_thr], title="Status", 
                    loc="upper right", **getattr(self, "LEGEND_KWARGS", {})
                )

        for i in range(n_groups, len(axes_flat)):
            axes_flat[i].set_visible(False)
            
        plt.suptitle(
            t="Stage 1: MV Distribution by Group", fontsize=14, weight="bold"
        )
        return fig

    def plot_qc_blank_scatter(self, ax=None):
        """Plots Log2 scatter of QC vs Blank intensities.

        This function visually distinguishes features retained or filtered
        based on the Blank/QC ratio. It also differentiates the underlying
        missing value mechanisms (MAR vs. MNAR) inherited from Stage 1.

        Args:
            ax: Optional matplotlib Axes object for plotting.

        Returns:
            A matplotlib Figure or Axes object containing the scatter plot.
        """
        blk_m = self.engine.stats.get("blank_mean")
        qc_m = self.engine.stats.get("qc_mean")

        # Extract classification indices inherited from Stage 1 stats.
        idx_mar = self.engine.stats.get("idx_mar", pd.Index([]))
        idx_mnar = self.engine.stats.get("idx_mnar", pd.Index([]))

        if blk_m is None or qc_m is None or blk_m.empty:
            return None if ax is None else ax

        # 1. Prepare data frame for plotting.
        df_p = pd.DataFrame({
            "QC": np.log2(qc_m.astype(float) + 1),
            "Blank": np.log2(blk_m.astype(float) + 1)
        })

        # Assign mechanism categories (Default to MAR).
        df_p["Feature Type"] = "MAR"
        
        # Safely label confirmed MNAR features intersecting current index.
        valid_mnar = idx_mnar.intersection(df_p.index)
        if not valid_mnar.empty:
            df_p.loc[valid_mnar, "Feature Type"] = "MNAR"

        # Determine filtering status based on threshold.
        tol_blk = self.fltr_params.get("blank_qc_ratio", 0.2)
        qc_safe = qc_m.replace(0, np.finfo(float).eps)
        df_p["Status"] = np.where(
            blk_m / qc_safe <= tol_blk, "Retained", "Filtered"
        )

        # 2. Initialize canvas.
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(5, 4))
        else:
            current_ax = ax
            fig = current_ax.figure

        # 3. Render scatter plot with dual visual mapping.
        # Hue maps to Status; Style maps to Feature Type.
        sns.scatterplot(
            data=df_p, x="QC", y="Blank", ax=current_ax, hue="Status",
            palette={"Retained": "tab:gray", "Filtered": "tab:red"},
            style="Feature Type", markers={"MAR": "o", "MNAR": "X"},
            s=50, edgecolor="k", linewidth=0.5, alpha=0.8
        )

        # 4. Draw decision boundary line.
        lims = [
            np.min([current_ax.get_xlim(), current_ax.get_ylim()]),
            np.max([current_ax.get_xlim(), current_ax.get_ylim()])
        ]
        x_l = np.linspace(max(0, lims[0]), lims[1], 200)
        current_ax.plot(
            x_l, np.log2(((2**x_l - 1) * tol_blk) + 1), color="k",
            linestyle="--", linewidth=1.5, label=f"Ratio={tol_blk}"
        )

        # 5. Apply standard formatting.
        self._apply_standard_format(
            ax=current_ax, title="Stage 2.1: Blank/QC Check",
            xlabel="Log2(Mean QC + 1)", ylabel="Log2(Mean Blank + 1)"
        )

        # 6. Configure multi-legend layout positioned on the right.
        self._format_multi_legends(
            ax=current_ax, group_titles=["Status", "Feature Type"],
            loc="upper left", start_bbox=(1.05, 1.0)
        )

        if ax is None:
            return fig
            
        return current_ax

    def plot_rsd_dist(self, idx_mnar, ax=None):
        """Plot RSD distribution with consistent bins for MAR and MNAR."""
        rsd_all = self.engine.stats.get("qc_rsd_all")
        if rsd_all is None or rsd_all.empty:
            return None if ax is None else ax
        
        if not isinstance(idx_mnar, pd.Index):
            idx_mnar = pd.Index(idx_mnar)
        
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(4, 4))
        else:
            current_ax = ax
            fig = current_ax.figure
        
        tol = self.fltr_params.get("rsd_qc_tol", 0.3)
        
        idx_mnar_valid = rsd_all.index.intersection(idx_mnar)
        
        types = pd.Series("MAR", index=rsd_all.index)
        if not idx_mnar_valid.empty:
            types.loc[idx_mnar_valid] = "MNAR"
            
        df_p = pd.DataFrame({
            "RSD": rsd_all,
            "Feature Type": types
        })
        
        max_rsd = float(rsd_all.max())
        bin_edges = np.linspace(0, max_rsd, 50)
        
        sns.histplot(
            data=df_p, x="RSD", hue="Feature Type", 
            palette={"MAR": "tab:gray", "MNAR": "tab:red"},
            hue_order=[t for t in ["MAR", "MNAR"] if t in df_p[
                "Feature Type"].unique()],
            bins=bin_edges, kde=True, ax=current_ax, legend=False,
            edgecolor="k", alpha=0.6
        )
            
        current_ax.axvline(
            x=tol, color="k", linestyle="--", linewidth=1.5
        )
        
        # [BUG FIX]: Safe manual handle injection
        handles = []
        if "MAR" in df_p["Feature Type"].values:
            handles.append(mpatches.Patch(
                facecolor="tab:gray", edgecolor="k", linewidth=1.0, label="MAR"))
        if "MNAR" in df_p["Feature Type"].values:
            handles.append(mpatches.Patch(
                facecolor="tab:red", edgecolor="k", linewidth=1.0, label="MNAR"))
            
        handles.append(
            mlines.Line2D(
                [], [], color="k", linestyle="--", 
                label=f"MAR Threshold ({tol})", linewidth=1.0
            )
        )
        
        current_ax.legend(
            handles=handles, title="Feature Type", loc="upper right",
            **getattr(self, "LEGEND_KWARGS", {})
        )

        self._apply_standard_format(
            ax=current_ax, title="Stage 2.2: QC RSD Check", 
            xlabel="RSD", ylabel="Feature Count"
        )
        
        if ax is None:
            return fig
            
        return current_ax

    def plot_mv_downgrade(self, mv_series, level, tol, ax=None):
        """Fallback histogram for QC or Global MV distribution."""
        if mv_series is None or mv_series.empty:
            return None if ax is None else ax
            
        tol_pct = tol * 100
        mv_pct = mv_series * 100
        
        return self.plot_single_mv_hist(
            series=mv_pct, tol=tol_pct, 
            title=f"Stage 1: MV Dist ({level})", 
            xl=f"MV Ratio in {level} (%)", yl="Feature Count", ax=ax
        )

    def plot_single_mv_hist(self, series, tol, title, xl, yl, ax=None):
        """Internal helper for single histogram plotting."""
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(4, 4))
        else:
            current_ax = ax
            fig = current_ax.figure
            
        df = pd.DataFrame({
            "MV": series, 
            "Status": np.where(series <= tol, "Retained", "Filtered")
        })
        
        sns.histplot(
            data=df, x="MV", hue="Status", multiple="stack", edgecolor="k", 
            palette={"Retained": "tab:gray", "Filtered": "tab:red"}, 
            hue_order=["Retained", "Filtered"],
            bins=np.arange(0, 105, 5), ax=current_ax, legend=False
        )
        
        current_ax.axvline(
            x=tol, color="k", linestyle="--", linewidth=1.5
        )
        
        # [BUG FIX]: Bypassing the overwrite behavior of the base class
        h_ret = mpatches.Patch(
            facecolor="tab:gray", edgecolor="k", label="Retained",
            linestyle="-", linewidth=1.0)
        h_flt = mpatches.Patch(
            facecolor="tab:red", edgecolor="k", label="Filtered",
            linestyle="-", linewidth=1.0)
        h_thr = mlines.Line2D(
            [], [], color="k", linestyle="--", label=f"Threshold ({tol:.1f}%)"
        )
        
        current_ax.legend(
            handles=[h_ret, h_flt, h_thr], title="Status", 
            loc="upper right", **getattr(self, "LEGEND_KWARGS", {})
        )
        
        self._apply_standard_format(
            ax=current_ax, title=title, xlabel=xl, ylabel=yl
        )
        
        if ax is None:
            return fig
            
        return current_ax

    # =========================================================================
    # Patchworklib Multi-panel Summary Plots
    # =========================================================================

    def plot_mv_filtering_summary_grid(self, filter_level, group_order=None):
        """Combine Stage 1 MV distribution plots based on filter level.
        
        The layout dynamically adjusts according to the active filter level:
        - "Group": Top rows contain individual group histograms (max 3/row).
        Elements are strictly left-aligned with empty bricks.
        - "QC": A one-row grid (QC and Global side-by-side).
        - "Global": Returns None, as a single plot does not require a grid.
        """
        if filter_level == "Global":
            return None

        try:
            import patchworklib as pw
        except ImportError:
            from loguru import logger
            logger.warning("patchworklib not found. Skipping Stage 1 grid.")
            return None

        pw.clear()

        stats = self.engine.stats
        fltr_params = self.fltr_params

        g_tol = fltr_params.get("mv_group_tol", 0.5)
        q_tol = fltr_params.get("mv_qc_tol", 0.8)
        gl_tol = fltr_params.get("mv_global_tol", 0.7)
        
        has_qc = not stats.get("mv_qc_series", pd.Series(dtype=float)).empty

        if filter_level == "Group":
            mv_df = stats.get("mv_group_df")
            
            # [BUG FIX]: Apply categorical or ASCII sorting before grid building
            if group_order is not None:
                valid_order = [g for g in group_order if g in mv_df.columns]
                missing = [g for g in mv_df.columns if g not in valid_order]
                missing_order = sorted(missing)
                group_cols = valid_order + missing_order
            else:
                group_cols = sorted(mv_df.columns.tolist())
                
            n_groups = len(group_cols)
            
            # Determine global column count for perfect left-alignment
            bottom_len = 2 if has_qc else 1
            max_cols = max(min(n_groups, 3), bottom_len)
            
            # Dynamically build individual bricks for each biological group
            g_bricks = []
            for i, grp in enumerate(group_cols):
                ax_g = pw.Brick(figsize=(4, 4), label=f"G{i}")
                self.plot_single_mv_hist(
                    series=mv_df[grp] * 100, 
                    tol=g_tol * 100, 
                    title=f"Stage 1 Group Dist: {grp}", 
                    xl="MV Ratio (%)", 
                    yl="Feature Count" if i % 3 == 0 else "", 
                    ax=ax_g
                )
                g_bricks.append(ax_g)
                
            # Chunk bricks into rows and pad with empty bricks for alignment
            g_rows = []
            for i in range(0, len(g_bricks), 3):
                chunk = g_bricks[i:i+3]
                current_row = chunk[0]
                for b in chunk[1:]:
                    current_row = current_row | b
                
                # Pad empty bricks to force left-alignment
                if len(chunk) < max_cols:
                    for j in range(len(chunk), max_cols):
                        emp = pw.Brick(figsize=(4, 4), label=f"E_g_{i}_{j}")
                        emp.axis("off")
                        current_row = current_row | emp
                        
                g_rows.append(current_row)
                
            # Build Global and QC bricks
            ax_gl = pw.Brick(figsize=(4, 4), label="GL")
            self.plot_mv_downgrade(
                mv_series=stats.get("mv_global_series"), level="Global", 
                tol=gl_tol, ax=ax_gl
            )

            bottom_bricks = []
            if has_qc:
                ax_qc = pw.Brick(figsize=(4, 4), label="QC")
                self.plot_mv_downgrade(
                    mv_series=stats.get("mv_qc_series"), level="QC", 
                    tol=q_tol, ax=ax_qc
                )
                bottom_bricks = [ax_qc, ax_gl]
            else:
                bottom_bricks = [ax_gl]
                
            row_bottom = bottom_bricks[0]
            for b in bottom_bricks[1:]:
                row_bottom = row_bottom | b
                
            # Pad bottom row to force left-alignment with the group rows
            if len(bottom_bricks) < max_cols:
                for j in range(len(bottom_bricks), max_cols):
                    emp = pw.Brick(figsize=(4, 4), label=f"E_b_{j}")
                    emp.axis("off")
                    row_bottom = row_bottom | emp

            # Combine everything: Group block on top, QC/Global on bottom
            combined_brick = g_rows[0]
            for r in g_rows[1:]:
                combined_brick = combined_brick / r
                
            combined_brick = combined_brick / row_bottom
            return combined_brick

        elif filter_level == "QC":
            ax1 = pw.Brick(figsize=(4, 4), label="A")
            ax2 = pw.Brick(figsize=(4, 4), label="B")

            self.plot_mv_downgrade(
                mv_series=stats.get("mv_qc_series"), level="QC", 
                tol=q_tol, ax=ax1
            )
            self.plot_mv_downgrade(
                mv_series=stats.get("mv_global_series"), level="Global", 
                tol=gl_tol, ax=ax2
            )
            combined_brick = ax1 | ax2
            
            return combined_brick

        return None

    def plot_quality_filtering_summary_grid(
        self
    ):
        """Combine Stage 2 plots into a single figure using patchworklib.
        
        This method aggregates Blank Scatter, RSD Distribution, and 
        Filtering Summary into a 1x3 or custom grid layout.
        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping summary grid.")
            return None

        # [CRITICAL]: Clear global patchworklib state to prevent data residue
        pw.clear()
        # Create bricks for each sub-plot using the patchworklib wrapper
        # We specify different names to avoid brick ID conflicts
        ax1 = pw.Brick(figsize=(4, 4))
        ax2 = pw.Brick(figsize=(4, 4))
        ax3 = pw.Brick(figsize=(4, 4))

        # Explicitly call each plotting method with the allocated axes
        self.plot_qc_blank_scatter(ax=ax1)
        
        # Note: idx_mnar needs to be retrieved from engine stats
        idx_mnar = self.engine.stats.get("idx_mnar", pd.Index([]))
        self.plot_rsd_dist(idx_mnar=idx_mnar, ax=ax2)
        
        self.plot_retained_count_steps(ax=ax3)

        # Use patchworklib operator '|' for horizontal concatenation
        # or '/' for vertical stacking. Here we use horizontal.
        combined_brick = (ax1 | ax2 | ax3)
        
        return combined_brick