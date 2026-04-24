# src/pimqc/core_classes.py

import copy
import numpy as np
import pandas as pd
from functools import cached_property
from typing import Dict, List, Tuple, Any, Optional, Union

from . import plot_utils as pu


class MetaboInt(pd.DataFrame):
    """Base class for metabolomics intensity dataset.

    This class manages intensity matrices with a multi-level column index
    and safely preserves custom attributes during pandas operations.
    """

    _metadata = ["attrs"]

    def __init__(
        self,
        *args: Any,
        pipeline_params: Optional[Dict[str, Any]] = None,
        mode: str = "POS",
        sample_name: str = "Sample Name",
        sample_type: str = "Sample Type",
        bio_group: str = "Bio Group",
        batch: str = "Batch",
        inject_order: str = "Inject Order",
        sample_dict: Optional[Dict[str, str]] = None,
        internal_standard: Optional[Union[List[str], str]] = None,
        outlier_marker: Optional[Union[List[str], str]] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the MetaboInt data structure.

        Args:
            *args: Variable length arguments passed to pandas DataFrame.
            pipeline_params: Global settings for the pipeline classes.
            mode: MS Polarity ("POS" or "NEG").
            sample_name: Column name for Sample Name.
            sample_type: Column name for Sample Type.
            bio_group: Column name for Biological Group.
            batch: Column name for Batch.
            inject_order: Column name for Injection Order.
            sample_dict: Mapping dictionary for specific sample types.
            internal_standard: List of internal standard metabolites.
            outlier_marker: List of outlier markers.
            **kwargs: Extra arguments passed to pandas DataFrame.
        """
        super().__init__(*args, **kwargs)

        if not hasattr(self, "attrs"):
            self.attrs: Dict[str, Any] = {}

        input_data = kwargs.get("data")
        if input_data is None and len(args) > 0:
            input_data = args[0]
            
        if input_data is not None and hasattr(input_data, "attrs"):
            self.attrs.update(copy.deepcopy(input_data.attrs))

        if "pipeline_stage" not in self.attrs:
            self.attrs["pipeline_stage"] = "Raw data"

        if sample_dict is None:
            sample_dict = {
                "Actual sample": "Sample",
                "Blank sample": "Blank",
                "QC sample": "QC",
            }

        base_configs = {
            "mode": mode,
            "sample_name": sample_name,
            "sample_type": sample_type,
            "bio_group": bio_group,
            "batch": batch,
            "inject_order": inject_order,
            "sample_dict": sample_dict,
            "internal_standard": self._to_list(internal_standard),
            "outlier_marker": self._to_list(outlier_marker)
        }

        # Explicitly load "MetaboInt" block
        if pipeline_params and "MetaboInt" in pipeline_params:
            base_configs.update(pipeline_params["MetaboInt"])

        self.attrs.update(base_configs)

    def _to_list(self, x: Any) -> List[Any]:
        """Convert input element to list."""
        if x is None:
            return []
        return [x] if isinstance(x, str) else list(x)

    @property
    def _constructor(self) -> type:
        """Override constructor to return MetaboInt."""
        return MetaboInt

    # def __finalize__(
    #     self, other: Any, method: Optional[str] = None, **kwargs: Any
    # ) -> "MetaboInt":
    #     """Copy custom attributes during object creation."""
    #     super().__finalize__(other, method=method, **kwargs)
    #     if hasattr(other, "attrs"):
    #         self.attrs = copy.deepcopy(other.attrs)
    #     return self

    def __finalize__(
        self, other: Any, method: Optional[str] = None, **kwargs: Any
    ) -> "MetaboInt":
        """Copy custom attributes safely, avoiding Pandas array bugs."""
        try:
            super().__finalize__(other, method=method, **kwargs)
        except ValueError:
            # Bypass Pandas bug: array-like dict values crash pd.concat
            pass

        if method == "concat" and hasattr(other, "objs"):
            for obj in other.objs:
                if hasattr(obj, "attrs") and obj.attrs:
                    self.attrs = copy.deepcopy(obj.attrs)
                    break
        elif hasattr(other, "attrs"):
            self.attrs = copy.deepcopy(other.attrs)
            
        return self

    @cached_property
    def _qc(self) -> "MetaboInt":
        """Subset containing only QC samples."""
        return self.loc[:,
            self.columns.get_level_values(
                level=self.attrs["sample_type"]
            ) == self.attrs["sample_dict"]["QC sample"]
        ]

    @cached_property
    def _blank(self) -> "MetaboInt":
        """Subset containing only Blank samples."""
        return self.loc[:,
            self.columns.get_level_values(
                level=self.attrs["sample_type"]
            ) == self.attrs["sample_dict"]["Blank sample"]
        ]

    @cached_property
    def _actual_sample(self) -> "MetaboInt":
        """Subset containing only Actual samples."""
        return self.loc[:,
            self.columns.get_level_values(
                level=self.attrs["sample_type"]
            ) == self.attrs["sample_dict"]["Actual sample"]
        ]

    @property
    def is_multi_batch_flag(self) -> bool:
        """Determine whether the current object contains multiple batches."""
        bt_col = self.attrs.get("batch", "Batch")
        
        if bt_col in self.columns.names:
            return len(self.columns.get_level_values(bt_col).unique()) > 1
        return False


    @cached_property
    def valid_is(self) -> List[str]:
        """List of valid internal standards in the current index (case-insensitive)."""
        # Retrieve the configured internal standards, return empty if not set.
        configured_is = self.attrs.get("internal_standard", [])
        if not configured_is:
            return []
            
        # Convert configured IS to lowercase and store in a set for O(1) lookup.
        target_is_lower = {str(item).lower() for item in configured_is}
        
        # Match using lowercase to ensure case-insensitivity, 
        # but retain the original naming format from the index.
        return [
            item for item in self.index 
            if str(item).lower() in target_is_lower
        ]
        
    @cached_property
    def valid_om(self) -> List[str]:
        """List of valid outlier markers in the current index."""
        return list(
            set(self.index).intersection(set(self.attrs["outlier_marker"]))
        )

    def int_order_info(self, feat_type: str = "IS") -> pd.DataFrame:
        """Extract Intensity-Order info of the specified feature type."""
        feats = []
        if feat_type in ("internal_standard", "IS"):
            feats = self.valid_is
        elif feat_type in ("outlier_marker", "OM"):
            feats = self.valid_om

        int_order_df = self.loc[feats].transpose()
        valid_samples = [
            self.attrs["sample_dict"]["Actual sample"],
            self.attrs["sample_dict"]["QC sample"]
        ]
        
        mask = int_order_df.index.get_level_values(
            level=self.attrs["sample_type"]
        ).isin(valid_samples)
        
        int_order_df = int_order_df.loc[mask].reset_index([
            self.attrs["sample_type"], 
            self.attrs["inject_order"]
        ])
        
        int_order_df[self.attrs["inject_order"]] = int_order_df[
            self.attrs["inject_order"]
        ].astype(int)
        
        int_order_df = int_order_df.sort_values(
            by=[self.attrs["sample_type"], self.attrs["inject_order"]],
            ascending=True
        )
        return int_order_df
    
    @staticmethod
    def calculate_boundaries(x: np.ndarray, boundary_type: str = "IQR"):
        """Calculate statistical boundaries of a 1-dimensional array.

        Args:
            x: Input numpy array.
            boundary_type: Method to calculate boundaries ('IQR' or 'sigma').

        Returns:
            Tuple[float, float, float]: Central line, lower limit, upper limit.
        """
        import numpy as np
        
        if boundary_type in ("mean-std", "sigma"):
            solid = float(np.nanmean(x))
            std_val = float(np.nanstd(x, ddof=1))
            return solid, solid - 3 * std_val, solid + 3 * std_val
            
        elif boundary_type == "IQR":
            solid = float(np.nanmedian(x))
            q1 = float(np.nanquantile(x, 0.25))
            q3 = float(np.nanquantile(x, 0.75))
            iqr = q3 - q1
            return solid, q1 - 1.5 * iqr, q3 + 1.5 * iqr
            
        return 0.0, 0.0, 0.0

    @cached_property
    def dataset_metrics(self):
        """Extracts comprehensive summary metrics of the current dataset.

        Calculates total feature counts, internal standard counts, sample
        distributions, and an ordered list of analytical batches.

        Returns:
            Dict[str, Any]: A nested dictionary containing structural
                metadata, ordered batch names, and sample distributions.
        """
        mode = self.attrs.get("mode","")
        sample_dict = self.attrs.get("sample_dict", {})
        qc_lbl = sample_dict.get("QC sample", "QC")
        blk_lbl = sample_dict.get("Blank sample", "Blank")
        act_lbl = sample_dict.get("Actual sample", "Sample")

        is_count = len(self.valid_is) if hasattr(self, "valid_is") else 0
        bt_col = self.attrs.get("batch", "Batch")
        st_col = self.attrs.get("sample_type", "Sample Type")
        io_col = self.attrs.get("inject_order", "Inject Order")

        ordered_batches = []
        if bt_col in self.columns.names:
            bt_vals = self.columns.get_level_values(bt_col)
            if isinstance(bt_vals.dtype, pd.CategoricalDtype):
                ordered_batches = bt_vals.unique().sort_values().tolist()
            else:
                ordered_batches = sorted(bt_vals.unique().tolist())

        metrics = {
            "mode":mode,
            "features": {
                "total": self.shape[0],
                "internal_standards": self.valid_is,
                "internal_standards_count": is_count
            },
            "samples": {
                "total": self.shape[1],
                "qc": self._qc.shape[1] if hasattr(self, "_qc") else 0,
                "blank": self._blank.shape[1] if hasattr(
                    self, "_blank") else 0,
                "actual": self._actual_sample.shape[1] if hasattr(
                    self, "_actual_sample") else 0
            },
            "batches":{
                "batch_count":len(ordered_batches),
                "ordered_batches": ordered_batches,
                "batch_distribution": {},
            }

        }

        if bt_col in self.columns.names and st_col in self.columns.names:
            # Prevent pandas ambiguity ValueError by disabling index generation
            col_df = self.columns.to_frame(index=False)
            dist_df = col_df.groupby(
                [bt_col, st_col]
            ).size().unstack(fill_value=0)

            for b_id in ordered_batches:
                if b_id in dist_df.index:
                    row = dist_df.loc[b_id]
                    
                    # Extract injection order range for the current batch
                    batch_mask = col_df[bt_col] == b_id
                    orders = col_df.loc[batch_mask, io_col].astype(int)
                    order_range = f"{orders.min()} ~ {orders.max()}"
                    
                    metrics["batches"]["batch_distribution"][str(b_id)] = {
                        "Total": int(row.sum()),
                        "QC": int(row.get(qc_lbl, 0)),
                        "Blank": int(row.get(blk_lbl, 0)),
                        "Sample": int(row.get(act_lbl, 0)),
                        "Inject Order": order_range
                    }

        return metrics