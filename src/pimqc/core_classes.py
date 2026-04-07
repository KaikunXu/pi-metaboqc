# src/pimqc/core_classes.py

import copy
import numpy as np
import pandas as pd
from functools import cached_property
from typing import Dict, List, Any, Optional, Union

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

    def __finalize__(
        self, other: Any, method: Optional[str] = None, **kwargs: Any
    ) -> "MetaboInt":
        """Copy custom attributes during object creation."""
        super().__finalize__(other, method=method, **kwargs)
        if hasattr(other, "attrs"):
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

    @cached_property
    def valid_is(self) -> List[str]:
        """List of valid internal standards in the current index."""
        return list(
            set(self.index).intersection(set(self.attrs["internal_standard"]))
        )

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