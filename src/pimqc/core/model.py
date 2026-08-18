"""Core MetaboInt data model and metadata-preserving dataframe behavior.

MetaboInt extends pandas.DataFrame with sample annotations, feature labels,
processing statistics, configuration state, and stage-tracking metadata. It
preserves this state through DataFrame operations and provides shared helpers
for sample selection, intensity boundaries, and metabolomics-specific checks.
"""

import copy
from functools import cached_property
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..constants import DEFAULT_RANDOM_SEED


class MetaboInt(pd.DataFrame):
    """Base class for metabolomics intensity dataset.

    This class manages intensity matrices with a multi-level column index
    and safely preserves structural mappings and pipeline state attributes
    across complex pandas mathematical operations.
    """

    _metadata = ["attrs", "stats"]

    def __init__(
        self,
        *args: object,
        pipeline_params: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None,
        sample_name: Optional[str] = None,
        sample_type: Optional[str] = None,
        bio_group: Optional[str] = None,
        batch: Optional[str] = None,
        inject_order: Optional[str] = None,
        sample_dict: Optional[Dict[str, str]] = None,
        internal_standard: Optional[Union[List[str], str]] = None,
        outlier_ref_feat: Optional[Union[List[str], str]] = None,
        global_seed: Optional[int] = None,
        **kwargs: object,
    ) -> None:
        """Initialize the MetaboInt data structure.

        Args:
            *args: Variable length arguments passed to DataFrame.
            pipeline_params: Global configuration dictionary from TOML.
            mode: Acquisition mode (e.g., 'POS' or 'NEG').
            sample_name: Multi-index level name for sample identifiers.
            sample_type: Multi-index level name for sample types.
            bio_group: Multi-index level name for biological groups.
            batch: Multi-index level name for analytical batches.
            inject_order: Multi-index level name for injection sequence.
            sample_dict: Mapping for internal sample type classifications.
            internal_standard: Explicitly specified internal standards.
            outlier_ref_feat: Explicitly specified outlier references.
            global_seed: Random seed for deterministic reproducibility.
            **kwargs: Keyword arguments passed to DataFrame constructor.
        """
        super().__init__(*args, **kwargs)

        if not hasattr(self, "attrs"):
            self.attrs: Dict[str, Any] = {}
        if not hasattr(self, "stats"):
            object.__setattr__(self, "stats", {})

        # Safely inherit attributes if instantiated from an existing MetaboInt
        input_data = kwargs.get("data")
        if input_data is None and len(args) > 0:
            input_data = args[0]

        if input_data is not None and hasattr(input_data, "attrs"):
            self.attrs.update(copy.deepcopy(input_data.attrs))
        if input_data is not None and hasattr(input_data, "stats"):
            self.stats.update(copy.deepcopy(input_data.stats))

        # =====================================================================
        # Category 1: Structural & Mapping Attributes
        # =====================================================================
        # Defaults fill only genuinely new objects. Pandas-derived stage
        # objects already carry structural attrs that must survive slicing.
        default_configs = {
            "mode": "ESI+",
            "sample_name": "Sample Name",
            "sample_type": "Sample Type",
            "bio_group": "Bio Group",
            "batch": "Batch",
            "inject_order": "Inject Order",
            "sample_dict": {
                "Actual sample": "Sample",
                "Blank sample": "Blank",
                "QC sample": "QC",
            },
            "internal_standard": [],
            "outlier_ref_feat": [],
            "global_seed": DEFAULT_RANDOM_SEED,
        }

        configured = {}
        if pipeline_params and "MetaboInt" in pipeline_params:
            configured.update(pipeline_params["MetaboInt"])

        # Explicit kwargs override TOML (Highest priority)
        local_args = locals()
        explicit_params = [
            "mode",
            "sample_name",
            "sample_type",
            "bio_group",
            "batch",
            "inject_order",
            "sample_dict",
            "global_seed",
        ]
        for param in explicit_params:
            if local_args[param] is not None:
                configured[param] = local_args[param]

        if internal_standard is not None:
            configured["internal_standard"] = self._to_list(internal_standard)
        if outlier_ref_feat is not None:
            configured["outlier_ref_feat"] = self._to_list(outlier_ref_feat)

        for key, default in default_configs.items():
            self.attrs.setdefault(key, copy.deepcopy(default))
        self.attrs.update(configured)

        # =====================================================================
        # Category 2: Lifecycle & State Attributes
        # =====================================================================
        # Initialize default baseline state ONLY if not inherited from upstream.
        # This acts as the genesis state for downstream dynamic visualizations.
        if "pipeline_stage" not in self.attrs:
            self.attrs["pipeline_stage"] = "Raw data"

        if "is_logged" not in self.attrs:
            self.attrs["is_logged"] = False
            self.attrs["log_base"] = "None"

        if "is_scaled" not in self.attrs:
            self.attrs["is_scaled"] = False
            self.attrs["scale_method"] = "None"

    def _to_list(self, x: object) -> List[Any]:
        """Convert input element to list safely."""
        if x is None:
            return []
        return [x] if isinstance(x, str) else list(x)

    @property
    def _constructor(self) -> type:
        """Override constructor to return MetaboInt."""
        return MetaboInt

    def _invalidate_cached_properties(self) -> None:
        """Clear cached values after an in-place dataframe mutation."""
        for cls in type(self).__mro__:
            for name, descriptor in vars(cls).items():
                if isinstance(descriptor, cached_property):
                    self.__dict__.pop(name, None)

    def __setitem__(self, key: object, value: object) -> None:
        """Invalidate derived values after assigning dataframe data."""
        super().__setitem__(key, value)
        self._invalidate_cached_properties()

    def _update_inplace(self, result: pd.DataFrame) -> None:
        """Invalidate derived values after pandas replaces the manager."""
        super()._update_inplace(result)
        self._invalidate_cached_properties()

    def drop(self, *args: object, **kwargs: object) -> pd.DataFrame:
        """Drop labels and clear caches when the operation mutates in place."""
        inplace = bool(kwargs.get("inplace", False))
        result = super().drop(*args, **kwargs)
        if inplace:
            self._invalidate_cached_properties()
        return result

    def __finalize__(
        self, other: object, method: Optional[str] = None, **kwargs: object
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
        if hasattr(other, "stats"):
            object.__setattr__(self, "stats", copy.deepcopy(other.stats))

        return self

    @cached_property
    def _qc(self) -> "MetaboInt":
        """Subset containing only QC samples."""
        return self.loc[
            :,
            self.columns.get_level_values(level=self.attrs["sample_type"])
            == self.attrs["sample_dict"]["QC sample"],
        ]

    @cached_property
    def _blank(self) -> "MetaboInt":
        """Subset containing only Blank samples."""
        return self.loc[
            :,
            self.columns.get_level_values(level=self.attrs["sample_type"])
            == self.attrs["sample_dict"]["Blank sample"],
        ]

    @cached_property
    def _actual_sample(self) -> "MetaboInt":
        """Subset containing only Actual samples."""
        return self.loc[
            :,
            self.columns.get_level_values(level=self.attrs["sample_type"])
            == self.attrs["sample_dict"]["Actual sample"],
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
        """
        List of valid internal standards in the current index
        (case-insensitive).
        """
        # Retrieve the configured internal standards, return empty if not set.
        configured_is = self.attrs.get("internal_standard", [])
        if not configured_is:
            return []

        # Convert configured IS to lowercase and store in a set for O(1) lookup.
        target_is_lower = {str(item).lower() for item in configured_is}

        # Match using lowercase to ensure case-insensitivity,
        # but retain the original naming format from the index.
        return [
            item for item in self.index if str(item).lower() in target_is_lower
        ]

    @cached_property
    def valid_orf(self) -> List[str]:
        """
        List of valid manually specified outlier reference features in
        the current index.
        """
        return list(
            set(self.index).intersection(set(self.attrs["outlier_ref_feat"]))
        )

    def int_order_info(self, feat_type: str = "IS") -> pd.DataFrame:
        """Extract Intensity-Order info of the specified feature type."""
        feats = []
        if feat_type in ("internal_standard", "IS", "is"):
            feats = self.valid_is
        elif feat_type in ("outlier_ref_feat", "ORF", "orf"):
            feats = self.valid_orf

        int_order_df = self.loc[feats].transpose()
        valid_samples = [
            self.attrs["sample_dict"]["Actual sample"],
            self.attrs["sample_dict"]["QC sample"],
        ]

        mask = int_order_df.index.get_level_values(
            level=self.attrs["sample_type"]
        ).isin(valid_samples)

        int_order_df = int_order_df.loc[mask].reset_index(
            [self.attrs["sample_type"], self.attrs["inject_order"]]
        )

        int_order_df[self.attrs["inject_order"]] = int_order_df[
            self.attrs["inject_order"]
        ].astype(int)

        int_order_df = int_order_df.sort_values(
            by=[self.attrs["sample_type"], self.attrs["inject_order"]],
            ascending=True,
        )
        return int_order_df

    @staticmethod
    def calculate_boundaries(
        x: np.ndarray, boundary_type: str = "IQR"
    ) -> tuple[float, float, float]:
        """Calculate statistical boundaries of a 1-dimensional array.

        Args:
            x: Input numpy array.
            boundary_type: Method to calculate boundaries ("IQR" or "sigma").

        Returns:
            Tuple[float, float, float]: Central line, lower limit, upper limit.
        """
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
    def dataset_metrics(self) -> Dict[str, Any]:
        """Extracts comprehensive summary metrics of the current dataset.

        Calculates total feature counts, internal standard counts, sample
        distributions, and analytical batches sorted by their starting
        injection order.

        Returns:
            Dict[str, Any]: A nested dictionary containing structural
                metadata, ordered batch names, and sample distributions.
        """
        # Local import to prevent circular dependency with __init__.py
        try:
            from . import __version__ as pkg_version
        except ImportError:
            pkg_version = "0+unknown"

        mode = self.attrs.get("mode", "ESI+")
        sample_dict = self.attrs.get("sample_dict", {})
        qc_lbl = sample_dict.get("QC sample", "QC")
        blk_lbl = sample_dict.get("Blank sample", "Blank")
        act_lbl = sample_dict.get("Actual sample", "Sample")

        is_count = len(self.valid_is) if hasattr(self, "valid_is") else 0
        bt_col = self.attrs.get("batch", "Batch")
        st_col = self.attrs.get("sample_type", "Sample Type")
        io_col = self.attrs.get("inject_order", "Inject Order")

        # Dynamic batch ordering based on chronological injection sequence
        ordered_batches = []
        if bt_col in self.columns.names and io_col in self.columns.names:
            col_df = self.columns.to_frame(index=False)
            # Find the minimum injection order for each unique batch
            batch_starts = col_df.groupby(bt_col)[io_col].min().astype(int)
            # Sort batch identifiers by their corresponding start order
            ordered_batches = batch_starts.sort_values().index.tolist()
        elif bt_col in self.columns.names:
            # Fallback to ASCII sorting if injection order is unavailable
            bt_vals = self.columns.get_level_values(bt_col)
            ordered_batches = sorted(bt_vals.unique().tolist())

        metrics = {
            "mode": mode,
            "pi-metaboqc_version": pkg_version,
            "features": {
                "total": self.shape[0],
                "internal_standards": self.valid_is,
                "internal_standards_count": is_count,
            },
            "samples": {
                "total": self.shape[1],
                "qc": self._qc.shape[1] if hasattr(self, "_qc") else 0,
                "blank": self._blank.shape[1] if hasattr(self, "_blank") else 0,
                "actual": (
                    self._actual_sample.shape[1]
                    if hasattr(self, "_actual_sample")
                    else 0
                ),
            },
            "batches": {
                "batch_count": len(ordered_batches),
                "ordered_batches": ordered_batches,
                "batch_distribution": {},
            },
        }

        if bt_col in self.columns.names and st_col in self.columns.names:
            col_df = self.columns.to_frame(index=False)
            dist_df = (
                col_df.groupby([bt_col, st_col]).size().unstack(fill_value=0)
            )

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
                        "Inject Order": order_range,
                    }

        return metrics
