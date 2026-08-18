"""Export and visualize completed sample and feature filtering results.

The runners keep high-missing sample removal, MAR/MNAR classification, and
low-quality feature rules inside ``MetaboIntFilter`` while moving CSV output and
dashboard construction into explicit post-transformation lifecycle phases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import numpy as np
import pandas as pd
from loguru import logger

from ..stage import StageResult, StageRunner
from ...plotting.filtering import FilteringPlotter

if TYPE_CHECKING:
    from .analysis import MetaboIntFilter


class SampleFilteringStageRunner(StageRunner["MetaboIntFilter", pd.DataFrame]):
    """Run and export the sample-level high-missingness filter."""

    def compute(self) -> StageResult[pd.DataFrame]:
        """Remove high-missingness analytical samples."""
        return self.processor.transform_sample_filtering()

    def export(self, result: StageResult[pd.DataFrame]) -> None:
        """Write the retained sample matrix and its attrition table."""
        assert self.output_dir is not None
        result.data.to_csv(
            self.output_dir / "Filtered_Data_High-MV_Samples.csv"
        )
        result.audit_tables["sample_tracking"].to_csv(
            self.output_dir / "Filtering_Tracking_High-MV_Samples.csv"
        )

    def render(self, result: StageResult[pd.DataFrame]) -> None:
        """Skip rendering because sample filtering has no standalone panel."""


class MissingValueFilteringStageRunner(
    StageRunner["MetaboIntFilter", pd.DataFrame]
):
    """Run missingness classification and its feature-retention dashboard."""

    def compute(self) -> StageResult[pd.DataFrame]:
        """Classify features and retain accepted MAR or MNAR features."""
        return self.processor.transform_mv_filtering()

    def export(self, result: StageResult[pd.DataFrame]) -> None:
        """Write sample and feature filtering matrices and tracking tables."""
        assert self.output_dir is not None
        # Preserve both attrition levels because the feature stage is computed
        # from the already sample-filtered matrix.
        result.metadata["sample_filtered"].to_csv(
            self.output_dir / "Filtered_Data_High-MV_Samples.csv"
        )
        result.audit_tables["sample_tracking"].to_csv(
            self.output_dir / "Filtering_Tracking_High-MV_Samples.csv"
        )
        result.data.to_csv(
            self.output_dir / "Filtered_Data_High-MV_Features.csv"
        )
        result.audit_tables["stage1_tracking"].to_csv(
            self.output_dir / "Filtering_Tracking_High-MV_Features.csv"
        )

    def render(self, result: StageResult[pd.DataFrame]) -> None:
        """Render the MAR/MNAR classification and retention dashboard."""
        assert self.output_dir is not None
        processor = result.render_context["processor"]
        qc_mask = result.metadata["qc_mask"]
        valid_groups = result.metadata["valid_groups"]
        active_tolerance = processor.attrs.get(
            "mv_group_tol" if valid_groups else "mv_qc_tol",
            0.5,
        )
        intensity_threshold = None
        if qc_mask.any():
            # Recreate the displayed MNAR intensity boundary from the same QC
            # percentile used by the processor.
            percentile = processor.attrs.get("mnar_intensity_pct", 0.1)
            raw_threshold = (
                processor.loc[:, qc_mask].median(axis=1).quantile(percentile)
            )
            intensity_threshold = np.log2(raw_threshold + 1)

        plotter = FilteringPlotter(
            processor,
            audit_tables=result.audit_tables,
        )
        dashboard = plotter.plot_mv_filtering_dashboard(
            tracking_df=result.audit_tables["stage1_tracking"],
            active_base_tol=active_tolerance,
            mnar_group_mv_tol=processor.attrs.get("mnar_group_mv_tol", 0.8),
            mnar_qc_mv_tol=processor.attrs.get("mnar_qc_mv_tol", 0.2),
            mnar_int_threshold=intensity_threshold,
            mnar_intensity_pct=processor.attrs.get("mnar_intensity_pct", 0.1),
        )
        if dashboard:
            path = self.output_dir / "MV_Classification_Dashboard.svg"
            plotter.save_and_show_pw(
                pw_obj=dashboard,
                file_path=str(path),
            )
            logger.info(f"High-MV Filter summary dashboard saved as: {path}")
        logger.success("High-missing value feature filtering completed.")


class QualityFilteringStageRunner(StageRunner["MetaboIntFilter", pd.DataFrame]):
    """Run blank-ratio and QC-RSD filtering and render its dashboard."""

    def __init__(
        self,
        processor: "MetaboIntFilter",
        output_dir: str | None,
        *,
        idx_mar: pd.Index | list[object] | None,
        idx_mnar: pd.Index | list[object] | None,
        runtime_overrides: Mapping[str, object] | None = None,
        allowed_override_keys: frozenset[str] | set[str] | None = None,
    ) -> None:
        """Initialize the quality-filtering lifecycle.

        Args:
            processor: Filtering processor containing the source matrix.
            output_dir: Optional directory for tables and dashboards.
            idx_mar: Optional MAR feature identifiers.
            idx_mnar: Optional MNAR feature identifiers.
            runtime_overrides: Named threshold overrides for this execution.
            allowed_override_keys: Permitted runtime configuration names.
        """
        super().__init__(
            processor,
            output_dir,
            runtime_overrides=runtime_overrides,
            allowed_override_keys=allowed_override_keys,
        )
        self.idx_mar = idx_mar
        self.idx_mnar = idx_mnar

    def compute(self) -> StageResult[pd.DataFrame]:
        """Apply the low-quality feature rules."""
        return self.processor.transform_quality_filtering(
            idx_mar=self.idx_mar,
            idx_mnar=self.idx_mnar,
        )

    def export(self, result: StageResult[pd.DataFrame]) -> None:
        """Write the retained matrix and feature-level audit table."""
        assert self.output_dir is not None
        path = self.output_dir / "Filtered_Data_Low-quality_Features.csv"
        result.data.to_csv(path, encoding="utf-8-sig", na_rep="NA")
        result.audit_tables["stage2_tracking"].to_csv(
            self.output_dir / "Filtering_Tracking_Low-quality_Features.csv",
            na_rep="N/A",
        )
        logger.info(
            f"Data after low-quality features filtering saved as: {path}"
        )

    def render(self, result: StageResult[pd.DataFrame]) -> None:
        """Render the low-quality filtering dashboard."""
        assert self.output_dir is not None
        plotter = FilteringPlotter(
            engine=result.data,
            audit_tables=result.audit_tables,
        )
        # Rendering failures should not invalidate an already completed and
        # exported filtering transformation.
        try:
            dashboard = plotter.plot_quality_filtering_dashboard()
            if dashboard:
                path = self.output_dir / "Low-quality_Filtering_Dashboard.svg"
                plotter.save_and_show_pw(
                    pw_obj=dashboard,
                    file_path=str(path),
                )
                logger.info(
                    f"Low-quality Filter summary dashboard saved as: {path}"
                )
        except Exception as error:
            logger.error(
                "Grid of low-quality features filtering generation failed: "
                f"{error}"
            )
        logger.success("Low-quality features filtering completed.")
