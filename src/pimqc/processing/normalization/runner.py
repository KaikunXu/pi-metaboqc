"""Export and visualize completed normalization results.

The runner invokes the normalization transformation, records its selection
passport, writes the normalized matrix and optional AUTO summary, and delegates
the stage-specific dashboard to the normalization plotter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from ..stage import StageResult, StageRunner
from ...plotting.normalization import NormalizationPlotter

if TYPE_CHECKING:
    from .analysis import MetaboIntNormalizer


class NormalizationStageRunner(
    StageRunner["MetaboIntNormalizer", "MetaboIntNormalizer"]
):
    """Run normalization without mixing calculations with artifact handling."""

    def compute(self) -> StageResult["MetaboIntNormalizer"]:
        """Apply normalization and collect its candidate-selection passport."""
        requested = self.processor.attrs.get("norm_method", "ROBUST_LOG_ONLY")
        blank_count = len(self.processor._blank.columns)
        if blank_count:
            logger.info(f"Permanently dropping {blank_count} Blank samples.")
        logger.info(f"Applying Normalization | Method: {requested}")

        normalized = self.processor.apply_normalization()
        normalized.attrs["pipeline_stage"] = "Normalization"
        # Keep the AUTO passport beside the matrix rather than coupling it to
        # the dashboard implementation.
        return StageResult(
            data=normalized,
            metrics=normalized.normalization_metrics,
            candidates=normalized.attrs.get("selection", {}).get(
                "candidate_summary"
            ),
            metadata={
                "output_suffix": self._output_suffix(normalized),
                **normalized.attrs.get("selection", {}),
            },
        )

    @staticmethod
    def _output_suffix(normalized: "MetaboIntNormalizer") -> str:
        method = normalized.attrs.get("norm_method", "ROBUST_LOG_ONLY")
        parts = [method]
        if normalized.attrs.get("is_logged", False) and method.upper() not in {
            "VSN",
            "ROBUST_LOG_ONLY",
        }:
            parts.append("Log2")
        return "_".join(parts)

    def export(self, result: StageResult["MetaboIntNormalizer"]) -> None:
        """Write the normalized matrix and optional AUTO summary."""
        suffix = result.metadata["output_suffix"]
        result.data.to_csv(
            self.output_dir / f"Normalized_Data_{suffix}.csv",
            na_rep="NA",
            encoding="utf-8-sig",
        )
        if result.candidates:
            # Candidate tables are exported only for AUTO runs.
            summary_path = self.output_dir / "Normalization_Auto_Summary.csv"
            pd.DataFrame(result.candidates).to_csv(
                summary_path,
                index=False,
                na_rep="NA",
                encoding="utf-8-sig",
            )
            logger.info(f"Auto normalization summary saved as: {summary_path}")

    def render(self, result: StageResult["MetaboIntNormalizer"]) -> None:
        """Render the normalization dashboard from the completed result."""
        logger.info("Generating diagnostic plots for normalization...")
        processor = result.render_context["processor"]
        plotter = NormalizationPlotter(
            raw_obj=processor,
            norm_obj=result.data,
        )
        dashboard = plotter.plot_normalization_dashboard()
        if dashboard is None:
            return
        path = self.output_dir / (
            f"Normalization_Dashboard_{result.metadata['output_suffix']}.svg"
        )
        plotter.save_and_show_pw(pw_obj=dashboard, file_path=str(path))
        logger.info(f"Normalization summary dashboard saved as: {path}")
