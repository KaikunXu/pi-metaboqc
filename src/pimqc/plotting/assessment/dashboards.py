"""Dashboard composition for quality-assessment diagnostics.

The module combines correlation, RSD, PCA, and outlier panels while adapting
the layout to the diagnostics available for each stage snapshot.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Any

from .. import plot_utils as pu


class AssessmentDashboardMixin:
    """Assemble the complete assessment dashboard."""

    def plot_assessment_dashboard(
        self,
        pca_res: dict[str, Any],
        rsd_data: dict[str, dict[str, int]],
        batch_corr: pd.DataFrame,
        corr_mat: pd.DataFrame,
        qc_mask: np.ndarray | None,
        batches: list[object] | pd.Index | np.ndarray,
        method: str,
        sample_type: str,
        batch: str,
        qc_label: str,
        actual_label: str,
        is_flags: pd.Series | None = None,
        orf_flags: pd.Series | None = None,
        sample_name: str = "Sample Name",
        target_param: str = "both",
    ) -> object | None:
        """Assemble the standard assessment dashboard.

        The panel set and legends adapt to available batch, reference-sample,
        and internal-standard metadata.
        """
        try:
            import patchworklib as pw
        except ImportError:
            return None

        pw.clear()

        def _bind_legends_to_axes(ax: plt.Axes | None) -> None:
            if ax is not None and hasattr(ax.figure, "legends"):
                for leg in list(ax.figure.legends):
                    ax.add_artist(leg)
                ax.figure.legends.clear()

        # Row 1 Assembly
        layout_width = 14.0
        ax1 = pw.Brick(figsize=pu.dashboard_brick_size(4.8, 4.0, layout_width))
        ax1.axis("off")
        ax_corr = ax1.inset_axes([0.0, 0.0, 0.83, 1.0])

        n_batches = batch_corr.shape[0] if batch_corr is not None else 0
        if n_batches <= 1:
            self.plot_qc_corr_heatmap(
                corr_matrix=corr_mat,
                corr_mask=qc_mask,
                batches=batches,
                method=method,
                cluster="none",
                ax=ax_corr,
            )
        else:
            self.plot_batch_corr_heatmap(
                batch_corr_matrix=batch_corr, method=method, ax=ax_corr
            )
        _bind_legends_to_axes(ax_corr)

        ax2 = pw.Brick(figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width))
        self.plot_rsd_bar(
            rsd_data=rsd_data,
            qc_label=qc_label,
            actual_label=actual_label,
            ax=ax2,
        )
        _bind_legends_to_axes(ax2)

        ax3 = pw.Brick(figsize=pu.dashboard_brick_size(5.2, 4.0, layout_width))
        ax3.axis("off")
        ax_pca = ax3.inset_axes([0.0, 0.0, 0.77, 1.0])

        self.plot_pca_scatter(
            pca_df=pca_res["pca_scatter"],
            pca_var=pca_res["pca_variance"],
            pca_diagnostics=pca_res["diagnostics"],
            sample_type=sample_type,
            batch=batch,
            qc_label=qc_label,
            actual_label=actual_label,
            ax=ax_pca,
        )
        _bind_legends_to_axes(ax_pca)

        # Row 2 Assembly
        ax4 = pw.Brick(figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width))
        self.plot_sd_od_scatter(
            metrics_df=pca_res["metrics_df"],
            sd_limit=pca_res["sd_limit"],
            od_limit=pca_res["od_limit"],
            ax=ax4,
            show_legend=False,
            annotate_thresholds=True,
            is_flags=is_flags,
            orf_flags=orf_flags,
        )

        ax5 = pw.Brick(figsize=pu.dashboard_brick_size(8.8, 4.0, layout_width))
        ax5.axis("off")
        ax5_top = ax5.inset_axes([0.0, 0.52, 1.0, 0.48])
        ax5_bot = ax5.inset_axes([0.0, 0.0, 1.0, 0.48], sharex=ax5_top)

        self._plot_stat_outliers_bar(
            outliers_df=pca_res["outliers"],
            sample_type=sample_type,
            batch=batch,
            sample_name=sample_name,
            actual_label=actual_label,
            target_param=target_param,
            sd_limit=pca_res["sd_limit"],
            od_limit=pca_res["od_limit"],
            ax1=ax5_top,
            ax2=ax5_bot,
            show_legend=False,
            is_flags=is_flags,
            orf_flags=orf_flags,
        )

        ax6 = pw.Brick(figsize=pu.dashboard_brick_size(1.2, 4.0, layout_width))
        self.plot_outlier_standalone_legend(
            metrics_df=pca_res["metrics_df"],
            sd_limit=pca_res["sd_limit"],
            od_limit=pca_res["od_limit"],
            ax=ax6,
            is_flags=is_flags,
            orf_flags=orf_flags,
            include_thresholds=False,
        )
        _bind_legends_to_axes(ax6)

        return (ax1 | ax2 | ax3) / (ax4 | ax5 | ax6)
