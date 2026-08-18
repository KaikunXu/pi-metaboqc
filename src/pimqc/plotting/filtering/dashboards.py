"""Dashboard composition for both filtering stages.

Standard and experimental manuscript layouts are assembled here from the
diagnostic panels and filtering flowchart defined in sibling modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from .. import plot_utils as pu


class FilteringDashboardMixin:
    """Assemble standard and experimental filtering dashboards."""

    def plot_mv_filtering_dashboard(
        self,
        tracking_df: pd.DataFrame,
        active_base_tol: float,
        mnar_group_mv_tol: float | None = None,
        mnar_qc_mv_tol: float = 0.2,
        mnar_int_threshold: float | None = None,
        mnar_intensity_pct: float = 0.1,
    ) -> object | None:
        """Assemble the high-missing-value filtering dashboard.

        The layout adapts to biological-group metadata and combines sample
        filtering, rescue diagnostics, MAR eligibility, and the stage
        decision flowchart.
        """
        try:
            import patchworklib as pw
        except ImportError:
            return None

        pw.clear()

        # Initialize data copy and evaluate biological grouping status
        df_curr = tracking_df.copy()
        has_group_info = ("Max_Group_MV_Pct" in df_curr.columns) and (
            df_curr["Max_Group_MV_Pct"].notna().any()
        )

        # Build the universal Sample MV Stripplot Brick
        layout_width = 12.0
        ax_sample = pw.Brick(
            figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
            label="sample_mv",
        )
        sample_track = self.audit_tables.get(
            "sample_tracking", pd.DataFrame()
        )
        sample_mv_tol = self.engine.attrs.get("sample_mv_tol", 0.5)
        self._plot_sample_mv_stripplot(
            sample_track, sample_mv_tol, ax=ax_sample, article_compact=True
        )

        # Dynamic layout assembly based on biological grouping
        if has_group_info:
            # Layout A: With Groups (1+2 Top, 1+1+1 Bottom)

            # Flowchart ratio is 2 units wide to match the bottom 2 plots
            ax_flow = pw.Brick(
                figsize=pu.dashboard_brick_size(8.0, 4.0, layout_width),
                label="flowchart",
            )
            self._plot_mv_filtering_flowchart(
                df=df_curr,
                ax=ax_flow,
                mnar_group_mv_tol=mnar_group_mv_tol,
                mnar_qc_mv_tol=mnar_qc_mv_tol,
                active_base_tol=active_base_tol,
                has_group_info=True,
                mnar_intensity_pct=mnar_intensity_pct,
                compact=True,
                margin_right=0.0,
            )

            # Subplot S1: Group Rescue Scatter
            ax_group_rescue = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="s1",
            )
            self._plot_group_rescue_scatter(
                df_curr,
                "Max_Group_MV_Pct",
                "Min_Group_MV_Pct",
                mnar_group_mv_tol,
                active_base_tol,
                ax_group_rescue,
                "Group-level MNAR Rescue",
                article_compact=True,
            )
            # Cascade remaining features downward
            mask_group = df_curr["Stage1_Status"].str.contains("Group")
            df_curr = df_curr[~mask_group]

            # Subplot S2: QC Rescue Scatter
            ax_qc_rescue = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="s2",
            )
            self._plot_qc_rescue_scatter(
                df_curr,
                mnar_qc_mv_tol,
                mnar_int_threshold,
                ax_qc_rescue,
                "QC-level MNAR Rescue",
                mnar_intensity_pct=mnar_intensity_pct,
                article_compact=True,
            )
            # Cascade remaining features downward
            mask_qc = df_curr["Stage1_Status"].str.contains("QC")
            df_curr = df_curr[~mask_qc]

            # Subplot S3: Base Threshold Check Histogram
            ax_base_check = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="s3",
            )
            self._plot_cutoff_histogram(
                df_curr,
                "Min_Group_MV_Pct",
                "Stage1_Status",
                active_base_tol,
                {"MAR": pu.PRIMARY_ACCENT_COLOR, "INVALID": pu.NEUTRAL_COLOR},
                ["MAR", "INVALID"],
                ax_base_check,
                ("MAR Eligibility Check"),
                "Min Group-level MV (%)",
                article_compact=True,
            )

            # Column-first topology binding to enforce strict vertical alignment
            # Prevents width stretching caused by the axis-off flowchart
            col_left = ax_sample / ax_group_rescue
            col_right = ax_flow / (ax_qc_rescue | ax_base_check)
            return col_left | col_right

        else:
            # Layout B: No Groups (1 Full-width Top, 1+1+1 Bottom)

            # Flowchart ratio is 3 units wide to span the entire top row
            ax_flow = pw.Brick(
                figsize=pu.dashboard_brick_size(12.0, 4.0, layout_width),
                label="flowchart",
            )
            self._plot_mv_filtering_flowchart(
                df=df_curr,
                ax=ax_flow,
                mnar_group_mv_tol=None,
                mnar_qc_mv_tol=mnar_qc_mv_tol,
                active_base_tol=active_base_tol,
                has_group_info=False,
                mnar_intensity_pct=mnar_intensity_pct,
                compact=True,
            )

            # Subplot S2: QC Rescue Scatter (Acts as Step 1 here)
            ax_qc_rescue = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="s2",
            )
            self._plot_qc_rescue_scatter(
                df_curr,
                mnar_qc_mv_tol,
                mnar_int_threshold,
                ax_qc_rescue,
                "QC-level MNAR Rescue",
                mnar_intensity_pct=mnar_intensity_pct,
                article_compact=True,
            )
            mask_qc = df_curr["Stage1_Status"].str.contains("QC")
            df_curr = df_curr[~mask_qc]

            # Subplot S3: Base Threshold Check Histogram (Acts as Step 2 here)
            ax_base_check = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="s3",
            )
            self._plot_cutoff_histogram(
                df_curr,
                "QC_MV_Pct",
                "Stage1_Status",
                active_base_tol,
                {"MAR": pu.PRIMARY_ACCENT_COLOR, "INVALID": pu.NEUTRAL_COLOR},
                ["MAR", "INVALID"],
                ax_base_check,
                "QC-level MV Check",
                "QC-level MV (%)",
                article_compact=True,
            )

            # Row-first topology binding: Full width top over 3 equal bottom
            row_bottom = ax_sample | ax_qc_rescue | ax_base_check
            return ax_flow / row_bottom

    # Manuscript-Only Filtering Dashboards
    def plot_high_mv_filter_article_dashboard(self) -> object | None:
        """Create a compact three-panel summary of high-MV feature screening.

        Experimental: The manuscript-only layout retains the three decision diagnostics used
        to classify group-rescued MNAR, QC-rescued MNAR, and MAR features. It
        is deliberately independent of the full Stage 1 dashboard so the
        standard report layout and its typography remain unchanged.
        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning(
                "patchworklib not found. Skipping article dashboard."
            )
            return None

        tracking_df = self.audit_tables.get(
            "stage1_tracking", pd.DataFrame()
        )
        if tracking_df.empty:
            logger.warning(
                "Stage 1 tracking data are unavailable for article export."
            )
            return None

        has_group_info = (
            "Max_Group_MV_Pct" in tracking_df.columns
            and tracking_df["Max_Group_MV_Pct"].notna().any()
        )
        if not has_group_info:
            logger.warning(
                "Group-level MNAR rescue is unavailable; skipping high-MV "
                "article dashboard."
            )
            return None

        sample_type = self.engine.attrs.get("sample_type", "Sample Type")
        sample_dict = self.engine.attrs.get("sample_dict", {})
        qc_label = sample_dict.get("QC sample", "QC")
        qc_mask = (
            self.engine.columns.get_level_values(sample_type) == qc_label
            if sample_type in self.engine.columns.names
            else np.zeros(self.engine.shape[1], dtype=bool)
        )
        mnar_int_threshold = None
        if qc_mask.any():
            mnar_intensity_pct = self.engine.attrs.get(
                "mnar_intensity_pct", 0.1
            )
            raw_threshold = (
                self.engine.loc[:, qc_mask]
                .median(axis=1)
                .quantile(mnar_intensity_pct)
            )
            mnar_int_threshold = np.log2(raw_threshold + 1)

        active_base_tol = self.engine.attrs.get("mv_group_tol", 0.5)
        mnar_group_mv_tol = self.engine.attrs.get("mnar_group_mv_tol", 0.8)
        mnar_qc_mv_tol = self.engine.attrs.get("mnar_qc_mv_tol", 0.2)
        mnar_intensity_pct = self.engine.attrs.get("mnar_intensity_pct", 0.1)

        pw.clear()
        # Patchworklib adds fixed label/legend padding. This width yields an
        # approximately 17.7 cm export, within the ACS double-column limit.
        panel_size = pu.article_brick_size(1.72, 1.72)
        ax_group = pw.Brick(figsize=panel_size, label="article_group_rescue")
        ax_qc = pw.Brick(figsize=panel_size, label="article_qc_rescue")
        ax_mar = pw.Brick(figsize=panel_size, label="article_mar_eligibility")

        self._plot_group_rescue_scatter(
            tracking_df,
            "Max_Group_MV_Pct",
            "Min_Group_MV_Pct",
            mnar_group_mv_tol,
            active_base_tol,
            ax_group,
            "Group-level MNAR Rescue",
            article_compact=True,
        )
        self._apply_article_panel_format(ax_group, "Group-level MNAR Rescue")

        after_group = tracking_df[
            ~tracking_df["Stage1_Status"].str.contains("Group", na=False)
        ]
        self._plot_qc_rescue_scatter(
            after_group,
            mnar_qc_mv_tol,
            mnar_int_threshold,
            ax_qc,
            "QC-level MNAR Rescue",
            mnar_intensity_pct=mnar_intensity_pct,
            article_compact=True,
        )
        self._apply_article_panel_format(ax_qc, "QC-level MNAR Rescue")

        after_qc = after_group[
            ~after_group["Stage1_Status"].str.contains("QC", na=False)
        ]
        self._plot_cutoff_histogram(
            after_qc,
            "Min_Group_MV_Pct",
            "Stage1_Status",
            active_base_tol,
            {"MAR": pu.PRIMARY_ACCENT_COLOR, "INVALID": pu.NEUTRAL_COLOR},
            ["MAR", "INVALID"],
            ax_mar,
            "MAR Eligibility Check",
            "Min group MV (%)",
            article_compact=True,
        )
        self._apply_article_panel_format(ax_mar, "MAR Eligibility Check")

        return ax_group | ax_qc | ax_mar

    def plot_quality_filtering_dashboard(self) -> object | None:
        """Assemble the low-quality feature-filtering dashboard.

        The dashboard contains three panels when blank samples are available
        and omits the blank/QC diagnostic otherwise.
        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping dashboard.")
            return None

        pw.clear()

        # Detect if Blank data exists to determine the topology
        blank_mean = self.audit_tables.get("blank_mean")
        has_blanks = blank_mean is not None and not blank_mean.empty

        idx_mar = self.audit_tables.get("idx_mar", pd.Index([]))

        # Topology A: 1x3 Grid (Blank samples exist)
        if has_blanks:
            layout_width = 12.0
            ax1 = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="qc_blank",
            )
            ax2 = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="qc_rsd",
            )
            ax3 = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="retention",
            )

            self._plot_qc_blank_scatter(ax=ax1, article_compact=True)
            self._plot_rsd_dist(idx_mar=idx_mar, ax=ax2, article_compact=True)
            self._plot_retained_count_steps(ax=ax3, article_compact=True)

            return ax1 | ax2 | ax3

        # Topology B: 1x2 Grid (No Blank samples)
        else:
            layout_width = 8.0
            ax2 = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="qc_rsd",
            )
            ax3 = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="retention",
            )

            self._plot_rsd_dist(idx_mar=idx_mar, ax=ax2, article_compact=True)
            self._plot_retained_count_steps(ax=ax3, article_compact=True)

            return ax2 | ax3

    def plot_low_quality_filter_article_dashboard(self) -> object | None:
        """
        Create a compact three-panel summary of low-quality feature filtering.

        Experimental: The QC-RSD panel deliberately reuses the MAR-only distribution used by
        the filtering engine. MNAR features remain absent from this diagnostic
        because they are exempt from the QC-RSD reproducibility filter.

        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning(
                "patchworklib not found. Skipping article dashboard."
            )
            return None

        blank_mean = self.audit_tables.get("blank_mean")
        idx_mar = self.audit_tables.get("idx_mar", pd.Index([]))
        if blank_mean is None or blank_mean.empty or len(idx_mar) == 0:
            logger.warning(
                "Blank/QC and MAR QC-RSD inputs are required for the "
                "article dashboard."
            )
            return None

        pw.clear()
        # Compensate for the smaller low-quality layout margin so the exported
        # dashboard matches the high-MV article dashboard at approximately 17.7
        # cm.
        panel_size = pu.article_brick_size(1.72, 1.72)
        ax_blank = pw.Brick(figsize=panel_size, label="article_blank_qc")
        ax_rsd = pw.Brick(figsize=panel_size, label="article_qc_rsd")
        ax_retention = pw.Brick(
            figsize=panel_size, label="article_feature_retention"
        )

        self._plot_qc_blank_scatter(
            ax=ax_blank,
            article_compact=True,
            legend_inside=True,
        )
        self._apply_article_panel_format(ax_blank, "Blank/QC Check")

        self._plot_rsd_dist(
            idx_mar=idx_mar,
            ax=ax_rsd,
            article_compact=True,
        )
        self._apply_article_panel_format(ax_rsd, "QC-RSD Check")

        self._plot_retained_count_steps(
            ax=ax_retention,
            article_compact=True,
        )
        self._apply_article_panel_format(
            ax_retention, "Feature Retention Across Filtering Steps"
        )

        return ax_blank | ax_rsd | ax_retention
