# src/pimqc/visualizer_classes.py
"""
Purpose of script: Base classes for visualization suites.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Avoid INFO level logging to console when saving figures as .pdf
import logging
logging.getLogger("fontTools").setLevel(logging.WARNING)

from . import plot_utils as pu


class BaseMetaboVisualizer:
    """Base class for all visualization suites in pi-metaboqc.

    This class provides global matplotlib and seaborn configurations
    to ensure consistent visual style across the pipeline, specifically
    targeting Adobe Illustrator compatibility and background consistency.
    """

    def __init__(self, metabo_obj) -> None:
        """Initialize the visualizer with global styles.

        Args:
            metabo_obj: A MetaboInt or inherited object containing data.
        """
        # ==========================================
        # Global Matplotlib & Seaborn Configuration
        # ==========================================
        # Ensure high-quality vector export
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42
        plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]
        
        # Force white style to ensure background consistency
        sns.set_style("ticks")
        plt.rcParams["axes.facecolor"] = "white"
        plt.rcParams["figure.facecolor"] = "white"
        
        # Data and Attribute Loading
        self.obj = metabo_obj
        self.attrs = metabo_obj.attrs
        self.params = self.attrs.get("pipeline_parameters", {})
        meta_params = self.params.get("MetaboInt", {})
        
        # Column Mapping from Metadata
        self.st_col = meta_params.get("sample_type", "Sample Type")
        self.bat_col = meta_params.get("batch", "Batch")
        self.io_col = meta_params.get("inject_order", "Inject Order")
        self.bg_col = meta_params.get("bio_group", "Bio Group")
        self.group_order = meta_params.get("group_order", None)
        
        # Label Mapping
        sample_dict = meta_params.get("sample_dict", {})
        self.qc_lbl = sample_dict.get("QC sample", "QC")
        self.act_lbl = sample_dict.get("Actual sample", "Sample")
        self.blk_lbl = sample_dict.get("Blank sample", "Blank")

        # Global Batch and Style Mapping for cross-module consistency
        self.all_batches = sorted(
            self.obj.columns.get_level_values(self.bat_col).unique()
        )
        standard_markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "X"]
        self.style_map = dict(
            zip(self.all_batches, standard_markers[:len(self.all_batches)])
        )
        
        # Global Palette Definition
        self.pal = {
            self.qc_lbl: "tab:red", 
            self.act_lbl: "tab:gray",
            True: "tab:red", 
            False: "tab:gray"
        }

        # Global Legend Style Configuration
        self.LEGEND_KWARGS = dict(
            frameon=True, 
            shadow=True, 
            edgecolor="black", 
            fontsize=10, 
            title_fontsize=11,
            borderpad=0.4,
            facecolor="white"
        )

    def _apply_standard_format(
        self,
        ax,
        title="",
        xlabel="",
        ylabel="",
        append_stage=True,
        custom_stage=None,
        **kwargs
    ):
        """Applies global standard formatting to a given matplotlib axis.

        Args:
            append_stage: Whether to dynamically append the pipeline stage.
            custom_stage: A specific stage label to override the default.
        """

        sns.despine(top=True, right=True, left=False, bottom=False, ax=ax)

        if append_stage:
            if custom_stage is not None:
                stage_label = custom_stage
            else:
                stage_label = ""
                # Iterates through instance attributes to find the data object
                for _, attr_value in vars(self).items():
                    if (
                        hasattr(attr_value, "attrs")
                        and isinstance(attr_value.attrs, dict)
                    ):
                        stage_label = attr_value.attrs.get(
                            "pipeline_stage", ""
                        )
                        if stage_label:
                            break

            if stage_label and f"[{stage_label}]" not in title:
                title = f"{title}\n[{stage_label}]"

        ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        pu.change_weight(ax=ax, axis="xy")
        pu.change_fontsize(ax=ax, axis="xy")


    def _format_single_legend(
        self, 
        ax, 
        loc = "upper right", 
        bbox_to_anchor = (1.05, 1.0),
        **kwargs
    ) -> None:
        """Format and position a standard single-group legend.
        
        Optimized for patchworklib compatibility by strictly binding the 
        legend to the Axes (ax) instead of the Figure (fig).
        """
        if ax.get_legend():
            ax.get_legend().remove()
            
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return
            
        legend_kwargs = self.LEGEND_KWARGS.copy()
        legend_kwargs.update(kwargs)
        
        # [CRITICAL FIX]: Use ax.legend instead of fig.legend. 
        # patchworklib only extracts Axes, so figure-bound legends will be lost.
        ax.legend(
            handles, labels, 
            loc=loc, 
            bbox_to_anchor=bbox_to_anchor, 
            **legend_kwargs
        )

    def _format_multi_legends(
        self, 
        ax, 
        group_titles = None,
        loc = "upper left",
        start_bbox = (1.05, 1.0),
        y_offset = 0.4,
        **kwargs
    ) -> None:
        """Dynamically split a seaborn combined legend based on titles."""
        # 1. Intercept the original combined legend generated by Seaborn
        leg = ax.get_legend()
        if not leg:
            return

        # 2. Safely extract handles and labels across Matplotlib versions
        labels = [t.get_text() for t in leg.get_texts()]
        if hasattr(leg, "legend_handles"):
            handles = leg.legend_handles
        elif hasattr(leg, "legendHandles"):
            handles = leg.legendHandles
        else:
            handles, _ = ax.get_legend_handles_labels()

        if not handles:
            return

        if group_titles is None:
            group_titles = [getattr(self, "st_col", "Sample Type"), 
                            getattr(self, "bat_col", "Batch")]
            
        # 3. Core elegant logic: Find slicing anchors based on titles
        title_indices = [i for i, l in enumerate(labels) if l in group_titles]
        
        # Extraction complete, destroy the original legend
        leg.remove()

        # Defense mechanism: Fallback to single legend if titles are not found
        if not title_indices:
            self._format_single_legend(
                ax=ax, loc=loc, bbox_to_anchor=start_bbox, **kwargs
            )
            return
            
        legend_kwargs = self.LEGEND_KWARGS.copy()
        legend_kwargs.update(kwargs)
        title_indices.append(len(labels))
        
        # 4. Iteratively generate and assemble split legends
        created_legends = []
        for i in range(len(title_indices) - 1):
            start_idx = title_indices[i]
            end_idx = title_indices[i + 1]
            
            sub_title = labels[start_idx]
            sub_handles = handles[start_idx + 1:end_idx]
            sub_labels = labels[start_idx + 1:end_idx]

            current_bbox = (start_bbox[0], start_bbox[1] - (i * y_offset))
            
            new_leg = ax.legend(
                sub_handles, sub_labels, title=sub_title,
                loc=loc, bbox_to_anchor=current_bbox, **legend_kwargs)
            created_legends.append(new_leg)
            
            # Add previous legends as independent Artists to the axes
            if i < len(title_indices) - 2:
                ax.add_artist(new_leg)
                
        # =====================================================================
        # Ultimate fix: Break the physical cropping curse of savefig 
        # with bbox_inches="tight"
        # =====================================================================
        # [BUG FIX]: Environment sniffing for patchworklib.
        # patchworklib utilizes a global figure registry and maintains its own 
        # tight-layout algorithms. Appending to ax.figure.legends causes severe 
        # visual leakage across multiple Jupyter cell executions. 
        # We strictly bypass this hack if a patchworklib environment is detected.
        is_patchwork = (
            type(ax).__module__.startswith("patchworklib") or 
            type(getattr(ax, "figure", None)).__module__.startswith(
                "patchworklib"))
        
        if getattr(ax, "figure", None) is not None and not is_patchwork:
            for leg_obj in created_legends:
                if leg_obj not in ax.figure.legends:
                    ax.figure.legends.append(leg_obj)

                
    def save_and_close_fig(
        self, fig, file_path, dpi = 300, **kwargs
    ) -> None:
        """Save figure to disk and close it safely to free memory.
        
        Robust against both standard matplotlib Figures.
        """
        if fig is None:
            return
            
        from loguru import logger

        # Detect if the object belongs to patchworklib (Brick or Layout)
        is_patchwork = type(fig).__module__.startswith("patchworklib")

        save_kwargs = {"dpi": dpi, "bbox_inches": "tight"}
        save_kwargs.update(kwargs)

        try:
            # Both standard Matplotlib and patchworklib support .savefig()
            fig.savefig(file_path, **save_kwargs)
        except Exception as e:
            logger.error(f"Failed to save figure {file_path}: {e}")
        finally:
            # [CRITICAL FIX]: Only attempt to close standard Matplotlib figures.
            # patchworklib objects do not exist in the plt state machine.
            if not is_patchwork:
                plt.close(fig)

    def save_and_show_pw(
        self, pw_obj, file_path=None, show_plot=True, dpi=300, showsize=None,
        **kwargs
    ):
        """Save and optionally display a patchworklib object cleanly.

        This method addresses the unpredictable display behavior of patchworklib
        in Jupyter Notebooks by rendering the layout to an in-memory buffer and
        displaying it as a clean static image.
        """
        if pw_obj is None:
            return

        import io
        import matplotlib.image as mpimg
        from loguru import logger

        save_kwargs = {"dpi": dpi, "bbox_inches": "tight"}
        save_kwargs.update(kwargs)

        if file_path:
            try:
                pw_obj.savefig(file_path, **save_kwargs)
            except Exception as e:
                logger.error(f"Failed to save patchwork figure: {e}")

        if show_plot:
            try:
                buf = io.BytesIO()
                pw_obj.savefig(buf, format="png", **save_kwargs)
                buf.seek(0)

                # Destroy implicit figures created by patchworklib
                plt.close("all")

                # Create a single clean Figure to display the unified buffer
                display_fig, display_ax = plt.subplots(figsize=showsize)
                display_ax.imshow(mpimg.imread(buf, format="png"))
                display_ax.axis("off")
                plt.subplots_adjust(
                    top=1, bottom=0, right=1, left=0, hspace=0, wspace=0
                )

                plt.show()
            except Exception as e:
                logger.error(f"Failed to display patchwork figure: {e}")
        else:
            # Ensure cleanup occurs if returning the object without showing it
            plt.close("all")