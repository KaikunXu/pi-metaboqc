# src/pimqc/visualizer_classes.py
"""
Purpose of script: Base classes for visualization suites.
"""
import os, re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Avoid INFO level logging to console when saving figures as .pdf
import logging
from loguru import logger
logging.getLogger("fontTools").setLevel(logging.WARNING)

from typing import Optional, Union

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
        # Ensure high-quality vector export across PDF, PS, and SVG formats
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42
        plt.rcParams["svg.fonttype"] = "none"
        plt.rcParams["savefig.dpi"] = 300
        plt.rcParams["savefig.bbox"] = "tight"
        
        # Hard-lock to Arial to prevent AI from throwing DejaVu errors
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["font.sans-serif"] = ["Arial"]
        
        # Force white style to ensure background consistency
        sns.set_style("ticks")
        plt.rcParams["axes.facecolor"] = "white"
        plt.rcParams["figure.facecolor"] = "white"
        
        # [CRITICAL UPDATE]: Comprehensive hook for SVG editability
        import matplotlib.axes
        if not hasattr(matplotlib.axes.Axes, "_pi_metaboqc_patched"):
            _orig_init = matplotlib.axes.Axes.__init__
            
            def _new_init(self_ax, *args, **kwargs):
                _orig_init(self_ax, *args, **kwargs)
                
                # 1. Rasterize elements with zorder < 2 (e.g., scatters)
                # This keeps the data as a bitmap while axes/text remain vector.
                self_ax.set_rasterization_zorder(2)
                
                # 2. Force lift text elements to the highest layer (zorder=10)
                # This ensures they are NEVER rasterized and are easy to select.
                self_ax.title.set_zorder(10)
                self_ax.xaxis.label.set_zorder(10)
                self_ax.yaxis.label.set_zorder(10)
                
                # 3. Disable clipping for labels to prevent AI 'Clipping Mask' 
                # lock. This allows the text tool (T) to directly access them.
                self_ax.xaxis.label.set_clip_on(False)
                self_ax.yaxis.label.set_clip_on(False)
                
                # 4. Handle Ticks and Tick-labels (Critical for Colorbars)
                # We iterate through major ticks to ensure labels are vector
                for axis in [self_ax.xaxis, self_ax.yaxis]:
                    axis.set_zorder(10)
                    for label in axis.get_ticklabels():
                        label.set_zorder(10)
                        label.set_clip_on(False)
                        
            matplotlib.axes.Axes.__init__ = _new_init
            matplotlib.axes.Axes._pi_metaboqc_patched = True

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

    @staticmethod
    def _clean_svg_fonts_for_ai(
        svg_data: str, target_font: str = "Arial"
    ) -> str:
        """Purify SVG font definitions safely for Adobe Illustrator compatibility.

        This method removes all fallback font declarations generated by Matplotlib
        and enforces a single target font. It uses strict regex boundaries to
        ensure the underlying XML/SVG tree structure remains completely intact.

        Args:
            svg_data: The raw SVG XML string to be cleaned.
            target_font: The desired font family name. Defaults to "Arial".

        Returns:
            The purified SVG XML string.
        """
        import re

        # 1. Clean inline CSS styles (e.g., style="font-family: 'DejaVu Sans';").
        # The regex matches "font-family:" followed by any whitespace, and then
        # consumes all characters until it hits a semicolon (;) or a double
        # quote ("). This safely removes single-quoted fallback fonts without
        # corrupting surrounding HTML attributes or XML tags.
        svg_data = re.sub(
            r'font-family:\s*[^;"]+', 
            f'font-family: {target_font}', 
            svg_data
        )

        # 2. Clean standard XML attributes (e.g., font-family="DejaVu Sans").
        svg_data = re.sub(
            r'font-family="[^"]+"', 
            f'font-family="{target_font}"', 
            svg_data
        )

        # 3. Clean single-quoted attributes (e.g., font-family='DejaVu Sans').
        svg_data = re.sub(
            r"font-family='[^']+'", 
            f"font-family='{target_font}'", 
            svg_data
        )

        return svg_data

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
        self, fig, file_path, **kwargs
    ) -> None:
        """Save standard Matplotlib/Seaborn figures with AI font compatibility."""
        if fig is None:
            return
            
        import io
        import os
        from pathlib import Path

        is_patchwork = type(fig).__module__.startswith("patchworklib")
        if is_patchwork:
            logger.warning("Passed patchwork object to save_and_close_fig. Use save_and_show_pw.")
            return

        path_obj = Path(file_path)
        svg_path = path_obj.with_suffix(".svg")
        os.makedirs(svg_path.parent, exist_ok=True)

        save_params = {"format": "svg", "transparent": True}
        save_params.update(kwargs)

        try:
            # Step 1: Save figure to a string buffer instead of disk
            buf = io.StringIO()
            fig.savefig(buf, **save_params)
            
            # Step 2: Intercept and physically clean the SVG string
            clean_svg = self._clean_svg_fonts_for_ai(buf.getvalue())
            
            # Step 3: Write the purified SVG to disk
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(clean_svg)
        except Exception as e:
            logger.error(f"Failed to save clean SVG figure at {svg_path}: {e}")
        finally:
            plt.close(fig)

    def save_and_show_pw(
        self, pw_obj, file_path=None, show_plot=True, width=600, **kwargs
    ):
        """Save and display patchworklib object with safe AI font styling."""
        if pw_obj is None:
            return

        import io
        from IPython.display import HTML, SVG, display

        try:
            buf = io.StringIO()
            pw_obj.savefig(buf, format="svg", **kwargs)
            
            # 1. Clean fonts but keep absolute dimensions for disk saving
            clean_svg = self._clean_svg_fonts_for_ai(buf.getvalue())
            plt.close("all")

            # 2. Write the robust, absolute-sized SVG to disk
            if file_path:
                path_obj = Path(file_path)
                svg_path = path_obj.with_suffix(".svg")
                os.makedirs(svg_path.parent, exist_ok=True)
                with open(svg_path, "w", encoding="utf-8") as f:
                    f.write(clean_svg)

            # 3. Dynamic layout adjustment strictly for Jupyter/VS Code display
            if show_plot:
                preview_svg = clean_svg
                # Convert absolute sizes to 100% only in the temporary string
                preview_svg = re.sub(
                    r'(<svg[^>]*?\s)width="[^"]+"', r'\1width="100%"', 
                    preview_svg, count=1
                )
                preview_svg = re.sub(
                    r'(<svg[^>]*?\s)height="[^"]+"', r'\1height="100%"', 
                    preview_svg, count=1
                )

                if width:
                    w_css = f"{width}px" if isinstance(width, int) else width
                    html_wrapper = (
                        f'<div style="width:{w_css}; max-width:100%; '
                        f'height:auto;">{preview_svg}</div>'
                    )
                    display(HTML(html_wrapper))
                else:
                    display(SVG(data=preview_svg))
                    
        except Exception as e:
            logger.error(f"Failed to process patchwork object: {e}")
            plt.close("all")

    # def save_and_show_pw(
    #     self, pw_obj, file_path=None, show_plot=True, showsize=None, **kwargs
    # ):
    #     """Save to SVG and/or display a patchworklib object in Jupyter."""
    #     if pw_obj is None:
    #         return

    #     import io
    #     import matplotlib.image as mpimg

    #     # Global rcParams will handle DPI and Bbox automatically
    #     save_params = {"transparent": True}
    #     save_params.update(kwargs)

    #     if file_path:
    #         path_obj = Path(file_path)
    #         svg_path = path_obj.with_suffix(".svg")
    #         os.makedirs(svg_path.parent, exist_ok=True)
            
    #         try:
    #             # Force SVG for disk export
    #             pw_obj.savefig(str(svg_path), format="svg", **save_params)
    #         except Exception as e:
    #             logger.error(f"Failed to save patchwork SVG: {e}")

    #     if show_plot:
    #         try:
    #             buf = io.BytesIO()
    #             # Mandatory PNG format for Jupyter memory buffer rendering
    #             pw_obj.savefig(buf, format="png", **save_params)
    #             buf.seek(0)

    #             plt.close("all")

    #             display_fig, display_ax = plt.subplots(figsize=showsize)
    #             display_ax.imshow(mpimg.imread(buf, format="png"))
    #             display_ax.axis("off")
    #             plt.subplots_adjust(
    #                 top=1, bottom=0, right=1, left=0, hspace=0, wspace=0
    #             )
    #             plt.show()
    #         except Exception as e:
    #             logger.error(f"Failed to display patchwork preview: {e}")
    #     else:
    #         plt.close("all")