# src/pimqc/visualizer_classes.py
"""
Purpose of script: Base classes for visualization suites.
"""
import io
import os, re
import itertools
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
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

        # Global Batch and Style Mapping (Smart Mixed Strategy)
        self.all_batches = sorted(
            self.obj.columns.get_level_values(self.bat_col).unique()
        )
        n_batches = len(self.all_batches)
        
        # Strategy A: Use standard filled markers for typical cohort sizes (<=15)
        if n_batches <= 10:
            available_markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "X"]
            marker_generator = itertools.cycle(available_markers)
            
            self.style_map = {
                batch_id: next(marker_generator) for batch_id in self.all_batches
            }
            
        # Strategy B: Switch to MathText (Alphanumeric) for large cohorts (>15)
        else:
            # Renders explicit numbers (e.g., '1', '2', '3') to guarantee 
            # absolute distinguishability and zero cognitive load in complex plots.
            self.style_map = {
                batch_id: f"${i}$" for i, batch_id in enumerate(
                    self.all_batches, start=1)
            }
        
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
        loc: str = "upper right",
        bbox_to_anchor: tuple = (1.05, 1.0),
        **kwargs
    ) -> None:
        """
        Format and position a standard single-group legend robustly.

        This implementation safely extracts handles from an existing legend
        object BEFORE removal. This prevents the loss of memory-only patches
        (explicit handles) that are not physically drawn on the Axes. It
        seamlessly falls back to scanning the Axes if no legend exists.
        """
        leg = ax.get_legend()
        handles = []
        labels = []

        # 1. Safely extract handles/labels from the existing legend (if any)
        if leg:
            labels = [t.get_text() for t in leg.get_texts()]
            # Support both Matplotlib 3.x and legacy handle attribute names
            handles = getattr(
                leg, "legend_handles", getattr(leg, "legendHandles", [])
            )
            
            # If extraction failed, fallback to physical axes scan
            if not handles:
                handles, labels = ax.get_legend_handles_labels()
                
            # Safely remove the old legend only AFTER extraction
            leg.remove()
        else:
            # 2. No existing legend, scan the physical axes
            handles, labels = ax.get_legend_handles_labels()

        # 3. Terminate if no valid graphical elements are found
        if not handles:
            return

        # 4. Merge global default styles with runtime parameter overrides
        legend_kwargs = getattr(self, "LEGEND_KWARGS", {}).copy()
        legend_kwargs.update(kwargs)

        # 5. Generate the new stylized legend and bind it to the Axes
        ax.legend(
            handles,
            labels,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            **legend_kwargs
        )

    # def _format_multi_legends(
    #     self, ax, group_titles: list, loc: str = "upper left", 
    #     start_bbox: tuple = (1.05, 1.0), group_pad: float = 0.04, **kwargs
    # ) -> list:
    #     """
    #     Splits handles into separate boxes using the add_artist trick.
    #     Calculates dynamic offsets based on font size to prevent overlap,
    #     while ensuring the final legend dictates the patchworklib bbox.
    #     """
    #     leg = ax.get_legend()
    #     if leg:
    #         labels = [t.get_text() for t in leg.get_texts()]
    #         handles = getattr(
    #             leg, "legend_handles", getattr(leg, "legendHandles", [])
    #         )
    #         if not handles:
    #             handles, _ = ax.get_legend_handles_labels()
    #         leg.remove()
    #     else:
    #         handles, labels = ax.get_legend_handles_labels()

    #     if not handles or not group_titles:
    #         return []

    #     title_idx = [i for i, l in enumerate(labels) if l in group_titles]
    #     if not title_idx:
    #         return []
    #     title_idx.append(len(labels))

    #     leg_kwargs = getattr(self, "LEGEND_KWARGS", {}).copy()
    #     leg_kwargs.update(kwargs)

    #     # Dynamic coordinate offset calculation based on absolute font size
    #     f_size = leg_kwargs.get("fontsize", 10)
    #     t_size = leg_kwargs.get("title_fontsize", 11)
        
    #     fig_h = ax.figure.get_size_inches()[1]
    #     ax_h = ax.get_position().height * fig_h
        
    #     dy_row = (f_size / 72.0 * 1.6) / ax_h
    #     dy_title = (t_size / 72.0 * 1.8) / ax_h

    #     created = []
    #     x_pos, y_pos = start_bbox

    #     for i in range(len(title_idx) - 1):
    #         s_idx, e_idx = title_idx[i], title_idx[i + 1]
    #         sub_h = handles[s_idx + 1:e_idx]
    #         sub_l = labels[s_idx + 1:e_idx]
            
    #         if not sub_h:
    #             continue

    #         new_leg = ax.legend(
    #             sub_h, sub_l, title=labels[s_idx], loc=loc, 
    #             bbox_to_anchor=(x_pos, y_pos), **leg_kwargs
    #         )
            
    #         # [CRITICAL]: add_artist trick. 
    #         # Leave the final legend as ax.legend_ to force PW bbox expansion.
    #         if i < len(title_idx) - 2:
    #             ax.add_artist(new_leg)
                
    #         created.append(new_leg)

    #         # Calculate precise step drop for the next box
    #         drop = dy_title + (len(sub_h) * dy_row) + group_pad
    #         if "upper" in loc:
    #             y_pos -= drop
    #         elif "lower" in loc:
    #             y_pos += drop

    #     is_pw = (
    #         type(ax).__module__.startswith("patchworklib") or 
    #         type(getattr(ax, "figure", None)).__module__.startswith(
    #             "patchworklib"
    #         )
    #     )
        
    #     if getattr(ax, "figure", None) is not None and not is_pw:
    #         for obj in created:
    #             if obj not in ax.figure.legends:
    #                 ax.figure.legends.append(obj)
                    
    #     return created
    
    def _format_multi_legends(
        self, ax, group_titles: list, loc: str = "upper left", 
        start_bbox: tuple = (1.05, 1.0), group_pad: float = 0.04, 
        ncols: int = 1, col_pad: float = 0.15, **kwargs
    ) -> list:
        """
        Splits handles into separate boxes using the add_artist trick.
        Calculates dynamic offsets based on font size to prevent overlap,
        supports dynamic multi-column grids (ncols), and ensures the final 
        legend dictates the patchworklib bbox.
        """
        import math
        
        leg = ax.get_legend()
        if leg:
            labels = [t.get_text() for t in leg.get_texts()]
            handles = getattr(
                leg, "legend_handles", getattr(leg, "legendHandles", [])
            )
            if not handles:
                handles, _ = ax.get_legend_handles_labels()
            leg.remove()
        else:
            handles, labels = ax.get_legend_handles_labels()

        if not handles or not group_titles:
            return []

        title_idx = [i for i, l in enumerate(labels) if l in group_titles]
        if not title_idx:
            return []
        title_idx.append(len(labels))

        leg_kwargs = getattr(self, "LEGEND_KWARGS", {}).copy()
        leg_kwargs.update(kwargs)

        # Dynamic coordinate offset calculation based on absolute font size
        f_size = leg_kwargs.get("fontsize", 10)
        t_size = leg_kwargs.get("title_fontsize", 11)
        
        fig_h = ax.figure.get_size_inches()[1]
        ax_h = ax.get_position().height * fig_h
        
        dy_row = (f_size / 72.0 * 1.6) / ax_h
        dy_title = (t_size / 72.0 * 1.8) / ax_h

        created = []
        n_groups = len(title_idx) - 1
        grps_per_col = math.ceil(n_groups / max(1, ncols))
        
        curr_x = start_bbox[0]
        
        # Safely acquire renderer for dynamic width reading
        try:
            renderer = ax.figure.canvas.get_renderer()
        except Exception:
            renderer = None
            
        global_grp_idx = 0

        for col_idx in range(ncols):
            curr_y = start_bbox[1]
            col_legends = []
            
            for _ in range(grps_per_col):
                if global_grp_idx >= n_groups:
                    break
                    
                s_idx = title_idx[global_grp_idx]
                e_idx = title_idx[global_grp_idx + 1]
                sub_h = handles[s_idx + 1:e_idx]
                sub_l = labels[s_idx + 1:e_idx]
                
                global_grp_idx += 1
                
                if not sub_h:
                    continue

                new_leg = ax.legend(
                    sub_h, sub_l, title=labels[s_idx], loc=loc, 
                    bbox_to_anchor=(curr_x, curr_y), **leg_kwargs
                )
                
                # [CRITICAL]: add_artist trick. 
                # Leave the final legend as ax.legend_ to force PW bbox expansion
                if global_grp_idx < n_groups:
                    ax.add_artist(new_leg)
                    
                created.append(new_leg)
                col_legends.append(new_leg)

                # Calculate precise step drop for the next box
                drop = dy_title + (len(sub_h) * dy_row) + group_pad
                if "upper" in loc:
                    curr_y -= drop
                elif "lower" in loc:
                    curr_y += drop

            # Calculate the maximum width of the current column to shift X
            if col_idx < ncols - 1 and col_legends:
                col_max_w = 0.0
                if renderer:
                    for l_obj in col_legends:
                        bbox = l_obj.get_window_extent(renderer)
                        ax_bbox = bbox.transformed(ax.transAxes.inverted())
                        col_max_w = max(col_max_w, ax_bbox.width)
                
                # Fallback horizontal shift if renderer fails
                shift_w = col_max_w if col_max_w > 0 else 0.35
                curr_x += shift_w + col_pad

        # Propagate to overall figure legends if not using patchworklib
        is_pw = (
            type(ax).__module__.startswith("patchworklib") or 
            type(getattr(ax, "figure", None)).__module__.startswith(
                "patchworklib"
            )
        )
        
        if getattr(ax, "figure", None) is not None and not is_pw:
            for obj in created:
                if obj not in ax.figure.legends:
                    ax.figure.legends.append(obj)
                    
        return created
    
    def _format_unified_multi_legends(
        self, ax, group_titles: list, loc: str = "upper left", 
        start_bbox: tuple = (1.05, 1.0), **kwargs
    ) -> list:
        """
        Merges handles into a single box but simulates a split visual style
        using invisible dummy handles for group headers.
        """
        import matplotlib.patches as mpatches

        leg = ax.get_legend()
        if leg:
            labels = [t.get_text() for t in leg.get_texts()]
            handles = getattr(
                leg, "legend_handles", getattr(leg, "legendHandles", [])
            )
            if not handles:
                handles, _ = ax.get_legend_handles_labels()
            leg.remove()
        else:
            handles, labels = ax.get_legend_handles_labels()

        if not handles or not group_titles:
            return []

        title_idx = [i for i, l in enumerate(labels) if l in group_titles]
        if not title_idx:
            return []
        title_idx.append(len(labels))

        leg_kwargs = getattr(self, "LEGEND_KWARGS", {}).copy()
        leg_kwargs.update(kwargs)

        combined_h, combined_l = [], []
        dummy = mpatches.Rectangle(
            (0, 0), 1, 1, fill=False, edgecolor='none', visible=False
        )

        for i in range(len(title_idx) - 1):
            s_idx, e_idx = title_idx[i], title_idx[i + 1]
            sub_h = handles[s_idx + 1:e_idx]
            sub_l = labels[s_idx + 1:e_idx]
            
            if not sub_h:
                continue
            
            # Inject simulated group header using text formatting
            combined_h.append(dummy)
            combined_l.append(f"--- {labels[s_idx]} ---")
            
            combined_h.extend(sub_h)
            combined_l.extend(sub_l)
            
            # Add a blank spacer if not the last group
            if i < len(title_idx) - 2:
                combined_h.append(dummy)
                combined_l.append("")

        leg_kwargs.update({
            "labelspacing": 0.4, "borderpad": 0.6, "handletextpad": 0.5
        })

        final_leg = ax.legend(
            combined_h, combined_l, loc=loc, 
            bbox_to_anchor=start_bbox, **leg_kwargs
        )
        
        return [final_leg]
    
    
    def save_and_close_fig(
        self, fig, file_path, **kwargs
    ) -> None:
        """Save standard Matplotlib/Seaborn figures with AI font compatibility."""
        if fig is None:
            return
            
        import io

        is_patchwork = type(fig).__module__.startswith("patchworklib")
        if is_patchwork:
            logger.warning(
                "Passed patchwork object to save_and_close_fig. "
                "Use save_and_show_pw.")
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
        self, pw_obj, file_path=None, show_plot=True, width=800, **kwargs
    ):
        """Save and display patchworklib object with safe AI font styling."""
        if pw_obj is None:
            return

        from . import io_utils as iu  # Local import for environment check

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
            if show_plot and iu.is_jupyter():
                # Safe import: Only triggered if inside a Jupyter environment
                from IPython.display import HTML, SVG, display
                
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
            # logger.error(f"Failed to process patchwork object: {e}")
            # Silently handle IPython import bugs in Python 3.13 terminal mode
            # Ensure memory is safely cleared without polluting console logs
            plt.close("all")
            pass