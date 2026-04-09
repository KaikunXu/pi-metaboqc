# src/pimqc/visualizer_classes.py
"""
Purpose of script: Base classes for visualization suites.
"""

from typing import Any, Dict, Optional

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

    def __init__(self, metabo_obj: Any) -> None:
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
        self.obj: Any = metabo_obj
        self.attrs: dict = metabo_obj.attrs
        
        # Column Mapping from Metadata
        self.st_col: str = self.attrs.get("sample_type", "Sample Type")
        self.bat_col: str = self.attrs.get("batch", "Batch")
        self.io_col: str = self.attrs.get("inject_order", "Inject Order")
        self.bg_col: str = self.attrs.get("bio_group", "Bio Group")
        
        # Label Mapping
        sample_dict: dict = self.attrs.get("sample_dict", {})
        self.qc_lbl: str = sample_dict.get("QC sample", "QC")
        self.act_lbl: str = sample_dict.get("Actual sample", "Sample")
        self.blk_lbl: str = sample_dict.get("Blank sample", "Blank")

        # Global Batch and Style Mapping for cross-module consistency
        self.all_batches = sorted(
            self.obj.columns.get_level_values(self.bat_col).unique()
        )
        standard_markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "X"]
        self.style_map = dict(
            zip(self.all_batches, standard_markers[:len(self.all_batches)])
        )
        
        # Global Palette Definition
        self.pal: dict = {
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
        self, ax: plt.Axes, title: str = "", 
        xlabel: str = "", ylabel: str = "",
        title_fontsize: int = 14,
        label_fontsize: int = 12,
        tick_fontsize: int = 10
    ) -> None:
        """Apply global standard formatting to a given matplotlib axis."""
        sns.despine(top=True, right=True, left=False, bottom=False, ax=ax)
        
        if title:
            ax.set_title(label=title)
        if xlabel:
            ax.set_xlabel(xlabel=xlabel)
        if ylabel:
            ax.set_ylabel(ylabel=ylabel)
            
        pu.change_weight(
            ax=ax, axis_ticks_weight="normal", axis_label_weight="normal", 
            title_weight="bold", axis="xy"
        )
        pu.change_fontsize(
            ax=ax, axis_ticks_fontsize=tick_fontsize, 
            axis_label_fontsize=label_fontsize, 
            title_fontsize=title_fontsize, axis="xy"
        )

    def _format_single_legend(
        self, 
        fig: plt.Figure, 
        ax: plt.Axes, 
        loc: str = "upper right", 
        bbox_to_anchor: tuple = (1.05, 1.0),
        **kwargs: Any
    ) -> None:
        """Format and position a standard single-group legend.
        
        Args:
            fig: The matplotlib Figure object.
            ax: The matplotlib Axes object where the data was plotted.
            loc: The location string for the legend.
            bbox_to_anchor: The bounding box coordinates to anchor the legend.
            **kwargs: Additional keyword arguments passed to fig.legend(), 
                which will override self.LEGEND_KWARGS if duplicated.
        """
        # Remove default built-in legend from the individual axis
        if ax.get_legend():
            ax.get_legend().remove()
            
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return
            
        # Merge global legend settings with user-provided specific kwargs
        legend_kwargs = self.LEGEND_KWARGS.copy()
        legend_kwargs.update(kwargs)
        
        fig.legend(
            handles, labels, 
            loc=loc, 
            bbox_to_anchor=bbox_to_anchor, 
            bbox_transform=ax.transAxes,
            **legend_kwargs
        )

    def _format_multi_legends(
        self, 
        fig: plt.Figure, 
        ax: plt.Axes, 
        group_titles: Optional[list] = None,
        loc: str = "upper left",
        start_bbox: tuple = (1.05, 1.0),
        y_offset: float = 0.4,
        **kwargs: Any
    ) -> None:
        """Dynamically split a seaborn combined legend into multiple sub-legends.
        
        This algorithm searches for the specified category names within the 
        legend labels, dynamically slices the handles/labels, and stacks them 
        vertically outside the plot.
        
        Args:
            fig: The matplotlib Figure object.
            ax: The matplotlib Axes object where the data was plotted.
            group_titles: List of expected title strings in the legend. 
                Defaults to [self.st_col, self.bat_col].
            loc: The location string for all sub-legends.
            start_bbox: The anchor coordinates for the top-most sub-legend.
            y_offset: The vertical distance subtracted from the Y-coordinate 
                for each subsequent sub-legend to prevent overlap.
            **kwargs: Additional keyword arguments passed to fig.legend().
        """
        if ax.get_legend():
            ax.get_legend().remove()
            
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return

        if group_titles is None:
            group_titles = [self.st_col, self.bat_col]

        # Scan for existing titles and record their index positions
        found_titles = []
        for title in group_titles:
            if title in labels:
                found_titles.append((labels.index(title), title))
                
        # Fallback to single legend if no group titles were matched
        if not found_titles:
            self._format_single_legend(
                fig=fig, ax=ax, loc=loc, bbox_to_anchor=start_bbox, **kwargs
            )
            return

        # Sort indices to ensure we slice the list sequentially from top to bottom
        found_titles.sort(key=lambda x: x[0])
        
        legend_kwargs = self.LEGEND_KWARGS.copy()
        legend_kwargs.update(kwargs)

        # Slice and render each sub-legend
        for i in range(len(found_titles)):
            start_idx = found_titles[i][0]
            title = found_titles[i][1]
            
            # Determine the end index for slicing this specific group
            if i + 1 < len(found_titles):
                end_idx = found_titles[i + 1][0]
            else:
                end_idx = len(labels)
                
            # Slice handles and labels (skipping the title string itself)
            h_sub = handles[start_idx + 1 : end_idx]
            l_sub = labels[start_idx + 1 : end_idx]
            
            if not h_sub:
                continue

            # Calculate dynamic vertical anchor to stack legends
            current_bbox = (start_bbox[0], start_bbox[1] - i * y_offset)
            
            fig.legend(
                h_sub, l_sub, 
                title=title, 
                loc=loc, 
                bbox_to_anchor=current_bbox,
                bbox_transform=ax.transAxes,
                **legend_kwargs
            )

    def save_and_close_fig(
        self,
        fig: plt.Figure,
        file_path: str,
        close_all: bool = False,
        **kwargs: Any
    ) -> None:
        """Save the figure to disk and cleanly close the canvas to free memory.

        Args:
            fig: The matplotlib Figure object to save.
            file_path: The complete output path including file extension.
            close_all: If True, closes all active matplotlib figures.
            **kwargs: Additional arguments passed to fig.savefig.
        """
        # Use tight bounding box by default to prevent label clipping.
        kwargs.setdefault("bbox_inches", "tight")
        
        # Set default DPI to 300 for high-quality, publication-ready output.
        kwargs.setdefault("dpi", 300)

        # 1. Save the figure to disk.
        fig.savefig(file_path, **kwargs)

        # 2. Clear the canvas and release memory.
        if close_all:
            plt.close("all")
        else:
            # Clear the internal elements of the current figure.
            fig.clf()
            # Close and unregister the figure from matplotlib state machine.
            plt.close(fig)