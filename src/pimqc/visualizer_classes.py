# src/pimqc/visualizer_classes.py
"""
Purpose of script: Base classes for visualization suites.
"""

from typing import Any, Dict

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

    def _format_complex_legend(self, fig: plt.Figure, ax: plt.Axes) -> None:
        """Format and reposition legends for scatter plots to the right side.
        
        This method splits the default Seaborn legend into separate sub-legends
        for Sample Type and Batch, positioning them outside the last subplot.
        It uses fig.legend with bbox_transform=ax.transAxes to ensure the 
        legend is fully captured by bbox_inches='tight' during saving.
        """
        # Remove default built-in legend from the individual axis
        if ax.get_legend():
            ax.legend().remove()
        
        handles, labels = ax.get_legend_handles_labels()
        
        # Check if both Sample Type and Batch labels are present in the handles
        if self.st_col in labels and self.bat_col in labels:
            c_loc = labels.index(self.st_col)
            s_loc = labels.index(self.bat_col)
            
            # Slice handles to exclude the label string itself (prevents duplication)
            h_color = handles[c_loc + 1 : s_loc]
            l_color = labels[c_loc + 1 : s_loc]
            
            h_style = handles[s_loc + 1 :]
            l_style = labels[s_loc + 1 :]
            
            kwargs = self.LEGEND_KWARGS.copy()
            
            # First Sub-legend: Sample Type (Color)
            # Anchored to the right of the specific subplot passed (ax)
            fig.legend(
                h_color, l_color, 
                title=self.st_col, 
                loc="upper left", 
                bbox_to_anchor=(1.05, 1.0),
                bbox_transform=ax.transAxes,
                **kwargs
            )
            
            # Second Sub-legend: Batch (Shape)
            # Positioned below the first legend
            fig.legend(
                h_style, l_style, 
                title=self.bat_col, 
                loc="upper left", 
                bbox_to_anchor=(1.05, 0.6),
                bbox_transform=ax.transAxes,
                **kwargs
            )
        else:
            # Fallback for plots with only one categorical variable
            fig.legend(
                handles, labels,
                loc="upper left", 
                bbox_to_anchor=(1.05, 1.0), 
                bbox_transform=ax.transAxes,
                **self.LEGEND_KWARGS
            )