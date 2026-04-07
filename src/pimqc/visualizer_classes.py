"""
Purpose of script: Base classes for visualization suites.
"""

from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from . import plot_utils as pu


class BaseMetaboVisualizer:
    """Base class for all visualization suites in pi-metaboqc."""

    def __init__(self, metabo_obj: Any) -> None:
        """Initialize with a computed metabolomics data object.
        
        Args:
            metabo_obj: A MetaboInt or inherited object containing data.
        """
        # ==========================================
        # Global Matplotlib export configuration (supports Adobe AI editing)
        # ==========================================
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42
        plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]
        
        self.obj = metabo_obj
        self.attrs = metabo_obj.attrs
        
        self.st_col = self.attrs.get("sample_type", "Sample Type")
        self.bat_col = self.attrs.get("batch", "Batch")
        self.io_col = self.attrs.get("inject_order", "Inject Order")
        self.bg_col = self.attrs.get("bio_group", "Bio Group")
        
        sample_dict = self.attrs.get("sample_dict", {})
        self.qc_lbl = sample_dict.get("QC sample", "QC")
        self.act_lbl = sample_dict.get("Actual sample", "Sample")
        self.blk_lbl = sample_dict.get("Blank sample", "Blank")
        
        self.pal = {
            self.qc_lbl: "tab:red", 
            self.act_lbl: "tab:gray",
            True: "tab:red", 
            False: "tab:gray"
        }

    def _apply_standard_format(
        self, ax: plt.Axes, title: str = "", 
        xlabel: str = "", ylabel: str = "",
        title_fontsize: int = 14,
        label_fontsize: int = 12,
        tick_fontsize: int = 10
    ) -> None:
        """Apply global standard formatting to a given matplotlib axis.
        
        Args:
            ax: The matplotlib axis object to format.
            title: The title string for the axis.
            xlabel: The label for the x-axis.
            ylabel: The label for the y-axis.
            title_fontsize: Font size for the title.
            label_fontsize: Font size for the axis labels.
            tick_fontsize: Font size for the axis tick marks.
        """
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
        """Format and reposition legends for scatter plots.
        
        Args:
            fig: The matplotlib figure object containing the axis.
            ax: The matplotlib axis object containing the legend.
        """
        if ax.get_legend():
            ax.legend().remove()
        
        handles, labels = ax.get_legend_handles_labels()
        if self.st_col in labels and self.bat_col in labels:
            c_loc = labels.index(self.st_col)
            s_loc = labels.index(self.bat_col)
            
            leg_size = pd.Series(data={
                "Color": s_loc - c_loc - 1,
                "Style": len(labels) - s_loc - 1
            }).sort_values(ascending=True)
            
            s_dict = {
                "Color": slice(c_loc, s_loc),
                "Style": slice(s_loc, len(labels))
            }
            pos_dict = {"Color": "upper right", "Style": "upper left"}

            for j, leg_type in enumerate(iterable=leg_size.index):
                sub_leg = fig.legend(
                    handles=handles[s_dict[leg_type]], 
                    labels=labels[s_dict[leg_type]],
                    frameon=True, loc=pos_dict[leg_type], 
                    shadow=True, fontsize=10, 
                    borderpad=0.4, facecolor="white",
                    bbox_to_anchor=(0.5, 0)
                )
                if j < len(s_dict) - 1:
                    fig.add_artist(a=sub_leg)