#src/pimqc/plot_utils.py
"""
Purpose of script: Utility functions for plotting.
Author: Kaikun Xu
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import List, Optional, Union
import warnings

def get_cmap(palette: str = "Set1") -> mpl.colors.Colormap:
    """Get a matplotlib colormap by name."""
    return mpl.colormaps[palette]

def custom_linear_cmap(
    color_list: List[str] = ["#1F77B4", "#FFFFFF", "#D62728"],
    n_colors: int = 100, 
    cmin: float = 0.0, cmax: float = 1.0
) -> mpl.colors.LinearSegmentedColormap:
    """Create a truncated custom linear segmented colormap."""
    base_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "Base_Cmap", colors=color_list, N=256
    )
    sampled_colors = base_cmap(np.linspace(cmin, cmax, n_colors))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "Truncated_Cmap", colors=sampled_colors, N=n_colors
    )
    cmap.set_bad(color="tab:gray")
    return cmap


def extract_qual_cmap(
    cmap: mpl.colors.Colormap, n_colors: Optional[int] = None) -> List[str]:
    """Extract hexadecimal colors from a qualitative colormap."""
    if n_colors is not None and n_colors >= cmap.N:
        warnings.warn(
            "The resampled number is greater than the total number.",
            category=UserWarning)
    n = n_colors if n_colors is not None else cmap.N
    colors = [mpl.colors.to_hex(cmap(i)).upper() for i in np.arange(0, n)]
    return colors

def extract_linear_cmap(
    cmap: mpl.colors.Colormap, 
    cmin: float = 0.0, 
    cmax: float = 1.0, 
    n_colors: Optional[int] = None
) -> List[str]:
    """Extract hexadecimal colors from a linear colormap given a range."""
    if n_colors is None:
        n_colors = cmap.N
    colors = [mpl.colors.to_hex(i).upper() for i in cmap(
        np.linspace(cmin, cmax, n_colors))]
    return colors

def change_axis_format(
    ax: plt.Axes, axis_format: str = "normal", axis: str = "xy") -> None:
    """Change the tick format of specified axes (percentage, scientific notation, etc.)."""
    if axis in ("x", "xy"):
        if axis_format in ("percentage", "percent", "pct"):
            ax.xaxis.set_major_locator(mticker.FixedLocator(ax.get_xticks()))
            ax.set_xticklabels(
                ["{:,.0f}".format(100 * x) for x in ax.get_xticks()])
        elif axis_format in ("scientific notation", "sci"):
            ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    if axis in ("y", "xy"):
        if axis_format in ("percentage", "percent", "pct"):
            ax.yaxis.set_major_locator(mticker.FixedLocator(ax.get_yticks()))
            ax.set_yticklabels(
                ["{:,.0f}".format(100 * x) for x in ax.get_yticks()])
        elif axis_format in ("scientific notation", "sci"):
            ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

def change_fontsize(
    ax: plt.Axes, 
    axis_ticks_fontsize: int = 14, 
    axis_label_fontsize: int = 14, 
    title_fontsize: int = 16, 
    axis: str = "xy"
) -> None:
    """Change the fontsize of axis ticks, labels, and title."""
    if axis in ("x", "xy"):
        ax.xaxis.label.set_fontsize(axis_label_fontsize)
        for tick in ax.get_xticklabels():
            tick.set_fontsize(axis_ticks_fontsize)
    if axis in ("y", "xy"):
        ax.yaxis.label.set_fontsize(axis_label_fontsize)
        for tick in ax.get_yticklabels():
            tick.set_fontsize(axis_ticks_fontsize)
    ax.title.set_fontsize(title_fontsize)

def change_weight(
    ax: plt.Axes, 
    axis_ticks_weight: str = "normal", 
    axis_label_weight: str = "normal", 
    title_weight: str = "bold", 
    axis: str = "xy"
) -> None:
    """Change the font weight of axis ticks, labels, and title."""
    if axis in ("x", "xy"):
        ax.xaxis.label.set_weight(axis_label_weight)
        for tick in ax.get_xticklabels():
            tick.set_weight(axis_ticks_weight)
    if axis in ("y", "xy"):
        ax.yaxis.label.set_weight(axis_label_weight)
        for tick in ax.get_yticklabels():
            tick.set_weight(axis_ticks_weight)
    ax.title.set_weight(title_weight)

def change_axis_rotation(
    ax: plt.Axes, rotation: float = 45, axis: str = "x") -> None:
    """Rotate the major tick labels of the specified axes."""
    if axis in ("x", "xy"):
        plt.setp(
            ax.xaxis.get_majorticklabels(), 
            rotation=rotation, 
            ha={0: "center", 90: "center"}.get(rotation, "right"), 
            va="top"
        )
    if axis in ("y", "xy"):
        plt.setp(
            ax.yaxis.get_majorticklabels(), 
            rotation=rotation, 
            ha={0: "right", 90: "center"}.get(rotation, "right"), 
            va={0: "center", 90: "center"}.get(rotation, "top")
        )

def show_values_on_bars(
    axs: Union[plt.Axes, np.ndarray], 
    value_format: str = "{:.2f}", 
    fontsize: float = 11, 
    position: str = "outside", 
    font_color: str = "k", 
    show_percentage: bool = False,
    pct_type: str = "total"
) -> None:
    """Annotate bar plots with their values.
    
    Args:
        axs: A single matplotlib Axes or a numpy array of Axes.
        value_format: Format string for the numerical value.
        fontsize: Font size of the text annotation.
        position: 'outside' or 'inside' the bar.
        font_color: Color of the text annotation.
        show_percentage: Whether to calculate and append percentage.
        pct_type: 'total' (plot-level) or 'group' (hue-container-level).
    """
    def _draw_label(ax: plt.Axes, p: mpl.patches.Patch, total: float):
        """Internal helper to draw a single text label."""
        height = p.get_height()
        value = value_format.format(height)
        
        if show_percentage and total > 0:
            value += "\n({:.1f}%)".format(100 * height / total)
        
        _x = p.get_x() + p.get_width() / 2
        
        if position == "outside":
            _y = p.get_y() + height
        else:
            _y = p.get_y() + height / 2
            
        ax.text(
            _x, _y, value, ha="center", 
            va="bottom" if (
                position == "outside" and height >= 0) else "center", 
            rotation=0, fontsize=fontsize, color=font_color
        )

    def _show_on_single_plot(ax: plt.Axes):
        # Logic 1: Percentage by group (for seaborn barplots with hue)
        if show_percentage and pct_type == "group" and ax.containers:
            for container in ax.containers:
                # Filter out patches that were manually removed
                valid_patches = [p for p in container if p in ax.patches]
                if not valid_patches:
                    continue
                
                # Calculate group total using only valid patches
                group_total = float(
                    np.sum([p.get_height() for p in valid_patches])
                )
                
                for p in valid_patches:
                    _draw_label(ax, p, group_total)
                    
        # Logic 2: Percentage by total across all bars (default legacy)
        else:
            valid_patches = ax.patches
            total_height = float(
                np.sum([p.get_height() for p in valid_patches])
            )
            for p in valid_patches:
                _draw_label(ax, p, total_height)
                
    if isinstance(axs, np.ndarray):
        for _, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

def confidence_ellipse(
    x: np.ndarray, 
    y: np.ndarray, 
    ax: plt.Axes, 
    n_std: float = 3.0, 
    facecolor: str = "none", 
    **kwargs
) -> mpl.patches.Ellipse:
    """Create a plot of the covariance confidence ellipse of `x` and `y`."""
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    
    if x.size != y.size:
        raise ValueError("x and y must be the same size.")
    
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    
    ellipse = Ellipse(
        (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
        facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    
    transf = transforms.Affine2D().rotate_deg(45).scale(
        scale_x, scale_y).translate(np.mean(x), np.mean(y))
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)