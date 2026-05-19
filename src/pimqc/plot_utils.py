#src/pimqc/plot_utils.py
"""
Purpose of script: Utility functions for plotting.
Author: Kaikun Xu
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from typing import List, Optional, Union
import warnings

def get_equivalent_hex(color, alpha=1.0, bg_color="white") -> str:
    """
    Convert a transparent color to its visually equivalent solid Hex color.
    
    This blends the target color with a background color (default white) 
    using the specified alpha value. This is highly useful for rendering 
    engines or export formats that drop or poorly support alpha channels.
    
    Args:
        + color: Color name ("tab:red"), hex ("#cccccc"), or RGB/RGBA tuple.
        Tuples can be 0.0-1.0 scale or 0-255 scale (e.g., (123, 234, 12)).
        + alpha (float): Transparency level (0.0 to 1.0).
        + bg_color (str): Background color to blend against.
        
    Returns:
        str: Solid 6-digit Hex color code (e.g., "#e6b3b3").
    """
    # 1. Normalize 0-255 scale RGB/RGBA tuples to 0.0-1.0 scale
    if isinstance(color, (tuple, list)):
        if any(c > 1.0 for c in color):
            color = tuple(c / 255.0 for c in color)
            
    # 2. Extract base RGBA (mcolors automatically handles names and hex)
    try:
        r, g, b, original_a = mcolors.to_rgba(color)
    except ValueError:
        raise ValueError(f"Invalid color format provided: {color}")
        
    bg_r, bg_g, bg_b, _ = mcolors.to_rgba(bg_color)
    
    # Determine final alpha (override original alpha if parameter is explicitly passed)
    final_alpha = alpha if alpha is not None else original_a
    
    # 3. Perform standard alpha blending against the background
    # Formula: Result = Foreground * alpha + Background * (1 - alpha)
    blend_r = r * final_alpha + bg_r * (1.0 - final_alpha)
    blend_g = g * final_alpha + bg_g * (1.0 - final_alpha)
    blend_b = b * final_alpha + bg_b * (1.0 - final_alpha)
    
    # 4. Convert blended RGB back to a solid 6-digit Hex string
    return mcolors.to_hex((blend_r, blend_g, blend_b))

def get_contrast_color(bg_color: Union[str, tuple]) -> str:
    """Calculate text color (black or white) to maximize contrast.
    
    Computes the perceived luminance of the background color, accounting 
    for alpha transparency (blended with a white canvas).
    
    Args:
        bg_color: A matplotlib-compatible color string or RGBA tuple.
        
    Returns:
        "white" if the background is dark, "k" (black) if it is light.
    """
    import matplotlib.colors as mcolors
    r, g, b, a = mcolors.to_rgba(bg_color)
    
    # Blend with a white background assuming transparency shows white
    r = r * a + 1.0 * (1 - a)
    g = g * a + 1.0 * (1 - a)
    b = b * a + 1.0 * (1 - a)
    
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.65 else "k"

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
    show_percentage: bool = False,
    pct_type: str = "total",
    stacked: bool = False,
    skip_zero: bool = True,
    threshold_pct: float = 0.10
) -> None:
    """Annotate bar plots with their values automatically.
    
    Args:
        axs: A single matplotlib Axes or a numpy array of Axes.
        value_format: Format string for the numerical value.
        fontsize: Font size of the text annotation.
        show_percentage: Whether to calculate and append percentage.
        pct_type: 'total' (plot-level) or 'group' (hue-container-level).
        stacked: Enable intelligent parsing for stacked bar charts.
        skip_zero: Do not annotate patches with a height of exactly 0.
        threshold_pct: Hide annotations if height is less than this
            percentage of the stack's total height (stacked=True only).
    """

    def _draw_label(
        ax: plt.Axes, 
        p: mpl.patches.Patch, 
        total: float, 
        stack_total: Optional[float] = None
    ):
        height = p.get_height()
        if skip_zero and height == 0:
            return
            
        # Check threshold for stacked patches to avoid clutter
        if stacked and stack_total is not None and stack_total > 0:
            if (height / stack_total) < threshold_pct:
                return
                
        value = value_format.format(height)
        if show_percentage and total > 0:
            value += "\n({:.1f}%)".format(100 * height / total)
            
        _x = p.get_x() + p.get_width() / 2
        
        # Split color logic based on placement position
        if stacked:
            _y = p.get_y() + height / 2
            va = "center"
            # Inner patches use auto-adaptive color based on background
            c = get_contrast_color(p.get_facecolor())
        else:
            _y = p.get_y() + height
            va = "bottom" if height >= 0 else "top"
            # Outside annotations must be black against the white canvas
            c = "k"
            
        ax.text(
            _x, _y, value, ha="center", va=va, 
            rotation=0, fontsize=fontsize, color=c
        )

    def _show_on_single_plot(ax: plt.Axes):
        if stacked and ax.containers:
            stack_totals = {}
            for container in ax.containers:
                for p in container:
                    if skip_zero and p.get_height() == 0:
                        continue
                    x_center = p.get_x() + p.get_width() / 2
                    cur_top = p.get_y() + p.get_height()
                    if x_center in stack_totals:
                        stack_totals[x_center] = max(
                            stack_totals[x_center], cur_top
                        )
                    else:
                        stack_totals[x_center] = cur_top
                        
            for container in ax.containers:
                for p in container:
                    x_center = p.get_x() + p.get_width() / 2
                    st_tot = stack_totals.get(x_center, 0)
                    _draw_label(ax, p, 0, stack_total=st_tot)
                    
            # Total value on top of the stacked bars (Always Black)
            max_h = max(stack_totals.values()) if stack_totals else 1
            for x_c, total_h in stack_totals.items():
                if skip_zero and total_h == 0:
                    continue
                ax.text(
                    x_c, total_h + (max_h * 0.02), 
                    value_format.format(total_h), 
                    ha="center", va="bottom", color="k", 
                    fontsize=fontsize + 1, fontweight="bold"
                )
        else:
            if show_percentage and pct_type == "group" and ax.containers:
                for container in ax.containers:
                    valid_p = [p for p in container if p in ax.patches]
                    if not valid_p: 
                        continue
                    grp_tot = float(
                        np.sum([p.get_height() for p in valid_p])
                    )
                    for p in valid_p:
                        _draw_label(ax, p, grp_tot)
            else:
                v_p = ax.patches
                tot_h = float(np.sum([p.get_height() for p in v_p]))
                for p in v_p:
                    _draw_label(ax, p, tot_h)
                    
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