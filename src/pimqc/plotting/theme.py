"""Apply pi-metaboqc plotting styles within reversible contexts.

Plot methods use Matplotlib and Seaborn context managers so constructing a
plotter cannot mutate an embedding application's global plotting state. The
module also prepares vector-export axes without monkey-patching Matplotlib.
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import wraps
from typing import ParamSpec, TypeVar

import matplotlib as mpl
import seaborn as sns

from . import plot_utils as pu

P = ParamSpec("P")
R = TypeVar("R")


def build_rc_params(font_fallbacks: list[str]) -> dict[str, object]:
    """Build plotting defaults without mutating global Matplotlib state."""
    primary_font = font_fallbacks[0]
    return {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.use14corefonts": False,
        "svg.fonttype": "none",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "text.usetex": False,
        "axes.unicode_minus": False,
        "axes.linewidth": pu.DEFAULT_AXIS_LINEWIDTH,
        "xtick.major.width": pu.DEFAULT_AXIS_LINEWIDTH,
        "ytick.major.width": pu.DEFAULT_AXIS_LINEWIDTH,
        "xtick.minor.width": pu.DEFAULT_AXIS_LINEWIDTH,
        "ytick.minor.width": pu.DEFAULT_AXIS_LINEWIDTH,
        "patch.linewidth": pu.DEFAULT_AXIS_LINEWIDTH,
        "hatch.linewidth": pu.DEFAULT_HATCH_LINEWIDTH,
        "font.family": "sans-serif",
        "font.sans-serif": font_fallbacks,
        "font.stretch": "normal",
        "font.style": "normal",
        "font.variant": "normal",
        "font.weight": "normal",
        "mathtext.fontset": "custom",
        "mathtext.rm": primary_font,
        "mathtext.it": f"{primary_font}:italic",
        "mathtext.bf": f"{primary_font}:bold",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    }


@contextmanager
def plot_style(font_fallbacks: list[str]) -> Generator[None, None, None]:
    """Apply package plotting defaults only within one plotting call."""
    with mpl.rc_context(rc=build_rc_params(font_fallbacks)):
        with sns.axes_style("ticks"):
            yield


def scoped_plot_method(func: Callable[P, R]) -> Callable[P, R]:
    """Run an instance plot method inside its scoped style context."""
    # Subclass discovery can encounter an inherited wrapper more than once;
    # preserve identity instead of nesting redundant style contexts.
    if getattr(func, "_pimqc_scoped_plot", False):
        return func

    @wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        instance = args[0]
        with plot_style(instance.runtime_font_fallbacks):
            return func(*args, **kwargs)

    wrapped._pimqc_scoped_plot = True
    return wrapped
