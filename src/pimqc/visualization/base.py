"""Shared plotting lifecycle and export behavior for visualizer classes.

BaseMetaboVisualizer centralizes figure sizing, styles, legends, notebook
rendering, SVG cleanup, and file export for all domain visualizers. It provides
consistent publication-oriented output while preserving stage-specific plotting
logic in each processing or dataset visualization module.
"""

import io
import os
import re
import itertools
from typing import Optional, Union
from pathlib import Path

from matplotlib import font_manager
import matplotlib.pyplot as plt
import seaborn as sns

# Avoid INFO level logging to console when saving figures as .pdf
import logging
from loguru import logger

from . import plot_utils as pu
from ..io import utils as iu

logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


class BaseMetaboVisualizer:
    """Base class for all visualization suites in pi-metaboqc.

    This class provides global matplotlib and seaborn configurations
    to ensure consistent visual style across the pipeline, specifically
    targeting Adobe Illustrator compatibility and background consistency.
    """

    FIG_SAVE_FORMAT = ["svg", "pdf"]  # or "svg"
    FIG_DISPLAY_FORMAT = "png"
    # QA panels are consumed by the SVG report stitcher; PDFs are unnecessary
    # intermediate duplicates and can be disabled centrally by subclasses.
    QA_PANEL_SAVE_FORMAT = "svg"
    QA_LEGEND_SAVE_FORMAT = "svg"
    JUPYTER_PNG_DPI = 450
    VECTOR_FONT_FAMILY = "Helvetica"
    VECTOR_FONT_FALLBACKS = [
        "Helvetica",
        "Arial",
        "Liberation Sans",
        "Nimbus Sans",
        "DejaVu Sans",
        "sans-serif",
    ]
    VECTOR_TEXT_LINEBREAK_MODE = "keep"
    VECTOR_TEXT_REPLACE_HYPHEN = True
    VECTOR_TEXT_REPLACE_SPACES = True
    VECTOR_TEXT_HYPHEN_REPLACEMENT = "\u2013"
    VECTOR_AUTO_SCI_NOTATION = True

    def __init__(
        self,
        metabo_obj: object,
        save_format: Optional[Union[str, list, tuple]] = None,
        display_format: Optional[str] = None,
    ) -> None:
        """
        Initialize the visualizer with global styles.

        Args:
        metabo_obj: A MetaboInt or inherited object containing data.
        save_format: (Optional) Override the global default save format(s).
        display_format: (Optional) Override the global default display format.

        """
        # ==========================================
        # Global Matplotlib & Seaborn Configuration
        # ==========================================
        self.runtime_font_family = self._resolve_runtime_font(
            self.VECTOR_FONT_FALLBACKS
        )
        self.runtime_font_fallbacks = [
            self.runtime_font_family,
            *[
                font_name
                for font_name in self.VECTOR_FONT_FALLBACKS
                if font_name != self.runtime_font_family
            ],
        ]

        # Ensure high-quality editable vector export across PDF, PS, and SVG.
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42
        plt.rcParams["pdf.use14corefonts"] = False
        plt.rcParams["svg.fonttype"] = "none"
        plt.rcParams["savefig.dpi"] = 300
        plt.rcParams["savefig.bbox"] = "tight"
        plt.rcParams["text.usetex"] = False
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["axes.linewidth"] = pu.DEFAULT_AXIS_LINEWIDTH
        plt.rcParams["xtick.major.width"] = pu.DEFAULT_AXIS_LINEWIDTH
        plt.rcParams["ytick.major.width"] = pu.DEFAULT_AXIS_LINEWIDTH
        plt.rcParams["xtick.minor.width"] = pu.DEFAULT_AXIS_LINEWIDTH
        plt.rcParams["ytick.minor.width"] = pu.DEFAULT_AXIS_LINEWIDTH
        plt.rcParams["patch.linewidth"] = pu.DEFAULT_AXIS_LINEWIDTH
        plt.rcParams["hatch.linewidth"] = pu.DEFAULT_HATCH_LINEWIDTH

        # Prefer Helvetica, with broadly available sans-serif fallbacks for
        # editable vector output across Matplotlib, PDF viewers, and AI.
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = self.runtime_font_fallbacks
        plt.rcParams["font.stretch"] = "normal"
        plt.rcParams["font.style"] = "normal"
        plt.rcParams["font.variant"] = "normal"
        plt.rcParams["font.weight"] = "normal"
        plt.rcParams["mathtext.fontset"] = "custom"
        plt.rcParams["mathtext.rm"] = self.runtime_font_family
        plt.rcParams["mathtext.it"] = f"{self.runtime_font_family}:italic"
        plt.rcParams["mathtext.bf"] = f"{self.runtime_font_family}:bold"

        # Force white style to ensure background consistency
        sns.set_style("ticks")
        plt.rcParams["axes.facecolor"] = "white"
        plt.rcParams["figure.facecolor"] = "white"

        # Configure Axes defaults for editable vector exports.
        import matplotlib.axes

        if not hasattr(matplotlib.axes.Axes, "_pi_metaboqc_patched"):
            _orig_init = matplotlib.axes.Axes.__init__

            def _new_init(
                self_ax: matplotlib.axes.Axes, *args: object, **kwargs: object
            ) -> None:
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
        self.default_save_fmt = save_format or self.FIG_SAVE_FORMAT
        self.default_display_fmt = display_format or self.FIG_DISPLAY_FORMAT

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

        # Strategy A: Use standard filled markers for typical cohort sizes
        # (<=15)
        if n_batches <= 10:
            available_markers = [
                "o",
                "s",
                "^",
                "D",
                "v",
                "<",
                ">",
                "p",
                "*",
                "X",
            ]
            marker_generator = itertools.cycle(available_markers)

            self.style_map = {
                batch_id: next(marker_generator)
                for batch_id in self.all_batches
            }

        # Strategy B: Switch to MathText (Alphanumeric) for large cohorts (>15)
        else:
            # Renders explicit numbers (e.g., '1', '2', '3') to guarantee
            # absolute distinguishability and zero cognitive load in complex
            # plots.
            self.style_map = {
                batch_id: f"${i}$"
                for i, batch_id in enumerate(self.all_batches, start=1)
            }

        # Global Palette Definition
        self.pal = {
            self.qc_lbl: pu.PRIMARY_ACCENT_COLOR,
            self.act_lbl: pu.NEUTRAL_COLOR,
            True: pu.PRIMARY_ACCENT_COLOR,
            False: pu.NEUTRAL_COLOR,
        }

        # Global Legend Style Configuration
        self.LEGEND_KWARGS = dict(
            frameon=True,
            shadow=False,
            edgecolor="black",
            fontsize=pu.DEFAULT_LEGEND_FONTSIZE,
            title_fontsize=pu.DEFAULT_LEGEND_TITLE_FONTSIZE,
            borderpad=0.4,
            facecolor="white",
            framealpha=1.0,
        )

    @staticmethod
    def _resolve_runtime_font(font_candidates: list[str]) -> str:
        """Return the first installed sans-serif font from a candidate list."""
        for font_name in font_candidates:
            if font_name == "sans-serif":
                continue
            try:
                font_manager.findfont(font_name, fallback_to_default=False)
                return font_name
            except ValueError:
                continue
        return "DejaVu Sans"

    @staticmethod
    def _clean_svg_fonts_for_ai(
        svg_input: str | Path, target_font: str = "Helvetica"
    ) -> str:
        """
        Purify SVG font definitions safely for Adobe Illustrator compatibility.

        This method removes all fallback font declarations generated by
        Matplotlib
        and enforces a single target font. It uses strict regex boundaries to
        ensure the underlying XML/SVG tree structure remains completely intact.

        Args:
        svg_input: Raw SVG XML string or path to an SVG file.
        target_font: The desired font family name or CSS fallback stack.

        Returns:
        The purified SVG XML string.

        """
        svg_text = str(svg_input)
        is_svg_path = False
        if not svg_text.lstrip().startswith("<"):
            svg_path = Path(svg_text)
            if svg_path.exists() and svg_path.suffix.lower() == ".svg":
                svg_text = svg_path.read_text(encoding="utf-8")
                is_svg_path = True

        # 1. Clean inline CSS styles (e.g., style="font-family: 'DejaVu
        # Sans';").
        # The regex matches "font-family:" followed by any whitespace, and then
        # consumes all characters until it hits a semicolon (;) or a double
        # quote ("). This safely removes single-quoted fallback fonts without
        # corrupting surrounding HTML attributes or XML tags.
        svg_text = re.sub(
            r'font-family:\s*[^;"]+', f"font-family: {target_font}", svg_text
        )
        svg_text = re.sub(
            r'font-stretch:\s*[^;"]+', "font-stretch: normal", svg_text
        )
        svg_text = re.sub(
            r'font-style:\s*[^;"]+', "font-style: normal", svg_text
        )

        # 2. Clean standard XML attributes (e.g., font-family="DejaVu Sans").
        svg_text = re.sub(
            r'font-family="[^"]+"', f'font-family="{target_font}"', svg_text
        )
        svg_text = re.sub(
            r'font-stretch="[^"]+"', 'font-stretch="normal"', svg_text
        )
        svg_text = re.sub(
            r'font-style="[^"]+"', 'font-style="normal"', svg_text
        )

        # 3. Clean single-quoted attributes (e.g., font-family='DejaVu Sans').
        svg_text = re.sub(
            r"font-family='[^']+'", f"font-family='{target_font}'", svg_text
        )
        svg_text = re.sub(
            r"font-stretch='[^']+'", "font-stretch='normal'", svg_text
        )
        svg_text = re.sub(
            r"font-style='[^']+'", "font-style='normal'", svg_text
        )

        if is_svg_path:
            svg_path.write_text(svg_text, encoding="utf-8")

        return svg_text

    def _svg_font_family_stack(self) -> str:
        """Return a CSS-compatible SVG font-family fallback stack."""
        return ", ".join(self.VECTOR_FONT_FALLBACKS)

    def _prepare_figure_for_vector_export(
        self,
        fig: plt.Figure,
        linebreak_mode: str | None = None,
    ) -> None:
        """Normalize text and numeric axes before editable vector export."""
        try:
            fig.canvas.draw()
        except Exception:
            pass

        pu.normalize_figure_text_for_vector_export(
            fig=fig,
            target_font=self.runtime_font_fallbacks,
            replace_hyphen=self.VECTOR_TEXT_REPLACE_HYPHEN,
            replace_spaces=self.VECTOR_TEXT_REPLACE_SPACES,
            linebreak_mode=linebreak_mode or self.VECTOR_TEXT_LINEBREAK_MODE,
            hyphen_replacement=self.VECTOR_TEXT_HYPHEN_REPLACEMENT,
        )

        if self.VECTOR_AUTO_SCI_NOTATION:
            for ax in fig.axes:
                pu.apply_smart_axis_notation(ax=ax, axis="xy")

        pu.solidify_figure_alpha(fig=fig, bg_color="white")

        try:
            fig.canvas.draw()
        except Exception:
            pass

    def _prepare_open_figures_for_vector_export(
        self,
        linebreak_mode: str | None = None,
    ) -> None:
        """Apply vector-export preparation to all currently open figures."""
        for fig_num in plt.get_fignums():
            self._prepare_figure_for_vector_export(
                fig=plt.figure(fig_num),
                linebreak_mode=linebreak_mode,
            )

    def _apply_standard_format(
        self,
        ax: plt.Axes,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        append_stage: bool = True,
        custom_stage: str | None = None,
        **kwargs: object,
    ) -> None:
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
                    if hasattr(attr_value, "attrs") and isinstance(
                        attr_value.attrs, dict
                    ):
                        stage_label = attr_value.attrs.get("pipeline_stage", "")
                        if stage_label:
                            break

            if stage_label and f"[{stage_label}]" not in title:
                title = f"{title}\n[{stage_label}]"

        title = pu.normalize_vector_text(
            title,
            replace_hyphen=self.VECTOR_TEXT_REPLACE_HYPHEN,
            replace_spaces=self.VECTOR_TEXT_REPLACE_SPACES,
            linebreak_mode="keep",
            hyphen_replacement=self.VECTOR_TEXT_HYPHEN_REPLACEMENT,
        )
        xlabel = pu.normalize_vector_text(
            xlabel,
            replace_hyphen=self.VECTOR_TEXT_REPLACE_HYPHEN,
            replace_spaces=self.VECTOR_TEXT_REPLACE_SPACES,
            linebreak_mode="keep",
            hyphen_replacement=self.VECTOR_TEXT_HYPHEN_REPLACEMENT,
        )
        ylabel = pu.normalize_vector_text(
            ylabel,
            replace_hyphen=self.VECTOR_TEXT_REPLACE_HYPHEN,
            replace_spaces=self.VECTOR_TEXT_REPLACE_SPACES,
            linebreak_mode="keep",
            hyphen_replacement=self.VECTOR_TEXT_HYPHEN_REPLACEMENT,
        )

        ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        pu.change_weight(ax=ax, axis="xy")
        pu.change_fontsize(
            ax=ax,
            axis_ticks_fontsize=kwargs.get(
                "tick_fontsize", pu.DEFAULT_AXIS_TICK_FONTSIZE
            ),
            axis_label_fontsize=kwargs.get(
                "label_fontsize", pu.DEFAULT_AXIS_LABEL_FONTSIZE
            ),
            title_fontsize=kwargs.get(
                "title_fontsize", pu.DEFAULT_TITLE_FONTSIZE
            ),
            axis="xy",
        )

        if self.VECTOR_AUTO_SCI_NOTATION:
            pu.apply_smart_axis_notation(ax=ax, axis="xy")

        ax.tick_params(
            axis="both",
            width=pu.DEFAULT_AXIS_LINEWIDTH,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(pu.DEFAULT_AXIS_LINEWIDTH)

    @staticmethod
    def _apply_article_panel_format(
        ax: plt.Axes,
        title: str | None = None,
        tick_fontsize: float = pu.ARTICLE_AXIS_TICK_FONTSIZE,
        label_fontsize: float = pu.ARTICLE_AXIS_LABEL_FONTSIZE,
        title_fontsize: float = pu.ARTICLE_TITLE_FONTSIZE,
        annotation_fontsize: float = pu.ARTICLE_ANNOTATION_FONTSIZE,
    ) -> plt.Axes:
        """
        Compress an axis for a dense manuscript composite without changing
        defaults.
        """
        if title is not None:
            ax.set_title(
                title, fontsize=title_fontsize, fontweight="bold", pad=2.0
            )
        else:
            ax.title.set_fontsize(title_fontsize)
            ax.title.set_fontweight("bold")
            ax.title.set_pad(2.0)

        pu.change_fontsize(
            ax=ax,
            axis_ticks_fontsize=tick_fontsize,
            axis_label_fontsize=label_fontsize,
            title_fontsize=title_fontsize,
        )
        ax.tick_params(axis="both", pad=1.0, length=2.0, width=0.6)
        for label in [ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()]:
            label.set_fontsize(tick_fontsize)
        for text_artist in ax.texts:
            text_artist.set_fontsize(annotation_fontsize)
        BaseMetaboVisualizer._apply_article_legend_style(
            ax=ax,
            fontsize=pu.ARTICLE_LEGEND_FONTSIZE,
            title_fontsize=pu.ARTICLE_LEGEND_TITLE_FONTSIZE,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(pu.DEFAULT_AXIS_LINEWIDTH)
        return ax

    @staticmethod
    def _format_colorbar_axes(
        ax: plt.Axes,
        tick_fontsize: float = pu.DEFAULT_COLORBAR_TICK_FONTSIZE,
        label_fontsize: float = pu.DEFAULT_COLORBAR_LABEL_FONTSIZE,
    ) -> None:
        """
        Keep colorbar ticks, labels, and frame edges compact and consistent.
        """
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontsize(tick_fontsize)
        ax.xaxis.label.set_fontsize(label_fontsize)
        ax.yaxis.label.set_fontsize(label_fontsize)
        ax.tick_params(axis="both", width=pu.DEFAULT_AXIS_LINEWIDTH, length=2)
        for spine in ax.spines.values():
            spine.set_linewidth(pu.DEFAULT_AXIS_LINEWIDTH)

    @staticmethod
    def _apply_article_legend_style(
        ax: plt.Axes,
        fontsize: float = pu.ARTICLE_LEGEND_FONTSIZE,
        title_fontsize: float = pu.ARTICLE_LEGEND_TITLE_FONTSIZE,
        frame_linewidth: float = pu.DEFAULT_LEGEND_FRAME_LINEWIDTH,
        handle_linewidth: float = pu.DEFAULT_AXIS_LINEWIDTH,
        marker_edgewidth: float = pu.DEFAULT_MARKER_EDGEWIDTH,
    ) -> None:
        """Apply a restrained, vector-friendly style to article-only legends.

        ``_format_multi_legends`` stores all but the final sublegend in
        ``ax.artists``. Styling both locations keeps grouped standalone legends
        visually consistent without changing the standard dashboard defaults.
        """
        legend_candidates = [ax.get_legend(), *ax.artists]
        seen: set[int] = set()
        for legend in legend_candidates:
            if legend is None or not hasattr(legend, "get_texts"):
                continue
            legend_id = id(legend)
            if legend_id in seen:
                continue
            seen.add(legend_id)

            for legend_text in legend.get_texts():
                legend_text.set_fontsize(fontsize)
                legend_text.set_ha("left")
                legend_text.set_multialignment("left")
            legend_title = legend.get_title()
            if legend_title is not None:
                legend_title.set_fontsize(title_fontsize)
                legend_title.set_ha("left")
                legend_title.set_multialignment("left")

            try:
                legend._legend_box.align = "left"
            except Exception:
                pass

            frame = legend.get_frame()
            if frame is not None:
                frame.set_linewidth(frame_linewidth)

            handles = getattr(
                legend, "legend_handles", getattr(legend, "legendHandles", [])
            )
            for handle in handles:
                if hasattr(handle, "set_linewidth"):
                    handle.set_linewidth(handle_linewidth)
                if hasattr(handle, "set_markeredgewidth"):
                    handle.set_markeredgewidth(marker_edgewidth)

    def _format_single_legend(
        self,
        ax: plt.Axes,
        loc: str = "upper right",
        bbox_to_anchor: tuple[float, float] | None = (1.05, 1.0),
        group_title: str | None = None,
        legend_cols: int | None = None,
        max_item_rows: int | None = 6,
        **kwargs: object,
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

        unsupported_kwargs = {"title", "ncol"}
        found_unsupported = sorted(
            unsupported_kwargs.intersection(legend_kwargs)
        )
        if found_unsupported:
            raise TypeError(
                "Unsupported legend helper parameter(s): "
                f"{', '.join(found_unsupported)}. Use group_title=..., "
                "legend_cols=..., and max_item_rows=... instead."
            )

        if legend_cols is None and max_item_rows:
            import math

            legend_cols = math.ceil(len(handles) / max_item_rows)

        if legend_cols is not None:
            legend_kwargs["ncol"] = max(1, int(legend_cols))
        if group_title is not None:
            legend_kwargs["title"] = group_title

        # 5. Generate the new stylized legend and bind it to the Axes
        legend = ax.legend(
            handles,
            labels,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            **legend_kwargs,
        )
        self._center_legend_title(legend)
        self._style_legend_artists(legend)

    @staticmethod
    def _style_legend_artists(legend: object | None) -> None:
        """Apply the shared compact edge treatment to one legend."""
        if legend is None:
            return
        frame = legend.get_frame() if hasattr(legend, "get_frame") else None
        if frame is not None:
            frame.set_linewidth(pu.DEFAULT_LEGEND_FRAME_LINEWIDTH)
        handles = getattr(
            legend, "legend_handles", getattr(legend, "legendHandles", [])
        )
        for handle in handles:
            if hasattr(handle, "set_linewidth"):
                handle.set_linewidth(pu.DEFAULT_AXIS_LINEWIDTH)
            if hasattr(handle, "set_markeredgewidth"):
                handle.set_markeredgewidth(pu.DEFAULT_MARKER_EDGEWIDTH)

    @staticmethod
    def _center_legend_title(legend: object | None) -> None:
        """Left-align a Matplotlib legend title and entries when present."""
        if legend is None:
            return
        try:
            title = legend.get_title()
            if title is not None:
                title.set_ha("left")
                title.set_multialignment("left")
            for legend_text in legend.get_texts():
                legend_text.set_ha("left")
                legend_text.set_multialignment("left")
            legend._legend_box.align = "left"
        except Exception:
            return

    def _format_multi_legends(
        self,
        ax: plt.Axes,
        group_titles: list[str],
        group_header_labels: list[str] | None = None,
        loc: str = "upper left",
        start_bbox: tuple[float, float] = (1.05, 1.0),
        row_gap: float = 0.04,
        layout_cols: int = 1,
        column_gap: float = 0.15,
        sublegend_cols: Optional[Union[int, dict[str, int]]] = None,
        max_item_rows: Optional[int] = 6,
        **kwargs: object,
    ) -> list[object]:
        """

        Splits handles into separate boxes using the add_artist trick.
        Calculates offsets from each rendered legend's bounding box to prevent
        overlap, supports both multi-column group grids (layout_cols) and
        adaptive
        internal columns within individual sublegends, and ensures the final
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

        header_labels = group_header_labels or group_titles
        if len(header_labels) != len(group_titles):
            raise ValueError(
                "group_header_labels must contain exactly one entry "
                "for each group title."
            )

        title_idx = [
            i for i, label in enumerate(labels) if label in header_labels
        ]
        if not title_idx:
            return []
        title_idx.append(len(labels))

        leg_kwargs = getattr(self, "LEGEND_KWARGS", {}).copy()
        leg_kwargs.update(kwargs)

        unsupported_kwargs = {
            "title",
            "group_gap",
            "group_pad",
            "ncols",
            "col_pad",
            "group_ncols",
            "max_rows_per_sublegend",
            "ncol",
        }
        found_unsupported = sorted(unsupported_kwargs.intersection(leg_kwargs))
        if found_unsupported:
            raise TypeError(
                "Unsupported legend helper parameter(s): "
                f"{', '.join(found_unsupported)}. "
                "Use group_titles=..., row_gap=..., "
                "layout_cols=..., column_gap=..., sublegend_cols=..., "
                "and max_item_rows=... instead."
            )

        layout_cols = max(1, int(layout_cols))

        # Dynamic coordinate offset calculation based on absolute font size
        f_size = leg_kwargs.get("fontsize", 10)
        t_size = leg_kwargs.get("title_fontsize", 11)

        fig_w = ax.figure.get_size_inches()[0]
        fig_h = ax.figure.get_size_inches()[1]
        ax_w = max(ax.get_position().width * fig_w, 1e-6)
        ax_h = max(ax.get_position().height * fig_h, 1e-6)

        dy_row = (f_size / 72.0 * 1.6) / ax_h
        dy_title = (t_size / 72.0 * 1.8) / ax_h
        min_row_gap = (f_size / 72.0 * 0.35) / ax_h
        min_column_gap = (f_size / 72.0 * 0.90) / ax_w
        row_gap = max(row_gap, min_row_gap)
        column_gap = max(column_gap, min_column_gap)

        created = []
        n_groups = len(title_idx) - 1
        groups_per_col = math.ceil(n_groups / layout_cols)

        curr_x = start_bbox[0]

        def _group_item_ncols(title: str, item_count: int) -> int:
            """Resolve the number of internal legend columns for one group."""
            if isinstance(sublegend_cols, dict):
                value = sublegend_cols.get(title, 1)
            elif isinstance(sublegend_cols, int):
                value = sublegend_cols
            elif max_item_rows:
                value = math.ceil(item_count / max_item_rows)
            else:
                value = 1

            return max(1, int(value))

        def _legend_axes_bbox(legend: object) -> object | None:
            """
            Return the legend extent in Axes coordinates after a canvas draw.
            """
            try:
                ax.figure.canvas.draw()
                renderer = ax.figure.canvas.get_renderer()
                return legend.get_window_extent(renderer).transformed(
                    ax.transAxes.inverted()
                )
            except Exception:
                return None

        global_grp_idx = 0

        for col_idx in range(layout_cols):
            curr_y = start_bbox[1]
            col_legends = []
            col_right = None

            for _ in range(groups_per_col):
                if global_grp_idx >= n_groups:
                    break

                s_idx = title_idx[global_grp_idx]
                e_idx = title_idx[global_grp_idx + 1]
                sub_h = handles[s_idx + 1 : e_idx]
                sub_l = labels[s_idx + 1 : e_idx]

                if not sub_h:
                    global_grp_idx += 1
                    continue

                group_title = group_titles[global_grp_idx]
                global_grp_idx += 1
                sub_ncol = _group_item_ncols(group_title, len(sub_h))
                new_leg = ax.legend(
                    sub_h,
                    sub_l,
                    title=group_title,
                    loc=loc,
                    bbox_to_anchor=(curr_x, curr_y),
                    ncol=sub_ncol,
                    **leg_kwargs,
                )
                self._center_legend_title(new_leg)
                self._style_legend_artists(new_leg)

                # Keep each legend attached to the parent Axes for patchwork
                # layouts.
                # Leave the final legend as ax.legend_ to force PW bbox
                # expansion
                if global_grp_idx < n_groups:
                    ax.add_artist(new_leg)

                created.append(new_leg)
                col_legends.append(new_leg)

                legend_bbox = _legend_axes_bbox(new_leg)
                if legend_bbox is not None:
                    col_right = (
                        legend_bbox.x1
                        if col_right is None
                        else max(col_right, legend_bbox.x1)
                    )
                    if "upper" in loc:
                        curr_y = legend_bbox.y0 - row_gap
                    elif "lower" in loc:
                        curr_y = legend_bbox.y1 + row_gap
                else:
                    sub_rows = math.ceil(len(sub_h) / sub_ncol)
                    drop = dy_title + (sub_rows * dy_row) + row_gap
                    if "upper" in loc:
                        curr_y -= drop
                    elif "lower" in loc:
                        curr_y += drop

            # Calculate the maximum width of the current column to shift X
            if col_idx < layout_cols - 1 and col_legends:
                if col_right is not None:
                    curr_x = col_right + column_gap
                else:
                    curr_x += 0.35 + column_gap

        # Propagate to overall figure legends if not using patchworklib
        is_pw = type(ax).__module__.startswith("patchworklib") or type(
            getattr(ax, "figure", None)
        ).__module__.startswith("patchworklib")

        if getattr(ax, "figure", None) is not None and not is_pw:
            for obj in created:
                if obj not in ax.figure.legends:
                    ax.figure.legends.append(obj)

        return created

    def _plot_grouped_standalone_legends(
        self,
        ax: plt.Axes,
        legend_groups: list[tuple[str, list[object]]],
        loc: str = "upper left",
        start_bbox: tuple[float, float] = (0.0, 1.0),
        row_gap: float = 0.04,
        layout_cols: int = 1,
        column_gap: float = 0.15,
        sublegend_cols: Optional[Union[int, dict[str, int]]] = None,
        max_item_rows: Optional[int] = 6,
        **kwargs: object,
    ) -> list[object]:
        """
        Draw grouped legends inside a standalone axis with adaptive spacing.
        """
        import matplotlib.patches as mpatches

        ax.axis("off")
        dummy = mpatches.Rectangle(
            (0, 0), 1, 1, fill=False, edgecolor="none", visible=False
        )
        seed_handles: list[object] = []
        seed_labels: list[str] = []
        group_titles: list[str] = []
        group_header_labels: list[str] = []

        for group_index, (group_title, group_handles) in enumerate(
            legend_groups
        ):
            clean_handles = [
                handle for handle in group_handles if handle is not None
            ]
            if not clean_handles:
                continue

            group_titles.append(group_title)
            # Matplotlib suppresses labels beginning with an underscore, even
            # when handles are passed explicitly.  Use a private-looking but
            # renderable sentinel so _format_multi_legends can recover every
            # group reliably before the temporary legend is removed.
            header_label = f"PIMQC_LEGEND_HEADER_{group_index}"
            group_header_labels.append(header_label)
            seed_handles.append(dummy)
            seed_labels.append(header_label)
            for handle in clean_handles:
                seed_handles.append(handle)
                seed_labels.append(str(handle.get_label()))

        if not group_titles:
            return []

        ax.legend(
            seed_handles,
            seed_labels,
            loc=loc,
            bbox_to_anchor=start_bbox,
            frameon=False,
        )
        return self._format_multi_legends(
            ax=ax,
            group_titles=group_titles,
            group_header_labels=group_header_labels,
            loc=loc,
            start_bbox=start_bbox,
            row_gap=row_gap,
            layout_cols=layout_cols,
            column_gap=column_gap,
            sublegend_cols=sublegend_cols,
            max_item_rows=max_item_rows,
            **kwargs,
        )

    def _format_unified_multi_legends(
        self,
        ax: plt.Axes,
        group_titles: list[str],
        loc: str = "upper left",
        start_bbox: tuple[float, float] = (1.05, 1.0),
        group_title: str | None = None,
        legend_cols: int | None = None,
        max_item_rows: int | None = 6,
        **kwargs: object,
    ) -> list[object]:
        """
        Merges handles into a single box but simulates a split visual style
        using invisible dummy handles for group headers.
        """
        import math
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

        title_idx = [
            i for i, label in enumerate(labels) if label in group_titles
        ]
        if not title_idx:
            return []
        title_idx.append(len(labels))

        leg_kwargs = getattr(self, "LEGEND_KWARGS", {}).copy()
        leg_kwargs.update(kwargs)

        unsupported_kwargs = {"title", "ncols", "ncol"}
        found_unsupported = sorted(unsupported_kwargs.intersection(leg_kwargs))
        if found_unsupported:
            raise TypeError(
                "Unsupported legend helper parameter(s): "
                f"{', '.join(found_unsupported)}. Use group_title=..., "
                "legend_cols=..., and max_item_rows=... instead."
            )

        combined_h, combined_l = [], []
        dummy = mpatches.Rectangle(
            (0, 0), 1, 1, fill=False, edgecolor="none", visible=False
        )

        for i in range(len(title_idx) - 1):
            s_idx, e_idx = title_idx[i], title_idx[i + 1]
            sub_h = handles[s_idx + 1 : e_idx]
            sub_l = labels[s_idx + 1 : e_idx]

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

        leg_kwargs.update(
            {"labelspacing": 0.4, "borderpad": 0.6, "handletextpad": 0.5}
        )
        if legend_cols is None and max_item_rows:
            legend_cols = math.ceil(len(combined_h) / max_item_rows)
        if legend_cols is not None:
            leg_kwargs["ncol"] = max(1, int(legend_cols))
        if group_title is not None:
            leg_kwargs["title"] = group_title

        final_leg = ax.legend(
            combined_h,
            combined_l,
            loc=loc,
            bbox_to_anchor=start_bbox,
            **leg_kwargs,
        )
        self._center_legend_title(final_leg)
        self._style_legend_artists(final_leg)

        return [final_leg]

    def _render_jupyter_display(
        self,
        obj: object,
        is_patchwork: bool,
        display_format: str = "svg",
        width: Optional[Union[int, str]] = "60%",
        transparent: bool = False,
    ) -> None:
        """
        Render plots within a Jupyter Notebook with explicit format control.

        Handles architectural discrepancies between Matplotlib and Patchworklib.
        SVG outputs utilize HTML wrapping for responsive CSS width control.
        PNG outputs revert to native IPython Image rendering to guarantee
        compatibility with native IDE image toolbars (e.g., VS Code copy/save),
        while injecting a solid white background to prevent dark-mode blending.

        Args:
        obj (Any): The figure or patchwork composite object to display.
        is_patchwork (bool): True if the object is from patchworklib.
        display_format (str): Inline format, strictly 'svg' or 'png'.
        width (Optional[Union[int, str]]): Target CSS display width (SVG only).
        transparent (bool): Active alpha transparency indicator.

        """
        from IPython.display import HTML, Image, display
        import tempfile
        import re

        display_fmt = display_format.lower()
        if display_fmt not in ["svg", "png"]:
            logger.warning(
                f"Unsupported display format: '{display_format}'. "
                "Defaulting to 'svg'."
            )
            display_fmt = "svg"

        # Resolve CSS width explicitly (applies only to SVG HTML wrapper)
        w_css = (
            f"{width}px"
            if isinstance(width, int)
            else (width if width else "100%")
        )

        # Define consistent HTML container style for responsive SVG rendering
        container_style = (
            f"width:{w_css}; max-width:100%; margin: 0; "
            f"height:auto; background-color: white;"
        )

        if is_patchwork:
            # Isolate Patchworklib rendering in a runtime work directory.
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = os.path.join(tmpdir, f"preview.{display_fmt}")

                if display_fmt == "svg":
                    self._prepare_open_figures_for_vector_export()
                    # Maintain SVG transparency for HTML rendering
                    obj.savefig(tmp_path, transparent=transparent)
                    self._clean_svg_fonts_for_ai(
                        tmp_path, target_font=self._svg_font_family_stack()
                    )

                    with open(tmp_path, "r", encoding="utf-8") as f:
                        raw_svg = f.read()

                    # Inject flexible responsiveness mappers
                    preview_svg = re.sub(
                        r'(<svg[^>]*?\s)width="[^"]+"',
                        r'\1width="100%"',
                        raw_svg,
                        count=1,
                    )
                    preview_svg = re.sub(
                        r'(<svg[^>]*?\s)height="[^"]+"',
                        r'\1height="auto"',
                        preview_svg,
                        count=1,
                    )

                    display(
                        HTML(
                            f'<div style="{container_style}">'
                            f"{preview_svg}</div>"
                        )
                    )

                elif display_fmt == "png":
                    # Force white background and opaque rendering for the
                    # runtime
                    # preview image to ensure visibility in dark-themed IDEs
                    obj.savefig(
                        tmp_path,
                        transparent=False,
                        facecolor="white",
                        dpi=self.JUPYTER_PNG_DPI,
                    )
                    with open(tmp_path, "rb") as f:
                        img_data = f.read()

                    # Output raw image MIME type to activate native VS Code
                    # toolbars
                    display(Image(data=img_data, width=width))
        else:
            # Native matplotlib figure pipeline via raw BytesIO memory buffers
            buf = io.BytesIO()
            if display_fmt == "svg":
                self._prepare_figure_for_vector_export(obj)
                obj.savefig(buf, format="svg", transparent=transparent)
                svg_data = buf.getvalue().decode("utf-8")
                svg_data = self._clean_svg_fonts_for_ai(
                    svg_data, target_font=self._svg_font_family_stack()
                )

                svg_data = re.sub(
                    r'(<svg[^>]*?\s)width="[^"]+"',
                    r'\1width="100%"',
                    svg_data,
                    count=1,
                )
                svg_data = re.sub(
                    r'(<svg[^>]*?\s)height="[^"]+"',
                    r'\1height="auto"',
                    svg_data,
                    count=1,
                )

                display(
                    HTML(f'<div style="{container_style}">{svg_data}</div>')
                )

            elif display_fmt == "png":
                dpi_setting = self.JUPYTER_PNG_DPI

                # Force white background for memory stream
                obj.savefig(
                    buf,
                    format="png",
                    transparent=False,
                    facecolor="white",
                    dpi=dpi_setting,
                )
                img_data = buf.getvalue()

                # Output raw image MIME type to activate native VS Code toolbars
                display(Image(data=img_data, width=width))

    def save_and_close_fig(
        self,
        fig: plt.Figure,
        file_path: Optional[str] = None,
        show_plot: bool = False,
        save_format: Optional[str] = None,
        display_format: Optional[str] = None,
        width: Optional[Union[int, str]] = "30%",
        transparent: bool = False,
        bbox_inches: Optional[str] = None,
        pad_inches: float = 0.1,
    ) -> None:
        """
        Save a Matplotlib figure and manage targeted Jupyter display.

        Args:
        fig (matplotlib.figure.Figure): Target plot object.
        file_path (Optional[str]): Physical storage destination path.
        show_plot (bool): Inline deployment control flag.
        save_format (Optional[str]): Extension layout for physical file.
        display_format (str): Layout specification for notebook preview.
        width (Optional[Union[int, str]]): Notebook canvas bounding width.
        transparent (bool): Alpha-channel background indicator.
        bbox_inches (Optional[str]): Optional Matplotlib export bounding box.
        pad_inches (float): Padding around a tight export bounding box.

        """
        actual_save_fmt = save_format or self.default_save_fmt
        actual_display_fmt = display_format or self.default_display_fmt

        # Execute Notebook inline display prior to figure state destruction
        if show_plot and iu.is_jupyter():
            self._render_jupyter_display(
                obj=fig,
                is_patchwork=False,
                display_format=actual_display_fmt,
                width=width,
                transparent=transparent,
            )

        if file_path and actual_save_fmt:
            filepath_str = str(file_path)
            base_path = (
                filepath_str.rsplit(".", 1)[0]
                if "." in Path(filepath_str).name
                else filepath_str
            )

            format_list = (
                [actual_save_fmt]
                if isinstance(actual_save_fmt, str)
                else list(actual_save_fmt)
            )
            vector_formats = {"svg", "pdf"}
            if any(
                str(fmt).lower().strip(".") in vector_formats
                for fmt in format_list
            ):
                self._prepare_figure_for_vector_export(fig)

            for fmt in format_list:
                clean_fmt = fmt.lower().strip(".")
                out_path = f"{base_path}.{clean_fmt}"

                fig.savefig(
                    out_path,
                    transparent=transparent,
                    format=clean_fmt,
                    bbox_inches=bbox_inches,
                    pad_inches=pad_inches,
                )

                if clean_fmt == "svg":
                    self._clean_svg_fonts_for_ai(
                        out_path, target_font=self._svg_font_family_stack()
                    )

        plt.close(fig)

    def save_and_show_pw(
        self,
        pw_obj: object,
        file_path: Optional[str] = None,
        show_plot: bool = True,
        save_format: Optional[str] = None,
        display_format: Optional[str] = None,
        width: Optional[Union[int, str]] = "60%",
        transparent: bool = False,
    ) -> None:
        """
        Save a Patchworklib brick composite and route targeted display.

        Order of execution is strategically inverted: inline notebook rendering
        is processed first, avoiding canvas state destruction before file
        export.

        Args:
        pw_obj (Any): Target patchworklib brick composite object.
        file_path (Optional[str]): Physical storage destination path.
        show_plot (bool): Inline deployment control flag.
        save_format (Optional[str]): Extension layout for physical file.
        display_format (str): Layout specification for notebook preview.
        width (Optional[Union[int, str]]): Notebook canvas bounding width.
        transparent (bool): Alpha-channel background indicator.

        """
        actual_save_fmt = save_format or self.default_save_fmt
        actual_display_fmt = display_format or self.default_display_fmt

        try:
            # CRITICAL OPTIMIZATION: Process notebook visualization first.
            # This captures the canvas state perfectly before physical file
            # compilation.
            if show_plot and iu.is_jupyter():
                self._render_jupyter_display(
                    obj=pw_obj,
                    is_patchwork=True,
                    display_format=actual_display_fmt,
                    width=width,
                    transparent=transparent,
                )
        except Exception as e:
            logger.error(f"Jupyter rendering stage encountered an error: {e}")
            pass

        # Execute physical storage compilation
        if file_path and actual_save_fmt:
            filepath_str = str(file_path)
            base_path = (
                filepath_str.rsplit(".", 1)[0]
                if "." in Path(filepath_str).name
                else filepath_str
            )

            format_list = (
                [actual_save_fmt]
                if isinstance(actual_save_fmt, str)
                else list(actual_save_fmt)
            )
            vector_formats = {"svg", "pdf"}
            if any(
                str(fmt).lower().strip(".") in vector_formats
                for fmt in format_list
            ):
                self._prepare_open_figures_for_vector_export()

            for fmt in format_list:
                clean_fmt = fmt.lower().strip(".")
                out_path = f"{base_path}.{clean_fmt}"

                pw_obj.savefig(out_path, transparent=transparent)

                if clean_fmt == "svg":
                    self._clean_svg_fonts_for_ai(
                        out_path, target_font=self._svg_font_family_stack()
                    )

        # Avoid invalid Tk toolbar handles when multiple patchworklib figures
        # are
        # created and saved sequentially in the same Windows session.
        if "tk" not in plt.get_backend().lower():
            plt.close("all")
