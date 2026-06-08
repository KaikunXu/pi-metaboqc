# src/pimqc/report_utils.py
"""
Script purpose: Build narrative reports from completed pipeline artifacts.

VisualAssetReporter.compile_assessor_report() scans the output workspace for
QA folders, collects stage-specific SVG assets, stitches compatible grids, and
places report-ready figures under the report asset directory. The narrative
reporter consolidates pipeline metrics and QA metrics into a single Jinja2
context, then generate_markdown() renders both comprehensive and brief reports.
export_report() converts generated Markdown files to PDF/HTML with Pandoc and
the configured rendering backend, handling dependency checks and fallbacks.
"""

import os
import sys
import math
from datetime import datetime

import subprocess
import ctypes
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from tabulate import tabulate

from loguru import logger
from typing import Union, Optional, Dict, Any


# =========================================================================
# Atomic Utility Functions
# =========================================================================
def _get_optimal_cols(n_docs: int, max_cols: int = 4) -> int:
    """Calculates optimal grid columns for subplot layout.

    Args:
        n_docs (int): Total number of documents to stitch.
        max_cols (int): Maximum allowed columns. Defaults to 4.

    Returns:
        int: Optimal number of columns bounded by max_cols.
    """
    if n_docs <= 0:
        return 1

    # Preset aesthetic mappings for typical plot counts to avoid
    # disproportionate grid aspect ratios (e.g., forcing 2x2 for 4 plots)
    layout_map = {
        1: 1,
        2: 2,
        3: 3,
        4: 2,
        5: 3,
        6: 3,
        7: 4,
        8: 4,
        9: 3,
        10: 4,
        11: 4,
        12: 4,
    }

    if n_docs in layout_map:
        cols = layout_map[n_docs]
    else:
        # Fallback for dynamic calculation on arbitrary large numbers
        cols = math.ceil(math.sqrt(n_docs))

    return min(max_cols, cols)


def stitch_svg_grids(
    svg_paths: list,
    file_path: str,
    cols: Union[int, str] = "auto",
    max_cols: int = 4,
    show_plot: bool = True,
    save_format: Optional[Union[str, list, tuple]] = ["svg", "pdf"],
    display_format: str = "png",
    width: Optional[Union[int, str]] = "60%",
) -> bool:
    """Stitches multiple SVGs into a grid, saves to disk, and displays inline.

    Assembles individual SVG plots into a unified master SVG grid. Provides
    dynamic file conversion to PDF/PNG via CairoSVG and aligns with the
    project's standard Jupyter rendering logic (supporting native VS Code
    image toolbars and responsive layouts).

    Args:
        svg_paths (list): List of paths to source SVG subplots.
        file_path (str): Destination base path for the stitched file(s).
        cols (Union[int, str]): Number of columns, or 'auto'.
        max_cols (int): Maximum allowed columns when using 'auto'.
        show_plot (bool): Whether to render the result in Jupyter/VS Code.
        save_format (Optional[Union[str, list, tuple]]): Disk export formats.
        display_format (str): Inline format for notebook preview ('svg' or 'png').
        width (Optional[Union[int, str]]): CSS width for Jupyter display.

    Returns:
        bool: True if the stitching and saving were successful, False otherwise.
    """
    try:
        import svgutils.transform as sg
    except ImportError:
        logger.error("Please install 'svgutils' via 'pip install svgutils'.")
        return False

    try:
        # 1. Validate and filter input paths
        valid_paths = [
            p for p in svg_paths if Path(p).exists() and Path(p).stat().st_size > 0
        ]
        if not valid_paths:
            return False

        # 2. Calculate optimal grid dimensions
        n_docs = len(valid_paths)
        active_cols = (
            _get_optimal_cols(n_docs, max_cols) if cols == "auto" else int(cols)
        )
        rows = (n_docs + active_cols - 1) // active_cols

        # 3. Robust UTF-8 reading to prevent GBK codec errors on Windows
        svg_figs = []
        for p in valid_paths:
            with open(p, "r", encoding="utf-8") as f:
                svg_figs.append(sg.fromstring(f.read()))

        # 4. Extract maximum dimensions for uniform grid alignment
        def parse_dim(val: str) -> float:
            import re

            match = re.search(r"(\d+\.?\d*)", str(val))
            return float(match.group(1)) if match else 0.0

        max_w = max(parse_dim(f.width) for f in svg_figs)
        max_h = max(parse_dim(f.height) for f in svg_figs)

        total_w = max_w * active_cols
        total_h = max_h * rows

        # 5. Initialize the master canvas with a white background
        fig = sg.SVGFigure(f"{total_w}", f"{total_h}")
        bg_svg = sg.fromstring(
            '<svg><rect width="100%" height="100%" fill="white"/></svg>'
        ).getroot()

        plots = [bg_svg]

        # 6. Position each subplot into the calculated grid slot
        for i, s_fig in enumerate(svg_figs):
            row, col = divmod(i, active_cols)
            plot = s_fig.getroot()
            plot.moveto(col * max_w, row * max_h)
            plots.append(plot)

        # 7. Assemble and set viewBox
        fig.append(plots)
        fig.root.set("viewBox", f"0 0 {total_w} {total_h}")

        # Extract raw byte string for memory-based format conversion
        merged_svg_bytes = fig.to_str()
        merged_svg_str = merged_svg_bytes.decode("utf-8")

        # 8. Format Normalization & Physical Storage Logic
        filepath_str = str(file_path)
        base_path = (
            filepath_str.rsplit(".", 1)[0]
            if "." in Path(filepath_str).name
            else filepath_str
        )

        if save_format:
            format_list = (
                [save_format] if isinstance(save_format, str) else list(save_format)
            )

            for fmt in format_list:
                clean_fmt = fmt.lower().strip(".")
                out_path = f"{base_path}.{clean_fmt}"

                if clean_fmt == "svg":
                    with open(out_path, "wb") as f:
                        f.write(merged_svg_bytes)
                else:
                    # Leverage CairoSVG for dynamic vector/raster conversion
                    try:
                        import cairosvg

                        if clean_fmt == "pdf":
                            cairosvg.svg2pdf(
                                bytestring=merged_svg_bytes, write_to=out_path
                            )
                        elif clean_fmt == "png":
                            cairosvg.svg2png(
                                bytestring=merged_svg_bytes, write_to=out_path
                            )
                    except ImportError:
                        logger.error(
                            f"Cannot save {clean_fmt.upper()}: 'cairosvg' is "
                            "not installed. Run `pip install cairosvg`."
                        )

        # 9. Environment-safe Jupyter rendering logic
        if show_plot:
            try:
                from . import io_utils as iu

                if iu.is_jupyter():
                    from IPython.display import HTML, Image, display
                    import re

                    display_fmt = display_format.lower()
                    if display_fmt not in ["svg", "png"]:
                        display_fmt = "svg"

                    w_css = (
                        f"{width}px"
                        if isinstance(width, int)
                        else (width if width else "100%")
                    )

                    if display_fmt == "svg":
                        # Strip absolute dimensions for responsive UI preview
                        preview_svg = re.sub(
                            r'(<svg[^>]*?\s)width="[^"]+"',
                            r'\1width="100%"',
                            merged_svg_str,
                            count=1,
                        )
                        preview_svg = re.sub(
                            r'(<svg[^>]*?\s)height="[^"]+"',
                            r'\1height="auto"',
                            preview_svg,
                            count=1,
                        )

                        container_style = (
                            f"width:{w_css}; max-width:100%; margin: 0 auto; "
                            f"height:auto; background-color: white;"
                        )
                        display(
                            HTML(f'<div style="{container_style}">{preview_svg}</div>')
                        )

                    elif display_fmt == "png":
                        try:
                            import cairosvg

                            # Force solid white background for VS Code dark mode
                            png_data = cairosvg.svg2png(
                                bytestring=merged_svg_bytes, background_color="white"
                            )
                            # Native Image rendering activates VS Code toolbars
                            display(Image(data=png_data, width=width))
                        except ImportError:
                            logger.error(
                                "Cannot preview PNG: 'cairosvg' missing. "
                                "Falling back to SVG display."
                            )
                            # Safe fallback to SVG if Cairo is missing
                            container_style = (
                                f"width:{w_css}; max-width:100%; margin: 0 auto; "
                                f"height:auto; background-color: white;"
                            )
                            display(
                                HTML(
                                    f'<div style="{container_style}">{merged_svg_str}</div>'
                                )
                            )

            except Exception as e:
                # Silently catch to keep terminal logs clean in headless mode
                logger.debug(f"Jupyter rendering bypassed: {e}")

        return True

    except Exception as e:
        logger.error(f"Failed to stitch SVG grid: {e}")
        return False


# =========================================================================
# Class 1: VisualAssetReporter (Handles QA Grids & Images)
# =========================================================================
class VisualAssetReporter:

    def __init__(self, base_dir: Union[str, Path]) -> None:
        """
        Initialize the reporter at the project workspace level.

        Args:
            base_dir: Root output directory containing step folders.
        """
        self.base_dir = Path(base_dir)
        self.qa_folders = self._detect_qa_folders()

    def _detect_qa_folders(self) -> list[str]:
        """Automatically scan for QA directories and sort them."""
        if not self.base_dir.exists():
            return []
        folders = [d for d in self.base_dir.iterdir() if d.is_dir() and "QA" in d.name]
        return sorted([d.name for d in folders])

    def compile_assessor_report(
        self,
        is_multi_batch: bool = True,
        report_folder: str = "13_Report_Markdown",
        cols: Union[int, str] = "auto",
    ) -> None:
        """Compile QA SVG plots into grids and deploy to report assets."""
        if not self.qa_folders:
            logger.error("No QA folders detected in the base directory.")
            return

        report_path = self.base_dir / report_folder
        assets_path = report_path / "assets"

        report_path.mkdir(parents=True, exist_ok=True)
        assets_path.mkdir(parents=True, exist_ok=True)

        if is_multi_batch:
            corr_prefix = "03_Batch_Correlation_Dashboard"
            corr_file = "Batch_Correlation_Heatmap.svg"
            logger.info("Multi-batch design detected. Assembling Batch Grid.")
        else:
            corr_prefix = "03_QC_Correlation_Dashboard"
            corr_file = "QC_Correlation_Heatmap.svg"
            logger.info("Single-batch design detected. Assembling QC Grid.")

        # Target files now expect the .svg extension natively
        target_map = {
            "01_QC_Sample_RSD_Dashboard": "RSD_Barplot.svg",
            "02_PCA_Scatter_Dashboard": "PCA_Scatter_QC_Sample.svg",
            corr_prefix: corr_file,
            "04_Outlier_Diagnosis_Dashboard": "Outlier_Scatter.svg",
        }

        for prefix, target_file in target_map.items():
            # Output directly to the assets folder as an SVG grid
            svg_out = assets_path / f"{prefix}.svg"
            input_svgs = []
            for folder in self.qa_folders:
                folder_path = self.base_dir / folder

                direct_file = folder_path / target_file
                if direct_file.exists():
                    input_svgs.append(direct_file)
                else:
                    subdirs = [d for d in folder_path.iterdir() if d.is_dir()]

                    subdirs.sort(key=lambda d: d.stat().st_mtime)

                    for sub in subdirs:
                        sub_file = sub / target_file
                        if sub_file.exists():
                            input_svgs.append(sub_file)

            if not input_svgs:
                logger.warning(f"Skipped {prefix}: No source SVGs found.")
                continue
            # Execute SVG stitching
            stitch_svg_grids(
                svg_paths=input_svgs,
                file_path=svg_out,
                cols=cols,
                save_format="svg",
                display_format="png",
            )
        logger.success(f"Report SVG assets compiled at: {assets_path}")


# =========================================================================
# Class 2: NarrativeStatsReporter (Handles attrs & Markdown Text)
# =========================================================================
class NarrativeStatsReporter:
    """Extracts metadata from MetaboInt objects to generate a single report."""

    # --- Define CSS as a class constant for cleaner maintenance ---
    REPORT_CSS = """
    /* ====================================================================
     * PI-METABOQC UNIFIED REPORT STYLESHEET
     * Optimized for WeasyPrint (PDF) and modern web browsers (HTML).
     * ==================================================================== */

    /* --- 1. Screen Reading Layout (Web Browsers) --- */
    @media screen {
        body {
            max-width: 1200px !important;
            margin: 0 auto !important;
            padding: 20px;
        }
    }

    /* --- 2. Default Image Constraints --- */
    .full-width-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 100%;
        height: auto;
    }

    /* --- 3. Out-of-bounds Rendering Magic (WeasyPrint PDF) --- */
    @media print {
        @page {
            /* Explicitly specify physical page margins for calculation */
            margin: 25mm; 
        }
        
        .full-width-image {
            /* Occupy exactly 95% of the total physical page width.
             * Math: 0.95 * (100% content width + 50mm total margins) */
            width: calc(95% + 47.5mm) !important; 
            
            /* Shift outward equally to perfectly center the 95% width.
             * Math: -(Desired Width - 100% Content Width) / 2 */
            margin-left: calc(2.5% - 23.75mm) !important;
            margin-right: calc(2.5% - 23.75mm) !important;
        }
    }

    /* --- 4. Academic Three-Line Table Style --- */
    table {
        border-collapse: collapse;
        width: auto;      /* Let the table shrink to fit its content */
        min-width: 80%;   /* Maintain a reasonably wide academic look */
        margin: 0 auto 24px auto; /* Center table and add bottom spacing */
        font-size: 14px;
    }
    th, td {
        padding: 8px 12px;
        text-align: left;
    }
    thead {
        border-top: 2px solid black;
        border-bottom: 1px solid black;
    }
    tbody {
        border-bottom: 2px solid black;
    }

    /* --- 5. Figure and Table Captions --- */
    figure {
        /* Remove default margin, keep bottom spacing */
        margin: 0 0 24px 0; 
        /* Ensure the containing block is centered */
        text-align: center; 
    }
    figcaption {
        text-align: center;
        font-size: 14px;
        color: #333333;
        margin-top: 10px;
        font-weight: 500;
    }
    caption {
        caption-side: top;  /* Force table title above the table */
        text-align: center; /* Center the title */
        font-weight: bold;
        color: #333333;
        margin-bottom: 8px; /* Spacing between title and table */
    }

    /* --- 6. CSS Auto-numbering Magic --- */
    body {
        /* Initialize chapter counter. (Assuming chapter 2 for QA) */
        counter-reset: chapter-counter 0; 
    }
    h2 {
        /* Increment chapter counter, reset table and figure counters */
        counter-increment: chapter-counter;
        counter-reset: table-counter figure-counter;
    }
    caption::before {
        /* Auto-generate Table prefix (e.g., "Table 2.1: ") */
        counter-increment: table-counter;
        content: "Table " counter(chapter-counter) "." 
                counter(table-counter) ": ";
    }
    figcaption::before {
        /* Auto-generate Figure prefix (e.g., "Figure 2.1: ") */
        counter-increment: figure-counter;
        content: "Figure " counter(chapter-counter) "." 
                counter(figure-counter) ": ";
        font-weight: bold;
    }

    # /* --- 7. Advanced Typography Support (GitHub-Light Stack) --- */
    # body {
    #     /* Standard GitHub-Light sans-serif font stack */
    #     font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, 
    #                 Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
    #     line-height: 1.5;
    #     color: #24292e;
    # }

    # code, pre, kbd, samp {
    #     /* Professional monospace stack for data & parameters */
    #     font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, 
    #                 Consolas, "Liberation Mono", monospace;
    #     font-size: 85%;
    #     background-color: rgba(27, 31, 35, 0.05);
    #     padding: 0.2em 0.4em;
    #     border-radius: 3px;
    # }

    # pre code {
    #     background-color: transparent;
    #     padding: 0;
    # }
    """
    # Define standard pipeline stages for ordered iteration
    _QA_STAGES = [
        ("raw_dataset", "Raw data"),
        ("high_mv_feature_filtering", "High-missing value features filtering"),
        ("intra_batch_correction", "Intra-batch correction"),
        ("inter_batch_correction", "Inter-batch correction"),
        ("global_correction", "Global SERRF correction"),
        ("low_quality_feature_filtering", "Low-quality features filtering"),
        ("missing_value_imputation", "Imputation"),
        ("normalization", "Normalization"),
    ]

    def __init__(self, base_dir: str) -> None:
        """Initialize the reporter with base directory and Jinja2 env."""
        self.base_dir = Path(base_dir)
        template_path = Path(__file__).parent / "templates"
        self.env = Environment(loader=FileSystemLoader(str(template_path)))

    def _create_batch_table(self, batch_dist: Dict[str, Any]) -> str:
        """Generates a Markdown table displaying batch sample distributions."""
        rows = []
        sum_total = 0
        sum_qc = 0
        sum_blank = 0
        sum_sample = 0

        if isinstance(batch_dist, dict):
            for b_id, b_info in batch_dist.items():
                total = b_info.get("Total", 0)
                qc = b_info.get("QC", 0)
                blank = b_info.get("Blank", 0)
                sample = b_info.get("Sample", 0)

                sum_total += total
                sum_qc += qc
                sum_blank += blank
                sum_sample += sample

                rows.append(
                    [b_id, total, qc, blank, sample, b_info.get("Inject Order", "N/A")]
                )

        if rows:
            rows.append(["All", sum_total, sum_qc, sum_blank, sum_sample, "/"])

        headers = ["Batch", "Total", "QC", "Blank", "Sample", "Inject Order"]
        table_str = tabulate(
            rows,
            headers=headers,
            tablefmt="github",
            stralign="center",
            numalign="center",
        )
        return f"\n\n{table_str}\n\n"

    def _create_rsd_summary_table(self, qa_metrics: Dict[str, Any]) -> str:
        """Generates a table detailing QC and Sample RSD distribution.

        Args:
            qa_metrics (Disct[str, Any]): Dictionary of quality assessment
                outputs for all pipeline stages.

        Returns:
            str: A formatted Markdown table string.
        """
        rows = []
        for stage_key, stage_name in self._QA_STAGES:
            qa_data = qa_metrics.get(stage_key, {})
            if not qa_data:
                continue

            rsd_dist = qa_data.get("rsd_distribution", {})
            if not rsd_dist:
                continue

            # Extract metrics for both Quality Control and Actual Samples
            qc_rsd = rsd_dist.get("qc", {})
            sample_rsd = rsd_dist.get("actual", {})

            # Build matrix rows with nested group logic for scannability
            if qc_rsd or sample_rsd:
                rows.append(
                    [
                        stage_name,
                        # QC Data
                        qc_rsd.get("0-10%", 0),
                        qc_rsd.get("10-20%", 0),
                        qc_rsd.get("20-30%", 0),
                        qc_rsd.get(">30%", 0),
                        # Sample Data
                        sample_rsd.get("0-10%", 0),
                        sample_rsd.get("10-20%", 0),
                        sample_rsd.get("20-30%", 0),
                        sample_rsd.get(">30%", 0),
                    ]
                )

        headers = [
            "Pipeline Stage",
            "QC 0-10%",
            "QC 10-20%",
            "QC 20-30%",
            "QC >30%",
            "Sample 0-10%",
            "Sample 10-20%",
            "Sample 20-30%",
            "Sample >30%",
        ]

        table_str = tabulate(
            rows,
            headers=headers,
            tablefmt="github",
            stralign="center",
            numalign="center",
        )
        return f"\n\n{table_str}\n\n" if rows else ""

    def _create_pca_summary_table(self, qa_metrics: Dict[str, Any]) -> str:
        """Generates a table detailing PCA drift and silhouette metrics."""
        rows = []
        for stage_key, stage_name in self._QA_STAGES:
            qa_data = qa_metrics.get(stage_key, {})
            if not qa_data:
                continue
            pca = qa_data.get("pca", {})
            if pca:
                pc1 = pca.get("pc1_variance")
                pc1_str = f"{pc1*100:.2f}%" if pc1 else "N/A"

                pc2 = pca.get("pc2_variance")
                pc2_str = f"{pc2*100:.2f}%" if pc2 else "N/A"

                disp = pca.get("relative_dispersion")
                disp_str = f"{disp:.4f}" if disp is not None else "N/A"

                silh = pca.get("batch_silhouette")
                silh_str = f"{silh:.4f}" if silh is not None else "N/A"

                shift = pca.get("centrality_shift")
                shift_str = f"{shift:.4f}" if shift is not None else "N/A"

                rows.append(
                    [stage_name, pc1_str, pc2_str, disp_str, silh_str, shift_str]
                )

        headers = [
            "Pipeline Stage",
            "PC1 Var",
            "PC2 Var",
            "Rel. Dispersion",
            "Batch Silh.",
            "Cent. Shift",
        ]
        table_str = tabulate(
            rows,
            headers=headers,
            tablefmt="github",
            disable_numparse=True,
        )
        return f"\n\n{table_str}\n\n" if rows else ""

    def _create_corr_summary_table(self, qa_metrics: Dict[str, Any]) -> str:
        """Generates a table summarizing pooled QC correlations.
        Dynamically adapts columns based on whether the dataset is single
        or multi-batch.
        """
        # 1. Determine if the dataset is multi-batch by scanning QA metrics
        is_multi_batch = False
        for stage_key, _ in self._QA_STAGES:
            qa_data = qa_metrics.get(stage_key, {})
            if (
                qa_data.get("correlation", {})
                .get("batch_level", {})
                .get("is_multi_batch", False)
            ):
                is_multi_batch = True
                break

        rows = []
        for stage_key, stage_name in self._QA_STAGES:
            qa_data = qa_metrics.get(stage_key, {})
            if not qa_data:
                continue

            corr_data = qa_data.get("correlation", {})
            sample_level = corr_data.get("sample_level", {})
            batch_level = corr_data.get("batch_level", {})

            # Extract inner-batch median (with fallback to legacy "median" key)
            inner = sample_level.get("inner_batch_median", "N/A")
            if inner == "N/A":
                inner = corr_data.get("median", "N/A")

            # Strictly format to 4 decimal places to ensure alignment
            inner_str = f"{inner:.4f}" if isinstance(inner, float) else str(inner)

            row = [stage_name, inner_str]

            # Append multi-batch specific metrics only if applicable
            if is_multi_batch:
                cross = sample_level.get("cross_batch_median", "N/A")
                worst_pair = batch_level.get("worst_batch_pair", "N/A")
                worst_corr = batch_level.get("worst_correlation", "N/A")

                cross_str = f"{cross:.4f}" if isinstance(cross, float) else str(cross)
                worst_corr_str = (
                    f"{worst_corr:.4f}"
                    if isinstance(worst_corr, float)
                    else str(worst_corr)
                )

                row.extend([cross_str, worst_pair, worst_corr_str])

            rows.append(row)

        # 2. Dynamically set headers based on batch design
        if is_multi_batch:
            headers = [
                "Pipeline Stage",
                "Inner-Batch Median",
                "Cross-Batch Median",
                "Worst Batch Pair",
                "Worst Batch Corr.",
            ]
        else:
            headers = ["Pipeline Stage", "Median Correlation"]

        # 3. Render table with disable_numparse=True to preserve trailing zeros
        table_str = tabulate(
            rows,
            headers=headers,
            tablefmt="github",
            disable_numparse=True,
            stralign="center",
            numalign="center",
        )
        return f"\n\n{table_str}\n\n" if rows else ""

    def _create_outlier_summary_table(self, qa_metrics: Dict[str, Any]) -> str:
        """
        Generates a table detailing multi-dimensional outliers across stages.
        """
        rows = []
        for stage_key, stage_name in self._QA_STAGES:
            q_data = qa_metrics.get(stage_key, {})
            if not q_data:
                continue

            # 1. SD-OD Extreme Outliers
            outliers = q_data.get("outliers", {})
            ext_samples = outliers.get("extreme_samples", [])
            ext_str = ", ".join(map(str, ext_samples)) if ext_samples else "None"

            # 2. IS Outliers
            is_qc = q_data.get("internal_standard_qc", {})
            is_samples = is_qc.get("is_outlier_samples", [])
            is_rate = is_qc.get("is_outlier_standard", "N/A")
            is_str = ", ".join(map(str, is_samples)) if is_samples else "None"

            # 3. ORF Outliers
            orf_qc = q_data.get("orf_qc", {})
            orf_samples = orf_qc.get("orf_outlier_samples", [])
            orf_rate = orf_qc.get("orf_outlier_standard", "N/A")
            orf_str = ", ".join(map(str, orf_samples)) if orf_samples else "None"

            rows.append([stage_name, ext_str, is_rate, is_str, orf_rate, orf_str])

        headers = [
            "Pipeline Stage",
            "SD-OD Extreme Outlier Samples",
            "IS Outliers (N/Total)",
            "IS Outlier Samples",
            "ORF Outliers (N/Total)",
            "ORF Outlier Samples",
        ]

        table_str = tabulate(
            rows,
            headers=headers,
            tablefmt="github",
            stralign="center",
            numalign="center",
        )
        return f"\n\n{table_str}\n\n" if rows else ""

    def consolidate_metrics(
        self, pipeline_metrics: Dict[str, Any], qa_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Consolidates pipeline and QA metrics into a unified context."""

        def get_val(d: Dict[str, Any], *keys: str, default: object = "N/A") -> object:
            """Safely traverse nested dictionaries to retrieve values."""
            for k in keys:
                if isinstance(d, dict) and k in d:
                    d = d[k]
                else:
                    return default
            return d

        batch_count = get_val(
            pipeline_metrics, "raw_dataset", "batches", "batch_count", default=1
        )

        stats = {
            "metadata": {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "mode": get_val(pipeline_metrics, "raw_dataset", "mode", default="N/A"),
                "is_multi_batch": batch_count > 1,
            }
        }

        # [REFACTOR]: Core stages iteration fully handles new 2-Stage Norm logic
        for stage_key, _ in self._QA_STAGES:
            pipe_data = pipeline_metrics.get(stage_key, {})
            qa_data = qa_metrics.get(stage_key, {})
            stats[stage_key] = {"pipeline_params": pipe_data, "qa_assessments": qa_data}
        if "signal_correction" in pipeline_metrics:
            stats["signal_correction"] = pipeline_metrics["signal_correction"]

        batch_dist = get_val(
            pipeline_metrics, "raw_dataset", "batches", "batch_distribution", default={}
        )

        stats["raw_dataset"]["batch_table"] = self._create_batch_table(batch_dist)

        mar_sel = get_val(
            pipeline_metrics,
            "missing_value_imputation",
            "strategies",
            "mar_method_selected",
            default="",
        )
        if mar_sel and mar_sel != "N/A":
            nrmse = get_val(
                pipeline_metrics,
                "missing_value_imputation",
                "performance",
                mar_sel,
                "nrmse_low",
                default="N/A",
            )
            stats["missing_value_imputation"]["best_nrmse_low"] = nrmse

        stats["summary_tables"] = {
            "rsd": self._create_rsd_summary_table(qa_metrics),
            "pca": self._create_pca_summary_table(qa_metrics),
            "correlation": self._create_corr_summary_table(qa_metrics),
            "outliers": self._create_outlier_summary_table(qa_metrics),
        }

        return stats

    def _debug_template_errors(self, template_name: str, context: dict) -> None:
        """
        Advanced debugger to identify template rendering issues without
        triggering false positives from Jinja2's static AST parser.
        """
        import jinja2
        from jinja2 import meta
        import traceback

        logger.info(f"--- Starting Template Debugger for '{template_name}' ---")

        # 1. Static Analysis (Informational only, downgraded to DEBUG level)
        try:
            template_src = self.env.loader.get_source(self.env, template_name)[0]
            parsed_content = self.env.parse(template_src)
            ref_vars = meta.find_undeclared_variables(parsed_content)

            missing_top = [var for var in ref_vars if var not in context]
            if missing_top:
                # Static AST analysis cannot fully resolve conditionals ({% if %})
                # or local assignments ({% set %}). Therefore, it often generates
                # false positives. Logging as debug information only.
                logger.debug(
                    "Static AST found potential undeclared variables "
                    f"(often false positives): {missing_top}"
                )
        except Exception as e:
            logger.debug(f"Static analysis skipped due to parser limitation: {e}")

        # 2. Runtime Analysis (The Ultimate Source of Truth)
        # Create an isolated, strict environment. Any truly undefined variable
        # evaluated at runtime will instantly trigger an exception.
        strict_env = jinja2.Environment(
            loader=self.env.loader,
            undefined=jinja2.StrictUndefined,  # Enforce strict evaluation
        )

        try:
            debug_template = strict_env.get_template(template_name)

            # Attempt a full render. Success here guarantees the template
            # logic is perfectly sound given the current context.
            debug_template.render(context)
            logger.success(
                f"Template '{template_name}' successfully passed strict "
                "runtime rendering."
            )

        except jinja2.exceptions.UndefinedError as e:
            # Catch actual runtime undefined variables (e.g., typos in keys)
            logger.error(f"RUNTIME UNDEFINED ERROR in '{template_name}': {e}")

        except jinja2.exceptions.TemplateSyntaxError as e:
            # Catch syntax errors (e.g., missing {% endif %})
            logger.error(
                f"SYNTAX ERROR in '{template_name}' at line {e.lineno}: " f"{e.message}"
            )

        except Exception as e:
            # Catch other runtime execution errors (e.g., type mismatches)
            logger.error(f"EXECUTION ERROR in '{template_name}': {e}")
            logger.debug(traceback.format_exc())

        logger.info("--- Template Debugger Finished ---")

    def _is_weasyprint_operational(self) -> bool:
        """Performs a hard check to verify if WeasyPrint C-libraries exist."""
        if sys.platform == "win32":
            gtk_bin = os.path.join(sys.prefix, "Library", "bin")
            if os.path.exists(gtk_bin) and (gtk_bin not in os.environ.get("PATH", "")):
                os.environ["PATH"] = (
                    f"{gtk_bin}{os.pathsep}{os.environ.get('PATH', '')}"
                )

        try:
            result = subprocess.run(
                ["weasyprint", "--version"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0:
                logger.debug(
                    f"WeasyPrint probe failed. Stderr: {result.stderr.strip()}"
                )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def _force_install_weasyprint_conda(self) -> bool:
        """Forces GTK3/Pango installation via Conda for WeasyPrint on Windows.

        Returns:
            bool: True if Conda injection succeeded, False otherwise.
        """
        # Verify that the pipeline is running inside a Conda environment
        if not os.path.exists(os.path.join(sys.prefix, "conda-meta")):
            logger.error("Not a Conda environment. Cannot auto-install GTK3.")
            return False

        logger.info("Conda detected. Auto-installing GTK3/Pango C-libraries...")
        conda_exe = os.environ.get("CONDA_EXE", "conda")

        try:
            subprocess.run(
                [
                    conda_exe,
                    "install",
                    "-c",
                    "conda-forge",
                    "weasyprint",
                    "pango",
                    "tinycss2",
                    "-y",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception as e:
            logger.error(f"Conda execution failed: {e}")
            return False

        # Closed-loop verification
        if self._is_weasyprint_operational():
            logger.success("WeasyPrint GTK3 libraries injected and verified.")
            return True
        else:
            logger.error("Conda installed WeasyPrint, but DLLs still fail.")
            return False

    def _is_pdflatex_available(self) -> bool:
        """Performs a hard check to verify if pdflatex is fully operational.

        Bypasses shutil.which to avoid false positives from broken paths or
        ghost registry entries.

        Returns:
            bool: True if the binary executes successfully, False otherwise.
        """
        try:
            result = subprocess.run(
                ["pdflatex", "--version"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # Only return True if the process exits without error
            return result.returncode == 0
        except (FileNotFoundError, OSError):
            # OSError catches cases where the file exists but not executable
            return False

    def _force_install_tinytex(self) -> bool:
        """Forces TinyTeX installation, bypassing pytinytex and exit codes.

        Executes the official PowerShell script on Windows. Ignores exit
        codes (often 1 due to fc-cache warnings) and hard-verifies the
        binary directory. Finally, broadcasts OS environment changes.

        Returns:
            bool: True if installation and binary verification succeed.
        """

        if sys.platform != "win32":
            try:
                subprocess.run(
                    ["pytinytex", "download"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return True
            except Exception as e:
                logger.error(f"pytinytex download failed: {e}")
                return False

        logger.info("Executing official PowerShell installer (takes time)...")
        ps_cmd = (
            "Invoke-WebRequest "
            "'https://tinytex.yihui.org/install-bin-windows.ps1' "
            "-OutFile 'ins.ps1'; & .\\ins.ps1; "
            "Remove-Item 'ins.ps1' -ErrorAction SilentlyContinue"
        )

        try:
            # check=False bypasses false negative exit code 1
            subprocess.run(
                ["powershell", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
                check=False,
                capture_output=True,
                text=True,
                creationflags=0x08000000,
            )
        except Exception as e:
            logger.error(f"PowerShell execution explicitly failed: {e}")
            return False
        finally:
            # [FIX]: Foolproof cleanup from the Python side.
            # This ensures cleanup even if the PowerShell script aborts early.
            if os.path.exists("ins.ps1"):
                try:
                    os.remove("ins.ps1")
                    logger.debug("Cleaned up orphaned ins.ps1 file.")
                except OSError:
                    pass

        # Hard verification of binary directory existence
        appdata = Path(os.environ.get("APPDATA", ""))
        progdata = Path(os.environ.get("ProgramData", ""))

        target_paths = [
            appdata / "TinyTeX" / "bin" / "windows",
            appdata / "TinyTeX" / "bin" / "win32",
            progdata / "TinyTeX" / "bin" / "windows",
            progdata / "TinyTeX" / "bin" / "win32",
        ]

        tt_bin = next((p for p in target_paths if p.is_dir()), None)

        if not tt_bin:
            logger.error("TinyTeX installed, but bin directory not found.")
            return False

        # Update process PATH immediately using standard path separator
        bin_path = str(tt_bin)
        curr_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{bin_path}{os.pathsep}{curr_path}"

        # Broadcast environment change to Windows OS
        try:
            hwnd_broadcast = 0xFFFF
            wm_settingchange = 0x001A
            smto_abortifhung = 0x0002
            result = ctypes.c_long()

            ctypes.windll.user32.SendMessageTimeoutW(
                hwnd_broadcast,
                wm_settingchange,
                0,
                "Environment",
                smto_abortifhung,
                5000,
                ctypes.byref(result),
            )
        except Exception as e:
            logger.debug(f"OS env broadcast failed (non-fatal): {e}")

        # [CRITICAL]: Closed-loop verification
        if self._is_pdflatex_available():
            logger.success("TinyTeX force-installed and verified.")
            return True
        else:
            logger.error("TinyTeX installed but pdflatex is still broken.")
            return False

    def _is_rsvg_operational(self) -> bool:
        """Performs a hard check to verify if rsvg-convert is installed.

        Pandoc requires this system binary to convert SVGs to PDFs on-the-fly
        when targeting XeLaTeX.

        Returns:
            bool: True if the binary executes successfully.
        """
        try:
            result = subprocess.run(
                ["rsvg-convert", "--version"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return result.returncode == 0
        except (FileNotFoundError, OSError):
            return False

    def _force_install_rsvg_conda(self) -> bool:
        """Forces librsvg installation via Conda for native SVG support.

        Returns:
            bool: True if Conda injection succeeded, False otherwise.
        """
        if not os.path.exists(os.path.join(sys.prefix, "conda-meta")):
            logger.error("Not a Conda env. Cannot auto-install librsvg.")
            return False

        logger.info("Auto-installing rsvg-convert (librsvg) via Conda...")
        conda_exe = os.environ.get("CONDA_EXE", "conda")

        try:
            subprocess.run(
                [conda_exe, "install", "-c", "conda-forge", "librsvg", "-y"],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception as e:
            logger.error(f"Conda execution failed for librsvg: {e}")
            return False

        if self._is_rsvg_operational():
            logger.success("rsvg-convert injected and verified.")
            return True
        else:
            logger.error("Conda installed librsvg, but binary fails.")
            return False

    def generate_markdown(
        self,
        pipeline_metrics: dict,
        qa_metrics: dict,
        report_folder: str = "08_Report_Summary",
    ) -> None:
        """
        Renders both comprehensive and brief Markdown QC reports.

        Iterates through defined template versions and generates distinct
        markdown documents based on the unified metrics context.
        """
        # Step 1: Consolidate double-source data
        context = self.consolidate_metrics(pipeline_metrics, qa_metrics)
        out_dir = self.base_dir / report_folder

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except IOError as e:
            logger.error(f"Failed to create report directory: {e}")
            return

        # Initialize a list to track all generated markdown paths
        self._generated_md_paths = []

        # Define the template mapping for dual output
        versions = {
            "Comprehensive": "report_comprehensive.md.j2",
            "Brief": "report_brief.md.j2",
        }

        # Step 2: Loop through versions and render
        for label, template_name in versions.items():
            logger.info(f"Generating {label.upper()} narrative report...")
            try:
                template = self.env.get_template(template_name)
                content = template.render(context)

                md_path = out_dir / f"Report_{label}.md"
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(content)

                self._generated_md_paths.append(md_path)
                logger.success(f"{label.capitalize()} report generated: {md_path}")
            except Exception as e:
                logger.error(f"Jinja2 rendering failed for {label}: {e}")
                self._debug_template_errors(template_name, context)

    def export_report(self, pdf_engine: Optional[str] = "weasyprint") -> bool:
        """Exports all generated markdown reports to PDF/HTML sequentially.

        Orchestrates the conversion from Markdown to PDF/HTML using Pandoc.
        Loops through all tracked markdown files and handles fallbacks.
        """
        md_paths = getattr(self, "_generated_md_paths", [])
        if not md_paths:
            logger.error("No Markdown found. Run `generate_markdown` first.")
            return False

        # --- Phase 0: Pandoc Environment Check ---
        try:
            import pypandoc

            try:
                pypandoc.get_pandoc_version()
            except OSError:
                logger.warning("Pandoc missing. Auto-downloading...")
                pypandoc.download_pandoc()
        except ImportError:
            logger.error("Missing dependency: pip install pypandoc[tinytex]")
            return False

        # --- Phase 1: Setup Temporary CSS File ---
        # Derive base directory from the first available markdown file
        md_dir = os.path.abspath(os.path.dirname(str(md_paths[0])))
        assets_path = Path(md_dir) / "assets"
        assets_path.mkdir(parents=True, exist_ok=True)
        css_path = assets_path / "report_style.css"

        with open(css_path, "w", encoding="utf-8") as f:
            f.write(self.REPORT_CSS)

        base_args = [
            "--standalone",
            "--embed-resources",
            "--quiet",
            f"--resource-path={md_dir}",
            f"--css={str(css_path)}",
        ]

        # =====================================================================
        # INTERNAL EXECUTORS (Refactored to accept explicit paths)
        # =====================================================================
        def _render_html(
            src_md: str, out_html: str, is_fallback: bool = False
        ) -> Optional[str]:
            """Internal helper for standard HTML rendering."""
            try:
                pypandoc.convert_file(
                    source_file=src_md,
                    to="html",
                    format="markdown",
                    outputfile=out_html,
                    extra_args=base_args,
                )
                status = "saved" if is_fallback else "generated"
                logger.success(f"HTML report {status}: {out_html}")
                return "HTML"
            except Exception as html_err:
                logger.error(f"HTML conversion failed: {html_err}")
                return None

        def _render_weasyprint(src_md: str, out_pdf: str) -> Optional[str]:
            """Internal helper for WeasyPrint PDF rendering via Pandoc."""
            try:
                logger.info("Attempting PDF export via WeasyPrint...")

                if sys.platform == "win32":
                    f_conf = os.path.join(
                        sys.prefix, "Library", "etc", "fonts", "fonts.conf"
                    )
                    if os.path.exists(f_conf):
                        os.environ["FONTCONFIG_FILE"] = f_conf
                        os.environ["FONTCONFIG_PATH"] = os.path.dirname(f_conf)

                if not self._is_weasyprint_operational():
                    if not self._force_install_weasyprint_conda():
                        return None

                wp_args = base_args + ["--pdf-engine=weasyprint", "--pdf-engine-opt=-q"]

                pypandoc.convert_file(
                    source_file=src_md,
                    to="pdf",
                    format="markdown",
                    outputfile=out_pdf,
                    extra_args=wp_args,
                )
                logger.success(f"PDF generated: {out_pdf}")
                return "WeasyPrint"
            except Exception as e:
                if "permission denied" in str(e).lower():
                    logger.error(f"Permission denied: Close {out_pdf}.")
                    raise e
                logger.warning(f"WeasyPrint engine failed: {e}")
                return None

        def _render_latex(
            src_md: str, out_pdf: str, is_fallback: bool = False
        ) -> Optional[str]:
            """Internal helper for XeLaTeX PDF rendering."""
            try:
                mode = "fallback" if is_fallback else "primary"
                logger.info(f"Attempting PDF export via LaTeX ({mode})...")

                if not self._is_pdflatex_available():
                    if not self._force_install_tinytex():
                        return None

                if not self._is_rsvg_operational():
                    self._force_install_rsvg_conda()

                lx_args = base_args + [
                    "--pdf-engine=xelatex",
                    "-V",
                    "geometry:margin=25mm",
                    "-V",
                    "tables=true",
                ]
                pypandoc.convert_file(
                    source_file=src_md,
                    to="pdf",
                    format="markdown",
                    outputfile=out_pdf,
                    extra_args=lx_args,
                )
                logger.success(f"PDF generated: {out_pdf}")
                return "XeLaTeX"
            except Exception as e:
                if "permission denied" in str(e).lower():
                    logger.error(f"Permission denied: Close {out_pdf}.")
                    raise e
                logger.warning(f"LaTeX engine failed: {e}")
                return None

        # =====================================================================
        # MAIN ROUTING LOGIC WITH GUARANTEED CLEANUP
        # =====================================================================
        overall_success = True

        try:
            for md_path in md_paths:
                md_str = str(md_path)
                file_label = os.path.basename(md_str)
                logger.info(f"--- Exporting PDF for: {file_label} ---")

                base_name = os.path.splitext(md_str)[0]
                pdf_path = base_name + ".pdf"
                html_path = base_name + ".html"

                target = pdf_engine.lower()
                final_engine = None

                # Execute conversion based on target and handle fallbacks
                if target == "html":
                    final_engine = _render_html(md_str, html_path, False)
                elif target == "xelatex":
                    final_engine = _render_latex(md_str, pdf_path) or _render_html(
                        md_str, html_path, True
                    )
                elif target == "weasyprint":
                    final_engine = (
                        _render_weasyprint(md_str, pdf_path)
                        or _render_latex(md_str, pdf_path, True)
                        or _render_html(md_str, html_path, True)
                    )
                else:
                    logger.error(f"Unsupported engine: {pdf_engine}")
                    overall_success = False
                    continue

                if final_engine:
                    logger.success(f"[{file_label}] completed using {final_engine}.")
                else:
                    logger.error(f"Failed to export [{file_label}].")
                    overall_success = False

            return overall_success

        finally:
            # Phase 2: Guaranteed Cleanup of the Temporary CSS File
            if css_path.exists():
                try:
                    css_path.unlink()
                except OSError as e:
                    logger.debug(f"Failed to remove temporary CSS: {e}")
