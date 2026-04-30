# src/pimqc/report_utils.py
"""
Purpose of script: Global utility classes and functions for assembling, 
rendering, and exporting automated project-level QC reports.
"""


import os
import sys
import time
from datetime import datetime

import subprocess
import ctypes
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from tabulate import tabulate

import logging
from loguru import logger
from typing import Union, Optional


# =========================================================================
# Atomic Utility Functions
# =========================================================================

# def clean_pdf_for_ai(doc):
#     """Optimizes PDF structure for Adobe Illustrator compatibility.

#     This function performs structural cleaning on the PDF content stream:
#     1. Removes Soft Masks (SMask) from image objects to eliminate 
#     transparency-related grouping issues in AI.

#     Note: Background removal via redaction is omitted as it spatially 
#     destroys overlaying vector data. Backgrounds should be rendered 
#     transparently during the initial PDF generation (e.g., in Matplotlib).

#     Args:
#         doc (fitz.Document): The PyMuPDF document object to be cleaned.

#     Returns:
#         fitz.Document: The processed document with simplified vector layers.
#     """
#     for page in doc:
#         # Step 1: Physical removal of Image Soft Masks (SMasks)
#         img_list = page.get_images()
#         for img in img_list:
#             xref = img[0]
#             smask_xref = img[1]
#             if smask_xref > 0:
#                 # Correct PyMuPDF method to nullify the SMask entry
#                 doc.xref_set_key(xref, "SMask", "null")

#     return doc


# def stitch_pdf_grids(
#     pdf_paths, 
#     output_name, 
#     cols="auto", 
#     max_cols=4
# ):
#     """Stitches multiple PDFs into a grid optimized for vector editing.

#     The implementation integrates a pre-processing structural cleaning 
#     sequence. It automatically resolves the optimal column layout if 
#     instructed, preserving a structured matrix for AI manipulation.

#     Args:
#         pdf_paths (list): List of paths to source PDF subplots.
#         output_name (str or Path): Destination path for stitched PDF.
#         cols (int or str): Number of columns, or 'auto' for dynamic 
#             calculation. Defaults to 'auto'.
#         max_cols (int): Maximum allowed columns when cols is 'auto'. 
#             Defaults to 4.

#     Returns:
#         bool: True if operation succeeded, False otherwise.
#     """
#     try:
#         import fitz
#     except ImportError:
#         logger.error("fitz (PyMuPDF) is required for robust stitching.")
#         return False

#     try:
#         valid_docs = []
#         for path in pdf_paths:
#             p = Path(path)
#             if p.exists() and p.stat().st_size > 0:
#                 doc = fitz.open(str(p))
#                 doc = clean_pdf_for_ai(doc)
#                 valid_docs.append(doc)

#         n_docs = len(valid_docs)
#         if not n_docs:
#             return False

#         # Resolve dynamic grid column assignment
#         if cols == "auto":
#             active_cols = _get_optimal_cols(n_docs, max_cols)
#         else:
#             active_cols = int(cols)

#         # Determine grid dimensions based on the largest source subplot
#         max_w = max(d[0].rect.width for d in valid_docs)
#         max_h = max(d[0].rect.height for d in valid_docs)
#         rows = (n_docs + active_cols - 1) // active_cols

#         out_doc = fitz.open()
#         out_page = out_doc.new_page(
#             width=max_w * active_cols, 
#             height=max_h * rows
#         )

#         for i, doc in enumerate(valid_docs):
#             row, col = i // active_cols, i % active_cols
            
#             x0, y0 = col * max_w, row * max_h
#             target_rect = fitz.Rect(x0, y0, x0 + max_w, y0 + max_h)
            
#             # Use overlay=True to simplify the XObject resource tree
#             out_page.show_pdf_page(target_rect, doc, 0, overlay=True)
#             doc.close()

#         # Save with garbage=4 to physically purge deleted objects
#         out_doc.save(
#             str(output_name), 
#             garbage=4, 
#             deflate=True, 
#             clean=True
#         )
#         out_doc.close()
#         return True

#     except Exception as e:
#         logger.error(f"Failed to stitch AI-optimized PDF grid: {e}")
#         return False


# def convert_pdf_to_png(
#     pdf_path, output_png_path=None, dpi_scale=2.0, show_jupyter=True):
#     """Render a PDF to PNG format and optionally display it in Jupyter."""
#     pdf_p = Path(pdf_path)
#     if not pdf_p.exists() or pdf_p.stat().st_size == 0:
#         logger.error(f"Invalid or empty PDF for conversion: {pdf_p}")
#         return None

#     try:
#         import fitz  # PyMuPDF
        
#         # Open with a new file handle
#         doc = fitz.open(str(pdf_p))
#         if doc.page_count == 0:
#             doc.close()
#             raise ValueError("Stitched PDF document contains no pages.")
            
#         page = doc[0]
#         mat = fitz.Matrix(dpi_scale, dpi_scale)
#         pix = page.get_pixmap(matrix=mat)
        
#         png_p = Path(output_png_path) if output_png_path else \
#                 pdf_p.with_suffix(".png")
#         pix.save(str(png_p))
#         doc.close()
        
#         if show_jupyter:
#             try:
#                 from IPython.display import Image, display
#                 display(Image(filename=str(png_p), width=800))
#             except (ImportError, NameError):
#                 pass
#         return png_p
#     except Exception as e:
#         logger.error(f"Failed to generate PNG preview: {e}")
#         return None

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
        1: 1, 2: 2, 3: 3, 4: 2,
        5: 3, 6: 3, 7: 3, 8: 4,
        9: 3, 10: 4, 11: 4, 12: 4
    }

    if n_docs in layout_map:
        cols = layout_map[n_docs]
    else:
        # Fallback for dynamic calculation on arbitrary large numbers
        cols = math.ceil(math.sqrt(n_docs))

    return min(max_cols, cols)


def stitch_svg_grids(
    svg_paths: list, 
    output_name: str, 
    cols: str = "auto", 
    max_cols: int = 4,
    show_jupyter: bool = True,
    width: int = 800
) -> bool:
    """Stitches multiple SVGs into a grid and displays it in Jupyter natively.

    Args:
        svg_paths: List of paths to source SVG subplots.
        output_name: Destination path for stitched SVG.
        cols: Number of columns, or 'auto'. Defaults to "auto".
        max_cols: Maximum allowed columns. Defaults to 4.
        show_jupyter: Whether to display the result in Jupyter/VS Code.
        width: Display width in Jupyter (e.g., 800, "100%").

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        import svgutils.transform as sg
    except ImportError:
        logger.error("Please install 'svgutils' via 'pip install svgutils'.")
        return False

    try:
        import re
        from pathlib import Path
        
        valid_paths = [
            p for p in svg_paths 
            if Path(p).exists() and Path(p).stat().st_size > 0
        ]
        if not valid_paths:
            return False

        n_docs = len(valid_paths)
        # Assuming _get_optimal_cols is defined elsewhere in your module
        active_cols = (
            _get_optimal_cols(n_docs, max_cols) 
            if cols == "auto" else int(cols)
        )
        rows = (n_docs + active_cols - 1) // active_cols

        svg_figs = [sg.fromfile(str(p)) for p in valid_paths]

        def parse_dim(val: str) -> float:
            match = re.search(r"(\d+\.?\d*)", str(val))
            return float(match.group(1)) if match else 0.0

        max_w = max(parse_dim(f.width) for f in svg_figs)
        max_h = max(parse_dim(f.height) for f in svg_figs)

        total_w = max_w * active_cols
        total_h = max_h * rows

        fig = sg.SVGFigure(f"{total_w}", f"{total_h}")

        plots = []
        for i, s_fig in enumerate(svg_figs):
            row, col = divmod(i, active_cols)
            plot = s_fig.getroot()
            plot.moveto(col * max_w, row * max_h)
            plots.append(plot)

        fig.append(plots)
        fig.root.set("viewBox", f"0 0 {total_w} {total_h}")
        fig.save(str(output_name))

        # --- New Jupyter Display Logic ---
        if show_jupyter:
            try:
                from IPython.display import HTML, display
                
                # Read the saved SVG back into memory
                with open(output_name, "r", encoding="utf-8") as f:
                    svg_content = f.read()

                # Convert absolute sizes to 100% strictly for UI preview
                preview_svg = re.sub(
                    r'(<svg[^>]*?\s)width="[^"]+"', r'\1width="100%"', 
                    svg_content, count=1
                )
                preview_svg = re.sub(
                    r'(<svg[^>]*?\s)height="[^"]+"', r'\1height="100%"', 
                    preview_svg, count=1
                )

                w_css = f"{width}px" if isinstance(width, int) else width
                html_wrapper = (
                    f'<div style="width:{w_css}; max-width:100%; '
                    f'height:auto;">{preview_svg}</div>'
                )
                display(HTML(html_wrapper))
            except Exception as e:
                logger.error(f"Failed to display stitched SVG grid: {e}")

        return True

    except Exception as e:
        logger.error(f"Failed to stitch SVG grid: {e}")
        return False

# =========================================================================
# Class 1: VisualAssetReporter (Handles QA Grids & Images)
# =========================================================================

class VisualAssetReporter: 

    def __init__(self, base_dir):
        """
        Initialize the reporter at the project workspace level.

        Args:
            base_dir: Root output directory containing step folders.
        """
        self.base_dir = Path(base_dir)
        self.qa_folders = self._detect_qa_folders()

    def _detect_qa_folders(self):
        """Automatically scan for QA directories and sort them."""
        if not self.base_dir.exists():
            return []
        folders = [
            d for d in self.base_dir.iterdir() 
            if d.is_dir() and "QA" in d.name
        ]
        return sorted([d.name for d in folders])

    # def compile_assessor_report_pdf(
    #     self, 
    #     is_multi_batch: bool = True, 
    #     report_folder: str = "13_Report_Markdown", 
    #     cols: int = 4, 
    #     show_jupyter: bool = True
    # ) -> None:
    #     """
    #     Compile QA PDF plots from different stages into grid matrices 
    #     and synchronize PNG assets for report embedding.
    #     """
        
    #     if not self.qa_folders:
    #         logger.error("No QA folders detected in the base directory.")
    #         return

    #     report_path = self.base_dir / report_folder
    #     assets_path = report_path / "assets"
        
    #     report_path.mkdir(parents=True, exist_ok=True)
    #     assets_path.mkdir(parents=True, exist_ok=True)

    #     # =================================================================
    #     # [Logic Branching]: Determine correlation plot type and prefix 
    #     # based on the experimental design (Single vs Multi-batch).
    #     # =================================================================
    #     if is_multi_batch:
    #         corr_prefix = "03_Batch_Correlation"
    #         corr_file = f"Batch_Corr_HM.pdf"
    #         logger.info(
    #             "Multi-batch design detected. Assembling Batch Correlation"
    #             "Grid.")
    #     else:
    #         corr_prefix = "03_QC_Correlation"
    #         corr_file = f"QC_Corr_HM.pdf"
    #         logger.info(
    #             "Single-batch design detected. Assembling QC Correlation"
    #             "Grid.")

    #     # Construct target mapping for batch stitching
    #     target_map = {
    #         "01_QC_Sample_RSD": f"RSD_Barplot.pdf",
    #         "02_PCA_Scatter": f"QC_AS_PCA_Scatter.pdf",
    #         corr_prefix: corr_file,  # <--- Dynamically injected
    #         "04_Outlier_Diagnosis": f"Outlier_Scatter.pdf"
    #     }
    #     # =================================================================

    #     for prefix, target_file in target_map.items():
    #         pdf_name = f"{prefix}_Grid.pdf"
    #         pdf_out = report_path / pdf_name

    #         # Collect source PDFs from all detected QA sub-directories
    #         input_pdfs = []
    #         for folder in self.qa_folders:
    #             file_path = self.base_dir / folder / target_file
                
    #             if file_path.exists():
    #                 input_pdfs.append(file_path)

    #         if not input_pdfs:
    #             logger.warning(f"Skipped {pdf_name}: No source files found.")
    #             continue

    #         # Execute PDF stitching (preserving vector layers for AI editing)
    #         success = stitch_pdf_grids(
    #             pdf_paths=input_pdfs, output_name=pdf_out, cols=cols)

    #         if not success:
    #             logger.warning(f"Failed to stitch grid for {pdf_name}.")
    #             continue
            
    #         if success:
    #             # Brief sleep to ensure OS handles file-locking after disk I/O
    #             time.sleep(0.3)
    #             # Convert vector PDF to high-resolution PNG for markdown embedding
    #             png_path = assets_path / f"{prefix}_Grid.png"
    #             convert_pdf_to_png(
    #                 pdf_path=pdf_out, output_png_path=png_path, 
    #                 dpi_scale=2.0, show_jupyter=show_jupyter
    #             )

    #     logger.success(f"Report assets globally compiled at: {report_path}")


    def compile_assessor_report(
        self, 
        is_multi_batch: bool = True, 
        report_folder: str = "13_Report_Markdown", 
        cols: int = 4
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
            corr_prefix = "03_Batch_Correlation"
            corr_file = "Batch_Corr_HM.svg"
            logger.info("Multi-batch design detected. Assembling Batch Grid.")
        else:
            corr_prefix = "03_QC_Correlation"
            corr_file = "QC_Corr_HM.svg"
            logger.info("Single-batch design detected. Assembling QC Grid.")

        # Target files now expect the .svg extension natively
        target_map = {
            "01_QC_Sample_RSD": "RSD_Barplot.svg",
            "02_PCA_Scatter": "QC_AS_PCA_Scatter.svg",
            corr_prefix: corr_file,
            "04_Outlier_Diagnosis": "Outlier_Scatter.svg"
        }

        for prefix, target_file in target_map.items():
            # Output directly to the assets folder as an SVG grid
            svg_out = assets_path / f"{prefix}_Grid.svg"

            input_svgs = []
            for folder in self.qa_folders:
                file_path = self.base_dir / folder / target_file
                if file_path.exists():
                    input_svgs.append(file_path)

            if not input_svgs:
                logger.warning(f"Skipped {prefix}: No source SVGs found.")
                continue

            # Execute SVG stitching
            stitch_svg_grids(
                svg_paths=input_svgs, output_name=svg_out, cols=cols
            )

        logger.success(f"Report SVG assets compiled at: {assets_path}")

# =========================================================================
# Class 2: NarrativeStatsReporter (Handles attrs & Markdown Text)
# =========================================================================

class NarrativeStatsReporter:
    """Extracts metadata from MetaboInt objects to generate a single report."""

    def __init__(self, base_dir: str):
        """Initialize with base directory and Jinja2 environment."""
        self.base_dir = Path(base_dir)
        template_path = Path(__file__).parent / "templates"
        self.env = Environment(loader=FileSystemLoader(str(template_path)))

    def _create_batch_table(self, batch_dist: dict) -> str:
        """Generates a Markdown table for batch distributions."""
        rows = []
        if isinstance(batch_dist, dict):
            for b_id, b_info in batch_dist.items():
                rows.append([
                    b_id,
                    b_info.get("Total", 0),
                    b_info.get("QC", 0),
                    b_info.get("Blank", 0),
                    b_info.get("Sample", 0),
                    b_info.get("Inject Order", "N/A")
                ])
        table_str = tabulate(
            rows, 
            headers=[
                "Batch", "Total", "QC", "Blank", "Sample", "Inject Order"
            ],
            tablefmt="github"
        )
        return f"\n\n{table_str}\n\n"

    def _create_rsd_table(self, rsd_data: dict) -> str:
        """Generates a Markdown table for RSD distributions."""
        rows = []
        for group in ["qc", "actual"]:
            dist = rsd_data.get(group, {})
            if dist:
                rows.append([
                    group.upper(),
                    dist.get("0-10%", 0),
                    dist.get("10-20%", 0),
                    dist.get("20-30%", 0),
                    dist.get(">30%", 0)
                ])
        if not rows:
            return "N/A"
        table_str = tabulate(
            rows,
            headers=["Group", "0-10%", "10-20%", "20-30%", ">30%"],
            tablefmt="github"
        )
        return f"\n\n{table_str}\n\n"

    def _create_pca_table(self, pca_data: dict) -> str:
        """Generates a Markdown table for PCA metrics."""
        if not pca_data:
            return "N/A"
        rows = [[
            f"{pca_data.get('pc1_variance', 0)*100:.2f}%",
            f"{pca_data.get('pc2_variance', 0)*100:.2f}%",
            f"{pca_data.get('relative_dispersion', 0):.4f}",
            f"{pca_data.get('batch_silhouette', 0):.4f}",
            f"{pca_data.get('centrality_shift', 0):.4f}"
        ]]
        table_str = tabulate(
            rows,
            headers=[
                "PC1 Var", "PC2 Var", "Rel Dispersion", 
                "Batch Silh", "Cent Shift"
            ],
            tablefmt="github"
        )
        return f"\n\n{table_str}\n\n"

    def consolidate_metrics(
        self, pipeline_metrics: dict, qa_metrics: dict) -> dict:
        """Consolidates pipeline and QA metrics into a unified context.

        Extracts all valid parameters, thresholds, feature retention rates,
        and algorithmic performance assessments (PCA, RSD, Outliers) from 
        both metric dictionaries without data loss.

        Args:
            pipeline_metrics: Dict containing pipeline parameters and results.
            qa_metrics: Dict containing QA assessment results.

        Returns:
            A deeply structured dictionary optimized for Jinja2 rendering.
        """
        def get_val(d, *keys, default="N/A"):
            for k in keys:
                if isinstance(d, dict) and k in d:
                    d = d[k]
                else:
                    return default
            return d

        # Base mode and timestamps
        stats = {
            "metadata": {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "mode": get_val(
                    pipeline_metrics, "raw_dataset", "mode", default="N/A")
            }
        }

        # List of generic processing stages that map 1:1 in both dictionaries
        core_stages = [
            "raw_dataset", 
            "high_mv_feature_filtering",
            "intra_signal_correction", 
            "inter_signal_correction",
            "low_quality_feature_filtering", 
            "missing_value_imputation"
        ]

        for stage in core_stages:
            p_data = pipeline_metrics.get(stage, {})
            q_data = qa_metrics.get(stage, {})
            
            stats[stage] = {
                "pipeline_params": p_data,
                "qa_assessments": q_data,
                # Pre-rendered Markdown tables for quick template injection
                "rsd_table": self._create_rsd_table(
                    q_data.get("rsd_distribution", {})),
                "pca_table": self._create_pca_table(
                    q_data.get("pca", {}))
            }

        # Specific extraction: Raw Dataset Batch Table
        batch_dist = get_val(
            pipeline_metrics, "raw_dataset", "batches", "batch_distribution",
            default={})
        stats["raw_dataset"]["batch_table"] = self._create_batch_table(
            batch_dist)

        # Specific extraction: Missing Value NRMSE (Best Method Extraction)
        mar_sel = get_val(
            pipeline_metrics, "missing_value_imputation", "strategies", 
            "mar_method_selected", default="")
        if mar_sel and mar_sel != "N/A":
            nrmse = get_val(
                pipeline_metrics, "missing_value_imputation", "performance", 
                mar_sel, "nrmse_low", default="N/A")
            stats["missing_value_imputation"]["best_nrmse_low"] = nrmse

        # Specific extraction: Normalization (1:2 Mapping resolution)
        # Pipeline has 'normalization', QA splits into sample & feature wise
        p_norm = pipeline_metrics.get("normalization", {})
        q_s_norm = qa_metrics.get("sample_wise_normalization", {})
        q_f_norm = qa_metrics.get("feature_wise_normalization", {})

        stats["normalization"] = {
            "pipeline_params": p_norm,
            "sample_wise_qa": q_s_norm,
            "feature_wise_qa": q_f_norm,
            "sample_rsd_table": self._create_rsd_table(
                q_s_norm.get("rsd_distribution", {})),
            "feature_rsd_table": self._create_rsd_table(
                q_f_norm.get("rsd_distribution", {})),
            "sample_pca_table": self._create_pca_table(
                q_s_norm.get("pca", {})),
            "feature_pca_table": self._create_pca_table(
                q_f_norm.get("pca", {}))
        }

        return stats
    
    def generate_markdown(
        self, 
        pipeline_metrics: dict, 
        qa_metrics: dict, 
        report_folder: str = "Comprehensive_Report_Markdown"
    ) -> None:
        """Renders the comprehensive Markdown QC report using integrated metrics.

        Args:
            pipeline_metrics: Dictionary of pipeline execution parameters.
            qa_metrics: Dictionary of quality assessment outputs.
            report_folder: Output directory name relative to base_dir.
        """
        # Step 1: Consolidate double-source data
        context = self.consolidate_metrics(pipeline_metrics, qa_metrics)
        
        # Step 2: Render phase (Restoring granular Jinja2 error handling)
        try:
            template = self.env.get_template("report_template.md.j2")
            content = template.render(context)
        except Exception as e:
            logger.error(f"Jinja2 template rendering failed: {e}")
            return

        # Step 3: I/O phase
        out_dir = self.base_dir / report_folder
        try:
            if not os.path.exists(out_dir):
                out_dir.mkdir(parents=True, exist_ok=True)
            
            # Safe extraction avoiding KeyError
            md_path = out_dir / "Comprehensive_QC_Report.md"
            
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(content)
    
            self._last_md_path = md_path
            logger.success(f"Markdown report generated at: {md_path}")
        except IOError as e:
            logger.error(
                f"File I/O operations failed during report saving: {e}")

    def _is_weasyprint_operational(self) -> bool:
        """Performs a hard check to verify if WeasyPrint C-libraries exist.

        Returns:
            bool: True if WeasyPrint and its GTK3/Pango DLLs load successfully.
        """
        try:
            result = subprocess.run(
                ["weasyprint", "--version"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # If DLLs are missing, Python throws an OS error inside subprocess
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
                    conda_exe, "install", "-c", "conda-forge", 
                    "weasyprint", "pango", "-y"
                ],
                check=True,
                capture_output=True,
                text=True
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
                text=True
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
                    check=True, capture_output=True, text=True
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
                [
                    "powershell", "-ExecutionPolicy", "Bypass",
                    "-Command", ps_cmd
                ],
                check=False,
                capture_output=True,
                text=True,
                creationflags=0x08000000
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
            progdata / "TinyTeX" / "bin" / "win32"
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
                hwnd_broadcast, wm_settingchange, 0, "Environment",
                smto_abortifhung, 5000, ctypes.byref(result)
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

    def export_report(self, pdf_engine: Optional[str] = "weasyprint") -> bool:
        """Exports the narrative report with multi-stage fallbacks.

        Orchestrates the conversion from Markdown to PDF/HTML using a
        routing logic. Supports direct HTML export, primary WeasyPrint
        rendering, and a robust LaTeX fallback with auto-installation.
        """
        import os
        import sys
        import ctypes
        import subprocess
        from pathlib import Path

        # 1. Path resolution and initial validation
        md_path = getattr(self, "_last_md_path", None)
        if not md_path:
            logger.error("No Markdown report found. Run `generate_markdown`.")
            return False

        md_path_str = str(md_path)
        base_name = os.path.splitext(md_path_str)[0]
        pdf_path = base_name + ".pdf"
        html_path = base_name + ".html"

        # --- Phase 0: Pandoc Environment Check ---
        try:
            import pypandoc
            try:
                pypandoc.get_pandoc_version()
            except OSError:
                logger.warning("Pandoc not found. Auto-downloading...")
                pypandoc.download_pandoc()
        except ImportError:
            logger.error("Missing dependency: pip install pypandoc[tinytex]")
            return False

        # --- Global Configuration ---
        base_args = ["--standalone", "--embed-resources", "--quiet"]

        # =====================================================================
        # INTERNAL EXECUTORS (Closures)
        # =====================================================================

        def _render_html(is_fallback: bool = False) -> bool:
            """Internal helper for standard HTML rendering."""
            try:
                pypandoc.convert_file(
                    source_file=md_path_str, to="html", format="markdown",
                    outputfile=html_path, extra_args=base_args
                )
                status = "saved" if is_fallback else "generated"
                logger.success(f"HTML report {status}: {html_path}")
                self._last_pdf_path = Path(html_path)
                return True
            except Exception as html_err:
                logger.error(f"HTML conversion failed: {html_err}")
                return False

        def _render_weasyprint() -> bool:
            """Internal helper for WeasyPrint PDF rendering."""
            try:
                logger.info("Attempting PDF export via WeasyPrint...")
                if not self._is_weasyprint_operational():
                    logger.warning("GTK3 DLLs missing. Fixing Conda env...")
                    if not self._force_install_weasyprint_conda():
                        return False

                if sys.platform == "win32":
                    f_conf = os.path.join(
                        sys.prefix, "Library", "etc", "fonts", "fonts.conf"
                    )
                    if os.path.exists(f_conf):
                        os.environ["FONTCONFIG_FILE"] = f_conf
                        os.environ["FONTCONFIG_PATH"] = os.path.dirname(f_conf)

                wp_args = base_args + [
                    "--pdf-engine=weasyprint", "--pdf-engine-opt=-q"]
                pypandoc.convert_file(
                    source_file=md_path_str, to="pdf", format="markdown",
                    outputfile=pdf_path, extra_args=wp_args
                )
                logger.success(f"PDF generated via WeasyPrint: {pdf_path}")
                self._last_pdf_path = Path(pdf_path)
                return True
            except Exception as e:
                # Detect file lock (Permission Denied)
                if "permission denied" in str(e).lower():
                    logger.error(f"Permission denied: Close {pdf_path}.")
                    raise e
                logger.warning(f"WeasyPrint engine failed: {e}")
                return False

        def _render_latex(is_fallback: bool = False) -> bool:
            """Internal helper for XeLaTeX PDF rendering."""
            try:
                mode = "fallback" if is_fallback else "primary"
                logger.info(f"Attempting PDF export via LaTeX ({mode})...")
                if not self._is_pdflatex_available():
                    logger.warning("pdflatex missing. Auto-installing...")
                    if not self._force_install_tinytex():
                        return False

                lx_args = base_args + ["--pdf-engine=xelatex"]
                pypandoc.convert_file(
                    source_file=md_path_str, to="pdf", format="markdown",
                    outputfile=pdf_path, extra_args=lx_args
                )
                logger.success(f"PDF generated via LaTeX: {pdf_path}")
                self._last_pdf_path = Path(pdf_path)
                return True
            except Exception as e:
                if "permission denied" in str(e).lower():
                    logger.error(f"Permission denied: Close {pdf_path}.")
                    raise e
                logger.warning(f"LaTeX engine failed: {e}")
                return False

        # =====================================================================
        # MAIN ROUTING LOGIC (The Orchestrator)
        # =====================================================================
        target_engine = pdf_engine.lower()

        try:
            # Route A: Direct HTML Fast-track
            if target_engine == "html":
                return _render_html(is_fallback=False)

            # Route B: Targeted LaTeX (with HTML fallback)
            if target_engine == "xelatex":
                return _render_latex() or _render_html(is_fallback=True)

            # Route C: Standard Fallback Pipeline (WP -> LaTeX -> HTML)
            if target_engine == "weasyprint":
                if _render_weasyprint():
                    return True
                if _render_latex(is_fallback=True):
                    return True
                return _render_html(is_fallback=True)

            logger.error(f"Unsupported engine: {pdf_engine}")
            return False

        except Exception:
            # Stop pipeline if a critical File Lock error was re-raised
            return False