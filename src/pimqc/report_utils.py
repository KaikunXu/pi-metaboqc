# src/pimqc/report_utils.py
"""
Purpose of script: Global utility classes and functions for assembling, 
rendering, and exporting automated project-level QC reports.
"""

import os
import time
from datetime import datetime
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from tabulate import tabulate
from loguru import logger

# =========================================================================
# Atomic Utility Functions
# =========================================================================

def clean_pdf_for_ai(doc):
    """Optimizes PDF structure for Adobe Illustrator compatibility.

    This function performs structural cleaning on the PDF content stream:
    1. Removes Soft Masks (SMask) from image objects to eliminate 
    transparency-related grouping issues in AI.
    
    Note: Background removal via redaction is omitted as it spatially 
    destroys overlaying vector data. Backgrounds should be rendered 
    transparently during the initial PDF generation (e.g., in Matplotlib).

    Args:
        doc (fitz.Document): The PyMuPDF document object to be cleaned.

    Returns:
        fitz.Document: The processed document with simplified vector layers.
    """
    for page in doc:
        # Step 1: Physical removal of Image Soft Masks (SMasks)
        img_list = page.get_images()
        for img in img_list:
            xref = img[0]
            smask_xref = img[1]
            if smask_xref > 0:
                # Correct PyMuPDF method to nullify the SMask entry
                doc.xref_set_key(xref, "SMask", "null")

    return doc


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


def stitch_pdf_grids(
    pdf_paths, 
    output_name, 
    cols="auto", 
    max_cols=4
):
    """Stitches multiple PDFs into a grid optimized for vector editing.

    The implementation integrates a pre-processing structural cleaning 
    sequence. It automatically resolves the optimal column layout if 
    instructed, preserving a structured matrix for AI manipulation.

    Args:
        pdf_paths (list): List of paths to source PDF subplots.
        output_name (str or Path): Destination path for stitched PDF.
        cols (int or str): Number of columns, or 'auto' for dynamic 
            calculation. Defaults to 'auto'.
        max_cols (int): Maximum allowed columns when cols is 'auto'. 
            Defaults to 4.

    Returns:
        bool: True if operation succeeded, False otherwise.
    """
    try:
        import fitz
    except ImportError:
        logger.error("fitz (PyMuPDF) is required for robust stitching.")
        return False

    try:
        valid_docs = []
        for path in pdf_paths:
            p = Path(path)
            if p.exists() and p.stat().st_size > 0:
                doc = fitz.open(str(p))
                doc = clean_pdf_for_ai(doc)
                valid_docs.append(doc)

        n_docs = len(valid_docs)
        if not n_docs:
            return False

        # Resolve dynamic grid column assignment
        if cols == "auto":
            active_cols = _get_optimal_cols(n_docs, max_cols)
        else:
            active_cols = int(cols)

        # Determine grid dimensions based on the largest source subplot
        max_w = max(d[0].rect.width for d in valid_docs)
        max_h = max(d[0].rect.height for d in valid_docs)
        rows = (n_docs + active_cols - 1) // active_cols

        out_doc = fitz.open()
        out_page = out_doc.new_page(
            width=max_w * active_cols, 
            height=max_h * rows
        )

        for i, doc in enumerate(valid_docs):
            row, col = i // active_cols, i % active_cols
            
            x0, y0 = col * max_w, row * max_h
            target_rect = fitz.Rect(x0, y0, x0 + max_w, y0 + max_h)
            
            # Use overlay=True to simplify the XObject resource tree
            out_page.show_pdf_page(target_rect, doc, 0, overlay=True)
            doc.close()

        # Save with garbage=4 to physically purge deleted objects
        out_doc.save(
            str(output_name), 
            garbage=4, 
            deflate=True, 
            clean=True
        )
        out_doc.close()
        return True

    except Exception as e:
        logger.error(f"Failed to stitch AI-optimized PDF grid: {e}")
        return False


def convert_pdf_to_png(
    pdf_path, output_png_path=None, dpi_scale=2.0, show_jupyter=True):
    """Render a PDF to PNG format and optionally display it in Jupyter."""
    pdf_p = Path(pdf_path)
    if not pdf_p.exists() or pdf_p.stat().st_size == 0:
        logger.error(f"Invalid or empty PDF for conversion: {pdf_p}")
        return None

    try:
        import fitz  # PyMuPDF
        
        # Open with a new file handle
        doc = fitz.open(str(pdf_p))
        if doc.page_count == 0:
            doc.close()
            raise ValueError("Stitched PDF document contains no pages.")
            
        page = doc[0]
        mat = fitz.Matrix(dpi_scale, dpi_scale)
        pix = page.get_pixmap(matrix=mat)
        
        png_p = Path(output_png_path) if output_png_path else \
                pdf_p.with_suffix(".png")
        pix.save(str(png_p))
        doc.close()
        
        if show_jupyter:
            try:
                from IPython.display import Image, display
                display(Image(filename=str(png_p), width=800))
            except (ImportError, NameError):
                pass
        return png_p
    except Exception as e:
        logger.error(f"Failed to generate PNG preview: {e}")
        return None


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
        self.mode = self._detect_mode()

    def _detect_qa_folders(self):
        """Automatically scan for QA directories and sort them."""
        if not self.base_dir.exists():
            return []
        folders = [
            d for d in self.base_dir.iterdir() 
            if d.is_dir() and "QA" in d.name
        ]
        return sorted([d.name for d in folders])

    def _detect_mode(self):
        """Infer polarity mode (POS/NEG) by scanning output files."""
        if not self.qa_folders:
            return "POS"
            
        target_dir = self.base_dir / self.qa_folders[0]
        for file in target_dir.glob("*.pdf"):
            if "_POS.pdf" in file.name:
                return "POS"
            elif "_NEG.pdf" in file.name:
                return "NEG"
        return "POS"

    def compile_assessor_report(
        self, 
        is_multi_batch: bool = True, 
        report_folder: str = "13_Report_Markdown", 
        cols: int = 4, 
        show_jupyter: bool = True
    ) -> None:
        """
        Compile QA PDF plots from different stages into grid matrices 
        and synchronize PNG assets for report embedding.
        """
        
        if not self.qa_folders:
            logger.error("No QA folders detected in the base directory.")
            return

        report_path = self.base_dir / report_folder
        assets_path = report_path / "assets"
        
        report_path.mkdir(parents=True, exist_ok=True)
        assets_path.mkdir(parents=True, exist_ok=True)

        # =================================================================
        # [Logic Branching]: Determine correlation plot type and prefix 
        # based on the experimental design (Single vs Multi-batch).
        # =================================================================
        if is_multi_batch:
            corr_prefix = "03_Batch_Correlation"
            corr_file = f"Batch_Corr_HM_{self.mode}.pdf"
            logger.info(
                "Multi-batch design detected. Assembling Batch Correlation"
                "Grid.")
        else:
            corr_prefix = "03_QC_Correlation"
            corr_file = f"QC_Corr_HM_{self.mode}.pdf"
            logger.info(
                "Single-batch design detected. Assembling QC Correlation"
                "Grid.")

        # Construct target mapping for batch stitching
        target_map = {
            "01_QC_Sample_RSD": f"RSD_Barplot_{self.mode}.pdf",
            "02_PCA_Scatter": f"QC_AS_PCA_Scatter_{self.mode}.pdf",
            corr_prefix: corr_file,  # <--- Dynamically injected
            "04_Outlier_Diagnosis": f"Outlier_Scatter_{self.mode}.pdf"
        }
        # =================================================================

        for prefix, target_file in target_map.items():
            pdf_name = f"{prefix}_Grid_{self.mode}.pdf"
            pdf_out = report_path / pdf_name

            # Collect source PDFs from all detected QA sub-directories
            input_pdfs = []
            for folder in self.qa_folders:
                file_path = self.base_dir / folder / target_file
                
                if file_path.exists():
                    input_pdfs.append(file_path)

            if not input_pdfs:
                logger.warning(f"Skipped {pdf_name}: No source files found.")
                continue

            # Execute PDF stitching (preserving vector layers for AI editing)
            success = stitch_pdf_grids(
                pdf_paths=input_pdfs, output_name=pdf_out, cols=cols)

            if not success:
                logger.warning(f"Failed to stitch grid for {pdf_name}.")
                continue
            
            if success:
                # Brief sleep to ensure OS handles file-locking after disk I/O
                time.sleep(0.3)
                # Convert vector PDF to high-resolution PNG for markdown embedding
                png_path = assets_path / f"{prefix}_Grid_{self.mode}.png"
                convert_pdf_to_png(
                    pdf_path=pdf_out, output_png_path=png_path, 
                    dpi_scale=2.0, show_jupyter=show_jupyter
                )

        logger.success(f"Report assets globally compiled at: {report_path}")

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

    def _extract_sequential_stats(self, obj_dict: dict) -> dict:
        """Extracts and restructures attributes from pipeline metrics dictionary.
        
        Parses the nested dictionary exported by the final pipeline execution
        to populate the Jinja2 template context using semantic module names.

        Args:
            obj_dict: The nested dictionary exported by pipeline execution.

        Returns:
            A restructured dictionary mapped for Jinja2 template rendering.
        """
        def get_val(d, *keys, default="N/A"):
            for k in keys:
                if isinstance(d, dict) and k in d:
                    d = d[k]
                else:
                    return default
            return d

        def safe_pct(val):
            return val * 100 if isinstance(val, (int, float)) else "N/A"

        # Map top-level module metrics
        raw_m = obj_dict.get("raw_dataset", {})
        mv_m = obj_dict.get("high_mv_feature_filtering", {})
        intra_m = obj_dict.get("intra_signal_correction", {})
        inter_m = obj_dict.get("inter_signal_correction", {})
        lq_m = obj_dict.get("low_quality_feature_filtering", {})
        imp_m = obj_dict.get("missing_value_imputation", {})
        norm_m = obj_dict.get("normalization", {})

        # --- Generate Markdown Table for Batch Distribution ---
        batch_dist = get_val(raw_m, "batches", "batch_distribution", default={})
        batch_rows = []
        if isinstance(batch_dist, dict):
            for b_id, b_info in batch_dist.items():
                batch_rows.append([
                    b_id,
                    b_info.get("Total", 0),
                    b_info.get("QC", 0),
                    b_info.get("Blank", 0),
                    b_info.get("Sample", 0),
                    b_info.get("Inject Order", "N/A")
                ])
        
        batch_table = tabulate(
            batch_rows, 
            headers=["Batch", "Total", "QC", "Blank", "Sample", "Inject Order"],
            tablefmt="github"
        )

        ord_batches = get_val(raw_m, "batches", "ordered_batches", default=[])
        ord_batches_str = (
            ", ".join(ord_batches) if isinstance(ord_batches, list) else "N/A"
        )

        # --- Construct Semantic Stats Dictionary ---
        stats = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "mode": get_val(raw_m, "mode", default=""),
            "raw_dataset": {
                "raw_feats": get_val(raw_m, "features", "total"),
                "is_count": get_val(
                    raw_m, "features", "internal_standards_count"),
                "total_samples": get_val(raw_m, "samples", "total"),
                "actual_samples": get_val(raw_m, "samples", "actual"),
                "qc_samples": get_val(raw_m, "samples", "qc"),
                "blank_samples": get_val(raw_m, "samples", "blank"),
                "batch_count": get_val(raw_m, "batches", "batch_count"),
                "ordered_batches": ord_batches_str,
                "batch_table": batch_table
            },
            
            "high_mv_feature_filtering": {
                "filtering_level": get_val(mv_m, "filtering_level"),
                "sample_mv_tol": get_val(mv_m, "thresholds", "sample_mv_tol"),
                "mnar_group_tol": get_val(mv_m, "thresholds", "mnar_group_tol"),
                "mnar_qc_tol": get_val(mv_m, "thresholds", "mnar_qc_tol"),
                "mar_count": get_val(
                    mv_m, "missing_classification", "mar_count"),
                "mnar_total": get_val(
                    mv_m, "missing_classification", "mnar_total"),
                "pre_mv_filter_count": get_val(
                    mv_m, "feature_retention", "pre_mv_filter_count"),
                "after_mv_filter_count": get_val(
                    mv_m, "feature_retention", "after_mv_filter_count"),
                "dropped_count": get_val(
                    mv_m, "feature_retention", "dropped_count"),
                "retention_rate_pct": get_val(
                    mv_m, "feature_retention", "retention_rate_pct")
            },
            
            "intra_signal_correction": {
                "base_est": get_val(intra_m, "methodology", "base_est"),
                "rsd_baseline": safe_pct(get_val(
                    intra_m, "performance", "median_qc_rsd_baseline")),
                "rsd_current": safe_pct(get_val(
                    intra_m, "performance", "median_qc_rsd_current")),
                "absolute_rsd_reduction": safe_pct(get_val(
                    intra_m, "performance", "absolute_rsd_reduction")),
                "relative_noise_reduction": safe_pct(get_val(
                    intra_m, "performance", "relative_noise_reduction"))
            },
            
            "inter_signal_correction": {
                "base_est": get_val(inter_m, "methodology", "base_est"),
                "rsd_baseline": safe_pct(get_val(
                    inter_m, "performance", "median_qc_rsd_baseline")),
                "rsd_current": safe_pct(get_val(
                    inter_m, "performance", "median_qc_rsd_current")),
                "absolute_rsd_reduction": safe_pct(get_val(
                    inter_m, "performance", "absolute_rsd_reduction")),
                "relative_noise_reduction": safe_pct(get_val(
                    inter_m, "performance", "relative_noise_reduction"))
            },
            
            "low_quality_feature_filtering": {
                "blank_ratio_tol": get_val(
                    lq_m, "thresholds", "blank_ratio_tol"),
                "qc_rsd_tol": get_val(
                    lq_m, "thresholds", "qc_rsd_tol"),
                "pre_stage2_total": get_val(
                    lq_m, "feature_retention", "pre_stage2", "total"),
                "post_blank_total": get_val(
                    lq_m, "feature_retention", "post_blank_check", "total"),
                "post_rsd_total": get_val(
                    lq_m, "feature_retention", "post_rsd_check", "total"),
                "dropped_by_blank": get_val(
                    lq_m, "filtering_breakdown", "dropped_by_blank", "total"),
                "dropped_by_rsd": get_val(
                    lq_m, "filtering_breakdown", "dropped_by_rsd", "total")
            },
            
            "missing_value_imputation": {
                "mar_method_selected": str(get_val(
                    imp_m, "strategies", "mar_method_selected")).upper(),
                "mnar_method": str(get_val(
                    imp_m, "strategies", "mnar_method")).upper(),
                "mnar_fraction": get_val(
                    imp_m, "strategies", "mnar_fraction"),
                "mar_count": get_val(
                    imp_m, "feature_distribution", "mar_count"),
                "mnar_count": get_val(
                    imp_m, "feature_distribution", "mnar_count"),
                "qc_jsd": get_val(
                    imp_m, "distribution_similarity", "QC", 
                    "Before vs Imputation", "jsd"),
                "sample_jsd": get_val(
                    imp_m, "distribution_similarity", "Sample", 
                    "Before vs Imputation", "jsd")
            },
            
            "normalization": {
                "sample_wise_method": str(get_val(
                    norm_m, "strategies", "sample_wise_method")).upper(),
                "feature_wise_method": str(get_val(
                    norm_m, "strategies", "feature_wise_method")).upper(),
                "quantile_norm_active": get_val(
                    norm_m, "strategies", "quantile_norm_active"),
                "vsn_scale": get_val(
                    norm_m, "vsn_parameters", "vsn_scale"),
                "qc_jsd_sample": get_val(
                    norm_m, "distribution_similarity", "QC", 
                    "Before vs Sample-Norm", "jsd"),
                "qc_jsd_feature": get_val(
                    norm_m, "distribution_similarity", "QC", 
                    "Sample-Norm vs Feature-Norm", "jsd"),
                "sample_jsd_sample": get_val(
                    norm_m, "distribution_similarity", "Sample", 
                    "Before vs Sample-Norm", "jsd"),
                "sample_jsd_feature": get_val(
                    norm_m, "distribution_similarity", "Sample", 
                    "Sample-Norm vs Feature-Norm", "jsd")
            }
        }

        # Dynamically extract NRMSE for the selected MAR method
        mar_sel = get_val(imp_m, "strategies", "mar_method_selected", "")
        if mar_sel and mar_sel != "N/A":
            nrmse = get_val(
                imp_m, "performance", mar_sel, "nrmse_low", default=0)
            stats["missing_value_imputation"]["nrmse_low"] = (
                f"{nrmse:.4f}" if isinstance(nrmse, (int, float)) else "N/A")
        else:
            stats["missing_value_imputation"]["nrmse_low"] = "N/A"

        return stats

    def generate_markdown(
        self, obj_dict: dict, report_folder: str = "15_Report_Markdown"):
        """Render a single comprehensive Markdown report from the object pool."""
        stats = self._extract_sequential_stats(obj_dict)
        
        try:
            template = self.env.get_template("report_template.md.j2")
            content = template.render(stats)
        except Exception as e:
            logger.error(f"Jinja2 rendering failed: {e}")
            return

        out_dir = self.base_dir / report_folder
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"Comprehensive_QC_Report_{
            stats['step1']['mode']}.md"
        
        with out_path.open("w", encoding="utf-8") as f:
            f.write(content)
        
        logger.success(f"Final comprehensive report generated: {out_path}")