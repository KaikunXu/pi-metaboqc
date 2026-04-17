# src/pimqc/report_utils.py
"""
Purpose of script: Global utility classes and functions for assembling, 
rendering, and exporting automated project-level QC reports.
"""

from pathlib import Path
from loguru import logger

# =========================================================================
# Atomic Utility Functions (Highly Reusable & Decoupled)
# =========================================================================

def stitch_pdf_grids(pdf_paths, output_name, cols=3):
    """Stitch PDF paths into a grid using fitz to prevent background masking.
    
    This implementation uses PyMuPDF's show_pdf_page, which handles 
    transparency and content layering more robustly than pypdf's 
    merge_page, preventing text from being obscured by opaque backgrounds.
    """
    try:
        import fitz
    except ImportError:
        logger.error("fitz (PyMuPDF) is required for robust PDF stitching.")
        return False

    valid_docs = []
    for path in pdf_paths:
        p = Path(path)
        if p.exists() and p.stat().st_size > 0:
            valid_docs.append(fitz.open(str(p)))

    if not valid_docs:
        return False

    # Determine dimensions based on the largest subplot
    max_w = max(doc[0].rect.width for doc in valid_docs)
    max_h = max(doc[0].rect.height for doc in valid_docs)

    rows = (len(valid_docs) + cols - 1) // cols
    grid_w = max_w * cols
    grid_h = max_h * rows

    out_doc = fitz.open()
    out_page = out_doc.new_page(width=grid_w, height=grid_h)

    for i, doc in enumerate(valid_docs):
        row, col = i // cols, i % cols
        
        # Calculate destination rectangle
        # fitz coordinate system origin is at the top-left
        x0 = col * max_w
        y0 = row * max_h
        x1 = x0 + doc[0].rect.width
        y1 = y0 + doc[0].rect.height
        
        target_rect = fitz.Rect(x0, y0, x1, y1)
        
        # Overlay the source page onto the grid page as a vector object
        # This method isolation prevents resource dictionary conflicts
        out_page.show_pdf_page(target_rect, doc, 0)
        doc.close()

    out_doc.save(str(output_name))
    out_doc.close()
    return True


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
# OOP Report Generator Classes
# =========================================================================

class ProjectReporter:
    """Global reporting suite that auto-detects pipeline steps and modes."""

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
            if d.is_dir() and "_QA_" in d.name
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
        self, report_folder="13_Report_Markdown", cols=2, show_jupyter=True):
        """Compile QA PDF plots into stitched grids and PNG assets."""
        import time
        
        if not self.qa_folders:
            logger.error("No QA folders detected in the base directory.")
            return

        report_path = self.base_dir / report_folder
        assets_path = report_path / "assets"
        
        report_path.mkdir(parents=True, exist_ok=True)
        assets_path.mkdir(parents=True, exist_ok=True)

        target_map = {
            "01_QC_Sample_RSD": f"RSD_Barplot_{self.mode}.pdf",
            "02_PCA_Scatter": f"QC_AS_PCA_Scatter_{self.mode}.pdf",
            "03_Batch_Correlation": f"Batch_Corr_HM_{self.mode}.pdf",
            "04_Outlier_Diagnosis": f"Outlier_Scatter_{self.mode}.pdf"
        }

        for prefix, target_file in target_map.items():
            pdf_name = f"{prefix}_Grid_{self.mode}.pdf"
            pdf_out = report_path / pdf_name

            # Business logic: Resolve target paths & handle alternative files
            input_pdfs = []
            for folder in self.qa_folders:
                file_path = self.base_dir / folder / target_file
                
                # Fallback for Batch correlation if QC Correlation is missing
                if not file_path.exists() and "QC_Corr" in target_file:
                    alt_target = target_file.replace(
                        "Batch_Corr_HM", "QC_Corr_HM")
                    alt_path = self.base_dir / folder / alt_target
                    if alt_path.exists():
                        file_path = alt_path

                if file_path.exists():
                    input_pdfs.append(file_path)

            if not input_pdfs:
                logger.warning(f"Skipped {pdf_name}: No source files found.")
                continue

            # Execute clean PDF stitching (pypdf vector preservation)
            success = stitch_pdf_grids(
                pdf_paths=input_pdfs, output_name=pdf_out, cols=cols)

            if not success:
                logger.warning(f"Failed to stitch grid for {pdf_name}.")
                continue
            
            if success:
                # Ensure the OS has finished disk writing before fitz reads it
                time.sleep(0.3)
                # Convert stitched PDF matrix to PNG for markdown embedding
                png_path = assets_path / f"{prefix}_Grid_{self.mode}.png"
                convert_pdf_to_png(
                    pdf_path=pdf_out, output_png_path=png_path, 
                    dpi_scale=2.0, show_jupyter=show_jupyter
                )

        logger.success(f"Report assets globally compiled at: {report_path}")