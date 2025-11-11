# app/services/ingest_pdf/extractor.py
from __future__ import annotations
from typing import Tuple
from dataclasses import dataclass
import fitz  # PyMuPDF

from sqlalchemy.orm import Session
from app.models.orm import Project
from app.services.ingest_pdf.anchors import load_project_config
from app.services.ingest_pdf.vector import extract_words, to_lines
from app.services.ingest_pdf.raster import ocr_pdf, lines_from_words
from app.services.ingest_pdf.extractor_logic import assemble_payload

@dataclass
class PDFPreflight:
    pages: int
    vector_text: bool

def preflight_pdf(pdf_bytes: bytes) -> PDFPreflight:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages = doc.page_count
        vector_text = any((doc.load_page(i).get_text("text") or "").strip() for i in range(pages))
        return PDFPreflight(pages=pages, vector_text=bool(vector_text))
    finally:
        doc.close()

def extract_pdf(db: Session, project_id: str, pdf_bytes: bytes) -> Tuple[str, float, object]:
    """
    Returns:
      status: "extracted"
      mean_ocr_conf: float
      payload: ExtractedPayload
    """
    # Load project anchors + heuristics + question schema
    proj = db.query(Project).filter(Project.project_id == project_id).first()
    if not proj:
        raise ValueError(f"Project '{project_id}' not found")
    cfg = load_project_config(proj.config_yaml)

    pf = preflight_pdf(pdf_bytes)
    mean_ocr_conf = 0.0

    try:
        ocr_res = ocr_pdf(pdf_bytes, dpi=cfg.heuristics.raster_dpi)
        mean_ocr_conf = ocr_res.mean_conf
        from app.services.ingest_pdf.raster import lines_from_words
        lines = lines_from_words(ocr_res.words, y_tol=4.0)
    except Exception as e:
        # Fallback to vector if OCR fails
        words = extract_words(pdf_bytes)
        lines = to_lines(words, y_tol=3.0)
        mean_ocr_conf = 0.0

    # (new) restrict to right panel if configured
    from app.services.ingest_pdf.layout import filter_right_panel
    lines = filter_right_panel(lines, min_x_ratio=getattr(cfg, "layout", {}).get("right_panel_min_x_ratio", None))

    payload = assemble_payload(lines, cfg, mean_ocr_conf=mean_ocr_conf)
    return payload.status, mean_ocr_conf, payload
