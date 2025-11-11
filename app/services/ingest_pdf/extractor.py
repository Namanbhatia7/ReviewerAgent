# app/services/ingest_pdf/extractor.py
from __future__ import annotations
from typing import Tuple, Optional
from dataclasses import dataclass
import fitz  # PyMuPDF
from app.models.schemas import ExtractedPayload, TextField

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

def extract_minimal(pdf_bytes: bytes, *, vector_text: bool) -> Tuple[ExtractedPayload, int, float]:
    """
    Returns:
      payload (ExtractedPayload), dpi (int), ocr_conf (float)
    Notes:
      - Minimal impl: does not OCR; sets placeholders.
      - We'll fill real fields after you wire anchors + OCR pipeline.
    """
    # Minimal payload: status + empty fields to satisfy schema
    payload = ExtractedPayload(
        status="extracted_minimal",
        prompt=TextField(text=""),
        questions=[],
        rating=None,
        explanation=TextField(text=""),
        page_instructions=None,
        notes={"vector_text": vector_text}
    )
    dpi = 0
    ocr_conf = 0.0
    return payload, dpi, ocr_conf
