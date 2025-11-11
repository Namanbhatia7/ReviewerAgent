# app/services/ingest_pdf/vector.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import fitz  # PyMuPDF

@dataclass
class Word:
    page: int
    text: str
    bbox: Tuple[float, float, float, float]  # x0,y0,x1,y1

@dataclass
class Line:
    page: int
    text: str
    bbox: Tuple[float, float, float, float]
    words: List[Word]

def extract_words(pdf_bytes: bytes) -> List[Word]:
    out: List[Word] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            # PyMuPDF returns word tuples: (x0,y0,x1,y1,"text", block_no, line_no, word_no)
            for w in page.get_text("words"):
                x0, y0, x1, y1, text, *_ = w
                if text.strip():
                    out.append(Word(page=pno + 1, text=text, bbox=(x0, y0, x1, y1)))
    finally:
        doc.close()
    return out

def to_lines(words: List[Word], y_tol: float = 3.0) -> List[Line]:
    lines: List[Line] = []
    if not words:
        return lines
    # group by (page, approx baseline y)
    words_sorted = sorted(words, key=lambda w: (w.page, round(w.bbox[1]/y_tol)))
    cur: List[Word] = []
    cur_key = None
    for w in words_sorted:
        key = (w.page, round(w.bbox[1]/y_tol))
        if cur_key is None or key == cur_key:
            cur.append(w); cur_key = key
        else:
            lines.append(_finalize_line(cur))
            cur = [w]; cur_key = key
    if cur:
        lines.append(_finalize_line(cur))
    return lines

def _finalize_line(ws: List[Word]) -> Line:
    page = ws[0].page
    x0 = min(w.bbox[0] for w in ws); y0 = min(w.bbox[1] for w in ws)
    x1 = max(w.bbox[2] for w in ws); y1 = max(w.bbox[3] for w in ws)
    text = " ".join(w.text for w in sorted(ws, key=lambda w: w.bbox[0]))
    return Line(page=page, text=text, bbox=(x0, y0, x1, y1), words=sorted(ws, key=lambda w: w.bbox[0]))
