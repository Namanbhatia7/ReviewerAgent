# app/services/ingest_pdf/raster.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR

from app.services.ingest_pdf.vector import Word, Line, _finalize_line

@dataclass
class OCRResult:
    words: List[Word]
    mean_conf: float

def ocr_pdf(pdf_bytes: bytes, dpi: int = 260) -> OCRResult:
    images = convert_from_bytes(pdf_bytes, dpi=dpi)
    ocr = PaddleOCR(lang="en", show_log=False)
    words: List[Word] = []
    confidences = []
    for idx, img in enumerate(images):
        page_no = idx + 1
        # PaddleOCR expects path or numpy array; convert PIL to np.array
        import numpy as np
        arr = np.array(img)
        res = ocr.ocr(arr, cls=True)
        # res is list[ [ (bbox, (text, conf)), ... ] ]
        for block in res:
            for item in block:
                (x1,y1),(x2,y2),(x3,y3),(x4,y4) = item[0]
                text, conf = item[1]
                if not text or text.isspace():
                    continue
                x0 = min(x1,x2,x3,x4); y0 = min(y1,y2,y3,y4)
                x1m = max(x1,x2,x3,x4); y1m = max(y1,y2,y3,y4)
                words.append(Word(page=page_no, text=str(text), bbox=(float(x0), float(y0), float(x1m), float(y1m))))
                confidences.append(float(conf))
    mean_conf = float(sum(confidences)/len(confidences)) if confidences else 0.0
    return OCRResult(words=words, mean_conf=mean_conf)

def lines_from_words(words: List[Word], y_tol: float = 4.0) -> List[Line]:
    # reuse _finalize_line and grouping similar to vector
    # Sort then group by (page, rounded y)
    if not words: return []
    words_sorted = sorted(words, key=lambda w: (w.page, round(w.bbox[1]/y_tol)))
    lines = []
    cur = []; cur_key = None
    for w in words_sorted:
        key = (w.page, round(w.bbox[1]/y_tol))
        if cur_key is None or key == cur_key:
            cur.append(w); cur_key = key
        else:
            lines.append(_finalize_line(cur)); cur = [w]; cur_key = key
    if cur: lines.append(_finalize_line(cur))
    return lines
