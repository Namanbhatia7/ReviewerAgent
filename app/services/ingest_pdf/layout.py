# app/services/ingest_pdf/layout.py
from __future__ import annotations
from typing import List, Optional
from app.services.ingest_pdf.vector import Line

def filter_right_panel(lines: List[Line], min_x_ratio: Optional[float]) -> List[Line]:
    if not lines or not min_x_ratio:
        return lines
    # Estimate page width by the widest line on each page, then keep lines whose x0 >= (width * min_x_ratio)
    by_page = {}
    for ln in lines:
        by_page.setdefault(ln.page, []).append(ln)
    out: List[Line] = []
    for page, lns in by_page.items():
        width = max(l.bbox[2] for l in lns) if lns else 0.0
        cutoff = width * float(min_x_ratio)
        out.extend([l for l in lns if l.bbox[0] >= cutoff])
    return out
