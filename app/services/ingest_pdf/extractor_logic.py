# app/services/ingest_pdf/extract_logic.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import re
from dataclasses import dataclass

from app.services.ingest_pdf.vector import Line, Word
from app.services.ingest_pdf.anchors import ProjectConfig, all_aliases_for
from app.models.schemas import ExtractedPayload, TextField, Question, Option, Rating
from app.services.ingest_pdf.cv_radios import radio_selected_near_label

# ---------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------
def _norm(s: str) -> str:
    return (s or "").strip().lower()

@dataclass
class AnchorHit:
    key: str
    line: Line

# ---------------------------------------------------------------
# Anchor and block utilities
# ---------------------------------------------------------------
def find_anchors(lines: List[Line], cfg: ProjectConfig) -> List[AnchorHit]:
    aliases_map: Dict[str, List[str]] = {k: [a.lower() for a in v] for k, v in cfg.anchors.items()}
    hits: List[AnchorHit] = []
    for ln in lines:
        low = _norm(ln.text)
        for key, aliases in aliases_map.items():
            if any(a in low for a in aliases):
                hits.append(AnchorHit(key=key, line=ln))
                break
    # Keep first occurrence per key
    seen = set()
    uniq: List[AnchorHit] = []
    for h in hits:
        if h.key in seen:
            continue
        seen.add(h.key)
        uniq.append(h)
    return uniq

def capture_block_after(lines: List[Line], start_line: Line, window_px: int) -> List[Line]:
    page = start_line.page
    top_y = start_line.bbox[3]  # below the anchor line
    return [
        ln for ln in lines
        if ln.page == page and ln.bbox[1] >= top_y and (ln.bbox[1] - top_y) <= window_px
    ]

def lines_to_text_and_bbox(blines: List[Line]) -> Tuple[str, Optional[Tuple[int,float,float,float,float]]]:
    if not blines:
        return "", None
    page = blines[0].page
    text = "\n".join(ln.text for ln in blines)
    x0 = min(ln.bbox[0] for ln in blines)
    y0 = min(ln.bbox[1] for ln in blines)
    x1 = max(ln.bbox[2] for ln in blines)
    y1 = max(ln.bbox[3] for ln in blines)
    return text, (page, x0, y0, x1 - x0, y1 - y0)

# ---------------------------------------------------------------
# Option parsing (text-based)
# ---------------------------------------------------------------
def parse_options(block_text: str, opt_markers: List[str], schema_opts: List[Dict[str, Any]]) -> List[Option]:
    """Parse options and infer selection from OCR text."""
    options: List[Option] = []
    lines = [l.strip() for l in block_text.splitlines() if l.strip()]
    norm_lines = [_norm(l) for l in lines]

    SEL_GLYPHS = {"[x]", "(x)", "✓", "✔", "●", "⦿", "◉", "•"}
    UNS_GLYPHS = {"[ ]", "( )", "○", "◯", "◌", "⭘"}
    EXPLICIT_PATTERNS = [
        r"(?:selected|your\s*answer|choice)\s*[:\-]\s*(.+)$",
        r"(?:answer)\s*[:\-]\s*(.+)$"
    ]

    def _line_has_selected_mark(line: str, raw: str) -> float:
        low = _norm(line)
        raw_low = _norm(raw)
        if raw_low not in low:
            return 0.0
        idx = low.find(raw_low)
        window = low[max(0, idx - 8): idx] + low[idx + len(raw_low): idx + len(raw_low) + 8]
        if any(g.lower() in window for g in [s.lower() for s in SEL_GLYPHS if s != "•"]):
            return 0.9
        if "•" in window:
            return 0.6
        if " selected" in low or " (selected" in low or low.startswith("selected:"):
            return 0.9
        return 0.0

    # Build option list
    raw_map = [(o["raw"], o.get("norm")) for o in schema_opts]
    for raw, norm in raw_map:
        best_conf = 0.0
        for orig, low in zip(lines, norm_lines):
            raw_low = _norm(raw)
            if raw_low in low:
                conf = _line_has_selected_mark(orig, raw)
                if conf == 0.0 and any(u in low for u in [u.lower() for u in UNS_GLYPHS]):
                    conf = max(conf, 0.1)
                best_conf = max(best_conf, conf)
        options.append(Option(text=raw, selected=(best_conf >= 0.5), selection_confidence=best_conf))

    # Explicit "Selected: ..." lines
    if not any(o.selected for o in options):
        chosen_norm = None
        for low in norm_lines:
            for pat in EXPLICIT_PATTERNS:
                m = re.search(pat, low, flags=re.I)
                if m:
                    chosen_norm = _norm(m.group(1))
                    break
            if chosen_norm:
                break
        if chosen_norm:
            for o in options:
                if _norm(o.text) in chosen_norm or chosen_norm in _norm(o.text):
                    o.selected = True
                    o.selection_confidence = max(o.selection_confidence, 0.95)

    # Inline line with both markers
    if not any(o.selected for o in options):
        for orig, low in zip(lines, norm_lines):
            if any(s in low for s in [s.lower() for s in SEL_GLYPHS]) and any(u in low for u in [u.lower() for u in UNS_GLYPHS]):
                sel_positions = []
                for g in SEL_GLYPHS:
                    gi = low.find(g.lower())
                    if gi >= 0:
                        sel_positions.append(gi)
                if sel_positions:
                    selected_idx = min(sel_positions)
                    best, best_dist = None, 1e9
                    for o in options:
                        oi = low.find(_norm(o.text))
                        if oi >= 0:
                            dist = abs(oi - selected_idx)
                            if dist < best_dist:
                                best = o
                                best_dist = dist
                    if best:
                        best.selected = True
                        best.selection_confidence = max(best.selection_confidence, 0.8)
                        break

    # Single-select cleanup
    selected_opts = [o for o in options if o.selected]
    if len(selected_opts) > 1:
        best = max(selected_opts, key=lambda x: x.selection_confidence)
        for o in options:
            o.selected = (o is best)
            if not o.selected:
                o.selection_confidence = min(o.selection_confidence, 0.2)

    for o in options:
        o.selection_confidence = max(0.0, min(1.0, o.selection_confidence))
    return options

# ---------------------------------------------------------------
# CV refinement: detect filled radio buttons
# ---------------------------------------------------------------
def refine_selection_with_cv(
    options: List[Option],
    option_label_bboxes: List[Optional[Tuple[int, float, float, float, float]]],
    page_images: Optional[Dict[int, "np.ndarray"]] = None,
):
    if not page_images:
        return
    for opt, obox in zip(options, option_label_bboxes):
        if obox is None:
            continue
        page, x0, y0, x1, y1 = obox
        img = page_images.get(page)
        if img is None:
            continue
        sel, conf = radio_selected_near_label(img, (x0, y0, x1, y1))
        if conf > opt.selection_confidence:
            opt.selected = sel
            opt.selection_confidence = conf

# ---------------------------------------------------------------
# Rating parser and payload assembly
# ---------------------------------------------------------------
def parse_rating(block_text: str) -> Optional[Rating]:
    patterns = [
        r"(\d(?:\.\d)?)\s*/\s*5",
        r"(\d(?:\.\d)?)\s+of\s+5",
        r"(?:score|rating)\s*[:=]?\s*(\d(?:\.\d)?)",
    ]
    for p in patterns:
        m = re.search(p, block_text, flags=re.I)
        if m:
            try:
                val = float(m.group(1))
                return Rating(value=val, label=m.group(0))
            except Exception:
                continue
    return None

def assemble_payload(
    lines: List[Line],
    cfg: ProjectConfig,
    mean_ocr_conf: float | None = None,
    page_images: Optional[Dict[int, "np.ndarray"]] = None,
) -> ExtractedPayload:
    window = cfg.heuristics.block_scan_window_px
    anchors_hits = find_anchors(lines, cfg)

    blocks: Dict[str, List[Line]] = {}
    for hit in anchors_hits:
        blocks[hit.key] = capture_block_after(lines, hit.line, window)

    warnings: List[str] = []
    prompt_text, prompt_bbox = lines_to_text_and_bbox(blocks.get("header_prompt", []))
    prompt = TextField(text=prompt_text, bbox=prompt_bbox) if prompt_text else None

    explain_blocks = {}
    for key in ["q2_prime", "q3_prime", "q4_prime", "q5_prime", "q6_prime"]:
        t, b = lines_to_text_and_bbox(blocks.get(key, []))
        if t:
            explain_blocks[key] = TextField(text=t, bbox=b)

    rating: Optional[Rating] = None
    for key in ["q5_prime", "q4_prime", "q3_prime", "q2_prime"]:
        tf = explain_blocks.get(key)
        if tf:
            r = parse_rating(tf.text)
            if r:
                r.bbox = tf.bbox
                rating = r
                break

    # Build questions
    questions: List[Question] = []
    for q in cfg.questions:
        qid = q.get("id")
        anchors = q.get("anchors", [])
        block_lines: List[Line] = []
        for a in anchors:
            if a in blocks:
                block_lines = blocks[a]
                break
        qtext, qbbox = lines_to_text_and_bbox(block_lines)
        schema_opts = q.get("options", []) or []
        options = parse_options(qtext, cfg.heuristics.option_markers, schema_opts) if schema_opts else []

        # approximate label bboxes for CV refinement
        opt_bboxes: List[Optional[Tuple[int, float, float, float, float]]] = []
        for o in schema_opts:
            ob = None
            raw_low = _norm(o["raw"])
            for ln in block_lines:
                if raw_low in _norm(ln.text):
                    page = ln.page
                    x0, y0, x1, y1 = ln.bbox
                    ob = (page, x0, y0, x1, y1)
                    break
            opt_bboxes.append(ob)
        refine_selection_with_cv(options, opt_bboxes, page_images=page_images)

        questions.append(
            Question(id=qid, text=qtext, bbox=qbbox, options=options, per_side=q.get("per_side"))
        )

    exp_tf = (
        explain_blocks.get("q6_prime")
        or explain_blocks.get("q5_prime")
        or explain_blocks.get("q4_prime")
        or explain_blocks.get("q3_prime")
        or explain_blocks.get("q2_prime")
    )
    explanation = exp_tf if exp_tf else TextField(text="")

    payload = ExtractedPayload(
        status="extracted",
        prompt=prompt,
        questions=questions,
        rating=rating,
        explanation=explanation,
        page_instructions=None,
        notes={
            "warnings": warnings,
            "unmatched_anchors": [k for k in cfg.anchors.keys() if k not in blocks],
            "mean_ocr_conf": mean_ocr_conf,
        },
    )
    return payload
