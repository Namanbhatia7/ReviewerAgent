# app/services/ingest_pdf/extract_logic.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import re
from dataclasses import dataclass
from app.services.ingest_pdf.vector import Line, Word
from app.services.ingest_pdf.anchors import ProjectConfig, all_aliases_for
from app.models.schemas import ExtractedPayload, TextField, Question, Option, Rating

@dataclass
class AnchorHit:
    key: str
    line: Line

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def find_anchors(lines: List[Line], cfg: ProjectConfig) -> List[AnchorHit]:
    aliases_map: Dict[str, List[str]] = {k: [a.lower() for a in v] for k, v in cfg.anchors.items()}
    hits: List[AnchorHit] = []
    for ln in lines:
        low = _norm(ln.text)
        for key, aliases in aliases_map.items():
            if any(a in low for a in aliases):
                hits.append(AnchorHit(key=key, line=ln))
                break
    # keep first occurrence per key (most templates list anchors once)
    seen = set(); uniq: List[AnchorHit] = []
    for h in hits:
        if h.key in seen: continue
        seen.add(h.key); uniq.append(h)
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
    x0 = min(ln.bbox[0] for ln in blines); y0 = min(ln.bbox[1] for ln in blines)
    x1 = max(ln.bbox[2] for ln in blines); y1 = max(ln.bbox[3] for ln in blines)
    return text, (page, x0, y0, x1 - x0, y1 - y0)


def parse_options(block_text: str, opt_markers: List[str], schema_opts: List[Dict[str, Any]]) -> List[Option]:
    """
    Parse options and infer selection for screenshot-style (raster) PDFs.

    Heuristics:
    - Detect selection glyphs on the same line as the option text: [x], (x), ✓, ✔, ●, ⦿
    - Treat unselected glyphs: [ ], ( ), ○, ◯, ◌, ⭘
    - Support explicit lines like "Selected: <option>", "Your answer: <option>", "Choice: <option>"
    - Resolve multiple positives by keeping the highest-confidence candidate.
    """
    # 1) Prep
    options: List[Option] = []
    lines = [l.strip() for l in block_text.splitlines() if l.strip()]
    # normalized parallel list for matching
    norm_lines = [ _norm(l) for l in lines ]

    # Selected/Unselected glyphs (expandable)
    SEL_GLYPHS = {"[x]", "(x)", "✓", "✔", "●", "⦿", "◉", "•"}   # '•' only counts if adjacent to the option text
    UNS_GLYPHS = {"[ ]", "( )", "○", "◯", "◌", "⭘"}

    # Phrases that explicitly state the chosen option
    EXPLICIT_PATTERNS = [
        r"(?:selected|your\s*answer|choice)\s*[:\-]\s*(.+)$",
        r"(?:answer)\s*[:\-]\s*(.+)$"
    ]

    # Helper: bounded window check for glyphs around an occurrence of raw option text
    def _line_has_selected_mark(line: str, raw: str) -> float:
        # search for raw inside line; if found, look ±8 chars for glyphs
        low = _norm(line); raw_low = _norm(raw)
        if raw_low not in low:
            return 0.0
        idx = low.find(raw_low)
        window = low[max(0, idx - 8): idx] + low[idx + len(raw_low): idx + len(raw_low) + 8]
        # confidence tiers
        if any(g.lower() in window for g in [s.lower() for s in SEL_GLYPHS if s not in {"•"}]):
            return 0.9
        # '•' is noisy in OCR; give smaller weight unless explicitly near text
        if "•" in window:
            return 0.6
        # textual hints on same line
        if " selected" in low or " (selected" in low or low.startswith("selected:"):
            return 0.9
        return 0.0

    # 2) Initial pass: create Option objects and look for line-level matches
    raw_map = [(o["raw"], o.get("norm")) for o in schema_opts]
    for raw, norm in raw_map:
        best_conf = 0.0
        # scan lines for this option
        for orig, low in zip(lines, norm_lines):
            raw_low = _norm(raw)
            if raw_low in low:
                conf = _line_has_selected_mark(orig, raw)
                # If explicit "[ ]" near it and no positive mark, downweight slightly
                if conf == 0.0 and any(m in low for m in [u.lower() for u in UNS_GLYPHS]):
                    conf = max(conf, 0.1)
                best_conf = max(best_conf, conf)
        options.append(Option(text=raw, selected=(best_conf >= 0.5), selection_confidence=best_conf))

    # 3) If none selected yet, look for explicit "Selected: <option>" style lines
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

    # 4) If still none, last resort: look for a single line that lists options with a bracket
    #    e.g., "(x) Good  ( ) Excellent  ( ) Fair"
    if not any(o.selected for o in options):
        for orig, low in zip(lines, norm_lines):
            # favor any line that contains both a selected and unselected marker
            if any(s in low for s in [s.lower() for s in SEL_GLYPHS]) and any(u in low for u in [u.lower() for u in UNS_GLYPHS]):
                # pick the option whose text occurs closest to a selected glyph
                sel_positions = []
                for g in SEL_GLYPHS:
                    gi = low.find(g.lower())
                    if gi >= 0:
                        sel_positions.append(gi)
                if sel_positions:
                    selected_idx = min(sel_positions)
                    # choose an option whose occurrence index is nearest to the selected glyph
                    best = None; best_dist = 1e9
                    for o in options:
                        oi = low.find(_norm(o.text))
                        if oi >= 0:
                            dist = abs(oi - selected_idx)
                            if dist < best_dist:
                                best = o; best_dist = dist
                    if best:
                        best.selected = True
                        best.selection_confidence = max(best.selection_confidence, 0.8)
                        break

    # 5) Enforce single-select radio behavior: keep the highest-confidence selection
    selected_opts = [o for o in options if o.selected]
    if len(selected_opts) > 1:
        best = max(selected_opts, key=lambda x: x.selection_confidence)
        for o in options:
            o.selected = (o is best)
            o.selection_confidence = o.selection_confidence if o is best else min(o.selection_confidence, 0.2)

    # 6) Clip confidence to [0,1] and return
    for o in options:
        if o.selection_confidence < 0.0: o.selection_confidence = 0.0
        if o.selection_confidence > 1.0: o.selection_confidence = 1.0

    return options

def parse_rating(block_text: str) -> Optional[Rating]:
    # Look for "4/5", "4 of 5", "score: 4", "rating 4"
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

def assemble_payload(lines: List[Line], cfg: ProjectConfig, mean_ocr_conf: float | None = None) -> ExtractedPayload:
    window = cfg.heuristics.block_scan_window_px
    anchors_hits = find_anchors(lines, cfg)
    # Map key->block
    blocks: Dict[str, List[Line]] = {}
    for hit in anchors_hits:
        blocks[hit.key] = capture_block_after(lines, hit.line, window)

    warnings: List[str] = []
    # Prompt/header
    prompt_text, prompt_bbox = lines_to_text_and_bbox(blocks.get("header_prompt", []))
    prompt = TextField(text=prompt_text, bbox=prompt_bbox) if prompt_text else None

    # Explanation blocks (q2'/q3'/q4'/q5')
    explain_blocks = {}
    for key in ["q2_prime", "q3_prime", "q4_prime", "q5_prime"]:
        t, b = lines_to_text_and_bbox(blocks.get(key, []))
        if t:
            explain_blocks[key] = TextField(text=t, bbox=b)

    # Rating block (if any explicit anchor like "rating" exists, add to YAML; else try q5')
    # We'll try to parse around explanation blocks too
    rating: Optional[Rating] = None
    for key in ["q5_prime", "q4_prime", "q3_prime", "q2_prime"]:
        tf = explain_blocks.get(key)
        if tf:
            r = parse_rating(tf.text)
            if r:
                r.bbox = tf.bbox
                rating = r; break

    # Build questions from schema
    questions: List[Question] = []
    for q in cfg.questions:
        qid = q.get("id")
        anchors = q.get("anchors", [])
        # choose first available anchor block
        block_lines: List[Line] = []
        for a in anchors:
            if a in blocks:
                block_lines = blocks[a]; break
        qtext, qbbox = lines_to_text_and_bbox(block_lines)
        # options
        schema_opts = q.get("options", []) or []
        options = parse_options(qtext, cfg.heuristics.option_markers, schema_opts) if schema_opts else []
        questions.append(Question(
            id=qid, text=qtext, bbox=qbbox, options=options, per_side=q.get("per_side")
        ))

    # Explanation (overall): prefer q5'
    exp_tf = explain_blocks.get("q5_prime") or explain_blocks.get("q4_prime") or explain_blocks.get("q3_prime") or explain_blocks.get("q2_prime")
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
        }
    )
    return payload
