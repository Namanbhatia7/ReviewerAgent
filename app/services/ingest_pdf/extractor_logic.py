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

def parse_options(block_text: str, opt_markers: List[str], schema_opts: List[Dict[str,Any]]) -> List[Option]:
    # Prefer matching normalized "raw" option texts in the block lines
    options: List[Option] = []
    lines = [l.strip() for l in block_text.splitlines() if l.strip()]
    raw_map = [ (o["raw"], o.get("norm")) for o in schema_opts ]
    for raw, norm in raw_map:
        selected = False
        conf = 0.0
        # detect “[x] raw”, “(x) raw”, “raw [x]” etc.
        for ln in lines:
            ln_low = _norm(ln)
            raw_low = _norm(raw)
            if raw_low in ln_low:
                # look for a marker indicating selection
                if any(m.lower() in ln_low for m in ["[x]", "(x)", "selected", "✓"]):
                    selected = True; conf = 0.9
                break
        options.append(Option(text=raw, selected=selected, selection_confidence=conf))
    # if none selected, try heuristic: a single line like "Selected: …"
    if not any(o.selected for o in options):
        for ln in lines:
            m = re.search(r"selected\s*:\s*(.+)$", ln, flags=re.I)
            if m:
                chosen = _norm(m.group(1))
                for o in options:
                    if _norm(o.text) in chosen or chosen in _norm(o.text):
                        o.selected = True; o.selection_confidence = max(o.selection_confidence, 0.9)
                break
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
