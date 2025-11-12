from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session
from app.models.orm import Project

# EasyOCR path (already in your raster.py)
from app.services.ingest_pdf.cv_radios import radio_selected_near_label
from app.services.ingest_pdf.raster import ocr_pdf, lines_from_words
from app.services.ingest_pdf.vector import Word, Line  # types only

logger = logging.getLogger(__name__)

# ------------------------- Small data types -------------------------

@dataclass
class AnchorHit:
    key: str                  # "q1", "q2", etc.
    label: str                # matched label string from OCR (or 'options_cluster')
    page: int
    bbox: Tuple[float, float, float, float]  # (x0,y0,x1,y1)
    score: float              # fuzzy score 0..100 (or synthetic 100 for option clusters)
    line_idx: int             # index in lines list for the matched line


# ------------------------- String utils -------------------------

_PUNCT_RX = re.compile(r"[^\w\s]+")

def _norm(s: str) -> str:
    """Light normalization for OCR noise: lowercase, NFKC, collapse space, strip punctuation/hyphen variants."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    s = s.lower()
    s = _PUNCT_RX.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _ratio(a: str, b: str) -> int:
    """Very fast token-set overlap ratio (0..100)."""
    sa = set(_norm(a).split())
    sb = set(_norm(b).split())
    if not sa or not sb:
        return 0
    inter = len(sa & sb)
    denom = max(len(sa), len(sb))
    return int(100 * inter / denom)


# ------------------------- Line, bbox helpers -------------------------

def _unify_line_text(line: Line) -> str:
    return " ".join(w.text for w in line.words if w.text).strip()

def _bbox_union_words(words: List[Word]) -> Tuple[float, float, float, float]:
    x0 = min(w.bbox[0] for w in words)
    y0 = min(w.bbox[1] for w in words)
    x1 = max(w.bbox[2] for w in words)
    y1 = max(w.bbox[3] for w in words)
    return (x0, y0, x1, y1)


# ------------------------- Anchor matching -------------------------

def _find_anchor_hits(lines: List[Line], anchors_cfg: Dict[str, List[str]], fuzz_threshold: int = 60) -> List[AnchorHit]:
    hits: List[AnchorHit] = []
    for idx, line in enumerate(lines):
        text = _unify_line_text(line)
        ntext = _norm(text)
        if not ntext:
            continue
        for akey, variants in anchors_cfg.items():
            best = 0
            best_label = ""
            for v in variants:
                sc = _ratio(text, v)
                if sc > best:
                    best = sc
                    best_label = v
            if best >= fuzz_threshold:
                try:
                    bbox = _bbox_union_words(line.words)
                except ValueError:
                    continue
                hits.append(
                    AnchorHit(
                        key=akey,
                        label=best_label,
                        page=line.words[0].page,
                        bbox=bbox,
                        score=best,
                        line_idx=idx,
                    )
                )

    # de-duplicate by (key,page): highest score wins; tie-break by top-most (smallest y)
    pruned: Dict[Tuple[str, int], AnchorHit] = {}
    for h in hits:
        k = (h.key, h.page)
        keep = pruned.get(k)
        if keep is None:
            pruned[k] = h
        else:
            if (h.score, -h.bbox[1]) > (keep.score, -keep.bbox[1]):
                pruned[k] = h

    out = list(pruned.values())
    out.sort(key=lambda a: (a.page, a.bbox[1]))
    logger.info("Anchor (text) hits: %s", [(h.key, h.page, round(h.score)) for h in out])
    return out


# ------------------------- Fallback: option-cluster anchors -------------------------

def _fallback_anchor_from_options(
    lines: List[Line],
    schema_opts: List[Dict[str, Any]],
    min_hits: int = 2,
    corridor_px: int = 180,
) -> Optional[Tuple[int, AnchorHit]]:
    """
    If the page doesn't print "Q1/Q2" labels, infer the anchor by finding a cluster of option
    texts from the schema within a short vertical corridor.

    Returns (line_index, AnchorHit) or None.
    """
    if not schema_opts:
        return None

    # pre-normalize schema "raw"
    raw_norms = [(_norm(o.get("raw", "")), o.get("raw", "")) for o in schema_opts if o.get("raw")]
    if not raw_norms:
        return None

    # map each line index -> (hit_count, y_top, y_bot, matched_raws)
    line_hits: List[Tuple[int, int, float, float, List[str]]] = []

    for idx, ln in enumerate(lines):
        text = _unify_line_text(ln)
        if not text:
            continue
        ntext = _norm(text)
        y_top = min(w.bbox[1] for w in ln.words)
        y_bot = max(w.bbox[3] for w in ln.words)

        matched_here = [raw for nraw, raw in raw_norms if nraw and (nraw in ntext or _ratio(ntext, nraw) >= 70)]
        if matched_here:
            line_hits.append((idx, len(matched_here), y_top, y_bot, matched_here))

    if not line_hits:
        return None

    # expand each hit into a vertical corridor and count distinct option raws inside it
    best: Tuple[int, int, float, float, List[str]] | None = None
    for idx, _, y_top, y_bot, matched in line_hits:
        corridor_top = y_top
        corridor_bot = y_top + corridor_px
        seen: set[str] = set(matched)
        # look forward a bit for more options on the same page
        page = lines[idx].words[0].page
        for j in range(idx + 1, len(lines)):
            l2 = lines[j]
            if l2.words[0].page != page:
                break
            y2 = min(w.bbox[1] for w in l2.words)
            if y2 > corridor_bot:
                break
            t2 = _norm(_unify_line_text(l2))
            for nraw, raw in raw_norms:
                if nraw and (nraw in t2 or _ratio(t2, nraw) >= 70):
                    seen.add(raw)
        hit_count = len(seen)
        if hit_count >= min_hits:
            candidate = (idx, hit_count, corridor_top, corridor_bot, list(seen))
            if best is None or hit_count > best[1] or (hit_count == best[1] and corridor_top < best[2]):
                best = candidate

    if not best:
        return None

    idx, hit_count, y_top, y_bot, seen_list = best
    try:
        bbox = _bbox_union_words(lines[idx].words)
    except ValueError:
        bbox = lines[idx].words[0].bbox

    # Build synthetic AnchorHit
    fake = AnchorHit(
        key="options_cluster",
        label="options_cluster",
        page=lines[idx].words[0].page,
        bbox=bbox,
        score=100.0,
        line_idx=idx,
    )
    logger.info(
        "Fallback option-cluster anchor at line %d (page %d), hits=%d, options=%s",
        idx,
        fake.page,
        hit_count,
        seen_list,
    )
    return idx, fake


# ------------------------- Block capture -------------------------

def _capture_block(lines: List[Line], start_idx: int, window_px: int) -> Tuple[str, Tuple[float, float, float, float], List[Line]]:
    """Collect lines under start line until window_px is exceeded or page changes."""
    start = lines[start_idx]
    page = start.words[0].page
    start_y = start.words[0].bbox[3]  # bottom of the anchor line
    buf: List[str] = []
    blk_lines: List[Line] = []
    y_max = start_y + window_px

    for ln in lines[start_idx + 1 :]:
        if ln.words[0].page != page:
            break
        y_top = min(w.bbox[1] for w in ln.words)
        if y_top > y_max:
            break
        text = _unify_line_text(ln)
        if not text.strip():
            continue
        blk_lines.append(ln)
        buf.append(text)

    text_block = "\n".join(buf).strip()
    bbox = _bbox_union_words([w for ln in blk_lines for w in ln.words]) if blk_lines else start.words[0].bbox
    return text_block, bbox, blk_lines


# ------------------------- Options parsing -------------------------
def _detect_filled_radio(
    page_img: "np.ndarray",
    line_bbox: Tuple[float, float, float, float],
    search_left_px: int = 140,
    corridor_px: int = 32,
) -> Tuple[bool, float]:
    """
    Heuristic radio detector:
    - Crops a small region immediately to the LEFT of the option line bbox.
    - Thresholds and finds round-ish dark blobs.
    - If a circular blob with sufficient fill is found, returns (True, confidence).

    Returns: (selected, confidence: 0.0..1.0)
    """
    import cv2
    import numpy as np

    x0, y0, x1, y1 = [int(v) for v in line_bbox]
    h, w = page_img.shape[:2]

    # Search region: a vertical corridor around the line, just to the LEFT of the text
    y_mid = (y0 + y1) // 2
    y_top = max(0, y_mid - corridor_px // 2)
    y_bot = min(h, y_mid + corridor_px // 2)
    x_right = max(0, x0 - 2)           # just left edge of text
    x_left = max(0, x_right - search_left_px)

    if x_left >= x_right or y_top >= y_bot:
        return (False, 0.0)

    roi = page_img[y_top:y_bot, x_left:x_right, :]
    if roi.size == 0:
        return (False, 0.0)

    # Preprocess: grayscale, strong contrast, adaptive threshold
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # normalize a bit
    gray = cv2.equalizeHist(gray)
    # invert once (radio dots are often darker than bg; adaptive on inverted helps)
    inv = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(inv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 15, 3)

    # Morph close to fill the dot
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Contours
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_score = 0.0
    found = False

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 12 or area > 300:  # ignore tiny noise and large shapes
            continue
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        circularity = 4.0 * np.pi * (area / (peri * peri) + 1e-6)  # 1.0 is a perfect circle
        if circularity < 0.55:
            continue

        # Filled-ness: compute mean inside the contour polygon on the inverted image.
        mask = np.zeros_like(bw)
        cv2.drawContours(mask, [c], -1, 255, -1)
        mean_inside = float(inv[mask == 255].mean()) if np.any(mask) else 0.0

        # Heuristic score combines circularity and intensity inside
        # Normalize mean_inside by 255 to 0..1
        score = 0.6 * min(1.0, circularity) + 0.4 * (mean_inside / 255.0)

        if score > best_score:
            best_score = score
            found = True

    if found:
        # Map score (~0.6..1.0) to a conservative confidence 0.6..0.95
        conf = max(0.6, min(0.95, best_score))
        return (True, float(conf))
    return (False, 0.0)


def _parse_options_from_block(
    block_text: str,
    opt_markers: List[str],
    schema_opts: List[Dict[str, Any]],
    *,
    blk_lines: Optional[List[Line]] = None,
    page_img: Optional["np.ndarray"] = None,
    page_no: Optional[int] = None,
    # CV radio detector knobs (will default; can override from YAML heuristics)
    corridor_px: int = 28,
    search_left_px: int = 140,
    radio_search_up_px: Optional[int] = None,
    radio_search_pad_px: Optional[int] = None,
    allow_multi: bool = False,
) -> List[Dict[str, Any]]:
    """
    Map schema options to block lines (fuzzy) and, if possible, detect a selected radio
    using cv_radios.radio_selected_near_label() near the matched line.
    """
    options: List[Dict[str, Any]] = []
    if not schema_opts:
        return options

    # Prepare lines for fuzzy matching
    raw_lines = [l.strip() for l in block_text.splitlines() if l.strip()]
    norm_lines = [_norm(l) for l in raw_lines]

    for sch in schema_opts:
        raw = sch["raw"]
        nraw = _norm(raw)

        # 1) find best matching line
        best_i = -1
        best_sc = 0
        for i, nl in enumerate(norm_lines):
            sc = _ratio(nl, nraw)
            if sc > best_sc:
                best_sc, best_i = sc, i

        selected = False
        conf = 0.0

        # 2) textual markers (rare in your UI, but keep it)
        if best_i >= 0:
            ln = raw_lines[best_i]
            ln_low = ln.lower()
            if any(m in ln_low for m in ("[x]", "(x)", "✓", "●", "■")):
                selected = True
                conf = 0.9
            else:
                m = re.search(r"selected\s*:\s*(.+)$", ln, flags=re.I)
                if m:
                    chosen = _norm(m.group(1))
                    if nraw in chosen or chosen in nraw:
                        selected = True
                        conf = max(conf, 0.9)

        # 3) visual radio detection with your cv_radios
        #    We look around the OCR line bbox that matched the option text.
        if (not selected) and (best_i >= 0) and blk_lines is not None and page_img is not None:
            try:
                # Union bbox of the matched OCR line (pixel coords on page_img)
                x0 = min(w.bbox[0] for w in blk_lines[best_i].words)
                y0 = min(w.bbox[1] for w in blk_lines[best_i].words)
                x1 = max(w.bbox[2] for w in blk_lines[best_i].words)
                y1 = max(w.bbox[3] for w in blk_lines[best_i].words)
                label_bbox = (x0, y0, x1, y1)

                # Your API: radio_selected_near_label(img, label_bbox, search_up_px=..., search_pad_px=...)
                sel, sconf = radio_selected_near_label(
                    page_img,
                    label_bbox,
                    # prefer YAML-tuned values if provided; else good defaults
                    search_up_px=radio_search_up_px if radio_search_up_px is not None else 72,
                    search_pad_px=radio_search_pad_px if radio_search_pad_px is not None else 60,
                )
                if sel:
                    selected = True
                    conf = max(conf, float(sconf))
            except Exception as e:
                logger.debug("cv_radios failed on option '%s': %r", raw, e)

        options.append(
            {
                "text": raw,
                "bbox": None,  # you can populate later from the matched line bbox if needed
                "selected": bool(selected),
                "selection_confidence": float(conf),
            }
        )

        _enforce_single_select(options, allow_multi=allow_multi)

    return options



# ------------------------- Public API -------------------------

def preflight_pdf(pdf_bytes: bytes, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """OCR preflight with DPI from YAML to surface quick stats."""
    dpi = cfg.get("extractors", {}).get("heuristics", {}).get("raster_dpi", 300)
    ocr = ocr_pdf(pdf_bytes, dpi=dpi)
    return {
        "pages": len(ocr.page_images),
        "mean_ocr_conf": ocr.mean_conf,
        "words": len(ocr.words),
    }

def _enforce_single_select(options: List[Dict[str, Any]], allow_multi: bool = False) -> None:
    """If multiple options are 'selected', keep only the highest confidence (radio group)."""
    if allow_multi:
        return
    if not options:
        return
    best_i, best_c = -1, -1.0
    for i, opt in enumerate(options):
        if opt.get("selected"):
            c = float(opt.get("selection_confidence") or 0.0)
            if c > best_c:
                best_c = c; best_i = i
    if best_i >= 0:
        for i, opt in enumerate(options):
            if i != best_i:
                opt["selected"] = False
                opt["selection_confidence"] = 0.0



def extract_minimal(pdf_bytes: bytes, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts:
      - header prompt (if configured)
      - questions[] with text block + options (selection via text markers only)
    and sets status="extracted".
    """
    heur = cfg.get("extractors", {}).get("heuristics", {})
    window = int(heur.get("block_scan_window_px", 1000))
    opt_markers = heur.get("option_markers", ["•", "-", "[ ]", "[x]", "( )", "(x)"])

    # 1) OCR
    dpi = heur.get("raster_dpi", 300)
    ocr = ocr_pdf(pdf_bytes, dpi=dpi)
    logger.info("OCR words=%d, mean_conf=%.3f", len(ocr.words), ocr.mean_conf)
    if not ocr.words:
        return {
            "status": "extracted",
            "prompt": "",
            "questions": [],
            "rating": None,
            "explanation": "",
            "debug": {"words": 0, "mean_conf": 0.0},
        }

    # 2) Lines
    lines = lines_from_words(ocr.words, y_tol=5.0)

    # 3) Primary anchors (text)
    anchors_cfg: Dict[str, List[str]] = cfg.get("extractors", {}).get("anchors", {})
    text_hits = _find_anchor_hits(lines, anchors_cfg, fuzz_threshold=60)
    key_to_hit = {h.key: h for h in text_hits}

    # 4) Header prompt (optional)
    prompt = ""
    if "header_prompt" in anchors_cfg and "header_prompt" in key_to_hit:
        hp = key_to_hit["header_prompt"]
        block_text, _, _ = _capture_block(lines, hp.line_idx, window_px=min(window, 1600))
        prompt = "\n".join(block_text.splitlines()[:10]).strip()

    # 5) Build questions
    out_questions = []
    for q in cfg.get("questions", []):
        qid = q["id"]
        qkeys = q.get("anchors") or []
        if not qkeys:
            m = re.match(r"^(q\d+)", qid)
            if m:
                qkeys = [m.group(1)]

        chosen_hit: Optional[AnchorHit] = None
        for k in qkeys:
            if k in key_to_hit:
                chosen_hit = key_to_hit[k]
                break

        q_text = ""
        q_bbox = None
        options = []

        if chosen_hit is None:
            # Fallback: infer anchor using options cluster on the same page as where options likely appear
            logger.debug("No text anchor for %s; attempting option-cluster fallback", qid)
            opt_hit = _fallback_anchor_from_options(lines, q.get("options", []), min_hits=2, corridor_px=180)
            if opt_hit is not None:
                chosen_hit = opt_hit[1]

        if chosen_hit is not None:
            block_text, block_bbox, blk_lines = _capture_block(lines, chosen_hit.line_idx, window_px=window)
            q_text = block_text
            q_bbox = (block_bbox[0], block_bbox[1], block_bbox[2], block_bbox[3], chosen_hit.page)

            page_img = None
            try:
                page_img = ocr.page_images[int(chosen_hit.page) - 1]
            except Exception:
                page_img = None
            
            allow_multi = bool(q.get("allow_multi", False))  # default single-choice

            options = _parse_options_from_block(
                block_text,
                opt_markers,
                q.get("options", []),
                blk_lines=blk_lines,
                page_img=page_img,
                page_no=int(chosen_hit.page),
                corridor_px=int(heur.get("option_mark_corridor_px", 28)),
                # forward YAML knobs to cv_radios (if present)
                radio_search_up_px=heur.get("radio_search_up_px"),
                radio_search_pad_px=heur.get("radio_search_pad_px"),
                allow_multi=allow_multi,
            )

        else:
            # still surface options with no selection, empty text
            options = _parse_options_from_block("", opt_markers, q.get("options", []))
            logger.debug("Question %s has neither text anchor nor option cluster; emitting empty text with options", qid)

        out_questions.append(
            {
                "id": qid,
                "text": q_text,
                "bbox": q_bbox,
                "options": options,
                "per_side": bool(q.get("per_side", False)),
            }
        )

    payload = {
        "status": "extracted",
        "prompt": {"text": prompt, "bbox": None},
        "questions": out_questions,
        "rating": None,
        "explanation": {"text": "", "bbox": None},
        "debug": {
            "ocr_words": len(ocr.words),
            "mean_conf": ocr.mean_conf,
            "anchors_found": [(h.key, h.page, round(h.score)) for h in text_hits],
        },
    }
    logger.info(
        "Extraction produced %d questions; anchors(text)=%s",
        len(out_questions),
        payload["debug"]["anchors_found"],
    )
    return payload


# ------------------------- Orchestration for API -------------------------

@dataclass
class PDFPreflight:
    pages: int
    vector_text: bool


def preflight_pages(pdf_bytes: bytes) -> PDFPreflight:
    """Lightweight page count & vector-text presence via PyMuPDF (no OCR)."""
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages = doc.page_count
        vector_text = any((doc.load_page(i).get_text("text") or "").strip() for i in range(pages))
        return PDFPreflight(pages=pages, vector_text=vector_text)
    finally:
        doc.close()


def extract_pdf(db: Session, project_id: str, pdf_bytes: bytes) -> Tuple[str, float, dict]:
    """
    Entry-point for /bundles/{id}/extract:
      - Load project YAML from DB
      - EasyOCR via ocr_pdf(...)
      - Anchor capture + option parsing
    """
    logger.info("extract_pdf: project_id=%s", project_id)

    proj = db.query(Project).filter(Project.project_id == project_id).first()
    if not proj:
        raise ValueError(f"Project '{project_id}' not found")

    import yaml
    cfg = yaml.safe_load(proj.config_yaml)
    logger.info("Loaded project config (questions=%d)", len(cfg.get("questions", [])))

    try:
        payload = extract_minimal(pdf_bytes, cfg)
        mean_conf = float(payload.get("debug", {}).get("mean_conf", 0.0))
        status = payload.get("status", "extracted")
        logger.info("Extraction done: status=%s, mean_conf=%.3f", status, mean_conf)
        return status, mean_conf, payload
    except Exception as e:
        logger.exception("Extraction failed")
        return "failed", 0.0, {
            "status": "failed",
            "error": str(e),
            "prompt": "",
            "questions": [],
            "rating": None,
            "explanation": "",
            "debug": {"exception": str(e)},
        }
