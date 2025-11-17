from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import time

import numpy as np

from sqlalchemy.orm import Session
from app.models.orm import Project

# Your helpers
from app.services.ingest_pdf.cv_radios import radio_selected_near_label
from app.services.ingest_pdf.raster import ocr_pdf as easyocr_ocr_pdf, lines_from_words
from app.services.ingest_pdf.vector import Word, Line  # typing only

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Debug paths
# ------------------------------------------------------------------------------
DEBUG_DIR = Path("./ocr_debug")
PAGES_DIR = DEBUG_DIR / "pages"
DEBUG_DIR.mkdir(exist_ok=True, parents=True)
PAGES_DIR.mkdir(exist_ok=True, parents=True)

# ------------------------------------------------------------------------------
# Small data types
# ------------------------------------------------------------------------------
@dataclass
class AnchorHit:
    key: str                  # "q1", "q2", etc.
    label: str                # matched label string from OCR
    page: int
    bbox: Tuple[float, float, float, float]  # (x0,y0,x1,y1)
    score: float              # fuzzy score 0..100
    line_idx: int             # index in 'lines' for the matched line

@dataclass
class PDFPreflight:
    pages: int
    vector_text: bool

# ------------------------------------------------------------------------------
# String utils
# ------------------------------------------------------------------------------
_PUNCT_RX = re.compile(r"[^\w\s]+")

def _norm(s: str) -> str:
    """Light normalization for OCR noise."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    s = s.lower()
    s = _PUNCT_RX.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _ratio(a: str, b: str) -> int:
    """Fast token-set overlap ratio (0..100)."""
    sa = set(_norm(a).split())
    sb = set(_norm(b).split())
    if not sa or not sb:
        return 0
    inter = len(sa & sb)
    denom = max(len(sa), len(sb))
    return int(100 * inter / denom)

# ------------------------------------------------------------------------------
# Line / geometry helpers
# ------------------------------------------------------------------------------
def _unify_line_text(line: Line) -> str:
    return " ".join(w.text for w in line.words if w.text).strip()

def _bbox_union_words(words: List[Word]) -> Tuple[float, float, float, float]:
    x0 = min(w.bbox[0] for w in words)
    y0 = min(w.bbox[1] for w in words)
    x1 = max(w.bbox[2] for w in words)
    y1 = max(w.bbox[3] for w in words)
    return (x0, y0, x1, y1)

# ------------------------------------------------------------------------------
# Debug drawings
# ------------------------------------------------------------------------------
def _draw_boxes(img: np.ndarray,
                boxes: List[Tuple[int, int, int, int]],
                color: Tuple[int, int, int],
                thick: int = 2,
                labels: Optional[List[str]] = None) -> np.ndarray:
    import cv2
    out = img.copy()
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thick)
        if labels and i < len(labels) and labels[i]:
            cv2.putText(out, labels[i], (x0, max(0, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out

def _save_png(img: np.ndarray, path: Path) -> None:
    try:
        import cv2
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)
        logger.info("Saved debug image: %s", path)
    except Exception as e:
        logger.warning("Failed to save image %s: %r", path, e)

# ------------------------------------------------------------------------------
# OCR via Tesseract (Poppler + pdf2image)
# ------------------------------------------------------------------------------
def _run_tesseract_on_pdf(pdf_input: bytes | str, dpi: int = 300) -> Dict[str, Any]:
    """
    Robust Tesseract OCR:
      - Supports file path (str) or in-memory bytes
      - Converts pages via Poppler (pdf2image)
      - Parses text AND word boxes via pytesseract
      - Saves page PNGs and a plain-text dump into ./ocr_debug
    """
    from pdf2image import convert_from_path, convert_from_bytes
    from PIL import Image
    import pytesseract
    import cv2

    # 1) rasterize
    if isinstance(pdf_input, (str, Path)):
        pages = convert_from_path(str(pdf_input), dpi=dpi)
    else:
        pages = convert_from_bytes(pdf_input, dpi=dpi)

    words: List[Word] = []
    confs: List[float] = []
    page_images_bgr: List[np.ndarray] = []
    text_dump: List[str] = []

    for pno, pil_img in enumerate(pages, start=1):
        # Save the raw page image (RGB->BGR for OpenCV), and also persist a PNG
        arr_rgb = np.array(pil_img)
        arr_bgr = arr_rgb[:, :, ::-1].copy()
        page_images_bgr.append(arr_bgr)
        _save_png(arr_bgr, PAGES_DIR / f"page_{pno:03d}.png")

        # Readable text for quick inspection
        page_text = pytesseract.image_to_string(pil_img, lang="eng", config="--psm 6")
        text_dump.append(f"\n--- Page {pno} ---\n{page_text.strip()}\n")

        # Structured OCR (words + boxes)
        gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        data = pytesseract.image_to_data(
            gray, lang="eng", config="--oem 1 --psm 6",
            output_type=pytesseract.Output.DICT
        )

        n = len(data["text"])
        for i in range(n):
            txt = str(data["text"][i] or "").strip()
            if not txt:
                continue
            try:
                conf = float(data["conf"][i])
            except Exception:
                conf = -1.0
            if conf < 0:
                continue
            x, y = int(data["left"][i]), int(data["top"][i])
            w, h = int(data["width"][i]), int(data["height"][i])
            words.append(Word(page=pno, text=txt, bbox=(float(x), float(y), float(x + w), float(y + h))))
            confs.append(conf / 100.0)

    # Write plain text dump
    dump_path = DEBUG_DIR / "ocr_text_dump.txt"
    try:
        with open(dump_path, "w", encoding="utf-8") as f:
            f.write("".join(text_dump))
        logger.info("Saved OCR text dump: %s", dump_path)
    except Exception as e:
        logger.warning("Failed writing OCR dump: %r", e)

    mean_conf = float(sum(confs) / len(confs)) if confs else 0.0
    return {"words": words, "mean_conf": mean_conf, "page_images": page_images_bgr}

# ------------------------------------------------------------------------------
# Anchor matching
# ------------------------------------------------------------------------------
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

    # keep best per (key,page)
    pruned: Dict[Tuple[str, int], AnchorHit] = {}
    for h in hits:
        k = (h.key, h.page)
        cur = pruned.get(k)
        if (cur is None) or ((h.score, -h.bbox[1]) > (cur.score, -cur.bbox[1])):
            pruned[k] = h
    out = list(pruned.values())
    out.sort(key=lambda a: (a.page, a.bbox[1]))
    logger.info("Anchor hits: %s", [(h.key, h.page, round(h.score)) for h in out])
    return out

# ------------------------------------------------------------------------------
# Block capture under an anchor
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# Options parsing and radio detection
# ------------------------------------------------------------------------------
def _enforce_single_select(options: List[Dict[str, Any]], allow_multi: bool = False) -> None:
    if allow_multi:
        return
    if not options:
        return
    best_i, best_c = -1, -1.0
    for i, opt in enumerate(options):
        if opt.get("selected"):
            c = float(opt.get("selection_confidence") or 0.0)
            if c > best_c:
                best_c = c
                best_i = i
    if best_i >= 0:
        for i, opt in enumerate(options):
            if i != best_i:
                opt["selected"] = False
                opt["selection_confidence"] = 0.0

def _parse_options_from_block(
    block_text: str,
    schema_opts: List[Dict[str, Any]],
    *,
    blk_lines: Optional[List[Line]] = None,
    page_img: Optional["np.ndarray"] = None,
    radio_search_up_px: Optional[int] = None,
    radio_search_pad_px: Optional[int] = None,
    allow_multi: bool = False,
) -> List[Dict[str, Any]]:
    """
    Map schema options to block lines (fuzzy) and detect selected radio near the matched line.
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
        out_bbox = None

        # 2) if we have a matched OCR line, try CV radio detection
        if (best_i >= 0) and blk_lines is not None and page_img is not None:
            try:
                x0 = int(min(w.bbox[0] for w in blk_lines[best_i].words))
                y0 = int(min(w.bbox[1] for w in blk_lines[best_i].words))
                x1 = int(max(w.bbox[2] for w in blk_lines[best_i].words))
                y1 = int(max(w.bbox[3] for w in blk_lines[best_i].words))
                out_bbox = (x0, y0, x1, y1)
                sel, sconf = radio_selected_near_label(
                    page_img,
                    (x0, y0, x1, y1),
                    search_up_px=radio_search_up_px if radio_search_up_px is not None else 72,
                    search_pad_px=radio_search_pad_px if radio_search_pad_px is not None else 60,
                )
                if sel:
                    selected = True
                    conf = float(sconf)
            except Exception as e:
                logger.debug("radio_selected_near_label failed for '%s': %r", raw, e)

        options.append(
            {
                "text": raw,
                "bbox": out_bbox,  # keep pixel bbox if we had a matched line; else None
                "selected": bool(selected),
                "selection_confidence": float(conf),
            }
        )

    # enforce single-select (radio group)
    _enforce_single_select(options, allow_multi=allow_multi)
    return options

# ------------------------------------------------------------------------------
# Main extractor
# ------------------------------------------------------------------------------
def preflight_pdf(pdf_bytes: bytes, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """OCR preflight with DPI from YAML to surface quick stats (Tesseract by default)."""
    heur = cfg.get("extractors", {}).get("heuristics", {})
    dpi = int(heur.get("raster_dpi", 300))
    engine = str(heur.get("ocr_engine", "tesseract")).lower()

    if engine == "easyocr":
        ocr = easyocr_ocr_pdf(pdf_bytes, dpi=dpi)
        return {"pages": len(ocr.page_images), "mean_ocr_conf": ocr.mean_conf, "words": len(ocr.words)}

    # default: tesseract
    o = _run_tesseract_on_pdf(pdf_bytes, dpi=dpi)
    return {"pages": len(o["page_images"]), "mean_ocr_conf": o["mean_conf"], "words": len(o["words"])}

def extract_minimal(pdf_bytes: bytes, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts:
      - header prompt (optional)
      - questions[] with block text + options (radio cv near label)
    Saves debug images and text dumps in ./ocr_debug.
    """
    ts = int(time.time())
    heur = cfg.get("extractors", {}).get("heuristics", {})
    window = int(heur.get("block_scan_window_px", 900))
    engine = str(heur.get("ocr_engine", "tesseract")).lower()
    radio_up = int(heur.get("radio_search_up_px", 72))
    radio_pad = int(heur.get("radio_search_pad_px", 60))
    dpi = int(heur.get("raster_dpi", 300))

    # 1) OCR
    if engine == "easyocr":
        ocr = easyocr_ocr_pdf(pdf_bytes, dpi=dpi)
        words = ocr.words
        mean_conf = float(ocr.mean_conf)
        page_images = ocr.page_images  # BGR np arrays
    else:
        o = _run_tesseract_on_pdf(pdf_bytes, dpi=dpi)
        words = o["words"]
        mean_conf = float(o["mean_conf"])
        page_images = o["page_images"]

    logger.info("OCR words=%d, mean_conf=%.3f", len(words), mean_conf)
    if not words:
        return {
            "status": "extracted",
            "prompt": {"text": "", "bbox": None},
            "questions": [],
            "rating": None,
            "explanation": {"text": "", "bbox": None},
            "debug": {"ocr_words": 0, "mean_conf": 0.0, "debug_dir": str(DEBUG_DIR)},
        }

    # 2) Lines
    lines = lines_from_words(words, y_tol=5.0)

    # 3) Anchors
    anchors_cfg: Dict[str, List[str]] = cfg.get("extractors", {}).get("anchors", {}) or {}
    text_hits = _find_anchor_hits(lines, anchors_cfg, fuzz_threshold=60)
    key_to_hit = {h.key: h for h in text_hits}

    # 4) Optional header prompt
    prompt_text = ""
    if "header_prompt" in anchors_cfg and "header_prompt" in key_to_hit:
        hp = key_to_hit["header_prompt"]
        block_text, _, _ = _capture_block(lines, hp.line_idx, window_px=min(window, 1600))
        prompt_text = "\n".join(block_text.splitlines()[:10]).strip()

    # Debug image: draw found anchor boxes on their pages
    try:
        for hit in text_hits:
            pidx = hit.page - 1
            if 0 <= pidx < len(page_images):
                (x0, y0, x1, y1) = tuple(int(v) for v in hit.bbox)
                img = _draw_boxes(page_images[pidx], [(x0, y0, x1, y1)], (0, 255, 255), 2, [f"anchor:{hit.key}"])
                _save_png(img, DEBUG_DIR / f"page_{hit.page:03d}_anchors.png")
    except Exception as e:
        logger.debug("Anchor debug draw failed: %r", e)

    # 5) Build questions
    out_questions: List[Dict[str, Any]] = []
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
        q_bbox_5: Optional[Tuple[int, float, float, float, float]] = None
        options: List[Dict[str, Any]] = []

        if chosen_hit is not None:
            # capture a block below the anchor
            block_text, block_bbox, blk_lines = _capture_block(lines, chosen_hit.line_idx, window_px=window)
            q_text = block_text
            # 5-tuple for Pydantic TextField-style bbox: (page, x0, y0, width, height)
            x0, y0, x1, y1 = block_bbox
            q_bbox_5 = (chosen_hit.page, float(x0), float(y0), float(x1 - x0), float(y1 - y0))

            # parse options with cv radios
            page_img = None
            try:
                page_img = page_images[chosen_hit.page - 1]  # BGR
            except Exception:
                page_img = None

            options = _parse_options_from_block(
                block_text,
                q.get("options", []),
                blk_lines=blk_lines,
                page_img=page_img,
                radio_search_up_px=radio_up,
                radio_search_pad_px=radio_pad,
                allow_multi=bool(q.get("allow_multi", False)),
            )

            # Debug draw: question block and option label bboxes
            try:
                if page_img is not None:
                    boxes = []
                    labels = []
                    if blk_lines:
                        bx0 = int(min(w.bbox[0] for ln in blk_lines for w in ln.words))
                        by0 = int(min(w.bbox[1] for ln in blk_lines for w in ln.words))
                        bx1 = int(max(w.bbox[2] for ln in blk_lines for w in ln.words))
                        by1 = int(max(w.bbox[3] for ln in blk_lines for w in ln.words))
                        boxes.append((bx0, by0, bx1, by1))
                        labels.append(f"{qid}:block")
                    # option label bboxes (if present)
                    for opt in options:
                        ob = opt.get("bbox")
                        if ob:
                            boxes.append((int(ob[0]), int(ob[1]), int(ob[2]), int(ob[3])))
                            lab = f"{'SEL' if opt.get('selected') else 'opt'}:{opt.get('text','')[:16]}"
                            labels.append(lab)
                    dbg = _draw_boxes(page_img, boxes, (0, 200, 0), 2, labels)
                    _save_png(dbg, DEBUG_DIR / f"page_{chosen_hit.page:03d}_{qid}_block_opts.png")
            except Exception as e:
                logger.debug("Question block debug draw failed for %s: %r", qid, e)

        else:
            # no anchor: return options with default unselected
            options = _parse_options_from_block("", q.get("options", []))

        out_questions.append({
            "id": qid,
            "text": q_text,
            "bbox": q_bbox_5,                 # 5-tuple or None
            "options": options,
            "per_side": bool(q.get("per_side", False)),
        })

    payload = {
        "status": "extracted",
        "prompt": {"text": prompt_text, "bbox": None},      # TextField-like dict
        "questions": out_questions,
        "rating": None,
        "explanation": {"text": "", "bbox": None},          # TextField-like dict
        "debug": {
            "ocr_words": len(words),
            "mean_conf": mean_conf,
            "anchors_found": [(h.key, h.page, round(h.score)) for h in text_hits],
            "debug_dir": str(DEBUG_DIR.resolve()),
        },
    }
    logger.info("Extraction produced %d questions; anchors(text)=%s",
                len(out_questions), payload["debug"]["anchors_found"])
    return payload

# ------------------------------------------------------------------------------
# API orchestration
# ------------------------------------------------------------------------------
def preflight_pages(pdf_bytes: bytes) -> PDFPreflight:
    """Lightweight page count & vector-text presence via PyMuPDF (no OCR)."""
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages = doc.page_count
        vector_text = any((doc.load_page(i).get_text("text") or "").strip() for i in range(pages))
        return PDFPreflight(pages=pages, vector_text=bool(vector_text))
    finally:
        doc.close()

def extract_pdf(db: Session, project_id: str, pdf_bytes: bytes) -> Tuple[str, float, dict]:
    """
    Entry-point for /bundles/{id}/extract:
      - Load project YAML from DB
      - OCR via Tesseract (default) or EasyOCR (cfg.heuristics.ocr_engine)
      - Anchor capture + block + CV radios
      - Debug artifacts to ./ocr_debug
    """
    logger.info("extract_pdf: project_id=%s", project_id)

    proj = db.query(Project).filter(Project.project_id == project_id).first()
    if not proj:
        raise ValueError(f"Project '{project_id}' not found")

    import yaml
    cfg = yaml.safe_load(proj.config_yaml) or {}
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
            "prompt": {"text": "", "bbox": None},
            "questions": [],
            "rating": None,
            "explanation": {"text": "", "bbox": None},
            "debug": {"exception": str(e), "debug_dir": str(DEBUG_DIR.resolve())},
        }