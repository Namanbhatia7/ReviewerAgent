from __future__ import annotations
from dataclasses import dataclass
from typing import List
import logging

from app.services.ingest_pdf.vector import Word, Line, _finalize_line

logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    words: List[Word]
    mean_conf: float
    page_images: List["np.ndarray"]  # numpy arrays (BGR) per page


# ------------------------ Rasterization backends ------------------------

def _render_with_pdf2image(pdf_bytes: bytes, dpi: int) -> List["np.ndarray"]:
    import numpy as np
    from pdf2image import convert_from_bytes
    logger.info("Rasterizing with pdf2image at %sdpi", dpi)
    pil_images = convert_from_bytes(pdf_bytes, dpi=dpi)
    arrs = [np.array(img) for img in pil_images]  # RGB
    logger.info("pdf2image produced %d page image(s)", len(arrs))
    return arrs

def _render_with_fitz(pdf_bytes: bytes, dpi: int) -> List["np.ndarray"]:
    import fitz  # PyMuPDF
    import numpy as np
    zoom = float(dpi) / 72.0
    mat = fitz.Matrix(zoom, zoom)
    logger.info("Rasterizing with PyMuPDF at %sdpi (fallback)", dpi)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        out: List["np.ndarray"] = []
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype="uint8").reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:
                img = img[:, :, :3]  # RGB
            out.append(img)
        logger.info("PyMuPDF produced %d page image(s)", len(out))
        return out
    finally:
        doc.close()


# ------------------------ Image preprocessing ------------------------

def _downscale_to_max_side(img: "np.ndarray", max_side: int = 2200) -> "np.ndarray":
    import cv2
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return img
    scale = max_side / float(long_side)
    new_w, new_h = int(w * scale), int(h * scale)
    out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    logger.debug("Downscaled page from (%d,%d) to (%d,%d)", w, h, new_w, new_h)
    return out

def _ensure_bgr_and_boost(img_rgb: "np.ndarray") -> "np.ndarray":
    """
    RGB->BGR, CLAHE on L channel, light sharpen. If the page is overall dark,
    auto-invert so white-on-black UIs become black-on-white for OCR.
    """
    import cv2, numpy as np
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    mean_val = float(bgr.mean())
    # CLAHE
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # Unsharp mask
    blurred = cv2.GaussianBlur(bgr2, (0, 0), 1.0)
    sharp = cv2.addWeighted(bgr2, 1.25, blurred, -0.25, 0)

    # Auto-invert if dark
    if mean_val < 100:
        logger.debug("Detected dark theme (mean=%.1f); inverting for OCR.", mean_val)
        return cv2.bitwise_not(sharp)
    return sharp

def _preprocess_pages(page_arrays_rgb: List["np.ndarray"], max_side: int) -> List["np.ndarray"]:
    out: List["np.ndarray"] = []
    for idx, rgb in enumerate(page_arrays_rgb, start=1):
        try:
            rgb2 = _downscale_to_max_side(rgb, max_side=max_side)
            bgr = _ensure_bgr_and_boost(rgb2)
            out.append(bgr)
        except Exception as e:
            logger.warning("Preprocess failed for page %d: %r; using basic RGB->BGR.", idx, e)
            import cv2
            out.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return out

def _invert_and_threshold(bgr: "np.ndarray") -> "np.ndarray":
    import cv2
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    inv2 = clahe.apply(inv)
    return cv2.cvtColor(inv2, cv2.COLOR_GRAY2BGR)

def _upsample_if_tiny(bgr: "np.ndarray", min_height: int = 1400) -> "np.ndarray":
    import cv2
    h, w = bgr.shape[:2]
    if h >= min_height:
        return bgr
    scale = min_height / float(h)
    return cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

def _generate_variants(bgr: "np.ndarray") -> list[tuple[str, "np.ndarray"]]:
    variants = [("normal", bgr)]
    try:
        inv = _invert_and_threshold(bgr)
        variants.append(("inverted", inv))
    except Exception:
        pass
    try:
        up = _upsample_if_tiny(bgr)
        if up is not bgr:
            variants.append(("upsampled", up))
    except Exception:
        pass
    return variants

def _crop_right_pane(bgr: "np.ndarray", frac_left_keep: float = 0.56) -> "np.ndarray":
    """Keep right pane (where the questions/radios are)."""
    h, w = bgr.shape[:2]
    x0 = int(w * frac_left_keep)
    return bgr[:, x0:, :]

def _debug_dump_pages(pages_bgr: List["np.ndarray"], tag: str = "ocr_debug_page"):
    try:
        import cv2, time
        ts = int(time.time())
        for i, img in enumerate(pages_bgr, start=1):
            cv2.imwrite(f"/tmp/{tag}_{i}_{ts}.png", img)
        logger.info("Dumped %d preprocessed page(s) to /tmp for debugging.", len(pages_bgr))
    except Exception:
        pass


# --------------------------- EasyOCR integration ---------------------------

def _easyocr_try(bgr: "np.ndarray", lang: str = "en") -> list[tuple[list[list[float]], str, float]]:
    """
    Run EasyOCR on a BGR image and normalize to [(pts4x2, text, score), ...].
    """
    import easyocr
    reader = easyocr.Reader([lang], gpu=False, verbose=False)
    res = reader.readtext(bgr)  # [ [box(4 pts), text, score], ... ]
    out = []
    for item in res:
        try:
            pts, text, score = item
            text = str(text).strip()
            if not text:
                continue
            # pts is already list-of-4-points (x,y)
            out.append((pts, text, float(score)))
        except Exception:
            continue
    return out


# --------------------------------- OCR core ---------------------------------

def ocr_pdf(pdf_bytes: bytes, dpi: int = 300, max_side: int = 2200) -> OCRResult:
    """
    Raster OCR pipeline (EasyOCR only):
      1) pdf2image → PyMuPDF fallback
      2) preprocess + auto-invert for dark UI
      3) try EasyOCR on (normal/inverted/upsampled) for right-crop first, then full page
    """
    # 1) rasterize
    page_arrays_rgb: List["np.ndarray"] | None = None
    raster_errors: List[str] = []
    try:
        page_arrays_rgb = _render_with_pdf2image(pdf_bytes, dpi)
    except Exception as e:
        logger.warning("pdf2image rasterization failed: %r", e)
        raster_errors.append(f"pdf2image: {e!r}")
    if page_arrays_rgb is None:
        try:
            page_arrays_rgb = _render_with_fitz(pdf_bytes, dpi)
        except Exception as e:
            logger.error("PyMuPDF rasterization failed: %r", e)
            raster_errors.append(f"pymupdf: {e!r}")
    if not page_arrays_rgb:
        msg = f"Rasterization failed → {'; '.join(raster_errors) or 'no backend produced images'}"
        logger.error(msg)
        raise RuntimeError(msg)

    # 2) preprocess
    pages_bgr = _preprocess_pages(page_arrays_rgb, max_side=max_side)
    logger.info("Prepared %d page image(s) for OCR (max_side=%d)", len(pages_bgr), max_side)

    # 3) EasyOCR on variants; prefer right pane crop (denser text)
    words: List[Word] = []
    confidences: List[float] = []
    logger.info("Running EasyOCR on %d page image(s)", len(pages_bgr))

    for page_no, full_bgr in enumerate(pages_bgr, start=1):
        right_bgr = _crop_right_pane(full_bgr, 0.56)
        variants = _generate_variants(right_bgr) + _generate_variants(full_bgr)

        page_cnt = 0
        for vname, vimg in variants:
            try:
                parsed = _easyocr_try(vimg)
                logger.debug("EasyOCR [%s] -> %d items on page %d", vname, len(parsed), page_no)
            except Exception as e:
                logger.debug("EasyOCR [%s] error p%d: %r", vname, page_no, e)
                parsed = []

            if not parsed:
                continue

            # Convert to Words
            import numpy as np
            for pts, text, score in parsed:
                try:
                    arr = np.array(pts, dtype="float32").reshape(-1, 2)
                    x0 = float(arr[:, 0].min()); y0 = float(arr[:, 1].min())
                    x1 = float(arr[:, 0].max()); y1 = float(arr[:, 1].max())
                    if text:
                        words.append(Word(page=page_no, text=text, bbox=(x0, y0, x1, y1)))
                        confidences.append(float(score))
                        page_cnt += 1
                except Exception:
                    continue

            if page_cnt:
                logger.info("Page %d parsed via EasyOCR [%s]: %d items", page_no, vname, page_cnt)
                break  # stop after first variant that yields text

        if not page_cnt:
            logger.warning("No OCR blocks parsed on page %d via EasyOCR", page_no)

    if not words:
        logger.warning("EasyOCR produced 0 words. Dumping preprocessed pages to /tmp.")
        _debug_dump_pages(pages_bgr)

    mean_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0
    logger.info("OCR complete (EasyOCR): words=%d, mean_conf=%.3f", len(words), mean_conf)
    return OCRResult(words=words, mean_conf=mean_conf, page_images=pages_bgr)


# ------------------------- Group words into lines -------------------------

def lines_from_words(words: List[Word], y_tol: float = 4.0) -> List[Line]:
    if not words:
        logger.warning("No OCR words to group; returning empty line list.")
        return []
    words_sorted = sorted(words, key=lambda w: (w.page, round(w.bbox[1] / y_tol)))
    lines: List[Line] = []
    cur: List[Word] = []
    cur_key = None
    for w in words_sorted:
        key = (w.page, round(w.bbox[1] / y_tol))
        if cur_key is None or key == cur_key:
            cur.append(w); cur_key = key
        else:
            lines.append(_finalize_line(cur)); cur = [w]; cur_key = key
    if cur:
        lines.append(_finalize_line(cur))
    logger.debug("Grouped %d words into %d lines", len(words), len(lines))
    return lines
