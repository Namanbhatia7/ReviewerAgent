from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2

BBox = Tuple[float, float, float, float]

def _safe_roi(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return img[0:0, 0:0]
    return img[y0:y1, x0:x1]

def _local_bg_mean(gray: np.ndarray, cx: int, cy: int, r: int, pad: int = 8) -> float:
    h, w = gray.shape[:2]
    x0 = max(0, cx - r - pad); y0 = max(0, cy - r - pad)
    x1 = min(w, cx + r + pad); y1 = min(h, cy + r + pad)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0: 
        return float(gray.mean())
    return float(roi.mean())

def radio_selected_near_label(
    page_img: "np.ndarray",
    label_bbox: BBox,
    search_up_px: int = 72,       # wider vertical reach
    search_pad_px: int = 60,      # wider horizontal reach
) -> Tuple[bool, float]:
    """
    Detect a selected radio above the label:
      - Uses HoughCircles, blob/contour, and inner-vs-local-background contrast.
      - Returns (selected, confidence [0..1]).
    """
    h, w = page_img.shape[:2]
    x0, y0, x1, y1 = map(int, label_bbox)

    # Radios sit above the label row in your UI; search a band above the label.
    roi_y0 = max(0, y0 - search_up_px)
    roi_y1 = max(0, y0 - 6)
    roi_x0 = max(0, x0 - search_pad_px)
    roi_x1 = min(w, x1 + search_pad_px)

    roi = _safe_roi(page_img, roi_x0, roi_y0, roi_x1, roi_y1)
    if roi.size == 0:
        return (False, 0.0)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Light equalization for dark theme; also denoise slightly
    gray = cv2.equalizeHist(gray)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    hough_conf, blob_conf, contrast_conf = 0.0, 0.0, 0.0
    selected = False

    # --- (A) Hough circles (looser thresholds) ---
    circles = cv2.HoughCircles(
        gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=22,
        param1=80, param2=12, minRadius=8, maxRadius=22
    )

    candidates = []
    if circles is not None:
        for (cx, cy, r) in circles[0]:
            cx, cy, r = int(cx), int(cy), int(r)
            candidates.append((cx, cy, r))

    # If Hough missed, add blob-like candidates using bright threshold
    if not candidates:
        # Threshold towards bright radios on dark background
        _, th = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.medianBlur(th, 3)
        cnts, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            (x, y), rad = cv2.minEnclosingCircle(c)
            area = cv2.contourArea(c)
            if area < 30 or area > 1200:
                continue
            cx, cy, r = int(x), int(y), int(rad)
            # circularity filter
            per = cv2.arcLength(c, True) + 1e-6
            circ = 4 * np.pi * (area / (per * per))
            if 0.4 <= circ <= 1.2 and 7 <= r <= 24:
                candidates.append((cx, cy, r))

    # Evaluate each candidate for "filled" vs "empty"
    best_score = 0.0
    for (cx, cy, r) in candidates:
        # Build inner and ring masks
        inner_r = max(3, int(r * 0.5))
        inner_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(inner_mask, (cx, cy), inner_r, 255, -1)

        ring_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(ring_mask, (cx, cy), int(r), 255, 2)

        inner_vals = gray[inner_mask == 255]
        ring_vals  = gray[ring_mask == 255]
        if inner_vals.size < 8 or ring_vals.size < 8:
            continue

        inner_mean = float(inner_vals.mean())
        ring_mean  = float(ring_vals.mean())
        bg_mean    = _local_bg_mean(gray, cx, cy, r, pad=10)

        # Scores:
        # 1) inner vs local bg (bright on dark) – strong in your UI
        s_bg = max(0.0, (inner_mean - bg_mean) / 40.0)  # scale factor tuned
        # 2) inner vs ring (sometimes small due to bright ring) – weaker cue
        s_ring = max(0.0, abs(inner_mean - ring_mean) / 35.0)
        # 3) solidity: is the inner disk consistently bright?
        _, inner_bin = cv2.threshold(inner_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fill_ratio = float((inner_vals > np.percentile(inner_vals, 60)).mean())
        s_fill = max(0.0, (fill_ratio - 0.5) * 1.6)  # >0.5 means mostly filled

        score = 0.55 * s_bg + 0.25 * s_ring + 0.20 * s_fill  # weighted fusion
        score = float(max(0.0, min(1.2, score)))  # cap
        if score > best_score:
            best_score = score

    if best_score >= 0.55:    # decision threshold tuned for your screenshot
        selected = True
        conf = min(0.95, 0.65 + (best_score - 0.55))  # 0.65..0.95
        return (selected, conf)

    # Not confident enough → unselected
    return (False, 0.0)
