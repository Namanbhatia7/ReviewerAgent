# app/services/ingest_pdf/cv_radios.py
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import cv2

BBox = Tuple[float, float, float, float]

def radio_selected_near_label(
    page_img: "np.ndarray",
    label_bbox: BBox,
    search_up_px: int = 42,
    search_pad_px: int = 24,
) -> Tuple[bool, float]:
    """
    Returns (selected, confidence) by checking for a filled dot in a circular radio near the label.
    - label_bbox: (x0, y0, x1, y1) in image pixel coords
    """
    h, w = page_img.shape[:2]
    x0, y0, x1, y1 = map(int, label_bbox)
    # define ROI slightly above the label where circles are drawn in your UI
    roi_y0 = max(0, y0 - search_up_px)
    roi_y1 = max(0, y0 - 6)
    roi_x0 = max(0, x0 - search_pad_px)
    roi_x1 = min(w, x1 + search_pad_px)
    if roi_y1 <= roi_y0 or roi_x1 <= roi_x0:
        return (False, 0.0)

    roi = page_img[roi_y0:roi_y1, roi_x0:roi_x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # normalize contrast a bit
    gray = cv2.equalizeHist(gray)
    # detect circles
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=18,
        param1=90, param2=18, minRadius=8, maxRadius=20
    )
    if circles is None:
        return (False, 0.0)

    # For each circle, check if inner area is filled (darker/lighter than ring)
    best_conf = 0.0
    sel = False
    for (cx, cy, r) in circles[0]:
        cx, cy, r = int(cx), int(cy), int(r)
        # sample inner disk and annulus
        mask_inner = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask_inner, (cx, cy), int(r*0.5), 255, -1)
        mask_ring = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask_ring, (cx, cy), int(r), 255, 2)

        inner_vals = gray[mask_inner == 255]
        ring_vals  = gray[mask_ring == 255]
        if inner_vals.size < 10 or ring_vals.size < 10:
            continue

        inner_mean = float(inner_vals.mean())
        ring_mean  = float(ring_vals.mean())

        # In dark UI, selected often has a **bright inner dot** relative to ring.
        # Use absolute difference to be robust to themes.
        contrast = abs(inner_mean - ring_mean)

        if contrast >= 12:  # threshold tuned for your screenshots
            conf = min(0.95, 0.6 + contrast/50.0)
            if conf > best_conf:
                best_conf = conf
                sel = True

    return (sel, best_conf if sel else 0.0)
