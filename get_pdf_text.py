#!/usr/bin/env python3
"""
extract_and_match_questions_right.py

- Renders PDF pages to images
- Runs pytesseract OCR to get text boxes
- Detects radio controls (OpenCV contour-based)
- For each radio group finds label text (below each radio)
- For each radio group finds question text to the RIGHT of the group (per your note)
- Fuzzy-matches that question text to a provided question list (JSON or Python list)
- Outputs JSON mapping: question_id, matched_question, match_score, selected_answer, page

Defaults:
- Input PDF: /mnt/data/rater_task_screens.pdf
- Question bank: provide a JSON file 'question_bank.json' in same folder as list of strings.
"""

import os
import json
from pathlib import Path
import fitz
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import Output
import difflib

# ---------- CONFIG ----------
PDF_PATH = "data/test_pdf/rater_task_screens.pdf"       # change as needed
QUESTION_BANK_PATH = "question_bank.json"           # JSON list of canonical questions
OUT_DIR = Path("out_radios_matched")
DPI = 200

# Detection / heuristic tuning - adjust for your screenshots
MIN_RADIO_AREA = 60
MAX_RADIO_AREA = 6000
FILL_THRESHOLD = 15            # percent dark pixels inside radio -> selected
GROUP_Y_GAP = 60               # determines grouping of stacked radios (px)
LABEL_VERTICAL_GAP = 6
LABEL_VERTICAL_MAX = 120
LABEL_HORIZONTAL_PADDING = 40

# How far to search to the RIGHT for question text (px)
QUESTION_SEARCH_RIGHT_MIN = 10
QUESTION_SEARCH_RIGHT_MAX = 1000   # large so it will find text further right if present

# Matching threshold (difflib ratio) (0-1). Lower => more permissive.
MATCH_THRESHOLD = 0.55

OUT_DIR.mkdir(parents=True, exist_ok=True)
# ------------------------------

def render_pdf_to_images(pdf_path, out_dir, dpi=200):
    doc = fitz.open(pdf_path)
    images = []
    for pno in range(len(doc)):
        page = doc[pno]
        mat = fitz.Matrix(dpi/72.0, dpi/72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = out_dir / f"page_{pno+1:03d}.png"
        pix.save(str(out_path))
        images.append(str(out_path))
    return images

def ocr_boxes(img_path):
    """Return list of OCR boxes dicts: x,y,w,h,text,conf"""
    img = Image.open(img_path).convert("RGB")
    data = pytesseract.image_to_data(img, output_type=Output.DICT)
    boxes = []
    for i in range(len(data['level'])):
        txt = data['text'][i].strip()
        if not txt:
            continue
        try:
            conf = float(data['conf'][i])
        except:
            conf = -1.0
        boxes.append({
            'x': int(data['left'][i]),
            'y': int(data['top'][i]),
            'w': int(data['width'][i]),
            'h': int(data['height'][i]),
            'text': txt,
            'conf': conf
        })
    return boxes

def detect_radios(img_path):
    """Find circular-ish contours (radios) and compute fill percentage."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("Could not open image: " + img_path)
    blur = cv2.GaussianBlur(img, (5,5), 0)
    th = cv2.adaptiveThreshold(blur, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    radios = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_RADIO_AREA or area > MAX_RADIO_AREA:
            continue
        x,y,wc,hc = cv2.boundingRect(c)
        ar = wc / float(hc) if hc>0 else 0
        if ar < 0.5 or ar > 1.6:
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circ = 4*np.pi*area/(perim*perim)
        if circ < 0.3:
            continue
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        interior_vals = cv2.bitwise_and(img, img, mask=mask)
        pixels = interior_vals[mask==255]
        if pixels.size == 0:
            continue
        dark = np.count_nonzero(pixels < 128)
        fill_pct = (dark / pixels.size) * 100.0
        radios.append({'x': int(x), 'y': int(y), 'w': int(wc), 'h': int(hc), 'fill_pct': float(fill_pct)})
    radios = sorted(radios, key=lambda r: (r['y'], r['x']))
    return radios

def group_radios(radios):
    if not radios:
        return []
    groups = []
    current = [radios[0]]
    for r in radios[1:]:
        if abs(r['y'] - current[-1]['y']) <= GROUP_Y_GAP:
            current.append(r)
        else:
            groups.append(current)
            current = [r]
    groups.append(current)
    return groups

def label_below_radio(radio, ocr_boxes):
    """Find OCR text directly below radio (within vertical range and horizontal padding)."""
    rx, ry, rw, rh = radio['x'], radio['y'], radio['w'], radio['h']
    radio_bottom = ry + rh
    candidates = []
    for b in ocr_boxes:
        bx, by, bw, bh = b['x'], b['y'], b['w'], b['h']
        if by <= radio_bottom:
            continue
        vert_gap = by - radio_bottom
        if vert_gap < LABEL_VERTICAL_GAP or vert_gap > LABEL_VERTICAL_MAX:
            continue
        # horizontal overlap with padding
        if (bx + bw) < (rx - LABEL_HORIZONTAL_PADDING) or bx > (rx + rw + LABEL_HORIZONTAL_PADDING):
            continue
        # score: prioritize smaller vertical gap and center alignment
        center_radio = rx + rw/2
        center_box = bx + bw/2
        score = vert_gap + abs(center_radio - center_box)*0.01
        candidates.append((score, b))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[0])
    best = candidates[0][1]
    # gather boxes close to best (same line)
    collected = [best['text']]
    base_y = best['y']
    for _, b in candidates[1:]:
        if abs(b['y'] - base_y) < max(20, best['h']):
            collected.append(b['text'])
    return " ".join(collected).strip()

def question_text_to_right_of_group(group, ocr_boxes, img_w):
    """
    For group bounding box, search OCR boxes to the RIGHT within horizontal window [group_right + min, +max]
    and vertical overlap - prefer boxes whose vertical center overlaps group's vertical span.
    Returns combined nearby boxes (joined by space).
    """
    gx1 = min(r['x'] for r in group)
    gx2 = max(r['x'] + r['w'] for r in group)
    gy1 = min(r['y'] for r in group)
    gy2 = max(r['y'] + r['h'] for r in group)
    candidates = []
    for b in ocr_boxes:
        bx, by, bw, bh = b['x'], b['y'], b['w'], b['h']
        # only boxes to the right
        if bx <= gx2 + QUESTION_SEARCH_RIGHT_MIN:
            continue
        if bx - gx2 > QUESTION_SEARCH_RIGHT_MAX:
            continue
        # vertical center overlap check
        b_center = by + bh/2
        g_center = (gy1 + gy2) / 2
        vert_dist = abs(b_center - g_center)
        # prefer boxes that vertically overlap or are near
        overlap = not ((by + bh) < gy1 or by > gy2)
        score = vert_dist - (100 if overlap else 0) + (bx - gx2)*0.001
        candidates.append((score, b))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[0])
    # take top candidate and gather peers on same horizontal band
    best = candidates[0][1]
    base_y = best['y']
    texts = [best['text']]
    for _, b in candidates[1:]:
        if abs(b['y'] - base_y) < max(40, best['h']):
            texts.append(b['text'])
    return " ".join(texts).strip()

def fuzzy_match_question(found_text, question_bank):
    """
    Uses difflib to find best match from question_bank for found_text.
    Returns (best_match, score [0..1]) or (None, 0) if nothing passes threshold.
    """
    if not found_text:
        return (None, 0.0)
    # compute ratios
    best = None
    best_score = 0.0
    for q in question_bank:
        score = difflib.SequenceMatcher(None, found_text.lower(), q.lower()).ratio()
        if score > best_score:
            best_score = score
            best = q
    if best_score >= MATCH_THRESHOLD:
        return (best, best_score)
    return (None, best_score)

def process_image(img_path, question_bank):
    boxes = ocr_boxes(img_path)
    radios = detect_radios(img_path)
    groups = group_radios(radios)
    page_w = Image.open(img_path).size[0]
    page_results = []
    local_qid = 0
    for g in groups:
        local_qid += 1
        # get labels for radios in group
        options = []
        for r in g:
            lbl = label_below_radio(r, boxes)
            options.append({'label': lbl, 'fill_pct': r['fill_pct'], 'bbox': (r['x'], r['y'], r['w'], r['h'])})
        # find selected label(s)
        selected = [opt['label'] or f"(option at {opt['bbox'][0]},{opt['bbox'][1]})"
                    for opt in options if opt['fill_pct'] >= FILL_THRESHOLD]
        if not selected and options:
            # fallback: highest fill_pct
            best = max(options, key=lambda o: o['fill_pct'])
            selected = [best['label'] or f"(option at {best['bbox'][0]},{best['bbox'][1]})"]
        # find question text to the right
        found_qtext = question_text_to_right_of_group(g, boxes, page_w)
        matched_q, match_score = fuzzy_match_question(found_qtext, question_bank)
        page_results.append({
            'local_question_id': local_qid,
            'found_question_text': found_qtext,
            'matched_question': matched_q,
            'match_score': float(match_score),
            'selected_answer': "; ".join(selected) if selected else "",
            'options': [o['label'] for o in options]
        })
    return page_results

def load_question_bank(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError("Question bank file not found: " + str(path))
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Question bank must be a JSON list of strings.")
    return data

def main():
    pdf = Path(PDF_PATH)
    if not pdf.exists():
        print("PDF not found:", PDF_PATH)
        return
    if not Path(QUESTION_BANK_PATH).exists():
        print("Question bank JSON not found:", QUESTION_BANK_PATH)
        print("Create a JSON file containing a list of question strings, e.g.:")
        print('["Question 1 text...", "Question 2 text...", ...]')
        return

    question_bank = load_question_bank(QUESTION_BANK_PATH)
    images = render_pdf_to_images(str(pdf), OUT_DIR, dpi=DPI)
    results_all = []
    global_qid = 0

    for page_idx, img in enumerate(images, start=1):
        page_results = process_image(img, question_bank)
        for pr in page_results:
            global_qid += 1
            results_all.append({
                'page': page_idx,
                'global_question_id': global_qid,
                'found_question_text': pr['found_question_text'],
                'matched_question': pr['matched_question'],
                'match_score': pr['match_score'],
                'selected_answer': pr['selected_answer'],
                'options': pr['options']
            })

    out_file = OUT_DIR / "matched_extraction.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False)

    print("Done. Wrote", out_file)
    print("Entries:", len(results_all))
    # Print summary of low-confidence matches (optional)
    low_conf = [r for r in results_all if (r['match_score'] < 0.7)]
    if low_conf:
        print("Warning: low-confidence matches (match_score < 0.7):", len(low_conf))
        for r in low_conf[:10]:
            print(r['page'], r['found_question_text'], "=>", r['matched_question'], r['match_score'])

if __name__ == "__main__":
    main()
