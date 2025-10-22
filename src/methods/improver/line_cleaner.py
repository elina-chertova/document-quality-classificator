import io
import os
import csv
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import cv2
import fitz
import numpy as np
from PIL import Image


@dataclass
class LineRemovalParams:
    dpi: int = 300
    jpeg_quality: int = 95
    strength: int = 30
    min_coverage: float = 0.95
    inpaint_radius: int = 3
    inpaint_method: str = "telea"
    clean_pad_px: int = 0
    max_side_px: int = 0


def _safe_find_contours(img: np.ndarray):
    res = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return res[0] if len(res) == 2 else res[1]


def _build_long_line_mask(binary: np.ndarray, strength: int, min_coverage: float) -> Tuple[np.ndarray, List[Tuple[int,int,int,int]]]:
    h, w = binary.shape[:2]
    horiz = binary.copy()
    vert = binary.copy()

    horiz_size = max(1, w // max(1, strength))
    vert_size  = max(1, h // max(1, strength))

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_size, 1))
    vert_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_size))

    horiz = cv2.erode(horiz, horiz_kernel)
    horiz = cv2.dilate(horiz, horiz_kernel)
    vert  = cv2.erode(vert,  vert_kernel)
    vert  = cv2.dilate(vert,  vert_kernel)

    mask = np.zeros_like(binary)
    bboxes: List[Tuple[int,int,int,int]] = []

    cnts_h = _safe_find_contours(horiz)
    for c in cnts_h:
        x, y, ww, hh = cv2.boundingRect(c)
        if ww / float(w) >= min_coverage:
            if hh > 0:
                y0 = max(0, y)
                y1 = min(h, y + hh)
                if y1 > y0:
                    cv2.rectangle(mask, (x, y0), (x + ww, y1), 255, -1)
                    bboxes.append((x, y0, ww, y1 - y0))

    cnts_v = _safe_find_contours(vert)
    for c in cnts_v:
        x, y, ww, hh = cv2.boundingRect(c)
        if hh / float(h) >= min_coverage:
            if ww > 0:
                x0 = max(0, x)
                x1 = min(w, x + ww)
                if x1 > x0:
                    cv2.rectangle(mask, (x0, y), (x1, y + hh), 255, -1)
                    bboxes.append((x0, y, x1 - x0, hh))

    return mask, bboxes


def detect_extra_line_image_fast(
    img_bgr: np.ndarray,
    min_coverage: float = 0.95,
    strength: int = 30,
) -> Dict[str, object]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        ~gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        15, -2
    )
    mask, bboxes = _build_long_line_mask(binary, strength=strength, min_coverage=min_coverage)
    has_line = cv2.countNonZero(mask) > 0
    orient = None
    if has_line and len(bboxes) > 0:
        hs = sum(1 for (_, _, w, h) in bboxes if w >= h)
        vs = len(bboxes) - hs
        orient = "h" if hs >= vs else "v"
    max_len_ratio = 0.0
    h, w = gray.shape[:2]
    for (x, y, ww, hh) in bboxes:
        max_len_ratio = max(max_len_ratio, max(ww / float(w), hh / float(h)))
    return {
        "has_line": has_line,
        "orientation": orient,
        "score": float(max_len_ratio),
        "bboxes": bboxes,
    }


def _inpaint_long_lines(
    img_bgr: np.ndarray,
    strength: int,
    min_coverage: float,
    pad: int,
    radius: int,
    method: str,
) -> Tuple[np.ndarray, List[Tuple[int,int,int,int]]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        ~gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        15, -2
    )
    mask, bboxes = _build_long_line_mask(binary, strength=strength, min_coverage=min_coverage)

    if pad > 0 and cv2.countNonZero(mask) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * pad + 1, 2 * pad + 1))
        mask = cv2.dilate(mask, k, iterations=1)

    if cv2.countNonZero(mask) == 0:
        return img_bgr, []

    inpaint_flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    cleaned = cv2.inpaint(img_bgr, mask, radius, inpaint_flag)
    return cleaned, bboxes


class PDFLineCleaner:
    def __init__(self, params: LineRemovalParams = LineRemovalParams(), log_csv_path: Optional[str] = None):
        self.params = params
        self.log_csv_path = log_csv_path
        self._ensure_log_header()

    def _ensure_log_header(self) -> None:
        if not self.log_csv_path:
            return
        new_file = not os.path.exists(self.log_csv_path)
        os.makedirs(os.path.dirname(self.log_csv_path) or ".", exist_ok=True)
        if new_file:
            with open(self.log_csv_path, mode="w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["source_pdf", "page_index", "had_lines", "orientation", "score", "num_boxes", "boxes_json", "action"]
                )

    def _log(self, row: List[object]) -> None:
        if not self.log_csv_path:
            return
        with open(self.log_csv_path, mode="a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def clean_pdf(self, input_pdf: str, output_pdf: str) -> None:
        p = self.params
        src = fitz.open(input_pdf)
        try:
            page = src.load_page(0)
            zoom = p.dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csRGB)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            cleaned_bgr, bboxes = _inpaint_long_lines(
                img_bgr,
                strength=p.strength,
                min_coverage=p.min_coverage,
                pad=p.clean_pad_px,
                radius=p.inpaint_radius,
                method=p.inpaint_method,
            )

            has_line = len(bboxes) > 0
            max_len_ratio = 0.0
            H, W = img_bgr.shape[:2]
            for (x, y, w, h) in bboxes:
                max_len_ratio = max(max_len_ratio, max(w / float(W), h / float(H)))
            orient = None
            if has_line:
                hs = sum(1 for (_, _, w, h) in bboxes if w >= h)
                vs = len(bboxes) - hs
                orient = "h" if hs >= vs else "v"

            pil = Image.fromarray(cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=p.jpeg_quality, subsampling=0, optimize=True)

            page.clean_contents()
            page.insert_image(page.rect, stream=buf.getvalue())

            self._log([
                input_pdf, 0, has_line, orient, f"{max_len_ratio:.4f}",
                len(bboxes), json.dumps(bboxes, ensure_ascii=False), "cleaned" if has_line else "noop"
            ])

            os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
            src.save(output_pdf, deflate=True, garbage=4)
        finally:
            src.close()
