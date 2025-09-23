import io
import os
import csv
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image


@dataclass
class LineRemovalParams:
    dpi: int = 200
    jpeg_quality: int = 95
    min_len_ratio: float = 0.5
    line_thickness: int = 3


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
                writer = csv.writer(f)
                writer.writerow([
                    "source_pdf",
                    "page_index",
                    "had_lines",
                    "orientation",
                    "score",
                    "num_boxes",
                    "boxes",
                    "action",
                ])

    def _log(self, row: List[object]) -> None:
        if not self.log_csv_path:
            return
        with open(self.log_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    @staticmethod
    def remove_lines_from_image(img_bgr: np.ndarray, min_len_ratio: float = 0.5, line_thickness: int = 3) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        min_h = max(10, int(min_len_ratio * w))
        min_v = max(10, int(min_len_ratio * h))

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (min_h, 1))
        hor = cv2.morphologyEx(binv, cv2.MORPH_OPEN, kh, iterations=1)

        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_v))
        ver = cv2.morphologyEx(binv, cv2.MORPH_OPEN, kv, iterations=1)

        mask = cv2.bitwise_or(hor, ver)

        band_px = max(4, int(0.01 * min(w, h)))
        dark_ratio_thr = 0.12
        if float(np.count_nonzero(binv[0:band_px, :])) / float(binv[0:band_px, :].size) > dark_ratio_thr:
            mask[0:band_px, :] = 255
        if float(np.count_nonzero(binv[h - band_px:h, :])) / float(binv[h - band_px:h, :].size) > dark_ratio_thr:
            mask[h - band_px:h, :] = 255
        if float(np.count_nonzero(binv[:, 0:band_px])) / float(binv[:, 0:band_px].size) > dark_ratio_thr:
            mask[:, 0:band_px] = 255
        if float(np.count_nonzero(binv[:, w - band_px:w])) / float(binv[:, w - band_px:w].size) > dark_ratio_thr:
            mask[:, w - band_px:w] = 255

        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (line_thickness, line_thickness)), 1)

        cleaned = cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)
        return cleaned

    def clean_pdf(self, input_pdf: str, output_pdf: str) -> None:
        p = self.params
        src = fitz.open(input_pdf)
        out = fitz.open()
        zoom = p.dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        pages_with_lines = 0
        total_pages = 0
        for page_index, page in enumerate(src):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            det = detect_extra_line_image(
                img,
                min_len_ratio=max(p.min_len_ratio, 0.5),
                max_thickness_px=max(2, int(2 * p.line_thickness)),
                table_many_lines_threshold=6,
            )

            if det.get("has_line"):
                cleaned = self.remove_lines_from_image(img, min_len_ratio=p.min_len_ratio, line_thickness=p.line_thickness)
            else:
                cleaned = img

            pil = Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=p.jpeg_quality)
            new_page = out.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, stream=buf.getvalue())

            # log
            self._log([
                input_pdf,
                page_index,
                bool(det.get("has_line")),
                det.get("orientation"),
                det.get("score"),
                len(det.get("bboxes", [])),
                det.get("bboxes"),
                "cleaned" if det.get("has_line") else "noop",
            ])
            action = "cleaned" if det.get("has_line") else "noop"
            if det.get("has_line"):
                pages_with_lines += 1
            total_pages += 1
            print(f"[LINES] {os.path.basename(input_pdf)} p{page_index+1}: had_lines={bool(det.get('has_line'))} orient={det.get('orientation')} score={det.get('score'):.2f} boxes={len(det.get('bboxes', []))} action={action}")

        os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
        out.save(output_pdf)
        out.close()
        src.close()
        print(f"[LINES] Done {os.path.basename(input_pdf)}: pages_with_lines={pages_with_lines}/{total_pages} → {output_pdf}")


@dataclass
class LineDetectParams:
    dpi: int = 300
    min_len_ratio: float = 0.9
    max_thickness_px: int = 8
    table_many_lines_threshold: int = 3


def detect_extra_line_image(
    img_bgr: np.ndarray,
    min_len_ratio: float = 0.9,
    max_thickness_px: int = 8,
    table_many_lines_threshold: int = 3,
) -> Dict[str, object]:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    hor = cv2.morphologyEx(binv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(min_len_ratio * w)), 1)))
    ver = cv2.morphologyEx(binv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(min_len_ratio * h)))))

    def extract_boxes(mask: np.ndarray, orient: str) -> List[Tuple[int, int, int, int]]:
        cnts, _ = cv2.findContours(cv2.dilate(mask, np.ones((3, 3), np.uint8), 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[Tuple[int, int, int, int]] = []
        for c in cnts:
            x, y, ww, hh = cv2.boundingRect(c)
            if orient == 'h' and ww >= int(min_len_ratio * w) and hh <= max_thickness_px:
                boxes.append((x, y, ww, hh))
            if orient == 'v' and hh >= int(min_len_ratio * h) and ww <= max_thickness_px:
                boxes.append((x, y, ww, hh))
        return boxes

    h_boxes = extract_boxes(hor, 'h')
    v_boxes = extract_boxes(ver, 'v')


    band_px = max(4, int(0.01 * min(w, h)))
    dark_ratio_thr = 0.12

    top_band = binv[0:band_px, :]
    if float(np.count_nonzero(top_band)) / float(top_band.size) > dark_ratio_thr:
        h_boxes.append((0, 0, w, band_px))

    bottom_band = binv[h - band_px:h, :]
    if float(np.count_nonzero(bottom_band)) / float(bottom_band.size) > dark_ratio_thr:
        h_boxes.append((0, h - band_px, w, band_px))

    left_band = binv[:, 0:band_px]
    if float(np.count_nonzero(left_band)) / float(left_band.size) > dark_ratio_thr:
        v_boxes.append((0, 0, band_px, h))

    right_band = binv[:, w - band_px:w]
    if float(np.count_nonzero(right_band)) / float(right_band.size) > dark_ratio_thr:
        v_boxes.append((w - band_px, 0, band_px, h))

    def is_table_like(boxes: List[Tuple[int, int, int, int]], orient: str) -> bool:
        if len(boxes) < table_many_lines_threshold:
            return False
        coords = [y for (_, y, _, _) in boxes] if orient == 'h' else [x for (x, _, _, _) in boxes]
        spread = (max(coords) - min(coords)) / (h if orient == 'h' else w)
        return spread > 0.6

    if is_table_like(h_boxes, 'h'):
        h_boxes = []
    if is_table_like(v_boxes, 'v'):
        v_boxes = []

    def score(boxes: List[Tuple[int, int, int, int]], orient: str) -> float:
        if not boxes:
            return 0.0
        lengths = [bw / w if orient == 'h' else bh / h for (_, _, bw, bh) in boxes]
        thicks = [bh if orient == 'h' else bw for (_, _, bw, bh) in boxes]
        s_len = max(lengths)
        s_th = max(0.0, 1.0 - (min(thicks) / max(1.0, float(max_thickness_px))))
        return 0.8 * s_len + 0.2 * s_th

    h_score, v_score = score(h_boxes, 'h'), score(v_boxes, 'v')
    if h_score == 0 and v_score == 0:
        return {'has_line': False, 'orientation': None, 'score': 0.0, 'bboxes': []}
    if h_score >= v_score:
        return {'has_line': True, 'orientation': 'h', 'score': float(h_score), 'bboxes': h_boxes}
    else:
        return {'has_line': True, 'orientation': 'v', 'score': float(v_score), 'bboxes': v_boxes}


def detect_extra_line_pdf(pdf_path: str, params: LineDetectParams = LineDetectParams()) -> List[Dict[str, object]]:
    doc = fitz.open(pdf_path)
    zoom = params.dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    results: List[Dict[str, object]] = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        res = detect_extra_line_image(img, params.min_len_ratio, params.max_thickness_px, params.table_many_lines_threshold)
        results.append(res)
    doc.close()
    return results


