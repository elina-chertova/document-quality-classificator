import os
import io
import shutil
import logging
import cv2
import fitz
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List


@dataclass
class PDFDeskewParams:
    dpi: int = 400
    angle_limit: float = 35.0
    jpeg_quality: int = 85
    keep_size: bool = False


class PDFDeskewer:
    def __init__(self, params: PDFDeskewParams):
        self.params = params
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            )
        self._log = logging.getLogger(self.__class__.__name__)

    def _rotate_bound_white(self, img: np.ndarray, angle: float, keep_size: bool = False) -> np.ndarray:
        h, w = img.shape[:2]
        c = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(c, angle, 1.0)
        if keep_size:
            return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nW, nH = int(h * sin + w * cos), int(h * cos + w * sin)
        M[0, 2] += (nW / 2) - c[0]
        M[1, 2] += (nH / 2) - c[1]
        return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    def _printed_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
        k = 1.0 - np.max(rgb, axis=2)
        k8 = (k * 255).astype(np.uint8)
        th_k = cv2.adaptiveThreshold(k8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -5)
        s = hsv[:, :, 1]
        sat_mask = (s < 0.40).astype(np.uint8) * 255
        bin0 = cv2.bitwise_and(th_k, sat_mask)
        bin0[k8 < 20] = 0
        bin0 = cv2.medianBlur(bin0, 3)
        return bin0

    def _score_projection(self, bin_img: np.ndarray) -> float:
        s = bin_img.sum(axis=1).astype(np.float32)
        s = cv2.GaussianBlur(s[:, None], (1, 61), sigmaX=0, sigmaY=9)[:, 0]
        grad = np.abs(np.diff(s)).sum()
        h, w = bin_img.shape[:2]
        return float(grad) / (w * h)

    def _estimate_angle_strict(self, img_bgr: np.ndarray) -> float:
        h, w = img_bgr.shape[:2]
        m = int(min(h, w) * 0.06)
        roi = img_bgr[m:h - m, m:w - m].copy()
        bin_print = self._printed_mask(roi)
        kx = max(20, roi.shape[1] // 50)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3))
        lines = cv2.dilate(bin_print, kernel, iterations=1)
        cnts, _ = cv2.findContours(lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[tuple[float, float]] = []
        min_area = (roi.shape[0] * roi.shape[1]) * 3e-4

        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_area:
                continue
            (_, _), (rw, rh), ang = cv2.minAreaRect(c)
            if rw <= 1 or rh <= 1:
                continue
            W, H = (rw, rh) if rw >= rh else (rh, rw)
            if H == 0:
                continue
            aspect = W / H
            if aspect < 3.0:
                continue
            if ang < -45:
                ang += 90.0
            if abs(ang) > self.params.angle_limit:
                continue
            weight = a * (aspect - 1.0)
            candidates.append((ang, weight))

        if candidates:
            candidates.sort(key=lambda t: t[0])
            total = sum(w for _, w in candidates)
            acc = 0.0
            sign_angle = 0.0
            for a, wgt in candidates:
                acc += wgt
                if acc >= total / 2:
                    sign_angle = a
                    break
        else:
            sign_angle = 0.0

        work = bin_print
        best_ang, best_score = sign_angle, -1.0
        search_pattern = [(0.5, 8.0), (0.2, 3.0), (0.05, 1.0)] if sign_angle != 0.0 else [(1.0, 35.0), (0.2, 6.0), (0.05, 2.0)]
        for step, span in search_pattern:
            rng = np.arange(best_ang - span, best_ang + span + 1e-9, step)
            for ang in rng:
                rot = self._rotate_bound_white(work, ang, keep_size=False)
                sc = self._score_projection(rot)
                if sc > best_score:
                    best_score, best_ang = sc, ang

        return float(best_ang)

    def _deskew_image(self, img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        ang = self._estimate_angle_strict(img_bgr)
        if abs(ang) < 0.05:
            return img_bgr, 0.0
        return self._rotate_bound_white(img_bgr, ang, keep_size=self.params.keep_size), ang

    def deskew_pdf(self, input_path: str, output_path: str, failed_folder: str = None) -> None:
        try:
            doc = fitz.open(input_path)
            out = fitz.open()
            zoom = self.params.dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)

            total_pages = len(doc)
            for page_index, page in enumerate(doc, start=1):
                pix = page.get_pixmap(matrix=mat, alpha=False)
                if pix.w == 0 or pix.h == 0 or len(pix.samples) == 0:
                    out.insert_pdf(doc, from_page=page.number, to_page=page.number)
                    self._log.info("%s | page %d/%d skipped (empty)", os.path.basename(input_path), page_index, total_pages)
                    continue

                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                elif pix.n == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                fixed, ang = self._deskew_image(img)
                self._log.info("%s | page %d/%d angle %.2f°", os.path.basename(input_path), page_index, total_pages, ang)

                rgb = cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                buf = io.BytesIO()
                pil.save(buf, format="JPEG", quality=self.params.jpeg_quality, optimize=True)
                new_page = out.new_page(width=page.rect.width, height=page.rect.height)
                new_page.insert_image(new_page.rect, stream=buf.getvalue())

            out.save(output_path)
            out.close()
            doc.close()
            self._log.info("saved deskewed PDF: %s", output_path)

        except Exception as e:
            self._log.error("failed %s: %s", os.path.basename(input_path), e)
            if failed_folder:
                os.makedirs(failed_folder, exist_ok=True)
                failed_path = os.path.join(failed_folder, os.path.basename(input_path))
                try:
                    shutil.copy(input_path, failed_path)
                    self._log.info("moved to failed folder: %s", failed_path)
                except Exception as copy_err:
                    self._log.error("could not move to failed: %s", copy_err)
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                    self._log.info("removed incomplete output: %s", output_path)
                except Exception as rm_err:
                    self._log.warning("could not remove output: %s", rm_err)

    def process_folder(self, input_folder: str, output_folder: str, failed_folder: str) -> None:
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(failed_folder, exist_ok=True)

        # Проверяем, что входная папка существует
        if not os.path.exists(input_folder):
            self._log.error("Input folder does not exist: %s", input_folder)
            return

        files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
        total = len(files)
        if total == 0:
            self._log.info("no PDFs found in %s", input_folder)
            return

        for index, fname in enumerate(files, start=1):
            in_path = os.path.join(input_folder, fname)
            out_path = os.path.join(output_folder, fname)
            remaining = total - index
            self._log.info("[%d/%d] processing %s (remaining: %d)", index, total, fname, remaining)
            self.deskew_pdf(in_path, out_path, failed_folder)
