"""
Классификатор качества PDF с использованием EasyOCR вместо pytesseract
"""

import os
import shutil
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from pdf2image import convert_from_path
import easyocr
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image


@dataclass
class PDFQualityResult:
    pdf_path: str
    category: str
    reason: str
    avg_blur: float
    median_ocr_conf: float
    mean_ocr_conf: float
    pct80: float
    words_count: int
    text_density: float
    roi_frac: float
    core_frac: float
    is_table_like: bool
    avg_skew_deg: float
    
    # Расширенные метрики (для совместимости с ExtendedPDFQualityResult)
    bbox_area_text_frac: float = 0.0
    conf_iqr: float = 0.0
    line_height_med: float = 0.0
    line_height_var: float = 0.0
    line_spacing_med: float = 0.0
    line_spacing_var: float = 0.0
    text_blocks_count: int = 0
    avg_block_width: float = 0.0
    avg_block_height: float = 0.0
    
    error: Optional[str] = None


class PDFQualityAssessorEasyOCR:
    def __init__(
        self,
        dpi: int = 400,
        tesseract_lang: str = "rus+eng",
        tesseract_config_text: str = "--oem 1 --psm 6",
        blur_low: float = 800.0,
        copy_to_dirs: bool = True,
        on_log: Optional[Callable[[str], None]] = None,
        max_workers: Optional[int] = 4,
        create_failed_dir: bool = True,
        min_roi_area_frac: float = 0.45,
        skew_bad_deg: float = 12.0,
        skew_warn_deg: float = 7.0,
    ):
        self.dpi = int(dpi)
        self.tesseract_lang = tesseract_lang
        self.tesseract_config_text = tesseract_config_text
        self.blur_low = float(blur_low)
        self.copy_to_dirs = copy_to_dirs
        self.on_log = on_log or (lambda msg: print(msg, flush=True))
        self.max_workers = max_workers
        self.create_failed_dir = create_failed_dir
        self.min_roi_area_frac = float(min_roi_area_frac)
        self.skew_bad_deg = float(skew_bad_deg)
        self.skew_warn_deg = float(skew_warn_deg)
        
        # Инициализация EasyOCR
        try:
            # Определяем языки для EasyOCR
            langs = []
            if "rus" in tesseract_lang.lower():
                langs.append('ru')
            if "eng" in tesseract_lang.lower() or "en" in tesseract_lang.lower():
                langs.append('en')
            if not langs:
                langs = ['en']
            
            self.reader = easyocr.Reader(langs, gpu=False)
            self.on_log("[INFO] EasyOCR инициализирован для классификации качества")
        except Exception as e:
            self.on_log(f"[WARNING] Ошибка инициализации EasyOCR: {e}")
            self.reader = None

    def _to_gray(self, image: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    def _find_document_roi(self, img: Image.Image) -> Tuple[Tuple[int, int, int, int], float]:
        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
        h, w = gray.shape[:2]
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = (0, 0, w, h); best_area = 0
        for c in cnts:
            x, y, ww, hh = cv2.boundingRect(c)
            area = ww * hh
            if area > best_area:
                best_area = area
                best = (x, y, x + ww, y + hh)
        roi_area_frac = best_area / float(w * h) if (w * h) else 1.0
        return best, roi_area_frac

    def _crop_roi(self, img: Image.Image) -> Tuple[Image.Image, float]:
        (x1, y1, x2, y2), frac = self._find_document_roi(img)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img.width, x2); y2 = min(img.height, y2)
        if x2 - x1 < img.width * 0.2 or y2 - y1 < img.height * 0.2:
            return img, frac
        return img.crop((x1, y1, x2, y2)), frac

    def _blur_score(self, image: Image.Image) -> float:
        gray = self._to_gray(image)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _text_density(self, image: Image.Image) -> float:
        gray = self._to_gray(image)
        gray = cv2.medianBlur(gray, 3)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 35, 15)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
        return float(np.count_nonzero(thr)) / float(thr.size)

    def _is_table_like(self, image: Image.Image) -> bool:
        gray = self._to_gray(image)
        thr = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
            31, 10
        )
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        h_lines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, v_kernel)

        frac = (np.count_nonzero(h_lines) + np.count_nonzero(v_lines)) / thr.size
        return frac > 0.010

    def _estimate_skew_deg(self, image: Image.Image) -> float:
        arr = np.array(image)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
        h, w = gray.shape[:2]
        m = int(min(h, w) * 0.06)
        roi = gray[m:h - m, m:w - m]
        thr = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 11)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, roi.shape[1] // 50), 3))
        lines = cv2.dilate(thr, kernel, iterations=1)
        cnts, _ = cv2.findContours(lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        def axial_dist(a: float) -> float:
            aa = abs(a)
            return min(aa, abs(90.0 - aa))

        dists = []
        area_min = (roi.shape[0] * roi.shape[1]) * 3e-4
        for c in cnts:
            a = cv2.contourArea(c)
            if a < area_min:
                continue
            (_, _), (rw, rh), ang = cv2.minAreaRect(c)
            if rw <= 1 or rh <= 1:
                continue
            dists.append(axial_dist(ang if ang <= 0 else ang - 90.0))
        return float(np.median(dists)) if dists else 0.0

    def _core_content_fraction(self, image: Image.Image) -> float:
        arr = np.array(image)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
        g = cv2.medianBlur(gray, 3)
        thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 41, 15)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        merged = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k, iterations=2)
        cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0.0
        h, w = gray.shape[:2]
        largest = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        return area / float(h * w)

    def _prep_for_ocr(self, image: Image.Image) -> Image.Image:
        im = image
        W, H = im.size
        long_side = max(W, H)
        if long_side < 1800:
            scale = 1800 / float(long_side)
            im = im.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
        arr = np.array(im)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = cv2.medianBlur(thr, 3)
        return Image.fromarray(thr)

    def _ocr_metrics_easyocr(self, image: Image.Image) -> Tuple[float, float, float, int]:
        """Использование EasyOCR для получения OCR метрик"""
        if not self.reader:
            return 0.0, 0.0, 0.0, 0
        
        try:
            # Конвертируем PIL Image в numpy array для EasyOCR
            img_array = np.array(image)
            
            # Выполняем OCR с помощью EasyOCR
            # EasyOCR возвращает список: [([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], text, confidence), ...]
            results = self.reader.readtext(img_array)
            
            if not results:
                return 0.0, 0.0, 0.0, 0
            
            # Извлекаем confidence scores и тексты
            confs = []
            words = 0
            
            for detection in results:
                if len(detection) >= 3:
                    # detection[0] - координаты
                    # detection[1] - текст
                    # detection[2] - confidence (0.0 - 1.0)
                    text = detection[1]
                    confidence = detection[2]
                    
                    if isinstance(confidence, (int, float)) and confidence > 0:
                        # EasyOCR возвращает confidence в диапазоне 0-1, конвертируем в 0-100
                        confs.append(float(confidence * 100))
                        if isinstance(text, str) and text.strip():
                            words += 1
            
            if not confs:
                return 0.0, 0.0, 0.0, 0
            
            # Вычисляем метрики аналогично pytesseract
            med = float(np.median(confs))
            mean = float(np.mean(confs))
            p80 = float(sum(c >= 80 for c in confs)) / float(len(confs))
            
            return med, mean, p80, words
            
        except Exception as e:
            self.on_log(f"[WARNING] EasyOCR error: {e}")
            return 0.0, 0.0, 0.0, 0

    def _categorize(self, blur: float, conf_med: float, pct80: float, words: int,
                    density: float, roi_frac: float, avg_skew_deg: float,
                    is_table: bool, core_frac: float) -> Tuple[str, str]:
        if roi_frac < self.min_roi_area_frac:
            return "trash", "roi<min"
        if avg_skew_deg >= self.skew_bad_deg:
            return "trash", "skew_bad"

        # EasyOCR возвращает более низкие confidence, корректируем пороги
        # Полностью мертвый OCR - очень мало слов и низкий confidence
        if words < 10 and conf_med < 10 and pct80 < 0.05:
            return "trash", "ocr_dead"
        if blur < 120 and pct80 < 0.10:
            return "trash", "blur_dead"

        if core_frac < 0.12 and pct80 < 0.15 and blur < 260 and conf_med < 20:
            return "trash", "miniature_poor"

        if is_table:
            # Для таблиц: более мягкие критерии с EasyOCR
            if conf_med >= 40 and pct80 >= 0.20 and blur >= 800 and words >= 50:
                return "good", "table_strong"
            if conf_med >= 20 and pct80 >= 0.08 and blur >= 220 and words >= 20:
                return "medium", "table_ok"
            # Если есть достаточно слов, даже с низким confidence - это не trash
            if words >= 30:
                return "medium", "table_readable"
            return "failed", "table_weak"

        # Для обычного текста: адаптированные пороги для EasyOCR
        if conf_med >= 60 and pct80 >= 0.30 and blur >= self.blur_low:
            return "good", "text_strong"
        if conf_med >= 35 and pct80 >= 0.15 and blur >= 400 and words >= 30:
            return "medium", "text_good"
        if conf_med >= 20 and pct80 >= 0.08 and blur >= 200 and words >= 20:
            return "medium", "text_ok"
        
        # Если распознали много слов, но с низким confidence - все равно readable
        if words >= 50:
            return "medium", "text_readable"

        return "failed", "text_weak"

    def assess_pdf(self, pdf_path: str) -> PDFQualityResult:
        try:
            pages = convert_from_path(pdf_path, dpi=self.dpi)
            if not pages:
                raise RuntimeError("PDF has 0 pages")

            roi_fracs: List[float] = []
            blur_scores: List[float] = []
            density_scores: List[float] = []
            core_fracs: List[float] = []
            skew_degs: List[float] = []
            table_flags: List[bool] = []
            conf_med_scores: List[float] = []
            conf_mean_scores: List[float] = []
            pct80_scores: List[float] = []
            words_scores: List[int] = []

            for page in pages:
                roi_img, roi_frac = self._crop_roi(page)
                roi_fracs.append(roi_frac)
                blur_scores.append(self._blur_score(roi_img))
                density_scores.append(self._text_density(roi_img))
                core_fracs.append(self._core_content_fraction(roi_img))
                skew_degs.append(self._estimate_skew_deg(roi_img))
                table_flags.append(self._is_table_like(roi_img))
                ocr_ready = self._prep_for_ocr(roi_img)
                mconf, meanconf, p80, words = self._ocr_metrics_easyocr(ocr_ready)
                conf_med_scores.append(mconf)
                conf_mean_scores.append(meanconf)
                pct80_scores.append(p80)
                words_scores.append(words)

            avg_blur = float(np.mean(blur_scores))
            med_conf = float(np.median(conf_med_scores))
            mean_conf = float(np.mean(conf_mean_scores))
            pct80 = float(np.mean(pct80_scores))
            total_words = int(np.sum(words_scores))
            avg_density = float(np.mean(density_scores))
            avg_skew = float(np.median(skew_degs)) if skew_degs else 0.0
            avg_roi_frac = float(np.median(roi_fracs)) if roi_fracs else 1.0
            avg_core = float(np.median(core_fracs)) if core_fracs else 0.0
            is_table_doc = any(table_flags)

            category, why = self._categorize(
                avg_blur, med_conf, pct80, total_words,
                avg_density, avg_roi_frac, avg_skew,
                is_table_doc, avg_core
            )

            self.on_log(
                f"{os.path.basename(pdf_path)} → blur:{avg_blur:.1f} conf_med:{med_conf:.1f} "
                f"pct80:{pct80:.2f} words:{total_words} dens:{avg_density:.3f} "
                f"roi:{avg_roi_frac:.2f} core:{avg_core:.2f} skew:{avg_skew:.1f} "
                f"table:{is_table_doc} → {category.upper()} (why={why})"
            )

            return PDFQualityResult(
                pdf_path=pdf_path,
                category=category,
                reason=why,
                avg_blur=avg_blur,
                median_ocr_conf=med_conf,
                mean_ocr_conf=mean_conf,
                pct80=pct80,
                words_count=total_words,
                text_density=avg_density,
                roi_frac=avg_roi_frac,
                core_frac=avg_core,
                is_table_like=is_table_doc,
                avg_skew_deg=avg_skew,
                # Расширенные метрики (базовые значения)
                bbox_area_text_frac=0.0,
                conf_iqr=0.0,
                line_height_med=0.0,
                line_height_var=0.0,
                line_spacing_med=0.0,
                line_spacing_var=0.0,
                text_blocks_count=0,
                avg_block_width=0.0,
                avg_block_height=0.0,
            )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            self.on_log(f"[FAILED] {os.path.basename(pdf_path)} → {err}")
            return PDFQualityResult(
                pdf_path=pdf_path,
                category="trash",
                reason=err,
                avg_blur=0.0,
                median_ocr_conf=0.0,
                mean_ocr_conf=0.0,
                pct80=0.0,
                words_count=0,
                text_density=0.0,
                roi_frac=0.0,
                core_frac=0.0,
                is_table_like=False,
                avg_skew_deg=0.0,
                # Расширенные метрики (нулевые значения)
                bbox_area_text_frac=0.0,
                conf_iqr=0.0,
                line_height_med=0.0,
                line_height_var=0.0,
                line_spacing_med=0.0,
                line_spacing_var=0.0,
                text_blocks_count=0,
                avg_block_width=0.0,
                avg_block_height=0.0,
                error=err,
            )

    def process_folder(
        self,
        input_folder: str,
        output_folder: Optional[str] = None,
        medium_subdir: str = "medium_quality",
        good_subdir: str = "good_quality",
        failed_subdir: str = "failed",
        trash_subdir: str = "trash",
        patterns: Tuple[str, ...] = (".pdf",),
    ) -> List[PDFQualityResult]:
        input_folder = os.path.abspath(input_folder)
        output_folder = os.path.abspath(output_folder) if output_folder else None

        if self.copy_to_dirs:
            if not output_folder:
                raise ValueError("output_folder is required when copy_to_dirs=True")
            medium_dir = os.path.join(output_folder, medium_subdir)
            good_dir = os.path.join(output_folder, good_subdir)
            failed_dir = os.path.join(output_folder, failed_subdir) if self.create_failed_dir else None
            trash_dir = os.path.join(output_folder, trash_subdir)
            os.makedirs(medium_dir, exist_ok=True)
            os.makedirs(good_dir, exist_ok=True)
            if failed_dir:
                os.makedirs(failed_dir, exist_ok=True)
            os.makedirs(trash_dir, exist_ok=True)
        else:
            medium_dir = good_dir = failed_dir = trash_dir = None  # type: ignore

        files = [
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.lower().endswith(patterns)
        ]
        if not files:
            self.on_log("[INFO] No PDF files found.")
            return []

        results: List[PDFQualityResult] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.assess_pdf, p): p for p in files}
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                if self.copy_to_dirs and output_folder:
                    if res.category == "good":
                        target = good_dir
                    elif res.category == "medium":
                        target = medium_dir
                    elif res.category == "failed":
                        target = failed_dir
                    else:
                        target = trash_dir
                    if target:
                        try:
                            shutil.copy(res.pdf_path, os.path.join(target, os.path.basename(res.pdf_path)))
                        except Exception as e:
                            self.on_log(f"[WARNING] Copy error {os.path.basename(res.pdf_path)} → {target}: {e}")
        return results

