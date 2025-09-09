import os
import shutil
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from pdf2image import convert_from_path
import pytesseract
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image


@dataclass
class ExtendedPDFQualityResult:
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

    bbox_area_text_frac: float
    conf_iqr: float
    line_height_med: float
    line_height_var: float
    line_spacing_med: float
    line_spacing_var: float
    text_blocks_count: int
    avg_block_width: float
    avg_block_height: float
    
    error: Optional[str] = None


class ExtendedPDFQualityAssessor:
    def __init__(
        self,
        dpi: int = 200,
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
            (_, _), (rw, rh), ang = cv2.minAreaRect(c)  # (-90, 0]
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

    def _ocr_metrics_extended(self, image: Image.Image) -> Tuple[float, float, float, int, float, float, float, float, float, float, int, float, float]:
        d = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT,
            lang=self.tesseract_lang, config=self.tesseract_config_text
        )
        
        confs, words = [], 0
        texts = d.get("text", [])
        conf_list = d.get("conf", [])
        left_list = d.get("left", [])
        top_list = d.get("top", [])
        width_list = d.get("width", [])
        height_list = d.get("height", [])

        text_bboxes = []
        total_bbox_area = 0
        
        for i in range(len(conf_list)):
            conf = conf_list[i]
            if isinstance(conf, str) and conf.lstrip("-").isdigit():
                conf = int(conf)
            if isinstance(conf, (int, float)) and conf > 0:
                confs.append(int(conf))
                t = texts[i] if i < len(texts) else ""
                if isinstance(t, str) and t.strip():
                    words += 1

                    if (i < len(left_list) and i < len(top_list) and 
                        i < len(width_list) and i < len(height_list)):
                        left = left_list[i]
                        top = top_list[i]
                        width = width_list[i]
                        height = height_list[i]
                        
                        if width > 0 and height > 0:
                            bbox_area = width * height
                            text_bboxes.append({
                                'left': left, 'top': top, 'width': width, 'height': height,
                                'area': bbox_area, 'conf': conf
                            })
                            total_bbox_area += bbox_area

        med = float(np.median(confs)) if confs else 0.0
        mean = float(np.mean(confs)) if confs else 0.0
        p80 = (float(sum(c >= 80 for c in confs)) / float(len(confs))) if confs else 0.0

        image_area = image.width * image.height
        bbox_area_text_frac = total_bbox_area / image_area if image_area > 0 else 0.0

        conf_iqr = float(np.percentile(confs, 75) - np.percentile(confs, 25)) if len(confs) > 1 else 0.0

        line_heights = [bbox['height'] for bbox in text_bboxes if bbox['height'] > 0]
        line_height_med = float(np.median(line_heights)) if line_heights else 0.0
        line_height_var = float(np.var(line_heights)) if len(line_heights) > 1 else 0.0

        line_spacings = []
        if len(text_bboxes) > 1:
            sorted_bboxes = sorted(text_bboxes, key=lambda x: x['top'])
            for i in range(1, len(sorted_bboxes)):
                prev_bottom = sorted_bboxes[i-1]['top'] + sorted_bboxes[i-1]['height']
                curr_top = sorted_bboxes[i]['top']
                spacing = curr_top - prev_bottom
                if spacing > 0:
                    line_spacings.append(spacing)
        
        line_spacing_med = float(np.median(line_spacings)) if line_spacings else 0.0
        line_spacing_var = float(np.var(line_spacings)) if len(line_spacings) > 1 else 0.0

        text_blocks_count = len(text_bboxes)

        avg_block_width = float(np.mean([bbox['width'] for bbox in text_bboxes])) if text_bboxes else 0.0
        avg_block_height = float(np.mean([bbox['height'] for bbox in text_bboxes])) if text_bboxes else 0.0
        
        return (med, mean, p80, words, bbox_area_text_frac, conf_iqr, 
                line_height_med, line_height_var, line_spacing_med, line_spacing_var,
                text_blocks_count, avg_block_width, avg_block_height)

    def _categorize_extended(self, blur: float, conf_med: float, pct80: float, words: int,
                           density: float, roi_frac: float, avg_skew_deg: float,
                           is_table: bool, core_frac: float, bbox_area_text_frac: float,
                           conf_iqr: float, line_height_var: float) -> Tuple[str, str]:
        if roi_frac < self.min_roi_area_frac:
            return "trash", "roi<min"
        if avg_skew_deg >= self.skew_bad_deg:
            return "trash", "skew_bad"

        if conf_med < 25 and pct80 < 0.10:
            return "trash", "ocr_dead"
        if blur < 120 and pct80 < 0.20:
            return "trash", "blur_dead"

        if bbox_area_text_frac < 0.03 and conf_med < 50:
            return "trash", "no_text_content"

        if conf_iqr >= 35 and pct80 < 0.30:
            return "failed", "unstable_ocr"

        if line_height_var > 100 and conf_med < 70:
            return "medium", "irregular_lines"

        if core_frac < 0.12 and pct80 < 0.25 and blur < 260 and conf_med < 55:
            return "trash", "miniature_poor"

        if is_table:
            if conf_med >= 65 and pct80 >= 0.45 and blur >= 800:
                return "good", "table_strong"
            if conf_med >= 55 and pct80 >= 0.30 and blur >= 220:
                return "medium", "table_ok"
            return "failed", "table_weak"

        if (conf_med >= 90 and pct80 >= 0.55 and blur >= self.blur_low and 
            bbox_area_text_frac >= 0.15 and conf_iqr <= 20):
            return "good", "text_strong"
        
        if (conf_med >= 70 and pct80 >= 0.35 and bbox_area_text_frac >= 0.05):
            return "medium", "text_ok"
        
        return "failed", "text_weak"

    def assess_pdf(self, pdf_path: str) -> ExtendedPDFQualityResult:
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

            bbox_area_text_fracs: List[float] = []
            conf_iqrs: List[float] = []
            line_height_meds: List[float] = []
            line_height_vars: List[float] = []
            line_spacing_meds: List[float] = []
            line_spacing_vars: List[float] = []
            text_blocks_counts: List[int] = []
            avg_block_widths: List[float] = []
            avg_block_heights: List[float] = []

            for page in pages:
                roi_img, roi_frac = self._crop_roi(page)
                roi_fracs.append(roi_frac)
                blur_scores.append(self._blur_score(roi_img))
                density_scores.append(self._text_density(roi_img))
                core_fracs.append(self._core_content_fraction(roi_img))
                skew_degs.append(self._estimate_skew_deg(roi_img))
                table_flags.append(self._is_table_like(roi_img))
                ocr_ready = self._prep_for_ocr(roi_img)

                (mconf, meanconf, p80, words, bbox_area_text_frac, conf_iqr,
                 line_height_med, line_height_var, line_spacing_med, line_spacing_var,
                 text_blocks_count, avg_block_width, avg_block_height) = self._ocr_metrics_extended(ocr_ready)
                
                conf_med_scores.append(mconf)
                conf_mean_scores.append(meanconf)
                pct80_scores.append(p80)
                words_scores.append(words)

                bbox_area_text_fracs.append(bbox_area_text_frac)
                conf_iqrs.append(conf_iqr)
                line_height_meds.append(line_height_med)
                line_height_vars.append(line_height_var)
                line_spacing_meds.append(line_spacing_med)
                line_spacing_vars.append(line_spacing_var)
                text_blocks_counts.append(text_blocks_count)
                avg_block_widths.append(avg_block_width)
                avg_block_heights.append(avg_block_height)

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

            avg_bbox_area_text_frac = float(np.mean(bbox_area_text_fracs))
            avg_conf_iqr = float(np.mean(conf_iqrs))
            avg_line_height_med = float(np.median(line_height_meds))
            avg_line_height_var = float(np.mean(line_height_vars))
            avg_line_spacing_med = float(np.median(line_spacing_meds))
            avg_line_spacing_var = float(np.mean(line_spacing_vars))
            total_text_blocks = int(np.sum(text_blocks_counts))
            avg_block_width = float(np.mean(avg_block_widths))
            avg_block_height = float(np.mean(avg_block_heights))

            category, why = self._categorize_extended(
                avg_blur, med_conf, pct80, total_words,
                avg_density, avg_roi_frac, avg_skew,
                is_table_doc, avg_core, avg_bbox_area_text_frac,
                avg_conf_iqr, avg_line_height_var
            )

            self.on_log(
                f"{os.path.basename(pdf_path)} → blur:{avg_blur:.1f} conf_med:{med_conf:.1f} "
                f"pct80:{pct80:.2f} words:{total_words} dens:{avg_density:.3f} "
                f"roi:{avg_roi_frac:.2f} core:{avg_core:.2f} skew:{avg_skew:.1f} "
                f"table:{is_table_doc} bbox_frac:{avg_bbox_area_text_frac:.3f} "
                f"conf_iqr:{avg_conf_iqr:.1f} line_var:{avg_line_height_var:.1f} "
                f"→ {category.upper()} (why={why})"
            )

            return ExtendedPDFQualityResult(
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
                bbox_area_text_frac=avg_bbox_area_text_frac,
                conf_iqr=avg_conf_iqr,
                line_height_med=avg_line_height_med,
                line_height_var=avg_line_height_var,
                line_spacing_med=avg_line_spacing_med,
                line_spacing_var=avg_line_spacing_var,
                text_blocks_count=total_text_blocks,
                avg_block_width=avg_block_width,
                avg_block_height=avg_block_height,
            )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            self.on_log(f"[FAILED] {os.path.basename(pdf_path)} → {err}")
            return ExtendedPDFQualityResult(
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
    ) -> List[ExtendedPDFQualityResult]:
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

        results: List[ExtendedPDFQualityResult] = []
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


# if __name__ == "__main__":
#     assessor = ExtendedPDFQualityAssessor(
#         dpi=200,
#         copy_to_dirs=True,
#         max_workers=4
#     )
#     results = assessor.process_folder(
#         input_folder="/Users/elinacertova/Downloads/documents_dataset/results/processed/",
#         output_folder="/Users/elinacertova/Downloads/documents_dataset/results/sorted_by_quality/",
#     )
#     cnt = {"trash": 0, "failed": 0, "medium": 0, "good": 0}
#     for r in results:
#         cnt[r.category] += 1
#     print(f"\nSummary: trash={cnt['trash']}, failed={cnt['failed']}, medium={cnt['medium']}, good={cnt['good']}")
