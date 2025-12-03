import cv2
import numpy as np
from PIL import Image
from typing import Tuple
from src.methods.classificator.classificator_easyocr import PDFQualityAssessorEasyOCR


class PDFQualityAssessorEasyOCROptimized(PDFQualityAssessorEasyOCR):
    def __init__(
            self,
            *args,
            skip_heavy_checks: bool = True,
            enable_profiling: bool = False,
            **kwargs
    ):
        self.skip_heavy_checks = skip_heavy_checks
        self.enable_profiling = enable_profiling
        super().__init__(*args, **kwargs)
        
        if self.enable_profiling:
            from typing import Dict, List
            self.timings: Dict[str, List[float]] = {
                'pdf_to_image': [],
                'crop_roi': [],
                'blur_score': [],
                'text_density': [],
                'core_fraction': [],
                'estimate_skew': [],
                'is_table_like': [],
                'prep_for_ocr': [],
                'ocr_easyocr': [],
                'categorize': [],
            }
    
    def _resize_for_analysis(
            self,
            image: Image.Image,
            max_size: int = 1200
    ) -> np.ndarray:
        arr = np.array(image)
        h, w = arr.shape[:2] if arr.ndim == 2 else arr.shape[0:2]
        
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return arr
    
    def _blur_score(
            self,
            image: Image.Image
    ) -> float:
        arr = self._resize_for_analysis(image, max_size=1000)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())
    
    def _text_density(
            self,
            image: Image.Image
    ) -> float:
        arr = self._resize_for_analysis(image, max_size=800)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
        gray = cv2.medianBlur(gray, 3)
        _, thr = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        return float(np.count_nonzero(thr)) / float(thr.size)
    
    def _core_content_fraction(
            self,
            image: Image.Image
    ) -> float:
        if self.skip_heavy_checks:
            return 0.5
        
        arr = self._resize_for_analysis(image, max_size=800)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
        _, thr = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        merged = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k, iterations=1)
        cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0.0
        h, w = gray.shape[:2]
        largest = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        return area / float(h * w)
    
    def _estimate_skew_deg(
            self,
            image: Image.Image
    ) -> float:
        if self.skip_heavy_checks:
            return 0.0
        
        arr = self._resize_for_analysis(image, max_size=1000)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
        h, w = gray.shape[:2]
        m = int(min(h, w) * 0.06)
        roi = gray[m:h - m, m:w - m]
        _, thr = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, roi.shape[1] // 70), 2))
        lines = cv2.dilate(thr, kernel, iterations=1)
        cnts, _ = cv2.findContours(lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        def axial_dist(a: float) -> float:
            aa = abs(a)
            return min(aa, abs(90.0 - aa))
        
        dists = []
        area_min = (roi.shape[0] * roi.shape[1]) * 5e-4
        for c in cnts:
            a = cv2.contourArea(c)
            if a < area_min:
                continue
            (_, _), (rw, rh), ang = cv2.minAreaRect(c)
            if rw <= 1 or rh <= 1:
                continue
            dists.append(axial_dist(ang if ang <= 0 else ang - 90.0))
        return float(np.median(dists)) if dists else 0.0
    
    def _is_table_like(
            self,
            image: Image.Image
    ) -> bool:
        arr = self._resize_for_analysis(image, max_size=1000)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
        _, thr = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
        h_lines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, v_kernel)
        frac = (np.count_nonzero(h_lines) + np.count_nonzero(v_lines)) / thr.size
        return frac > 0.010
    
    def _crop_roi(self, img: Image.Image) -> Tuple[Image.Image, float]:
        arr = self._resize_for_analysis(img, max_size=1000)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
        h, w = gray.shape[:2]
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = (0, 0, w, h)
        best_area = 0
        for c in cnts:
            x, y, ww, hh = cv2.boundingRect(c)
            area = ww * hh
            if area > best_area:
                best_area = area
                best = (x, y, x + ww, y + hh)
        roi_area_frac = best_area / float(w * h) if (w * h) else 1.0
        
        scale = max(img.width, img.height) / max(w, h)
        x1, y1, x2, y2 = best
        x1 = int(x1 * scale)
        y1 = int(y1 * scale)
        x2 = int(x2 * scale)
        y2 = int(y2 * scale)
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.width, x2)
        y2 = min(img.height, y2)
        
        if x2 - x1 < img.width * 0.2 or y2 - y1 < img.height * 0.2:
            return img, roi_area_frac
        
        return img.crop((x1, y1, x2, y2)), roi_area_frac
    
    def _prep_for_ocr(self, image: Image.Image) -> Image.Image:
        Image.MAX_IMAGE_PIXELS = None
        im = image
        W, H = im.size
        long_side = max(W, H)
        if long_side < 1500:
            scale = 1500 / float(long_side)
            im = im.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
        arr = np.array(im)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(thr)


def get_optimized_assessor(
    dpi: int = 200,
    device: str = "cuda",
    skip_heavy_checks: bool = True,
    max_workers: int = 4,
) -> PDFQualityAssessorEasyOCROptimized:
    return PDFQualityAssessorEasyOCROptimized(
        dpi=dpi,
        copy_to_dirs=False,
        max_workers=max_workers,
        device=device,
        skip_heavy_checks=skip_heavy_checks,
    )

