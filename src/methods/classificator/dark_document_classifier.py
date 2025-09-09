import os
import shutil
from dataclasses import dataclass
from typing import List, Optional, Callable
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class DarkDocumentResult:
    pdf_path: str
    is_dark: bool
    mean_brightness: float
    median_brightness: float
    dark_pixels_percent: float
    very_dark_pixels_percent: float
    contrast: float
    reason: str
    error: Optional[str] = None


class DarkDocumentClassifier:
    """
    Классификатор для детекции слишком темных документов.
    Анализирует яркость PDF-документов и перемещает темные в отдельную папку.
    """

    def __init__(
        self,
        dpi: int = 200,
        brightness_threshold: float = 80.0,
        dark_pixels_threshold: float = 30.0,
        contrast_threshold: float = 15.0,
        very_dark_pixels_threshold: float = 15.0,
        copy_to_dirs: bool = True,
        on_log: Optional[Callable[[str], None]] = None,
        max_workers: Optional[int] = None,
        create_dark_dir: bool = True,
    ):
        self.dpi = int(dpi)
        self.brightness_threshold = float(brightness_threshold)
        self.dark_pixels_threshold = float(dark_pixels_threshold)
        self.contrast_threshold = float(contrast_threshold)
        self.very_dark_pixels_threshold = float(very_dark_pixels_threshold)
        self.copy_to_dirs = copy_to_dirs
        self.on_log = on_log or (lambda msg: print(msg, flush=True))
        self.max_workers = max_workers
        self.create_dark_dir = create_dark_dir

    def _analyze_brightness(self, image: Image.Image) -> tuple[float, float, float, float, float]:
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        mean_brightness = float(np.mean(gray))
        median_brightness = float(np.median(gray))

        contrast = float(np.std(gray))

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        total_pixels = gray.size

        dark_pixels = np.sum(hist[:100])
        dark_pixels_percent = float(dark_pixels / total_pixels * 100)

        very_dark_pixels = np.sum(hist[:50])
        very_dark_pixels_percent = float(very_dark_pixels / total_pixels * 100)

        return mean_brightness, median_brightness, dark_pixels_percent, very_dark_pixels_percent, contrast

    def _is_dark_document(self, mean_brightness: float, dark_pixels_percent: float,
                         very_dark_pixels_percent: float, contrast: float) -> tuple[bool, str]:
        if mean_brightness < 100:
            return True, f"very_low_brightness ({mean_brightness:.1f} < 100)"

        if mean_brightness < 220:
            return True, f"medium_low_brightness ({mean_brightness:.1f} < 220)"

        if mean_brightness < 150 and dark_pixels_percent > 20:
            return True, f"low_brightness_high_dark_pixels (brightness:{mean_brightness:.1f} < 150, dark:{dark_pixels_percent:.1f}% > 20%)"

        if dark_pixels_percent > 40:
            return True, f"too_many_dark_pixels ({dark_pixels_percent:.1f}% > 40%)"

        if very_dark_pixels_percent > 25:
            return True, f"too_many_very_dark_pixels ({very_dark_pixels_percent:.1f}% > 25%)"

        if mean_brightness < 120 and contrast < 20:
            return True, f"low_brightness_low_contrast (brightness:{mean_brightness:.1f} < 120, contrast:{contrast:.1f} < 20)"

        if contrast < 10:
            return True, f"extremely_low_contrast ({contrast:.1f} < 10)"

        return False, "normal_brightness"

    def classify_pdf(self, pdf_path: str) -> DarkDocumentResult:
        try:
            pages = convert_from_path(pdf_path, dpi=self.dpi)
            if not pages:
                raise RuntimeError("PDF has 0 pages")

            all_mean_brightness = []
            all_median_brightness = []
            all_dark_pixels_percent = []
            all_very_dark_pixels_percent = []
            all_contrast = []

            for page in pages:
                mean_bright, median_bright, dark_pixels_pct, very_dark_pixels_pct, contrast = self._analyze_brightness(page)
                all_mean_brightness.append(mean_bright)
                all_median_brightness.append(median_bright)
                all_dark_pixels_percent.append(dark_pixels_pct)
                all_very_dark_pixels_percent.append(very_dark_pixels_pct)
                all_contrast.append(contrast)

            avg_mean_brightness = float(np.mean(all_mean_brightness))
            avg_median_brightness = float(np.mean(all_median_brightness))
            avg_dark_pixels_percent = float(np.mean(all_dark_pixels_percent))
            avg_very_dark_pixels_percent = float(np.mean(all_very_dark_pixels_percent))
            avg_contrast = float(np.mean(all_contrast))

            is_dark, reason = self._is_dark_document(avg_mean_brightness, avg_dark_pixels_percent,
                                                   avg_very_dark_pixels_percent, avg_contrast)

            self.on_log(
                f"{os.path.basename(pdf_path)} → "
                f"mean_brightness:{avg_mean_brightness:.1f} "
                f"median_brightness:{avg_median_brightness:.1f} "
                f"dark_pixels:{avg_dark_pixels_percent:.1f}% "
                f"very_dark_pixels:{avg_very_dark_pixels_percent:.1f}% "
                f"contrast:{avg_contrast:.1f} → "
                f"{'DARK' if is_dark else 'NORMAL'} ({reason})"
            )

            return DarkDocumentResult(
                pdf_path=pdf_path,
                is_dark=is_dark,
                mean_brightness=avg_mean_brightness,
                median_brightness=avg_median_brightness,
                dark_pixels_percent=avg_dark_pixels_percent,
                very_dark_pixels_percent=avg_very_dark_pixels_percent,
                contrast=avg_contrast,
                reason=reason
            )

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            self.on_log(f"[FAILED] {os.path.basename(pdf_path)} → {err}")
            return DarkDocumentResult(
                pdf_path=pdf_path,
                is_dark=False,
                mean_brightness=0.0,
                median_brightness=0.0,
                dark_pixels_percent=0.0,
                very_dark_pixels_percent=0.0,
                contrast=0.0,
                reason="error",
                error=err
            )

    def process_folder(
        self,
        input_folder: str,
        output_folder: Optional[str] = None,
        normal_subdir: str = "normal_brightness",
        dark_subdir: str = "dark_documents",
        patterns: tuple[str, ...] = (".pdf",),
    ) -> List[DarkDocumentResult]:
        input_folder = os.path.abspath(input_folder)
        output_folder = os.path.abspath(output_folder) if output_folder else None

        if self.copy_to_dirs:
            if not output_folder:
                raise ValueError("output_folder is required when copy_to_dirs=True")

            normal_dir = os.path.join(output_folder, normal_subdir)
            dark_dir = os.path.join(output_folder, dark_subdir)

            os.makedirs(normal_dir, exist_ok=True)
            if self.create_dark_dir:
                os.makedirs(dark_dir, exist_ok=True)
        else:
            normal_dir = dark_dir = None

        files = [
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.lower().endswith(patterns)
        ]

        if not files:
            self.on_log("[INFO] No PDF files found.")
            return []

        self.on_log(f"[INFO] Found {len(files)} PDF files to process.")

        results: List[DarkDocumentResult] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.classify_pdf, pdf_path): pdf_path for pdf_path in files}

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                if self.copy_to_dirs and output_folder:
                    if result.is_dark and dark_dir:
                        target_dir = dark_dir
                    else:
                        target_dir = normal_dir

                    if target_dir:
                        try:
                            target_path = os.path.join(target_dir, os.path.basename(result.pdf_path))
                            shutil.copy(result.pdf_path, target_path)
                        except Exception as e:
                            self.on_log(f"[WARNING] Copy error {os.path.basename(result.pdf_path)} → {target_dir}: {e}")

        dark_count = sum(1 for r in results if r.is_dark)
        normal_count = len(results) - dark_count
        error_count = sum(1 for r in results if r.error)

        self.on_log(f"\n[SUMMARY] Processed {len(results)} files:")
        self.on_log(f"  - Normal brightness: {normal_count}")
        self.on_log(f"  - Dark documents: {dark_count}")
        self.on_log(f"  - Errors: {error_count}")

        return results

#
# if __name__ == "__main__":
#     classifier = DarkDocumentClassifier(
#         dpi=200,
#         brightness_threshold=120.0,
#         dark_pixels_threshold=15.0,
#         copy_to_dirs=True,
#         max_workers=4
#     )
#
#     results = classifier.process_folder(
#         input_folder="/Users/elinacertova/Downloads/documents_dataset/results/rotated",
#         output_folder="/Users/elinacertova/Downloads/documents_dataset/results/dark",
#     )
#
#     print("\n" + "="*80)
#     print("DETAILED RESULTS:")
#     print("="*80)
#
#     for result in results:
#         status = "DARK" if result.is_dark else "NORMAL"
#         print(f"{os.path.basename(result.pdf_path):<40} | {status:<6} | "
#               f"mean:{result.mean_brightness:6.1f} | "
#               f"dark:{result.dark_pixels_percent:5.1f}% | "
#               f"very_dark:{result.very_dark_pixels_percent:5.1f}% | "
#               f"contrast:{result.contrast:5.1f} | "
#               f"reason: {result.reason}")
