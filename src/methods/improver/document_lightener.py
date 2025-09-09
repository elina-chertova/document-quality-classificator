import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from PIL import Image
from pdf2image import convert_from_path
import shutil


@dataclass
class LightenParams:
    target_long_side: int = 2200
    bg_kernel_frac: float = 0.06
    clahe_clip: float = 2.0
    denoise_h: int = 6
    sharpen_amount: float = 1.4
    edge_low: int = 50
    edge_high: int = 150
    keep_color: bool = False


def _ensure_odd(n: int) -> int:
    return n if n % 2 else n + 1


def _resize_long_side(img: np.ndarray, target: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w)
    if s >= target:
        return img
    scale = target / float(s)
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LANCZOS4)


def _l_channel(img_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    return L, a, b


def _merge_l_channel(L: np.ndarray, a: Optional[np.ndarray], b: Optional[np.ndarray], keep_color: bool) -> np.ndarray:
    if keep_color and a is not None and b is not None:
        lab = cv2.merge([L, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(L, cv2.COLOR_GRAY2BGR)



@dataclass
class AdvancedLightenParams:
    target_long_side: int = 2400
    bg_radius_frac: float = 0.08
    bg_gauss_sigma_frac: float = 0.02
    text_block_frac: float = 0.035
    text_open_size: int = 3
    p_low: float = 2.0
    p_high: float = 99.0
    clahe_clip: float = 1.5
    denoise_h_bg: int = 5
    sharpen_amount: float = 1.0
    edge_low: int = 50
    edge_high: int = 150
    keep_color: bool = True


def _percentile_stretch(gray: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    lo = np.percentile(gray, p_low)
    hi = np.percentile(gray, p_high)
    if hi <= lo:
        return gray
    out = (gray.astype(np.float32) - lo) / (hi - lo)
    out = np.clip(out, 0, 1) * 255.0
    return out.astype(np.uint8)


def lighten_document_image(img_bgr: np.ndarray, p: LightenParams = LightenParams()) -> np.ndarray:
    img_bgr = _resize_long_side(img_bgr, p.target_long_side)
    L, a, b = _l_channel(img_bgr)
    k = max(31, _ensure_odd(int(min(L.shape[:2]) * p.bg_kernel_frac)))
    bg = cv2.medianBlur(L, k)
    flat = cv2.divide(L, bg, scale=255)
    clahe = cv2.createCLAHE(clipLimit=p.clahe_clip, tileGridSize=(8, 8))
    flat = clahe.apply(flat)
    den = cv2.fastNlMeansDenoising(flat, h=p.denoise_h, templateWindowSize=7, searchWindowSize=21)
    blur = cv2.GaussianBlur(den, (0, 0), 1.0)
    usm = cv2.addWeighted(den, 1 + p.sharpen_amount, blur, -p.sharpen_amount, 0)
    edges = cv2.Canny(den, p.edge_low, p.edge_high)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)
    outL = den.copy()
    outL[edges > 0] = usm[edges > 0]
    outL = cv2.normalize(outL, None, 0, 255, cv2.NORM_MINMAX)
    outL = cv2.fastNlMeansDenoising(outL, h=4)
    out = _merge_l_channel(outL, a, b, p.keep_color)
    return out


def _lighten_background_only_bgr(img_bgr: np.ndarray, p: LightenParams) -> np.ndarray:

    L, a, b = _l_channel(img_bgr)
    k = max(31, _ensure_odd(int(min(L.shape[:2]) * p.bg_kernel_frac)))
    bg = cv2.medianBlur(L, k)
    flat = cv2.divide(L, bg, scale=255)
    stretched = _percentile_stretch(flat, 2.0, 99.8)
    outL = flat.copy()
    _, bgmask = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bgmask = cv2.bitwise_not(bgmask)
    outL[bgmask > 0] = stretched[bgmask > 0]
    outL = cv2.GaussianBlur(outL, (0, 0), 0.6)
    outL = cv2.normalize(outL, None, 0, 255, cv2.NORM_MINMAX)
    edges = cv2.Canny(L, p.edge_low, p.edge_high)
    blur = cv2.GaussianBlur(outL, (0, 0), 0.8)
    usm = cv2.addWeighted(outL, 1.15, blur, -0.15, 0)
    outL[edges > 0] = usm[edges > 0]
    return _merge_l_channel(outL, a, b, p.keep_color)

def lighten_document_image_advanced(img_bgr: np.ndarray, p: AdvancedLightenParams = AdvancedLightenParams()) -> np.ndarray:
    img = _resize_long_side(img_bgr, p.target_long_side)
    H, W = img.shape[:2]
    smin = min(H, W)

    L, a, b = _l_channel(img)

    r = max(25, int(smin * p.bg_radius_frac))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    bg = cv2.morphologyEx(L, cv2.MORPH_OPEN, kernel)

    sigma = max(3.0, smin * p.bg_gauss_sigma_frac)
    bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=sigma, sigmaY=sigma)

    bg_safe = np.maximum(bg, 10).astype(np.uint16)
    L_u16 = L.astype(np.uint16)
    flat = cv2.divide(L_u16, bg_safe, scale=255).astype(np.uint8)

    blk = _ensure_odd(max(25, int(smin * p.text_block_frac)))
    text_mask = cv2.adaptiveThreshold(
        flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blk, 10
    )
    if p.text_open_size > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (p.text_open_size, p.text_open_size))
        text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, k, iterations=1)
        text_mask = cv2.dilate(text_mask, np.ones((2, 2), np.uint8), 1)
    bg_mask = cv2.bitwise_not(text_mask)

    bg_smooth = cv2.fastNlMeansDenoising(flat, h=p.denoise_h_bg, templateWindowSize=7, searchWindowSize=21)
    bg_smooth = cv2.GaussianBlur(bg_smooth, (0, 0), 1.0)
    outL = flat.copy()
    outL[bg_mask > 0] = bg_smooth[bg_mask > 0]

    stretched = _percentile_stretch(outL, p.p_low, p.p_high)
    outL_bg = outL.copy()
    outL_bg[bg_mask > 0] = stretched[bg_mask > 0]
    clahe = cv2.createCLAHE(clipLimit=p.clahe_clip, tileGridSize=(8, 8))
    outL = clahe.apply(outL_bg)

    edges = cv2.Canny(outL, p.edge_low, p.edge_high)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)
    blur = cv2.GaussianBlur(outL, (0, 0), 1.0)
    usm = cv2.addWeighted(outL, 1 + p.sharpen_amount, blur, -p.sharpen_amount, 0)
    text_edges = cv2.bitwise_and(edges, text_mask)
    outL[text_edges > 0] = usm[text_edges > 0]

    return _merge_l_channel(outL, a, b, p.keep_color)


def save_image(out_bgr: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ext = os.path.splitext(out_path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(out_path, out_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(out_path, out_bgr)


def process_path(input_path: str, output_path: str, params: LightenParams = LightenParams()) -> None:
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {input_path}")
    out = lighten_document_image(img, params)
    save_image(out, output_path)


def process_folder(input_dir: str, output_dir: str, params: LightenParams = LightenParams()) -> None:
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    os.makedirs(output_dir, exist_ok=True)
    for name in os.listdir(input_dir):
        if not name.lower().endswith(exts):
            continue
        inp = os.path.join(input_dir, name)
        out = os.path.join(output_dir, name.rsplit(".", 1)[0] + "_clean.png")
        try:
            print(f"[INFO] {name} → enhancing")
            process_path(inp, out, params)
            print(f"[OK]   saved: {out}")
        except Exception as e:
            print(f"[FAILED] {name}: {e}")


class PDFDocumentLightener:
    def __init__(
        self,
        dpi: int = 200,
        lighten_params: Optional[object] = None,
        on_log: Optional[callable] = None
    ):
        self.dpi = dpi
        self.lighten_params: object = lighten_params if lighten_params is not None else LightenParams()
        self.on_log = on_log or (lambda msg: print(msg, flush=True))

    def lighten_pdf(self, pdf_path: str, output_path: str, passes: int = 1) -> bool:
        try:
            pages = convert_from_path(pdf_path, dpi=self.dpi)
            if not pages:
                self.on_log(f"[ERROR] PDF has 0 pages: {pdf_path}")
                return False

            temp_dir = os.path.join(os.path.dirname(output_path), "temp_images")
            os.makedirs(temp_dir, exist_ok=True)
            
            enhanced_pages = []
            
            for i, page in enumerate(pages):
                img_array = np.array(page)
                if len(img_array.shape) == 3:
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = img_array

                enhanced_bgr = img_bgr
                n_passes = max(1, int(passes))
                for p_idx in range(n_passes):
                    if isinstance(self.lighten_params, LightenParams):
                        if p_idx == 0:
                            enhanced_bgr = lighten_document_image(enhanced_bgr, self.lighten_params)  # type: ignore[arg-type]
                        else:
                            enhanced_bgr = _lighten_background_only_bgr(enhanced_bgr, self.lighten_params)
                    else:
                        enhanced_bgr = lighten_document_image_advanced(enhanced_bgr, self.lighten_params)  # type: ignore[arg-type]
                        break

                enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
                enhanced_pil = Image.fromarray(enhanced_rgb)
                enhanced_pages.append(enhanced_pil)
                
                self.on_log(f"[INFO] Processed page {i+1}/{len(pages)}")

            if enhanced_pages:
                enhanced_pages[0].save(
                    output_path,
                    save_all=True,
                    append_images=enhanced_pages[1:],
                    format='PDF',
                    resolution=self.dpi
                )

                shutil.rmtree(temp_dir, ignore_errors=True)
                
                self.on_log(f"[OK] Enhanced PDF saved: {output_path}")
                return True
            else:
                self.on_log(f"[ERROR] No pages processed: {pdf_path}")
                return False
                
        except Exception as e:
            self.on_log(f"[ERROR] Failed to process {pdf_path}: {e}")
            return False

    def process_dark_folder(
        self, 
        input_folder: str, 
        output_folder: str,
        lighten_params: Optional[LightenParams] = None,
        passes: int = 1
    ) -> dict:
        if lighten_params is not None:
            self.lighten_params = lighten_params
            
        os.makedirs(output_folder, exist_ok=True)

        pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            self.on_log("[INFO] No PDF files found in input folder")
            return {"processed": 0, "success": 0, "failed": 0, "errors": []}
        
        self.on_log(f"[INFO] Found {len(pdf_files)} PDF files to lighten")
        
        results = {
            "processed": len(pdf_files),
            "success": 0,
            "failed": 0,
            "errors": []
        }
        
        for pdf_file in pdf_files:
            input_path = os.path.join(input_folder, pdf_file)
            output_path = os.path.join(output_folder, f"{pdf_file}")
            
            self.on_log(f"[INFO] Processing: {pdf_file}")
            
            if self.lighten_pdf(input_path, output_path, passes=passes):
                results["success"] += 1
            else:
                results["failed"] += 1
                results["errors"].append(pdf_file)
        
        self.on_log(f"\n[SUMMARY] Processed: {results['processed']}, Success: {results['success']}, Failed: {results['failed']}")
        
        return results


# if __name__ == "__main__":
#     lightener = PDFDocumentLightener(
#         dpi=200,
#         lighten_params=LightenParams(
#             target_long_side=2200,
#             bg_kernel_frac=0.06,
#             clahe_clip=2.0,
#             keep_color=False
#         )
#     )
#
#     results = lightener.process_dark_folder(
#         input_folder="/Users/elinacertova/Downloads/documents_dataset/results/dark/dark_documents",
#         output_folder="/Users/elinacertova/Downloads/documents_dataset/results/dark/lightened_documents"
#     )
#
#     print(f"Results: {results}")
