import os
import shutil
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
from typing import Optional


def bilateral_filter_lighten(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    mean_val = np.mean(filtered)
    if mean_val < 150:
        alpha = 200 / mean_val
        filtered = np.clip(filtered * alpha, 0, 255).astype(np.uint8)
    
    return cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)


class PDFBilateralLightener:
    def __init__(
        self,
        dpi: int = 300,
        on_log: Optional[callable] = None
    ):
        self.dpi = dpi
        self.on_log = on_log or (lambda msg: print(msg, flush=True))
    
    def lighten_pdf(self, pdf_path: str, output_path: str) -> bool:
        try:
            pages = convert_from_path(pdf_path, dpi=self.dpi)
            if not pages:
                self.on_log(f"[ERROR] PDF has 0 pages: {pdf_path}")
                return False
            
            enhanced_pages = []
            
            for i, page in enumerate(pages):
                img_array = np.array(page)
                if len(img_array.shape) == 3:
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = img_array

                enhanced_bgr = bilateral_filter_lighten(img_bgr)

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
        output_folder: str
    ) -> dict:
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
            output_path = os.path.join(output_folder, pdf_file)
            
            self.on_log(f"[INFO] Processing: {pdf_file}")
            
            if self.lighten_pdf(input_path, output_path):
                results["success"] += 1
            else:
                results["failed"] += 1
                results["errors"].append(pdf_file)
        
        self.on_log(f"\n[SUMMARY] Processed: {results['processed']}, Success: {results['success']}, Failed: {results['failed']}")
        
        return results

