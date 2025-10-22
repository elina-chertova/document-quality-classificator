import os
import fitz
import numpy as np
import cv2
from PIL import Image
import io

from src.methods.improver.contrast_enhancer import enhance_text_contrast


def enhance_contrast_documents(
    input_dir: str,
    output_dir: str,
    brightness_thresh: float = 120,
    darkness_boost: float = 1.5,
    dpi: int = 300,
):
    Image.MAX_IMAGE_PIXELS = None
    
    os.makedirs(output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    enhanced_count = 0
    normal_count = 0
    failed_count = 0
    
    for filename in files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            doc = fitz.open(input_path)
            out_doc = fitz.open()
            
            page_enhanced = False
            page_avg_dark_values = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
                pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csRGB)
                
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                enhanced_img, was_enhanced, avg_dark = enhance_text_contrast(
                    img_bgr, 
                    brightness_thresh=brightness_thresh,
                    darkness_boost=darkness_boost
                )
                
                page_avg_dark_values.append(avg_dark)
                if was_enhanced:
                    page_enhanced = True
                
                enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
                
                pil_img = Image.fromarray(enhanced_rgb)
                img_bytes = io.BytesIO()
                pil_img.save(img_bytes, format='JPEG', quality=95)
                img_bytes.seek(0)
                
                img_page = out_doc.new_page(width=pix.width, height=pix.height)
                img_page.insert_image(img_page.rect, stream=img_bytes.getvalue())
            
            out_doc.save(output_path)
            out_doc.close()
            doc.close()
            
            avg_dark_doc = np.mean(page_avg_dark_values) if page_avg_dark_values else 0.0
            
            if page_enhanced:
                print(f"[ENHANCED] {filename} → контраст усилен (avg_dark={avg_dark_doc:.1f})")
                enhanced_count += 1
            else:
                print(f"[OK] {filename} → контраст нормальный (avg_dark={avg_dark_doc:.1f})")
                normal_count += 1
                
        except Exception as e:
            import traceback
            print(f"[FAILED] {filename}: {e}")
            traceback.print_exc()
            failed_count += 1
    
    print(f"\nИтого:")
    print(f"  Усилен контраст: {enhanced_count}")
    print(f"  Нормальный контраст: {normal_count}")
    print(f"  Ошибки: {failed_count}")

