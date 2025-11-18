import os
import io
import shutil

import cv2
import fitz
import numpy as np
from PIL import Image

from src.methods.improver.text_enhancer import enhance_text_regions


def _mean_lightness(img_bgr: np.ndarray) -> float:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return float(np.mean(lab[:, :, 0]))


def enhance_text_documents(
    input_dir: str,
    output_dir: str,
    dpi: int = 300,
    min_lightness_for_enhancement: float = 130.0,
    skip_filenames: set[str] | None = None,
):
    Image.MAX_IMAGE_PIXELS = None
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

    enhanced = normal = skipped = failed = 0

    for filename in files:
        src = os.path.join(input_dir, filename)
        dst = os.path.join(output_dir, filename)
        try:
            doc = fitz.open(src)
            page = doc.load_page(0)
            mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
            pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csRGB)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            avg_l_source = _mean_lightness(img_bgr)
            if avg_l_source < min_lightness_for_enhancement or (skip_filenames and filename in skip_filenames):
                shutil.copy2(src, dst)
                doc.close()
                reason = (
                    f"слишком темный (avg L={avg_l_source:.1f})"
                    if avg_l_source < min_lightness_for_enhancement
                    else "отмечен как осветленный"
                )
                print(f"[TEXT-SKIP] {filename} → {reason}, пропускаем")
                skipped += 1
                continue

            enhanced_img, was_modified, avg_l = enhance_text_regions(img_bgr)
            result_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)

            pil_img = Image.fromarray(result_rgb)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=95)
            buf.seek(0)

            out_doc = fitz.open()
            img_page = out_doc.new_page(width=pix.width, height=pix.height)
            img_page.insert_image(img_page.rect, stream=buf.getvalue())
            out_doc.save(dst)
            out_doc.close()
            doc.close()

            if was_modified:
                print(f"[TEXT+] {filename} → усилен текст (avg L={avg_l:.1f})")
                enhanced += 1
            else:
                print(f"[TEXT] {filename} → без изменений (avg L={avg_l:.1f})")
                normal += 1
        except Exception as e:
            print(f"[FAILED] {filename}: {e}")
            failed += 1

    print("\nИтого по text enhancer:")
    print(f"  Усилен текст: {enhanced}")
    print(f"  Без изменений: {normal}")
    print(f"  Пропущено (тёмные): {skipped}")
    print(f"  Ошибок: {failed}")

