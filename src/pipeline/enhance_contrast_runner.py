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
            page = doc.load_page(0)

            mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
            pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csRGB)

            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            enhanced_img, was_enhanced, avg_dark = enhance_text_contrast(img_bgr)

            img_to_use = enhanced_img if was_enhanced else img_bgr
            result_rgb = cv2.cvtColor(img_to_use, cv2.COLOR_BGR2RGB)

            pil_img = Image.fromarray(result_rgb)
            img_bytes = io.BytesIO()
            pil_img.save(img_bytes, format='JPEG', quality=95)
            img_bytes.seek(0)

            out_doc = fitz.open()
            img_page = out_doc.new_page(width=pix.width, height=pix.height)
            img_page.insert_image(img_page.rect, stream=img_bytes.getvalue())

            out_doc.save(output_path)
            out_doc.close()
            doc.close()

            if was_enhanced:
                print(f"[ENHANCED] {filename} → контраст усилен (avg_dark={avg_dark:.1f})")
                enhanced_count += 1
            else:
                print(f"[OK] {filename} → контраст нормальный (avg_dark={avg_dark:.1f})")
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

# перетестировать на quality comparison и удалить
# import os
# import fitz
# import numpy as np
# import cv2
# from PIL import Image
# import io
#
# from src.methods.improver.contrast_enhancer import enhance_text_contrast
#
#
# def enhance_contrast_documents(
#         input_dir: str,
#         output_dir: str,
#         brightness_thresh: float = 120,
#         darkness_boost: float = 1.5,
#         dpi: int = 300,
# ):
#     Image.MAX_IMAGE_PIXELS = None
#
#     os.makedirs(output_dir, exist_ok=True)
#
#     files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
#
#     enhanced_count = 0
#     normal_count = 0
#     failed_count = 0
#
#     for filename in files:
#         input_path = os.path.join(input_dir, filename)
#         output_path = os.path.join(output_dir, filename)
#
#         try:
#             doc = fitz.open(input_path)
#             page = doc.load_page(0)
#
#             mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
#             pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csRGB)
#
#             img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
#             img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#
#             enhanced_img, was_enhanced, avg_dark = enhance_text_contrast(
#                 img_bgr,
#                 brightness_thresh=brightness_thresh,
#                 darkness_boost=darkness_boost
#             )
#
#             enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
#
#             pil_img = Image.fromarray(enhanced_rgb)
#             img_bytes = io.BytesIO()
#             pil_img.save(img_bytes, format='JPEG', quality=95)
#             img_bytes.seek(0)
#
#             out_doc = fitz.open()
#             img_page = out_doc.new_page(width=pix.width, height=pix.height)
#             img_page.insert_image(img_page.rect, stream=img_bytes.getvalue())
#
#             out_doc.save(output_path)
#             out_doc.close()
#             doc.close()
#
#             if was_enhanced:
#                 print(f"[ENHANCED] {filename} → контраст усилен (avg_dark={avg_dark:.1f})")
#                 enhanced_count += 1
#             else:
#                 print(f"[OK] {filename} → контраст нормальный (avg_dark={avg_dark:.1f})")
#                 normal_count += 1
#
#         except Exception as e:
#             import traceback
#             print(f"[FAILED] {filename}: {e}")
#             traceback.print_exc()
#             failed_count += 1
#
#     print(f"\nИтого:")
#     print(f"  Усилен контраст: {enhanced_count}")
#     print(f"  Нормальный контраст: {normal_count}")
#     print(f"  Ошибки: {failed_count}")
