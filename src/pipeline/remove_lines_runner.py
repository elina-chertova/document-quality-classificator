import os
import shutil
import fitz
import numpy as np

from src.methods.improver.line_cleaner import (
    PDFLineCleaner,
    LineRemovalParams,
    detect_extra_line_image_fast,
)


def remove_lines(input_dir: str, lines_cleaned_folder: str, no_lines_ok_folder: str, combined_output_folder: str | None = None, log_csv: str | None = None):
    os.makedirs(lines_cleaned_folder, exist_ok=True)
    os.makedirs(no_lines_ok_folder, exist_ok=True)
    if combined_output_folder:
        os.makedirs(combined_output_folder, exist_ok=True)

    params = LineRemovalParams(
        dpi=350,
        jpeg_quality=95,
        strength=32,
        min_coverage=0.95,
        clean_pad_px=1,
        inpaint_radius=1,
        inpaint_method="ns",
    )

    cleaner = PDFLineCleaner(params, log_csv_path=log_csv)

    for name in os.listdir(input_dir):
        if not name.lower().endswith(".pdf"):
            continue

        inp = os.path.join(input_dir, name)

        has_any = True
        try:
            doc = fitz.open(inp)
            page = doc.load_page(0)
            mat = fitz.Matrix(params.dpi / 72.0, params.dpi / 72.0)
            pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csRGB)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            doc.close()

            det = detect_extra_line_image_fast(
                img,
                min_coverage=params.min_coverage,
                strength=params.strength,
            )
            has_any = bool(det.get("has_line"))
        except Exception as e:
            print(f"[WARN] Detection failed for {name}: {e}")
            has_any = True

        if has_any:
            out = os.path.join(lines_cleaned_folder, name)
            print(f"[INFO] Lines detected → cleaning {inp} → {out}")
            try:
                cleaner.clean_pdf(inp, out)
                if combined_output_folder:
                    combined_dst = os.path.join(combined_output_folder, name)
                    shutil.copy2(out, combined_dst)
            except Exception as e:
                print(f"[FAILED] {inp}: {e}")
        else:
            dst = os.path.join(no_lines_ok_folder, name)
            print(f"[INFO] No lines detected → copying {inp} → {dst}")
            try:
                shutil.copy2(inp, dst)
                if combined_output_folder:
                    combined_dst = os.path.join(combined_output_folder, name)
                    shutil.copy2(dst, combined_dst)
            except Exception as e:
                print(f"[FAILED COPY] {inp}: {e}")



# import os
#
# import shutil
# # from src.methods.improver.line_cleaner import PDFLineCleaner, LineRemovalParams, detect_extra_line_pdf, LineDetectParams
# from src.methods.improver.line_cleaner import PDFLineCleaner, LineRemovalParams
#
#
# def remove_lines(input_dir, lines_cleaned_folder, no_lines_ok_folder, log_csv: str | None = None):
#     input_dir = input_dir
#     cleaned_dir = lines_cleaned_folder
#     ok_dir = no_lines_ok_folder
#     os.makedirs(cleaned_dir, exist_ok=True)
#     os.makedirs(ok_dir, exist_ok=True)
#
#     params = LineRemovalParams(
#         dpi=400,
#         jpeg_quality=95,
#         min_len_ratio=0.9,  # детекция длинных линий
#         clean_min_len_ratio=0.55,  # при очистке считаем линиями ≥55% ширины/высоты
#         clean_line_thickness=3
#     )
#
#     # params = LineRemovalParams(dpi=400, jpeg_quality=95, min_len_ratio=0.55, line_thickness=3)
#     cleaner = PDFLineCleaner(params, log_csv_path=log_csv)
#
#     for name in os.listdir(input_dir):
#         if not name.lower().endswith(".pdf"):
#             continue
#         inp = os.path.join(input_dir, name)
#         try:
#             det_pages = detect_extra_line_pdf(inp, params=LineDetectParams(dpi=params.dpi, min_len_ratio=0.9, max_thickness_px=8, table_many_lines_threshold=3))
#             has_any = any(bool(p.get('has_line')) for p in det_pages)
#         except Exception:
#             has_any = True
#
#         if has_any:
#             out = os.path.join(cleaned_dir, name)
#             print(f"[INFO] Lines detected → cleaning {inp} → {out}")
#             try:
#                 cleaner.clean_pdf(inp, out)
#             except Exception as e:
#                 print(f"[FAILED] {inp}: {e}")
#         else:
#             dst = os.path.join(ok_dir, name)
#             print(f"[INFO] No lines detected → copying {inp} → {dst}")
#             try:
#                 shutil.copy2(inp, dst)
#             except Exception as e:
#                 print(f"[FAILED COPY] {inp}: {e}")
