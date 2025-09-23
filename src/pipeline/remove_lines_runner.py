import os

import shutil
from src.methods.improver.line_cleaner import PDFLineCleaner, LineRemovalParams, detect_extra_line_pdf, LineDetectParams
from .config import PipelineConfig


def remove_lines(root_dir: str = "/Users/elinacertova/Downloads/documents_dataset", log_csv: str | None = None):
    cfg = PipelineConfig()
    if root_dir:
        cfg.paths.root_dir = root_dir

    input_dir = cfg.paths.input_folder
    cleaned_dir = cfg.paths.lines_cleaned_folder
    ok_dir = cfg.paths.no_lines_ok_folder
    os.makedirs(cleaned_dir, exist_ok=True)
    os.makedirs(ok_dir, exist_ok=True)

    params = LineRemovalParams(dpi=300, jpeg_quality=95, min_len_ratio=0.55, line_thickness=3)
    cleaner = PDFLineCleaner(params, log_csv_path=log_csv)

    for name in os.listdir(input_dir):
        if not name.lower().endswith(".pdf"):
            continue
        inp = os.path.join(input_dir, name)
        # быстрый проход детектора по PDF
        try:
            det_pages = detect_extra_line_pdf(inp, params=LineDetectParams(dpi=params.dpi, min_len_ratio=0.9, max_thickness_px=8, table_many_lines_threshold=3))
            has_any = any(bool(p.get('has_line')) for p in det_pages)
        except Exception:
            has_any = True  # на всякий случай чистим, если не смогли детектировать

        if has_any:
            out = os.path.join(cleaned_dir, name)
            print(f"[INFO] Lines detected → cleaning {inp} → {out}")
            try:
                cleaner.clean_pdf(inp, out)
            except Exception as e:
                print(f"[FAILED] {inp}: {e}")
        else:
            dst = os.path.join(ok_dir, name)
            print(f"[INFO] No lines detected → copying {inp} → {dst}")
            try:
                shutil.copy2(inp, dst)
            except Exception as e:
                print(f"[FAILED COPY] {inp}: {e}")


if __name__ == "__main__":
    pass


