"""
Раннер для запуска улучшений качества документов на датасете.
Выбирайте один из пайплайнов: ocrmypdf, scantailor_then_ocr, unpaper_tesseract.
"""
import os
from glob import glob
from typing import Literal

from src.pipeline.config import PipelineConfig
from src.methods.improver.ocr.ocrmypdf_tools import process_with_ocrmypdf
from src.methods.improver.ocr.scantailor_pipeline import pipeline_scantailor_then_ocr
from src.methods.improver.ocr.unpaper_tesseract_pipeline import unpaper_tesseract_pipeline


def improve_dataset(
    mode: Literal["ocrmypdf", "scantailor_then_ocr", "unpaper_tesseract"] = "ocrmypdf",
    input_dir: str | None = None,
    output_dir: str | None = None,
    root_dir: str = "/Users/elinacertova/Downloads/documents_dataset",
    lang: str | list[str] = "rus+eng",
) -> list[str]:
    cfg = PipelineConfig()
    if root_dir:
        cfg.paths.root_dir = root_dir
    src_dir = input_dir or cfg.paths.output_folder
    dst_dir = output_dir or os.path.join(cfg.paths.root_dir, "results", f"improved_{mode}")
    os.makedirs(dst_dir, exist_ok=True)

    outputs: list[str] = []
    pdfs = sorted(glob(os.path.join(src_dir, "*.pdf")))
    for pdf in pdfs:
        base_name = os.path.basename(pdf)
        if mode == "ocrmypdf":
            outp = os.path.join(dst_dir, base_name)
            try:
                outputs.append(process_with_ocrmypdf(pdf, outp, languages=lang))
            except Exception as e:
                print(f"[IMPROVE] FAIL ocrmypdf {pdf}: {e}")
        elif mode == "scantailor_then_ocr":
            outp = os.path.join(dst_dir, base_name)
            try:
                outputs.append(pipeline_scantailor_then_ocr(pdf, outp, lang=lang))
            except Exception as e:
                print(f"[IMPROVE] FAIL scantailor {pdf}: {e}")
        else:
            outp = os.path.join(dst_dir, base_name)
            try:
                outputs.append(unpaper_tesseract_pipeline(pdf, outp, lang=lang))
            except Exception as e:
                print(f"[IMPROVE] FAIL unpaper {pdf}: {e}")
    print(f"[IMPROVE] Done {mode}: {len(outputs)} files → {dst_dir}")
    return outputs


if __name__ == "__main__":
    pass


