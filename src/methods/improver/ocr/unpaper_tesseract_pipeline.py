"""
Пайплайн: unpaper → tesseract постранично → склейка в PDF (pypdf).
Требуются системные бинарники: pdftoppm, unpaper, tesseract.
"""
from __future__ import annotations
import os
import subprocess
import tempfile
import shutil
from glob import glob
from typing import Iterable
from pypdf import PdfReader, PdfWriter


def _find_binary(candidates: Iterable[str]) -> str:
    for name in candidates:
        path = shutil.which(name)
        if path:
            return path
    raise FileNotFoundError(f"Не найден ни один из бинарников: {', '.join(candidates)}")


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def unpaper_tesseract_pipeline(
    input_pdf: str,
    output_pdf: str | None = None,
    lang: str = "rus+eng",
    dpi: int = 300,
    layout: str = "single",
    unpaper_extra: list[str] | None = None,
) -> str:
    pdftoppm = _find_binary(["pdftoppm"])
    unpaper = _find_binary(["unpaper"])
    tesseract = _find_binary(["tesseract"])

    if output_pdf is None:
        base = os.path.splitext(os.path.basename(input_pdf))[0]
        output_pdf = f"{base}__unpaper_ocr.pdf"
    if unpaper_extra is None:
        unpaper_extra = []

    with tempfile.TemporaryDirectory() as work:
        _run([pdftoppm, "-r", str(dpi), input_pdf, os.path.join(work, "page")])

        ppm_pages = sorted(glob(os.path.join(work, "page-*.ppm")))
        if not ppm_pages:
            raise RuntimeError("Не удалось разложить PDF на страницы (pdftoppm).")
        cleaned: list[str] = []
        for p in ppm_pages:
            base = os.path.splitext(os.path.basename(p))[0]
            out_img = os.path.join(work, f"{base}_clean.pgm")
            cmd = [
                unpaper,
                "--layout", layout,
                "--no-grayfilter",
                "--deskew-scan-direction", "left,right",
                "--interpolate", "cubic",
                p, out_img
            ]
            if unpaper_extra:
                cmd = [unpaper] + unpaper_extra + [
                    "--layout", layout, "--no-grayfilter", "--deskew-scan-direction", "left,right",
                    "--interpolate", "cubic",
                    p, out_img
                ]
            _run(cmd)
            cleaned.append(out_img)

        per_page_pdfs: list[str] = []
        for img in cleaned:
            out_base = img
            _run([tesseract, img, out_base, "-l", lang, "pdf"])
            per_page_pdfs.append(out_base + ".pdf")

        writer = PdfWriter()
        for p in per_page_pdfs:
            reader = PdfReader(p)
            for pg in reader.pages:
                writer.add_page(pg)
        with open(output_pdf, "wb") as f:
            writer.write(f)

    return output_pdf


