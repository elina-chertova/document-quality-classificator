"""
Пайплайн: ScanTailor CLI → сборка PDF → ocrmypdf.
Требуются системные бинарники: pdftoppm, scantailor-cli (или scantailor), img2pdf, ocrmypdf.
"""
from __future__ import annotations
import os
import subprocess
import tempfile
import shutil
from typing import Iterable
import img2pdf
import ocrmypdf


def _find_binary(candidates: Iterable[str]) -> str:
    for name in candidates:
        path = shutil.which(name)
        if path:
            return path
    raise FileNotFoundError(f"Не найден ни один из бинарников: {', '.join(candidates)}")


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def pipeline_scantailor_then_ocr(
    input_pdf: str,
    output_pdf: str | None = None,
    lang: str = "rus+eng",
    dpi: int = 300,
    margins: tuple[int, int, int, int] = (0, 0, 0, 0),
    enable_page_detection: bool = True,
    deskew: bool = True,
    content_detection: bool = True,
    ocrmypdf_optimize: int = 3,
    ocrmypdf_clean: bool = True,
    ocrmypdf_remove_background: bool = True,
    ocrmypdf_jobs: int = 0,
) -> str:
    pdftoppm = _find_binary(["pdftoppm"])
    scantailor = _find_binary(["scantailor-cli", "scantailor"])
    if output_pdf is None:
        base = os.path.splitext(os.path.basename(input_pdf))[0]
        output_pdf = f"{base}__st_ocr.pdf"

    with tempfile.TemporaryDirectory() as work:
        _run([pdftoppm, "-png", "-r", str(dpi), input_pdf, os.path.join(work, "page")])

        # Используем документированные ключи cli
        st_args = [
            scantailor,
            "--layout=0",
            "--deskew=auto" if deskew else "--deskew=manual",
            "--content-detection=normal" if content_detection else "--content-detection=cautious",
            f"--margins={margins[0]}",
            f"--alignment-vertical=center",
            f"--alignment-horizontal=center",
            f"--output-dpi={dpi}",
        ]
        st_args.extend([work, os.path.join(work, "out")])
        _run(st_args)

        images: list[str] = []
        for root, _, files in os.walk(os.path.join(work, "out")):
            for f in files:
                if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
                    images.append(os.path.join(root, f))
        images.sort()
        if not images:
            raise RuntimeError("ScanTailor не вернул обработанные изображения.")
        clean_pdf = os.path.join(work, "clean.pdf")
        with open(clean_pdf, "wb") as f:
            f.write(img2pdf.convert(images))

        # без fast_web_view для совместимости
        ocrmypdf.ocr(
            input_file=clean_pdf,
            output_file=output_pdf,
            language=lang,
            deskew=True,
            clean=ocrmypdf_clean,
            remove_background=ocrmypdf_remove_background,
            optimize=ocrmypdf_optimize,
            oversample=dpi,
            jobs=ocrmypdf_jobs,
            progress_bar=False,
        )
    return output_pdf


