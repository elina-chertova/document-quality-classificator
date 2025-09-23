"""
Инструменты на базе ocrmypdf: одиночная обработка и батч по директории.
Требуются системные зависимости ocrmypdf и tesseract с моделями языков.
"""
import os
import shutil
from glob import glob
from typing import List, Sequence

import ocrmypdf


def process_with_ocrmypdf(
    input_pdf: str,
    output_pdf: str | None = None,
    languages: Sequence[str] = ("rus", "eng"),
    oversample_dpi: int = 300,
    optimize: int = 1,
    deskew: bool = True,
    clean: bool = False,
    clean_final: bool = False,
    remove_background: bool = True,
    rotate_pages: bool = True,
    tesseract_timeout: int = 300,
    progress_bar: bool = False,
    skip_text: bool = False,
    force_ocr: bool = False,
    jobs: int | None = None,
) -> str:
    if output_pdf is None:
        base = os.path.splitext(os.path.basename(input_pdf))[0]
        output_pdf = f"{base}__ocr.pdf"

    # Автоотключение clean, если нет unpaper в PATH
    if (clean or clean_final) and not shutil.which("unpaper"):
        print("[ocrmypdf] 'unpaper' не найден в PATH — отключаю clean/clean_final")
        clean = False
        clean_final = False

    # Если нет pngquant (используется при optimize>=1), отключаем оптимизацию
    if optimize and optimize > 0 and not shutil.which("pngquant"):
        print("[ocrmypdf] 'pngquant' не найден в PATH — отключаю optimize>0 → optimize=0")
        optimize = 0

    # Приведение языков: допускаем 'rus+eng' или 'rus,eng'
    if isinstance(languages, str):
        raw = languages.replace(",", "+")
        languages = tuple(p.strip() for p in raw.split("+") if p.strip()) or ("eng",)

    # Формируем общие kwargs
    kwargs = dict(
        input_file=input_pdf,
        output_file=output_pdf,
        language=list(languages),
        output_type="pdfa",
        rotate_pages=rotate_pages,
        deskew=deskew,
        remove_background=remove_background,
        clean=clean,
        clean_final=clean_final,
        oversample=oversample_dpi,
        optimize=optimize,
        tesseract_timeout=tesseract_timeout,
        progress_bar=progress_bar,
        skip_text=skip_text,
        force_ocr=force_ocr,
    )


    # jobs: если None — не передаём (OCRmyPDF сам выберет), иначе задаём число
    if jobs is not None and isinstance(jobs, int) and jobs > 0:
        kwargs["jobs"] = jobs

    # Вызов OCRmyPDF с авто-ретраем без remove_background для совместимости версий
    try:
        ocrmypdf.ocr(**kwargs)
    except NotImplementedError as e:
        msg = str(e)
        if "--remove-background is temporarily not implemented" in msg and kwargs.get("remove_background"):
            kwargs["remove_background"] = False
            ocrmypdf.ocr(**kwargs)
        else:
            raise
    return output_pdf


def batch_ocrmypdf_dir(
    input_dir: str,
    output_dir: str,
    languages: Sequence[str] = ("rus", "eng"),
    oversample_dpi: int = 300,
    optimize: int = 1,
    deskew: bool = True,
    clean: bool = False,
    clean_final: bool = False,
    remove_background: bool = True,
    rotate_pages: bool = True,
    jobs: int | None = None,
    skip_text: bool = True,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    processed: List[str] = []
    for pdf in sorted(glob(os.path.join(input_dir, "*.pdf"))):
        base = os.path.splitext(os.path.basename(pdf))[0]
        outp = os.path.join(output_dir, f"{base}__ocr.pdf")
        try:
            process_with_ocrmypdf(
                pdf,
                outp,
                languages=languages,
                oversample_dpi=oversample_dpi,
                optimize=optimize,
                deskew=deskew,
                clean=clean,
                clean_final=clean_final,
                remove_background=remove_background,
                rotate_pages=rotate_pages,
                jobs=jobs,
                skip_text=skip_text,
                progress_bar=False,
            )
            processed.append(outp)
        except Exception as e:
            print(f"[WARN] ошибка на {pdf}: {e}")
    return processed


