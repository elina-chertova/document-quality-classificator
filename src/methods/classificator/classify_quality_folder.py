"""
Классификация качества PDF в папке без сохранения CSV.
Выводит на экран: имя файла → категория и причина.
"""

import os
from typing import Tuple, List

from src.methods.classificator.classificator_extended import ExtendedPDFQualityAssessor


def classify_quality_folder(input_folder: str) -> List[Tuple[str, str, str]]:
    """
    Классифицирует все PDF в папке и возвращает список кортежей
    (filename, category, reason).
    """
    input_folder = os.path.abspath(input_folder)
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Папка не найдена: {input_folder}")

    files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    files.sort()

    assessor = ExtendedPDFQualityAssessor(
        dpi=400,
        copy_to_dirs=False,
        max_workers=4,
    )

    results: List[Tuple[str, str, str]] = []
    for index, fname in enumerate(files, start=1):
        pdf_path = os.path.join(input_folder, fname)
        res = assessor.assess_pdf(pdf_path)
        results.append((fname, res.category, res.reason))
        print(f"[{index}/{len(files)}] {fname} → {res.category.upper()} (why={res.reason})")

    cnt = {"trash": 0, "failed": 0, "medium": 0, "good": 0}
    for _, cat, _ in results:
        cnt[cat] = cnt.get(cat, 0) + 1
    print(f"\nИтог: trash={cnt.get('trash', 0)}, failed={cnt.get('failed', 0)}, medium={cnt.get('medium', 0)}, good={cnt.get('good', 0)}")

    return results


def classify_quality_and_copy(input_folder: str, output_folder: str) -> List[Tuple[str, str, str]]:
    """
    Классифицирует все PDF и раскладывает копии по подпапкам в output_folder
    (good/medium/failed/trash)
    """
    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Папка не найдена: {input_folder}")

    assessor = ExtendedPDFQualityAssessor(
        dpi=400,
        copy_to_dirs=True,
        max_workers=4,
    )

    results_objs = assessor.process_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        medium_subdir="medium",
        good_subdir="good",
        failed_subdir="failed",
        trash_subdir="trash",
    )

    results: List[Tuple[str, str, str]] = []
    for obj in results_objs:
        fname = os.path.basename(obj.pdf_path)
        results.append((fname, obj.category, obj.reason))

    cnt = {"trash": 0, "failed": 0, "medium": 0, "good": 0}
    for _, cat, _ in results:
        cnt[cat] = cnt.get(cat, 0) + 1
    print(f"Готово. Сохранено в {output_folder} → trash={cnt.get('trash', 0)}, failed={cnt.get('failed', 0)}, medium={cnt.get('medium', 0)}, good={cnt.get('good', 0)}")

    return results

