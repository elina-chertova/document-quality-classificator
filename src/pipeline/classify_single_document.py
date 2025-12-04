from src.methods.classificator.classificator_easyocr import PDFQualityAssessorEasyOCR
from PyPDF2 import PdfReader, PdfWriter
import os
import csv

from src.methods.classificator.classificator_easyocr_optimized import PDFQualityAssessorEasyOCROptimized


def classify_single_document(
    pages_dir: str,
    document_name: str,
    output_base_dir: str,
    output_csv_path: str,
    dpi: int = 400,
    max_workers: int = 4,
    classifier_dpi: int | None = 300,
    device: str | None = None,
    optimized: bool = False
) -> list[dict]:
    pages_dir = os.path.abspath(pages_dir)
    output_base_dir = os.path.abspath(output_base_dir)
    output_csv_path = os.path.abspath(output_csv_path)

    print("\n8. Классификация документов...")
    class_dpi = classifier_dpi if classifier_dpi is not None else min(dpi, 300)
    if optimized:
        assessor = PDFQualityAssessorEasyOCROptimized(
            dpi=class_dpi,
            copy_to_dirs=False,
            max_workers=max_workers,
            device=device,
        )
    else:
        assessor = PDFQualityAssessorEasyOCR(
            dpi=class_dpi,
            copy_to_dirs=False,
            max_workers=max_workers,
            device=device,
        )

    results: list[dict] = []
    files = sorted(f for f in os.listdir(pages_dir) if f.lower().endswith(".pdf"))

    for fname in files:
        pdf_path = os.path.join(pages_dir, fname)
        try:
            result = assessor.assess_pdf(pdf_path)

            page_num = (
                fname.replace(".pdf", "").split("_page_")[-1]
                if "_page_" in fname
                else "1"
            )
            results.append(
                {
                    "document": document_name,
                    "page": page_num,
                    "category": result.category,
                    "reason": result.reason,
                    "confidence": result.median_ocr_conf,
                    "words": result.words_count,
                }
            )
            print(f"   {fname} → {result.category.upper()}")
        except Exception as e:
            print(f"   [ОШИБКА] {fname}: {e}")
            page_num = (
                fname.replace(".pdf", "").split("_page_")[-1]
                if "_page_" in fname
                else "1"
            )
            results.append(
                {
                    "document": document_name,
                    "page": page_num,
                    "category": "error",
                    "reason": str(e),
                    "confidence": 0.0,
                    "words": 0,
                }
            )

    print(f"\n9. Сохранение результатов в CSV...")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["document", "page", "category", "reason", "confidence", "words"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"   Результаты сохранены: {output_csv_path}")

    print(f"\n10. Сборка страниц в итоговый PDF...")
    final_dir = os.path.join(output_base_dir, "final_documents")
    merged_pdf_path = os.path.join(final_dir, f"{document_name}.pdf")

    writer = PdfWriter()
    for fname in files:
        pdf_path = os.path.join(pages_dir, fname)
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                writer.add_page(page)
        except Exception as e:
            print(f"[WARNING] Пропускаем {fname} при объединении: {e}")

    if len(writer.pages) > 0:
        os.makedirs(os.path.dirname(merged_pdf_path), exist_ok=True)
        with open(merged_pdf_path, "wb") as f:
            writer.write(f)
        print(f"   Итоговый документ сохранён: {merged_pdf_path}")
    else:
        print("   [ПРЕДУПРЕЖДЕНИЕ] Не удалось собрать итоговый PDF (страницы отсутствуют?)")

    print("\n" + "=" * 60)
    print("ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 60)
    print(f"Всего страниц: {len(results)}")

    category_counts: dict[str, int] = {}
    for r in results:
        cat = r["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")

    return results