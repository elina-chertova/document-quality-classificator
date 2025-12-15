import os
import csv
import shutil
from pathlib import Path

from PyPDF2 import PdfReader, PdfWriter

from src.pipeline.split_pages import split_pages
from src.pipeline.rotate_right_runner import rotate_right
from src.pipeline.deskew import deskew_documents
from src.pipeline.remove_lines_runner import remove_lines
from src.pipeline.dark_docs_to_light import dark_documents_to_light
from src.pipeline.text_enhancement_runner import enhance_text_documents
from src.methods.classificator.classificator_easyocr import PDFQualityAssessorEasyOCR
from src.methods.classificator.classificator_easyocr_optimized import PDFQualityAssessorEasyOCROptimized


def process_single_document_smart(
    input_pdf_path: str,
    output_base_dir: str,
    output_csv_path: str,
    dpi: int = 400,
    max_workers: int = 4,
    classifier_dpi: int | None = 300,
    device: str | None = None,
    optimized: bool = False,
    compare_ocr: bool = True,
):
    input_pdf_path = os.path.abspath(input_pdf_path)
    output_base_dir = os.path.abspath(output_base_dir)
    output_csv_path = os.path.abspath(output_csv_path)

    if os.path.exists(output_base_dir):
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)

    if not os.path.exists(input_pdf_path):
        raise FileNotFoundError(f"Файл не найден: {input_pdf_path}")

    document_name = Path(input_pdf_path).stem

    temp_input_dir = os.path.join(output_base_dir, "temp_input")
    splitted_dir = os.path.join(output_base_dir, "splitted")
    rotated_dir = os.path.join(output_base_dir, "rotated")
    deskewed_dir = os.path.join(output_base_dir, "deskewed")
    lines_cleaned_dir = os.path.join(output_base_dir, "lines_cleaned")
    no_lines_dir = os.path.join(output_base_dir, "lines_not_detected")
    combined_dir = os.path.join(output_base_dir, "combined")
    lightened_dir = os.path.join(output_base_dir, "lightened")
    dark_dir = os.path.join(output_base_dir, "dark")
    lightened_combined_dir = os.path.join(output_base_dir, "lightened_combined")
    text_enhanced_dir = os.path.join(output_base_dir, "text_enhanced")
    final_dir = os.path.join(output_base_dir, "final_documents")
    failed_dir = os.path.join(output_base_dir, "failed")

    os.makedirs(temp_input_dir, exist_ok=True)

    temp_pdf = os.path.join(temp_input_dir, os.path.basename(input_pdf_path))
    shutil.copy2(input_pdf_path, temp_pdf)

    print("=" * 60)
    print(f"ОБРАБОТКА ДОКУМЕНТА: {document_name}")
    print("=" * 60)

    print("\n1. Разделение по страницам...")
    split_pages(input_dir=temp_input_dir, output_dir=splitted_dir)

    print("\n2. Поворот изображений...")
    rotate_right(input_dir=splitted_dir, output_dir=rotated_dir, failed_dir=failed_dir)

    print("\n3. Классификация качества документов...")
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

    classification_results: list[dict] = []
    files = sorted(f for f in os.listdir(rotated_dir) if f.lower().endswith(".pdf"))

    good_pages: list[str] = []
    pages_to_enhance: list[str] = []
    page_to_result_map: dict[str, dict] = {}

    for fname in files:
        pdf_path = os.path.join(rotated_dir, fname)
        try:
            result = assessor.assess_pdf(pdf_path)

            page_num = (
                fname.replace(".pdf", "").split("_page_")[-1]
                if "_page_" in fname
                else "1"
            )
            result_dict = {
                "document": document_name,
                "page": page_num,
                "category": result.category,
                "reason": result.reason,
                "confidence_before": result.median_ocr_conf,
                "confidence_after": result.median_ocr_conf,
                "confidence_improvement": 0.0,
                "words_before": result.words_count,
                "words_after": result.words_count,
                "words_improvement": 0,
            }
            classification_results.append(result_dict)
            page_to_result_map[fname] = result_dict

            if result.category == "good":
                good_pages.append(fname)
                print(f"   {fname} → GOOD (confidence: {result.median_ocr_conf:.1f}%, пропускаем улучшения)")
            else:
                pages_to_enhance.append(fname)
                print(f"   {fname} → {result.category.upper()} (confidence: {result.median_ocr_conf:.1f}%, будет улучшено)")

        except Exception as e:
            print(f"   [ОШИБКА] {fname}: {e}")
            page_num = (
                fname.replace(".pdf", "").split("_page_")[-1]
                if "_page_" in fname
                else "1"
            )
            result_dict = {
                "document": document_name,
                "page": page_num,
                "category": "error",
                "reason": str(e),
                "confidence_before": 0.0,
                "confidence_after": 0.0,
                "confidence_improvement": 0.0,
                "words_before": 0,
                "words_after": 0,
                "words_improvement": 0,
            }
            classification_results.append(result_dict)
            page_to_result_map[fname] = result_dict
            pages_to_enhance.append(fname)

    print(f"\n   Статистика: {len(good_pages)} хороших, {len(pages_to_enhance)} требуют улучшения")

    os.makedirs(final_dir, exist_ok=True)

    if good_pages:
        print(f"\n4. Копирование хороших страниц в финальную папку...")
        for fname in good_pages:
            src = os.path.join(rotated_dir, fname)
            dst = os.path.join(final_dir, fname)
            shutil.copy2(src, dst)

    if pages_to_enhance:
        print(f"\n5. Улучшение страниц среднего/плохого качества...")

        pages_to_enhance_dir = os.path.join(output_base_dir, "pages_to_enhance")
        os.makedirs(pages_to_enhance_dir, exist_ok=True)

        for fname in pages_to_enhance:
            src = os.path.join(rotated_dir, fname)
            dst = os.path.join(pages_to_enhance_dir, fname)
            shutil.copy2(src, dst)

        print("\n   5.1. Выравнивание текста...")
        deskewed_enhance_dir = os.path.join(output_base_dir, "deskewed_enhance")
        deskew_documents(input_dir=pages_to_enhance_dir, output_dir=deskewed_enhance_dir, failed_dir=failed_dir)

        print("\n   5.2. Удаление линий...")
        lines_cleaned_enhance_dir = os.path.join(output_base_dir, "lines_cleaned_enhance")
        no_lines_enhance_dir = os.path.join(output_base_dir, "lines_not_detected_enhance")
        combined_enhance_dir = os.path.join(output_base_dir, "combined_enhance")
        remove_lines(
            input_dir=deskewed_enhance_dir,
            lines_cleaned_folder=lines_cleaned_enhance_dir,
            no_lines_ok_folder=no_lines_enhance_dir,
            combined_output_folder=combined_enhance_dir
        )

        print("\n   5.3. Осветление темных документов...")
        lightened_enhance_dir = os.path.join(output_base_dir, "lightened_enhance")
        dark_enhance_dir = os.path.join(output_base_dir, "dark_enhance")
        lightened_combined_enhance_dir = os.path.join(output_base_dir, "lightened_combined_enhance")
        dark_meta = dark_documents_to_light(
            input_folder=combined_enhance_dir,
            output_folder=lightened_enhance_dir,
            dark_folder=dark_enhance_dir,
            combined_output_folder=lightened_combined_enhance_dir,
            lightening_method='bilateral_filter'
        )
        dark_meta = dark_meta or {}
        dark_filenames = set(dark_meta.get("dark_docs", []))

        print("\n   5.4. Усиление текста...")
        text_enhanced_enhance_dir = os.path.join(output_base_dir, "text_enhanced_enhance")
        enhance_text_documents(
            input_dir=lightened_combined_enhance_dir,
            output_dir=text_enhanced_enhance_dir,
            skip_filenames=dark_filenames
        )

        if compare_ocr:
            print("\n   5.5. Оценка OCR качества после улучшения...")
            enhanced_dir_for_ocr = text_enhanced_enhance_dir
            if not os.path.exists(enhanced_dir_for_ocr) or len(os.listdir(enhanced_dir_for_ocr)) == 0:
                enhanced_dir_for_ocr = lightened_combined_enhance_dir

            for fname in pages_to_enhance:
                enhanced_pdf_path = None
                for check_dir in [text_enhanced_enhance_dir, lightened_combined_enhance_dir]:
                    candidate = os.path.join(check_dir, fname)
                    if os.path.exists(candidate):
                        enhanced_pdf_path = candidate
                        break

                if enhanced_pdf_path and os.path.exists(enhanced_pdf_path):
                    try:
                        result_after = assessor.assess_pdf(enhanced_pdf_path)
                        if fname in page_to_result_map:
                            result_dict = page_to_result_map[fname]
                            conf_before = result_dict["confidence_before"]
                            conf_after = result_after.median_ocr_conf
                            improvement = conf_after - conf_before
                            
                            words_before = result_dict["words_before"]
                            words_after = result_after.words_count
                            words_improvement = words_after - words_before

                            result_dict["confidence_after"] = conf_after
                            result_dict["confidence_improvement"] = improvement
                            result_dict["words_after"] = words_after
                            result_dict["words_improvement"] = words_improvement
                            
                            print(f"      {fname}: {conf_before:.1f}% → {conf_after:.1f}% ({improvement:+.1f}%)")
                    except Exception as e:
                        print(f"      [WARNING] Не удалось оценить OCR для {fname} после улучшения: {e}")
        else:
            for fname in pages_to_enhance:
                if fname in page_to_result_map:
                    result_dict = page_to_result_map[fname]
                    result_dict["confidence_after"] = result_dict["confidence_before"]
                    result_dict["confidence_improvement"] = 0.0
                    result_dict["words_after"] = result_dict["words_before"]
                    result_dict["words_improvement"] = 0

        print("\n   5.6. Копирование улучшенных страниц в финальную папку...")
        for fname in pages_to_enhance:
            src = os.path.join(text_enhanced_enhance_dir, fname)
            if os.path.exists(src):
                dst = os.path.join(final_dir, fname)
                shutil.copy2(src, dst)
            else:
                src_fallback = os.path.join(lightened_combined_enhance_dir, fname)
                if os.path.exists(src_fallback):
                    dst = os.path.join(final_dir, fname)
                    shutil.copy2(src_fallback, dst)
                else:
                    print(f"   [WARNING] Не найден улучшенный файл для {fname}, используем оригинал")
                    src_original = os.path.join(rotated_dir, fname)
                    if os.path.exists(src_original):
                        dst = os.path.join(final_dir, fname)
                        shutil.copy2(src_original, dst)

    print(f"\n6. Сохранение результатов классификации в CSV...")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "document", "page", "category", "reason",
            "confidence_before", "confidence_after", "confidence_improvement",
            "words_before", "words_after", "words_improvement"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(classification_results)

    print(f"   Результаты сохранены: {output_csv_path}")

    improved_pages = [r for r in classification_results if r["confidence_improvement"] > 0]
    degraded_pages = [r for r in classification_results if r["confidence_improvement"] < 0]
    unchanged_pages = [r for r in classification_results if r["confidence_improvement"] == 0 and r["category"] != "error"]
    
    ocr_comparison_stats = {
        "improved_count": len(improved_pages),
        "degraded_count": len(degraded_pages),
        "unchanged_count": len(unchanged_pages),
        "avg_improvement": 0.0,
        "max_improvement": 0.0,
        "avg_degradation": 0.0,
        "total_pages_compared": len([r for r in classification_results if r["category"] != "error"]) if compare_ocr else 0,
    }
    
    if improved_pages:
        ocr_comparison_stats["avg_improvement"] = sum(r["confidence_improvement"] for r in improved_pages) / len(improved_pages)
        ocr_comparison_stats["max_improvement"] = max(r["confidence_improvement"] for r in improved_pages)
    if degraded_pages:
        ocr_comparison_stats["avg_degradation"] = sum(abs(r["confidence_improvement"]) for r in degraded_pages) / len(degraded_pages)
    
    if compare_ocr and (improved_pages or degraded_pages or unchanged_pages):
        print(f"\n   Статистика улучшений OCR:")
        if improved_pages:
            print(f"      Улучшено: {ocr_comparison_stats['improved_count']} страниц (среднее: +{ocr_comparison_stats['avg_improvement']:.1f}%, максимум: +{ocr_comparison_stats['max_improvement']:.1f}%)")
        if degraded_pages:
            print(f"      Ухудшилось: {ocr_comparison_stats['degraded_count']} страниц (среднее: -{ocr_comparison_stats['avg_degradation']:.1f}%)")
        if unchanged_pages:
            print(f"      Без изменений: {ocr_comparison_stats['unchanged_count']} страниц")

    print(f"\n7. Сборка страниц в итоговый PDF...")
    merged_pdf_path = os.path.join(final_dir, f"{document_name}.pdf")

    writer = PdfWriter()
    final_files = sorted(f for f in os.listdir(final_dir) if f.lower().endswith(".pdf"))
    for fname in final_files:
        pdf_path = os.path.join(final_dir, fname)
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
    print(f"Всего страниц: {len(classification_results)}")

    category_counts: dict[str, int] = {}
    for r in classification_results:
        cat = r["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")

    return {
        "results": classification_results,
        "ocr_comparison": ocr_comparison_stats,
        "total_pages": len(classification_results),
        "category_counts": category_counts,
    }

