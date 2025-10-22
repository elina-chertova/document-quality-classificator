import os
import csv
import shutil
from pathlib import Path

from src.pipeline.split_pages import split_pages
from src.pipeline.rotate_right_runner import rotate_right
from src.pipeline.deskew import deskew_documents
from src.pipeline.remove_lines_runner import remove_lines
from src.pipeline.dark_docs_to_light import dark_documents_to_light
from src.pipeline.enhance_contrast_runner import enhance_contrast_documents
from src.methods.classificator.classificator_easyocr import PDFQualityAssessorEasyOCR


def process_single_document(
    input_pdf_path: str,
    output_base_dir: str,
    output_csv_path: str,
    dpi: int = 400,
    max_workers: int = 4,
):
    input_pdf_path = os.path.abspath(input_pdf_path)
    output_base_dir = os.path.abspath(output_base_dir)
    output_csv_path = os.path.abspath(output_csv_path)
    
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
    contrast_enhanced_dir = os.path.join(output_base_dir, "contrast_enhanced")
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
    
    print("\n3. Выравнивание текста...")
    deskew_documents(input_dir=rotated_dir, output_dir=deskewed_dir, failed_dir=failed_dir)
    
    print("\n4. Удаление линий...")
    remove_lines(
        input_dir=deskewed_dir,
        lines_cleaned_folder=lines_cleaned_dir,
        no_lines_ok_folder=no_lines_dir,
        combined_output_folder=combined_dir
    )
    
    print("\n5. Осветление темных документов...")
    dark_documents_to_light(
        input_folder=combined_dir,
        output_folder=lightened_dir,
        dark_folder=dark_dir,
        combined_output_folder=lightened_combined_dir
    )
    
    print("\n6. Усиление контраста...")
    enhance_contrast_documents(
        input_dir=lightened_combined_dir,
        output_dir=contrast_enhanced_dir
    )
    
    print("\n7. Классификация документов...")
    assessor = PDFQualityAssessorEasyOCR(
        dpi=dpi,
        copy_to_dirs=False,
        max_workers=max_workers,
    )
    
    results = []
    files = sorted([f for f in os.listdir(contrast_enhanced_dir) if f.lower().endswith('.pdf')])
    
    for fname in files:
        pdf_path = os.path.join(contrast_enhanced_dir, fname)
        try:
            result = assessor.assess_pdf(pdf_path)
            page_num = fname.replace('.pdf', '').split('_page_')[-1] if '_page_' in fname else '1'
            results.append({
                'document': document_name,
                'page': page_num,
                'category': result.category,
                'reason': result.reason,
                'confidence': result.median_ocr_conf,
                'words': result.words_count,
            })
            print(f"   {fname} → {result.category.upper()}")
        except Exception as e:
            print(f"   [ОШИБКА] {fname}: {e}")
            page_num = fname.replace('.pdf', '').split('_page_')[-1] if '_page_' in fname else '1'
            results.append({
                'document': document_name,
                'page': page_num,
                'category': 'error',
                'reason': str(e),
                'confidence': 0.0,
                'words': 0,
            })
    
    print(f"\n8. Сохранение результатов в CSV...")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['document', 'page', 'category', 'reason', 'confidence', 'words']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"   Результаты сохранены: {output_csv_path}")
    
    print("\n" + "=" * 60)
    print("ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 60)
    print(f"Всего страниц: {len(results)}")
    
    category_counts = {}
    for r in results:
        cat = r['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")
    
    return results

