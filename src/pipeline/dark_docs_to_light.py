import os
import sys
import shutil

from src.methods.classificator.dark_document_classifier import DarkDocumentClassifier
from src.methods.improver.document_lightener import PDFDocumentLightener
from src.methods.improver.bilateral_lightener import PDFBilateralLightener
from src.pipeline.config import PipelineConfig
from src.pipeline.processing import classify_documents, copy_normal_documents, lighten_dark_documents


def dark_documents_to_light(
    input_folder: str,
    output_folder: str,
    dark_folder: str,
    combined_output_folder: str | None = None,
    dpi: int = 200,
    brightness_threshold: float = 100.0,
    dark_pixels_threshold: float = 0.3,
    contrast_threshold: float = 40.0,
    very_dark_pixels_threshold: float = 0.1,
    copy_to_dirs: bool = True,
    max_workers: int = 4,
    lightener_dpi: int = 300,
    passes: int = 2,
    lightening_method: str = "bilateral_filter",  # "bilateral_filter" или "original"
):
    print("=" * 60)
    print("ОБЪЕДИНЕННАЯ ОБРАБОТКА ДОКУМЕНТОВ")
    print("=" * 60)
    
    cfg = PipelineConfig()

    classifier = DarkDocumentClassifier(
        dpi=dpi,
        brightness_threshold=brightness_threshold,
        dark_pixels_threshold=dark_pixels_threshold,
        contrast_threshold=contrast_threshold,
        very_dark_pixels_threshold=very_dark_pixels_threshold,
        copy_to_dirs=copy_to_dirs,
        max_workers=max_workers,
    )

    # Выбор метода осветления
    if lightening_method == "bilateral_filter":
        lightener = PDFBilateralLightener(dpi=lightener_dpi)
        print(f"Метод осветления: bilateral_filter (лучший по тестам)")
    else:
        lightener = PDFDocumentLightener(
            dpi=lightener_dpi,
            lighten_params=cfg.lightener.params,
        )
        print(f"Метод осветления: original")
    
    print(f"Входная папка: {input_folder}")
    print(f"Выходная папка: {output_folder}")
    print(f"Папка темных: {dark_folder}")
    print()
    
    if not os.path.exists(input_folder):
        print(f"ОШИБКА: Входная папка не найдена: {input_folder}")
        return 1

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(dark_folder, exist_ok=True)
    if combined_output_folder:
        os.makedirs(combined_output_folder, exist_ok=True)
    
    try:
        print("1. Классификация документов...")
        classification_results, dark_docs, normal_docs, error_docs = classify_documents(
            input_folder=input_folder,
            dark_folder=dark_folder,
            classifier=classifier,
        )
        
        print(f"   Нормальные: {len(normal_docs)}")
        print(f"   Темные: {len(dark_docs)}")
        print(f"   Ошибки: {len(error_docs)}")
        
        print("\n2. Копирование нормальных документов...")
        normal_count = copy_normal_documents(normal_docs, output_folder)
        print(f"   Скопировано: {normal_count}")
        
        print("\n3. Осветление темных документов...")

        if lightening_method == "bilateral_filter":
            lightening_results = lightener.process_dark_folder(
                input_folder=os.path.join(dark_folder, "dark_documents"),
                output_folder=output_folder
            )
        else:
            lightening_results = lighten_dark_documents(
                dark_folder=dark_folder,
                output_folder=output_folder,
                lightener=lightener,
                lighten_params=cfg.lightener.params,
                passes=passes,
            )
        print(f"   Обработано: {lightening_results['processed']}")
        print(f"   Успешно: {lightening_results['success']}")
        print(f"   Ошибки: {lightening_results['failed']}")
        if lightening_results['errors']:
            print(f"   Файлы с ошибками:")
            for error_file in lightening_results['errors']:
                print(f"     {error_file}")

        if combined_output_folder:
            print("\n4. Копирование всех результатов в комбинированную папку...")
            copied_count = 0
            for fname in os.listdir(output_folder):
                if fname.lower().endswith('.pdf'):
                    src = os.path.join(output_folder, fname)
                    dst = os.path.join(combined_output_folder, fname)
                    try:
                        shutil.copy2(src, dst)
                        copied_count += 1
                    except Exception as e:
                        print(f"   [ОШИБКА] Не удалось скопировать {fname}: {e}")
            print(f"   Скопировано в комбинированную папку: {copied_count}")

        print("\n" + "=" * 60)
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
        print("=" * 60)
        print(f"Всего документов: {len(classification_results)}")
        print(f"Нормальные (скопированы): {normal_count}")
        print(f"Темные (осветлены): {len(dark_docs)}")
        print(f"Ошибки классификации: {len(error_docs)}")
        print(f"\nВсе обработанные документы сохранены в:")
        print(f"  {output_folder}")
        if combined_output_folder:
            print(f"\nКомбинированные результаты также сохранены в:")
            print(f"  {combined_output_folder}")
        
        return 0
        
    except Exception as e:
        print(f"ОШИБКА: {e}")
        return 1
