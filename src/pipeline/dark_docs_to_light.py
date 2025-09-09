import os
import sys

from src.methods.classificator.dark_document_classifier import DarkDocumentClassifier
from src.methods.improver.document_lightener import PDFDocumentLightener
from src.pipeline.config import PipelineConfig
from src.pipeline.processing import classify_documents, copy_normal_documents, lighten_dark_documents


def dark_documents_to_light():
    print("=" * 60)
    print("ОБЪЕДИНЕННАЯ ОБРАБОТКА ДОКУМЕНТОВ")
    print("=" * 60)
    
    cfg = PipelineConfig()

    classifier = DarkDocumentClassifier(
        dpi=cfg.classifier.dpi,
        brightness_threshold=cfg.classifier.brightness_threshold,
        dark_pixels_threshold=cfg.classifier.dark_pixels_threshold,
        contrast_threshold=cfg.classifier.contrast_threshold,
        very_dark_pixels_threshold=cfg.classifier.very_dark_pixels_threshold,
        copy_to_dirs=cfg.classifier.copy_to_dirs,
        max_workers=cfg.classifier.max_workers,
    )

    lightener = PDFDocumentLightener(
        dpi=cfg.lightener.dpi,
        lighten_params=cfg.lightener.params,
    )

    input_folder = cfg.paths.input_folder
    output_folder = cfg.paths.output_folder
    dark_folder = cfg.paths.dark_folder
    
    print(f"Входная папка: {input_folder}")
    print(f"Выходная папка: {output_folder}")
    print(f"Папка темных: {dark_folder}")
    print()
    
    if not os.path.exists(input_folder):
        print(f"ОШИБКА: Входная папка не найдена: {input_folder}")
        return 1

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(dark_folder, exist_ok=True)
    
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
        lightening_results = lighten_dark_documents(
            dark_folder=dark_folder,
            output_folder=output_folder,
            lightener=lightener,
            lighten_params=cfg.lightener.params,
            passes=2,
        )
        print(f"   Обработано: {lightening_results['processed']}")
        print(f"   Успешно: {lightening_results['success']}")
        print(f"   Ошибки: {lightening_results['failed']}")
        if lightening_results['errors']:
            print(f"   Файлы с ошибками:")
            for error_file in lightening_results['errors']:
                print(f"     {error_file}")

        print("\n" + "=" * 60)
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
        print("=" * 60)
        print(f"Всего документов: {len(classification_results)}")
        print(f"Нормальные (скопированы): {normal_count}")
        print(f"Темные (осветлены): {len(dark_docs)}")
        print(f"Ошибки классификации: {len(error_docs)}")
        print(f"\nВсе обработанные документы сохранены в:")
        print(f"  {output_folder}")
        
        return 0
        
    except Exception as e:
        print(f"ОШИБКА: {e}")
        return 1
