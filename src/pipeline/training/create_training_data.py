
import os
import pandas as pd
from typing import Optional, Callable

from src.methods.classificator.classificator_easyocr import PDFQualityAssessorEasyOCR


def create_training_data(
    input_base_dir: str,
    output_csv_path: str,
    dpi: int = 400,
    blur_low: float = 800.0,
    min_roi_area_frac: float = 0.45,
    skew_bad_deg: float = 12.0,
    skew_warn_deg: float = 7.0,
    device: Optional[str] = None,
    max_workers: Optional[int] = 4,
    on_log: Optional[Callable[[str], None]] = None,
) -> int:
    """
    Создает обучающие данные для классификатора качества документов.

    Args:
        input_base_dir: Базовая папка, содержащая подпапки 'good', 'medium', 'failed'
        output_csv_path: Путь для сохранения CSV файла с метриками
        dpi: DPI для обработки PDF (по умолчанию 400, как в PDFQualityAssessorEasyOCR)
        blur_low: Минимальный порог размытия (по умолчанию 800.0, как в PDFQualityAssessorEasyOCR)
        min_roi_area_frac: Минимальная доля области интереса
        skew_bad_deg: Порог угла наклона (плохой)
        skew_warn_deg: Порог угла наклона (предупреждение)
        device: Устройство для OCR (cuda/cpu)
        max_workers: Количество потоков для параллельной обработки
        on_log: Функция для логирования (опционально)

    Returns:
        0 при успехе, 1 при ошибке
    """
    print("=== СОЗДАНИЕ ОБУЧАЮЩИХ ДАННЫХ ===")

    try:
        print("✓ Классификатор с EasyOCR загружен")
    except ImportError as e:
        print(f"✗ Ошибка импорта классификатора: {e}")
        return 1

    classifier = PDFQualityAssessorEasyOCR(
        dpi=dpi,
        copy_to_dirs=False,
        blur_low=blur_low,
        min_roi_area_frac=min_roi_area_frac,
        skew_bad_deg=skew_bad_deg,
        skew_warn_deg=skew_warn_deg,
        device=device,
        max_workers=max_workers,
        on_log=on_log,
    )

    base_path = os.path.abspath(input_base_dir)

    if not os.path.exists(base_path):
        print(f"✗ Папка не найдена: {base_path}")
        return 1

    print(f"✓ Папка найдена: {base_path}")

    all_data = []
    total_files = 0
    processed_files = 0

    for label in ['failed', 'medium', 'good']:
        label_path = os.path.join(base_path, label)

        if not os.path.exists(label_path):
            print(f"⚠ Папка {label} не найдена, пропускаем")
            continue

        print(f"\n--- Обрабатываем {label.upper()} ---")

        pdf_files = [f for f in os.listdir(label_path) if f.lower().endswith('.pdf')]
        total_files += len(pdf_files)

        print(f"Найдено {len(pdf_files)} PDF файлов")

        for i, pdf_file in enumerate(pdf_files):
            pdf_path = os.path.join(label_path, pdf_file)

            print(f"  [{i + 1}/{len(pdf_files)}] {pdf_file}...", end=" ")

            try:
                result = classifier.assess_pdf(pdf_path)

                row_data = {
                    'filename': pdf_file,
                    'true_label': label,
                    'predicted_label': result.category,
                    'reason': result.reason or '',
                    'avg_blur': result.avg_blur,
                    'median_ocr_conf': result.median_ocr_conf,
                    'mean_ocr_conf': result.mean_ocr_conf,
                    'pct80': result.pct80,
                    'words_count': result.words_count,
                    'text_density': result.text_density,
                    'roi_frac': result.roi_frac,
                    'core_frac': result.core_frac,
                    'is_table_like': result.is_table_like,
                    'avg_skew_deg': result.avg_skew_deg,

                    'bbox_area_text_frac': result.bbox_area_text_frac,
                    'conf_iqr': result.conf_iqr,
                    'line_height_med': result.line_height_med,
                    'line_height_var': result.line_height_var,
                    'line_spacing_med': result.line_spacing_med,
                    'line_spacing_var': result.line_spacing_var,
                    'text_blocks_count': result.text_blocks_count,
                    'avg_block_width': result.avg_block_width,
                    'avg_block_height': result.avg_block_height,

                    'error': result.error or '',
                    'correct': 1 if result.category == label else 0
                }

                all_data.append(row_data)
                processed_files += 1
                print("✓")

            except Exception as e:
                print(f"✗ Ошибка: {e}")
                error_row = {
                    'filename': pdf_file,
                    'true_label': label,
                    'predicted_label': 'error',
                    'reason': f'Processing error: {str(e)}',
                    'avg_blur': 0.0,
                    'median_ocr_conf': 0.0,
                    'mean_ocr_conf': 0.0,
                    'pct80': 0.0,
                    'words_count': 0,
                    'text_density': 0.0,
                    'roi_frac': 0.0,
                    'core_frac': 0.0,
                    'is_table_like': False,
                    'avg_skew_deg': 0.0,
                    'bbox_area_text_frac': 0.0,
                    'conf_iqr': 0.0,
                    'line_height_med': 0.0,
                    'line_height_var': 0.0,
                    'line_spacing_med': 0.0,
                    'line_spacing_var': 0.0,
                    'text_blocks_count': 0,
                    'avg_block_width': 0.0,
                    'avg_block_height': 0.0,

                    'error': str(e),
                    'correct': 0
                }
                all_data.append(error_row)
                processed_files += 1

    print(f"\n=== РЕЗУЛЬТАТЫ ===")
    print(f"Всего файлов: {total_files}")
    print(f"Обработано: {processed_files}")
    print(f"Ошибок: {total_files - processed_files}")

    if not all_data:
        print("✗ Нет данных для сохранения")
        return 1

    csv_path = os.path.abspath(output_csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    print(f"\nСохраняем данные в {csv_path}...")

    try:
        df = pd.DataFrame(all_data)

        df.to_csv(csv_path, index=False, encoding='utf-8')

        print(f"✓ Данные сохранены: {len(all_data)} строк")

        print(f"\n=== СТАТИСТИКА ===")
        print(f"Размер файла: {os.path.getsize(csv_path) / 1024:.1f} KB")

        for label in ['failed', 'medium', 'good']:
            label_data = df[df['true_label'] == label]
            if len(label_data) > 0:
                correct = len(label_data[label_data['correct'] == 1])
                print(f"{label}: {len(label_data)} файлов, {correct} правильно классифицированы")

        total_correct = df['correct'].sum()
        total_rows = len(df)
        accuracy = total_correct / total_rows if total_rows > 0 else 0
        print(f"Общая точность: {total_correct}/{total_rows} = {accuracy:.4f}")

        print(f"\n=== ПЕРВЫЕ 5 СТРОК ===")
        print(df.head().to_string())

        print(f"\n✓ Готово! CSV файл создан: {csv_path}")
        return 0

    except Exception as e:
        print(f"✗ Ошибка сохранения: {e}")
        return 1


