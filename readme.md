Установка:
```bash
pip install -r requirements.txt
```



## OCR и улучшение качества

### OCR (Surya)
```python
from src.pipeline.ocr_runner import run_ocr
run_ocr(root_dir="/Users/elinacertova/Downloads/documents_dataset")
```

### OCR (VLLM/Qwen, первая страница → txt)
```python
from src.pipeline.vllm_ocr_runner import run_vllm_ocr
run_vllm_ocr(root_dir="/Users/elinacertova/Downloads/documents_dataset")
```

### Дополнительные методы улучшения текста

```python

from src.pipeline.quality_improvements_runner import improve_dataset
improve_dataset(mode="ocrmypdf", root_dir="/Users/elinacertova/Downloads/documents_dataset")
improve_dataset(mode="scantailor_then_ocr", root_dir="/Users/elinacertova/Downloads/documents_dataset")  # пока не работает
improve_dataset(mode="unpaper_tesseract", root_dir="/Users/elinacertova/Downloads/documents_dataset")  # пока не работает
```


### Подсчёт CER
```python
from src.pipeline.cer_runner import run_cer
run_cer(
    hyp_dir="/path/to/hyp_txt",   # распознанные тексты
    ref_dir="/path/to/ref_txt",   # эталоны (.txt) с теми же именами
    csv_out="/path/to/cer_results.csv",
)
```

### Перебор параметров и CER (quality sweep) - пока не тестировать
```python
from src.pipeline.quality_sweep import run_quality_sweep
run_quality_sweep(
    input_dir="/path/to/input_pdfs",    # папка с PDF
    refs_dir="/path/to/refs_txt",      # эталонные .txt с теми же именами
    root_dir="/Users/elinacertova/Downloads/documents_dataset",
)
```
Результаты: `…/results/sweep/ocrmypdf_*`, для каждой комбинации создаётся `cer.csv` и выводится сводка в stdout.

### Ключевые файлы и точки входа

- Конфигурация путей/параметров: `src/pipeline/config.py`
- Оркестратор пайплайна (split → rotate → deskew → финальная обработка): `src/pipeline/main.py`
- Разделение на страницы: `src/pipeline/split_pages.py` (функция `split_pages`)
- Поворот на 90/180/270: `src/pipeline/rotate_right_runner.py` (функция `rotate_right`)
- Автодескью (< 90°): `src/pipeline/deskew.py` (функция `deskew_documents`)
- Пороговый классификатор (без ML): `src/methods/classificator/classificator.py` (`PDFQualityAssessor`)
- Расширенный классификатор (расширенные метрики, для подготовки CSV): `src/methods/classificator/classificator_extended.py`
- Подготовка обучающего CSV: `create_training_data.py`
- Тюнинг и сохранение лучшей ML‑модели: `tune_extended_classifier.py` (сохраняет `final_quality_classifier_model.pkl`)

### Пути (меняются в одном месте)

Все пути собраны в `src/pipeline/config.py` в `PathsConfig`:
- `root_dir`: базовая директория датасета
- Производные пути:
  - `full_dataset_folder` — исходные PDF (до сплита)
  - `split_folder` — результат сплита
  - `rotated_folder` — повёрнутые на 90/180/270
  - `failed_rotate_folder` — ошибки поворота
  - `deskewed_folder` — выровненные под углами < 90°
  - `failed_deskew_folder` — ошибки deskew
  - `input_folder` — вход для финальной обработки (обычно `deskewed_folder`)
  - `output_folder` — итоговые обработанные документы
  - `dark_folder` — папка для тёмных (если нужно)
  - `example_quality_base` — база примеров качества (good/medium/failed)
  - `training_csv_path` — путь к обучающему CSV
  - `trained_model_path` — путь для сохранённой ML‑модели (`final_quality_classifier_model.pkl`)

Обновите `root_dir` один раз — остальные пути пересчитаются автоматически.


## Пайплайн: порядок запуска и пути

1. Запуск оркестратора (используются пути из `config.py`):
```bash
python -m src.pipeline.main
```
Внутри выполняются:
1) split страниц → 2) rotate 90/180/270 → 3) deskew < 90° → 4) финальная обработка/копирование.


### Пороговая классификация (без ML)

Разложить документы по подпапкам `good/medium/failed/trash` c помощью простого порогового классификатора:
```python
from src.methods.classificator.classificator import PDFQualityAssessor

assessor = PDFQualityAssessor(
    dpi=400,
    copy_to_dirs=True,
    max_workers=4,
)

assessor.process_folder(
    input_folder="/Users/elinacertova/Downloads/documents_dataset/results/processed",
    output_folder="/Users/elinacertova/Downloads/documents_dataset/results/output_sorted",
    medium_subdir="medium",
    good_subdir="good",
    failed_subdir="failed",
    trash_subdir="trash",
)
```
или
```yaml
python simple_classificator.py
```


### Подготовка обучающих данных (CSV)
Для того, чтобы код корректно работал, необходимо заранее иметь директорию с субдиректориями good, medium, failed (используются как true labels).

Создаёт `classification_analysis.csv`  ( используются папки`good/medium/failed`):
```bash
python create_training_data.py
```
Пути `example_quality_base` и `training_csv_path` берутся из `config.py`.

### Тюнинг и сохранение лучшей ML‑модели

Подбирает пороги и ML‑модели, выводит метрики и сохраняет лучшую модель:
```bash
python tune_extended_classifier.py
```
Итог сохраняется в `final_quality_classifier_model.pkl` (путь в `config.py → trained_model_path`).


### Инференс
```yaml
python src/pipeline/inference.py
```

### Сравнение pytesseract и Surya
```bash
python -m src.compare_ocr_backends
```
Результат: создаются подпапки `tesseract/` и `surya/` c TXT, и `compare.csv` с длинами текстов и CER (если указаны эталоны).


## Детекция печатей (YOLO Stamp Detector)

### Детекция печатей на одном изображении
```python
from src.methods.detector import detect_stamps_single

result = detect_stamps_single(
    image_path="document.pdf",
    conf_threshold=0.8,  # По умолчанию 0.8 (80%)
    visualize=True,
)
print(f"Найдено печатей: {result['num_stamps']}")
```

### Детекция печатей в папке
```python
from src.methods.detector import detect_stamps_folder

summary = detect_stamps_folder(
    input_folder="documents/",
    conf_threshold=0.8,
    recursive=False,
)
print(f"Обработано: {summary['total_images']}, найдено печатей: {summary['total_stamps']}")
```

### Затирание печатей (анонимизация документов)
```python
from src.methods.detector import remove_stamps_from_image, remove_stamps_from_folder

# Одно изображение - автоматически детектирует и затирает печати
result = remove_stamps_from_image("document.pdf")
print(f"Затерто печатей: {result['num_stamps']}")
print(f"Очищенный PDF: {result['cleaned_pdf_path']}")

# Папка - массовое затирание
summary = remove_stamps_from_folder("documents/", recursive=True)
print(f"Обработано: {summary['total_images']}, затерто: {summary['total_stamps']}")
```