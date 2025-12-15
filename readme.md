Установка:
```bash
pip install -r requirements.txt
```

### Выбор cpu/gpu прописывается в .env
```shell
DQC_DEVICE=cuda:0
```

## Предобработка папки документов (improver #1):
1. Поделить документ по страницам
```python
from src.pipeline.split_pages import split_pages


split_pages(input_dir="/Users/elinacertova/Downloads/dataset_tester_dir/docs", 
            output_dir="/Users/elinacertova/Downloads/dataset_tester_dir/splitted")
```
2. Поворот изображения на прямые углы
```python
from src.pipeline.rotate_right_runner import rotate_right


rotate_right(input_dir='/Users/elinacertova/Downloads/dataset_tester_dir/splitted',
                 output_dir='/Users/elinacertova/Downloads/dataset_tester_dir/rotated',
                 failed_dir='/Users/elinacertova/Downloads/dataset_tester_dir/failed')    
```

3. Выравнивание текста (поворот на углы меньше 45 градусов)
```python
from src.pipeline.deskew import deskew_documents


deskew_documents(input_dir='/Users/elinacertova/Downloads/dataset_tester_dir/rotated',
                 output_dir='/Users/elinacertova/Downloads/dataset_tester_dir/deskewed',
                 failed_dir='/Users/elinacertova/Downloads/dataset_tester_dir/failed')
```

4. Удаление линий из-за проблем сканирование
```python
from src.pipeline.remove_lines_runner import remove_lines


remove_lines(input_dir='/Users/elinacertova/Downloads/dataset_tester_dir/deskewed',
             lines_cleaned_folder='/Users/elinacertova/Downloads/dataset_tester_dir/lines_cleaned',
             no_lines_ok_folder='/Users/elinacertova/Downloads/dataset_tester_dir/lines_not_detected',
             combined_output_folder='/Users/elinacertova/Downloads/dataset_tester_dir/combined')
```

5. Осветление темных документов
```python
from src.pipeline.dark_docs_to_light import dark_documents_to_light

dark_documents_to_light(input_folder='/Users/elinacertova/Downloads/dataset_tester_dir/combined',
                        output_folder='/Users/elinacertova/Downloads/dataset_tester_dir/lightened',
                        dark_folder='/Users/elinacertova/Downloads/dataset_tester_dir/dark',
                        combined_output_folder='/Users/elinacertova/Downloads/dataset_tester_dir/lightened_combined')
```

6. Усиление текста (для светлых страниц)
```python
from src.pipeline.text_enhancement_runner import enhance_text_documents


enhance_text_documents(
    input_dir='/Users/elinacertova/Downloads/dataset_tester_dir/lightened_combined',
    output_dir='/Users/elinacertova/Downloads/dataset_tester_dir/text_enhanced'
)
```

### Классификация по подобранным порогам для папки
```python
from src.pipeline.quality_classifier_runner import classify_by_quality


classify_by_quality(input_folder='/Users/elinacertova/Downloads/dataset_tester_dir/lightened_combined',
                    output_folder='/Users/elinacertova/Downloads/dataset_tester_dir/classified')
```

## Пайплайн обработки до классификатора для единичного документа  
Ненужные модули можно закомментировать в коде в process_single_document.
Результаты сохраняются в CSV файл.
```python
from pathlib import Path
from src.pipeline.classify_single_document import classify_single_document
from src.pipeline.process_single_document import process_single_document


input_pdf_path = '/Users/elinacertova/Downloads/single_doc_test/Scan_20250213_120013.pdf'
output_base_dir = '/Users/elinacertova/Downloads/single_doc_test/output2'
output_csv_path = '/Users/elinacertova/Downloads/single_doc_test/results.csv'
pages_dir = process_single_document(
  input_pdf_path=input_pdf_path,
  output_base_dir=output_base_dir,
  output_csv_path=output_csv_path
)

document_name = Path(input_pdf_path).stem

results = classify_single_document(
    pages_dir=pages_dir,
    document_name=document_name,
    output_base_dir=output_base_dir,
    output_csv_path=output_csv_path,
    dpi=400,
    max_workers=4,
    classifier_dpi=300,
    optimized=False
)
```
## Пайплайн `Классификация -> improver`
```python
from src.pipeline.process_single_document_smart import process_single_document_smart

results = process_single_document_smart(
    input_pdf_path="/Scan_20250213_120013.pdf",
    output_base_dir="/single_doc_test/output3",
    output_csv_path="results.csv",
    dpi=400,
    max_workers=4,
    classifier_dpi=300,
    device="gpu",
    optimized=False
)
```

## Классификация с использованием ML
### Подготовка данных для обучения
Папка `/classific_testing` должна содержать размеченные заранее данные формата
- `/classific_testing/failed`
- `/classific_testing/medium`
- `/classific_testing/good`

с постраничными документами внутри.
```python
from src.pipeline.training.create_training_data import create_training_data


create_training_data(
    input_base_dir="/classific_testing",
    output_csv_path="classification_analysis.csv",
    device="gpu"
)
```
### Обучение
```python
from src.pipeline.training.tune_extended_classifier import tune_extended_classifier


result = tune_extended_classifier(
    csv_path="classification_analysis.csv",
    model_path="final_quality_classifier_model.pkl",
)

if 'error' in result:
    print(f"Ошибка: {result['error']}")
else:
    print(f"\nЛучший метод: {result['best_method']}, точность: {result['best_accuracy']:.4f}")
```

### Инференс
```python
from src.pipeline.training.inference_quality import predict_quality_for_pdf_pages

results = predict_quality_for_pdf_pages(
    pdf_path="Договор_купли_продажи_недвижимого_имущества_пример_2025_для_двух.pdf",
    model_path="final_quality_classifier_model.pkl",
    dpi=400,
    device="gpu",
)

print(f"Файл: {results[0]['file']}")
print(f"Всего страниц: {len(results)}\n")

for r in results:
    print(f"Страница {r['page']}: {r['predicted'].upper()}")
    print(f"  Вероятности: {r['proba']}")
    print(f"  OCR confidence: {r['median_ocr_conf']:.1f}, blur: {r['avg_blur']:.1f}, words: {r['words_count']}")
    print()
```


## Детекция печатей в папке
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
# ----------------------------------------
## Протестируйте до этого момента 
# ----------------------------------------




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
# from src.methods.classificator.classificator import PDFQualityAssessor
# 
# assessor = PDFQualityAssessor(
#     dpi=400,
#     copy_to_dirs=True,
#     max_workers=4,
# )
# 
# assessor.process_folder(
#     input_folder="/Users/elinacertova/Downloads/documents_dataset/results/processed",
#     output_folder="/Users/elinacertova/Downloads/documents_dataset/results/output_sorted",
#     medium_subdir="medium",
#     good_subdir="good",
#     failed_subdir="failed",
#     trash_subdir="trash",
# )

from src.methods.classificator.classificator_easyocr import PDFQualityAssessorEasyOCR

assessor = PDFQualityAssessorEasyOCR(
    dpi=200,
    tesseract_lang="rus+eng", 
    copy_to_dirs=True,
    max_workers=4
)

result = assessor.assess_pdf("document.pdf")
print(f"Category: {result.category}")
print(f"Words: {result.words_count}")
print(f"Confidence: {result.median_ocr_conf:.2f}")

results = assessor.process_folder(
    input_folder="input/",
    output_folder="output/"
)
```


### Подготовка обучающих данных (CSV)
Для того, чтобы код корректно работал, необходимо заранее иметь директорию с субдиректориями good, medium, failed (используются как true labels).

Создаёт `classification_analysis.csv`  ( используются папки`good/medium/failed`):
```bash
python src/pipeline/training/create_training_data.py
```
Пути `example_quality_base` и `training_csv_path` берутся из `config.py`.

### Тюнинг и сохранение лучшей ML‑модели



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
