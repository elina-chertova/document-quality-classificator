

## Пайплайн: порядок запуска и пути

Ниже — минимальный сквозной сценарий обработки: от подготовки PDF до финальной классификации качества. Точки входа и обязательные пути указаны явно.

### 1) Разделение многостраничных PDF на страницы
- Точка входа: `src/pipeline/main.py` (закомментированные примеры)
- Пример путей - датасет, который содержит все PDF документы:
  - input_dir: `/Users/elinacertova/Downloads/documents_dataset/all`


Запустите (включает разделение пдф по страницам, все повороты и запуск осветления в случае необходимости, все новые файлы будут скопированы в новую директорию, можно ее посмотреть в конфиге):
```bash
python src/pipeline/main.py
```


### 2а) Подготовка обучающего CSV (если будем использовать такой вариант). 
Для того, чтобы код корректно работал, необходимо заранее иметь директорию с поддиректориями good, medium, failed (используются как true labels)

- Точка входа: `create_training_data.py` (создает `classification_analysis.csv` на основе `classificator_extended.py`).
- Итоговый CSV: `classification_analysis.csv` в корне проекта.

### 3) Тюнинг/оценка моделей (опционально)
- Точка входа: `tune_extended_classifier.py` — подберёт пороги и обучит/оценит ML-модели, выведет метрики и сохранит сводку в `results_summary`.

### 4) Обучение финальной ML‑модели качества
- Точка входа: `src/final_quality_classifier.py`
- Требуется файл: `classification_analysis.csv`

Запуск обучения (создаст `src/final_quality_classifier_model.pkl`):
```bash
python src/final_quality_classifier.py
```

### 7) Классификация примеров (валидация)
- Точка входа: `src/test_final_classifier.py`
- Использует модель `src/final_quality_classifier_model.pkl`

Запуск:
```bash
python src/test_final_classifier.py
```

### 8) Классификация произвольной папки документов
Вариант А — использовать готовую обученную модель:
```bash
python src/classify_documents.py \
  /Users/elinacertova/Downloads/documents_dataset/results/processed \
  /Users/elinacertova/Downloads/documents_dataset/results/final_res
```
Результат: файлы будут разложены по подпапкам `failed/`, `medium/`, `good` в выходной директории.

Вариант Б — обучить на лету из CSV и сразу классифицировать датасет:
```bash
python src/use_classifier.py
```

### Ключевые пути, которые нужно проверить/править под вашу среду
- Исходные PDF: `/Users/elinacertova/Downloads/documents_dataset/all`
- Разделённые страницы: `/Users/elinacertova/Downloads/documents_dataset/results/all_splitted`
- Правильные повороты (90/180/270): `/Users/elinacertova/Downloads/documents_dataset/results/rotated`
- Автодескью (< 90°): `/Users/elinacertova/Downloads/documents_dataset/results/deskewed`
- Итоговая классификация: `/Users/elinacertova/Downloads/documents_dataset/results/final_res`
- Папка для неудачных кейсов: `/Users/elinacertova/Downloads/documents_dataset/results/failed` и `documents_dataset/final/failed_2`
- CSV для обучения/оценки: `classification_analysis.csv`

Если путь отличается — отредактируйте соответствующие значения прямо в скриптах перед запуском.