"""
Модуль для детекции печатей на документах с использованием YOLO.
"""

import os
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
import json
import tempfile

from ultralytics import YOLO
from PIL import Image, ImageDraw
from pdf2image import convert_from_path
import cv2
import numpy as np
import img2pdf


class StampDetector:
    """
    Детектор печатей на документах с использованием YOLO.
    
    Поддерживает обработку как отдельных изображений, так и целых папок.
    """
    
    def __init__(
        self,
        model_path: str = "/Users/elinacertova/PycharmProjects/documents_preprocessing/src/models/yolo_stamp_detector.pt",
        conf_threshold: float = 0.8,
        iou_threshold: float = 0.45,
        save_annotated: bool = True,
        save_json: bool = True,
        remove_stamps: bool = False,
    ):
        """
        Инициализация детектора печатей.
        
        Args:
            model_path: Путь к модели YOLO (.pt файл)
            conf_threshold: Порог уверенности для детекции (0.0-1.0)
            iou_threshold: Порог IoU для NMS (Non-Maximum Suppression)
            save_annotated: Сохранять ли изображения с аннотациями
            save_json: Сохранять ли JSON с координатами детекций
            remove_stamps: Затирать ли печати белым прямоугольником
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")
        
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.save_annotated = save_annotated
        self.save_json = save_json
        self.remove_stamps = remove_stamps
        
        # Загрузка модели YOLO
        print(f"Загрузка модели YOLO из {model_path}...")
        self.model = YOLO(model_path)
        print("Модель успешно загружена!")
    
    def _convert_pdf_to_image(self, pdf_path: str, dpi: int = 200) -> Image.Image:
        """
        Конвертация PDF в изображение (первая страница).
        
        Args:
            pdf_path: Путь к PDF файлу
            dpi: DPI для конвертации
            
        Returns:
            PIL Image объект
        """
        pages = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)
        if not pages:
            raise RuntimeError(f"PDF файл пуст или не читается: {pdf_path}")
        return pages[0]
    
    def _remove_stamps_from_image(self, image: Image.Image, detections: List[Dict]) -> Image.Image:
        """
        Затирает печати белым прямоугольником на изображении.
        
        Args:
            image: PIL Image объект
            detections: Список детекций с bbox координатами
            
        Returns:
            PIL Image объект с затертыми печатями
        """
        if not detections:
            return image
        
        # Создаем копию изображения
        img_cleaned = image.copy()
        draw = ImageDraw.Draw(img_cleaned)
        
        # Затираем каждую печать белым прямоугольником
        for detection in detections:
            bbox = detection['bbox']
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
            x2, y2 = int(bbox['x2']), int(bbox['y2'])
            
            # Рисуем белый прямоугольник
            draw.rectangle([x1, y1, x2, y2], fill='white', outline='white')
        
        return img_cleaned
    
    def detect_single_image(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        visualize: bool = True,
        dpi: int = 200,
    ) -> Dict:
        """
        Детекция печатей на одном изображении или PDF.
        
        Args:
            image_path: Путь к изображению или PDF
            output_dir: Директория для сохранения результатов (если None, создается рядом с изображением)
            visualize: Показать ли результат визуализации
            dpi: DPI для конвертации PDF (если это PDF файл)
            
        Returns:
            Dict с информацией о детекциях:
                - image_path: путь к исходному изображению
                - num_stamps: количество найденных печатей
                - detections: список словарей с координатами и уверенностью
                - annotated_path: путь к аннотированному изображению (если save_annotated=True)
                - json_path: путь к JSON файлу (если save_json=True)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл не найден: {image_path}")
        
        print(f"\nОбработка: {image_path}")
        
        # Определение выходной директории
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(image_path), "stamp_detection_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # Проверка, является ли файл PDF
        is_pdf = image_path.lower().endswith('.pdf')
        temp_image_path = None
        original_pil_image = None
        
        try:
            if is_pdf:
                # Конвертируем PDF в изображение
                print(f"  Конвертация PDF в изображение (DPI={dpi})...")
                pil_image = self._convert_pdf_to_image(image_path, dpi=dpi)
                original_pil_image = pil_image.copy()  # Сохраняем для возможного затирания печатей
                
                # Сохраняем во временный файл для YOLO
                temp_image_path = os.path.join(tempfile.gettempdir(), f"temp_yolo_{os.getpid()}.jpg")
                pil_image.save(temp_image_path, "JPEG")
                source_for_yolo = temp_image_path
            else:
                source_for_yolo = image_path
                # Загружаем изображение для возможного затирания печатей
                if self.remove_stamps:
                    original_pil_image = Image.open(image_path)
            
            # Запуск детекции
            results = self.model.predict(
                source=source_for_yolo,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                save=False,  # Мы сами сохраним результаты
                verbose=False,
            )
        finally:
            # Удаляем временный файл, если он был создан
            if temp_image_path and os.path.exists(temp_image_path):
                try:
                    os.remove(temp_image_path)
                except:
                    pass
        
        # Обработка результатов
        result = results[0]
        detections = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # Координаты bbox
            confidences = result.boxes.conf.cpu().numpy()  # Уверенность
            classes = result.boxes.cls.cpu().numpy()  # Классы
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                x1, y1, x2, y2 = box
                detections.append({
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    },
                    "confidence": float(conf),
                    "class": int(cls),
                    "class_name": self.model.names[int(cls)] if hasattr(self.model, 'names') else f"class_{int(cls)}",
                })
        
        # Подготовка результата
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        result_dict = {
            "image_path": image_path,
            "image_name": image_name,
            "num_stamps": len(detections),
            "detections": detections,
        }
        
        # Сохранение аннотированного изображения
        if self.save_annotated:
            annotated_path = os.path.join(output_dir, f"{image_name}_annotated.jpg")
            annotated_img = result.plot()  # Получаем изображение с аннотациями
            cv2.imwrite(annotated_path, annotated_img)
            result_dict["annotated_path"] = annotated_path
            print(f"Аннотированное изображение сохранено: {annotated_path}")
        
        # Сохранение JSON с детекциями
        if self.save_json:
            json_path = os.path.join(output_dir, f"{image_name}_detections.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)
            result_dict["json_path"] = json_path
            print(f"JSON с детекциями сохранен: {json_path}")
        
        # Затирание печатей белым прямоугольником (если включено)
        if self.remove_stamps and len(detections) > 0:
            if original_pil_image is not None:
                print(f"  Затирание {len(detections)} печатей...")
                cleaned_image = self._remove_stamps_from_image(original_pil_image, detections)
                
                # Сохранение очищенного изображения
                cleaned_image_path = os.path.join(output_dir, f"{image_name}_cleaned.jpg")
                cleaned_image.save(cleaned_image_path, "JPEG", quality=95)
                result_dict["cleaned_image_path"] = cleaned_image_path
                print(f"✓ Изображение без печатей сохранено: {cleaned_image_path}")
                
                # Если исходный файл был PDF, создаем также очищенный PDF
                if is_pdf:
                    cleaned_pdf_path = os.path.join(output_dir, f"{image_name}_cleaned.pdf")
                    try:
                        # Конвертируем очищенное изображение обратно в PDF
                        cleaned_image_rgb = cleaned_image.convert('RGB')
                        cleaned_image_rgb.save(cleaned_pdf_path, "PDF", resolution=100.0)
                        result_dict["cleaned_pdf_path"] = cleaned_pdf_path
                        print(f"✓ PDF без печатей сохранен: {cleaned_pdf_path}")
                    except Exception as e:
                        print(f"  Предупреждение: не удалось создать PDF: {e}")
            else:
                print(f"  Предупреждение: исходное изображение не загружено, затирание невозможно")
        
        print(f"Найдено печатей: {len(detections)}")
        
        # Визуализация (опционально)
        if visualize:
            self._visualize_result(annotated_img if self.save_annotated else result.plot())
        
        return result_dict
    
    def detect_folder(
        self,
        input_folder: str,
        output_dir: Optional[str] = None,
        image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf'),
        recursive: bool = False,
        dpi: int = 200,
        remove_stamps: Optional[bool] = None,
    ) -> Dict:
        """
        Детекция печатей для всех изображений в папке.
        
        Args:
            input_folder: Путь к папке с изображениями
            output_dir: Директория для сохранения результатов
            image_extensions: Кортеж расширений файлов для обработки
            recursive: Рекурсивный поиск изображений в подпапках
            dpi: DPI для конвертации PDF файлов
            remove_stamps: Затирать ли печати (если None, используется значение из __init__)
            
        Returns:
            Dict со статистикой обработки:
                - total_images: общее количество обработанных изображений
                - total_stamps: общее количество найденных печатей
                - results: список результатов для каждого изображения
                - summary_path: путь к сводному JSON файлу
        """
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"Папка не найдена: {input_folder}")
        
        # Сохраняем оригинальное значение remove_stamps
        original_remove_stamps = self.remove_stamps
        if remove_stamps is not None:
            self.remove_stamps = remove_stamps
        
        # Определение выходной директории
        if output_dir is None:
            output_dir = os.path.join(input_folder, "stamp_detection_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # Поиск изображений
        image_files = []
        if recursive:
            for root, dirs, files in os.walk(input_folder):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        image_files.append(os.path.join(root, file))
        else:
            image_files = [
                os.path.join(input_folder, f)
                for f in os.listdir(input_folder)
                if f.lower().endswith(image_extensions)
            ]
        
        if not image_files:
            print(f"В папке {input_folder} не найдено изображений с расширениями {image_extensions}")
            return {
                "total_images": 0,
                "total_stamps": 0,
                "results": [],
            }
        
        print(f"\nНайдено {len(image_files)} изображений для обработки")
        print(f"Результаты будут сохранены в: {output_dir}\n")
        
        # Обработка каждого изображения
        results = []
        total_stamps = 0
        
        for i, image_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] Обработка: {os.path.basename(image_path)}")
            try:
                result = self.detect_single_image(
                    image_path=image_path,
                    output_dir=output_dir,
                    visualize=False,  # Не показываем визуализацию для массовой обработки
                    dpi=dpi,
                )
                results.append(result)
                total_stamps += result["num_stamps"]
            except Exception as e:
                print(f"❌ Ошибка при обработке {image_path}: {e}")
                results.append({
                    "image_path": image_path,
                    "error": str(e),
                    "num_stamps": 0,
                })
        
        # Создание сводной статистики
        summary = {
            "total_images": len(image_files),
            "total_stamps": total_stamps,
            "average_stamps_per_image": total_stamps / len(image_files) if image_files else 0,
            "results": results,
        }
        
        # Сохранение сводного JSON
        summary_path = os.path.join(output_dir, "detection_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        summary["summary_path"] = summary_path
        
        print(f"\n" + "="*60)
        print(f"ИТОГОВАЯ СТАТИСТИКА:")
        print(f"  Обработано изображений: {summary['total_images']}")
        print(f"  Найдено печатей: {summary['total_stamps']}")
        print(f"  Среднее количество печатей на изображение: {summary['average_stamps_per_image']:.2f}")
        print(f"  Сводный отчет сохранен: {summary_path}")
        print("="*60)
        
        # Восстанавливаем оригинальное значение remove_stamps
        self.remove_stamps = original_remove_stamps
        
        return summary
    
    def _visualize_result(self, annotated_img: np.ndarray):
        """
        Показать результат детекции (для интерактивного использования).
        
        Args:
            annotated_img: Изображение с аннотациями (BGR формат)
        """
        try:
            # Конвертация BGR -> RGB для отображения
            img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            # Показ изображения
            cv2.imshow("Stamp Detection Result", annotated_img)
            print("Нажмите любую клавишу для продолжения...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Не удалось показать изображение: {e}")
    
    def get_model_info(self) -> Dict:
        """
        Получить информацию о загруженной модели.
        
        Returns:
            Dict с информацией о модели
        """
        info = {
            "model_path": self.model_path,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
        }
        
        if hasattr(self.model, 'names'):
            info["class_names"] = self.model.names
        
        if hasattr(self.model, 'model'):
            info["model_type"] = str(type(self.model.model))
        
        return info


# Функции для удобного использования

def detect_stamps_single(
    image_path: str,
    model_path: str = "/Users/elinacertova/PycharmProjects/documents_preprocessing/src/models/yolo_stamp_detector.pt",
    output_dir: Optional[str] = None,
    conf_threshold: float = 0.8,
    visualize: bool = True,
    dpi: int = 200,
) -> Dict:
    """
    Удобная функция для детекции печатей на одном изображении или PDF.
    
    Args:
        image_path: Путь к изображению или PDF
        model_path: Путь к модели YOLO
        output_dir: Директория для сохранения результатов
        conf_threshold: Порог уверенности
        visualize: Показать результат
        dpi: DPI для конвертации PDF
        
    Returns:
        Dict с результатами детекции
    """
    detector = StampDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
    )
    return detector.detect_single_image(
        image_path=image_path,
        output_dir=output_dir,
        visualize=visualize,
        dpi=dpi,
    )


def detect_stamps_folder(
    input_folder: str,
    model_path: str = "/Users/elinacertova/PycharmProjects/documents_preprocessing/src/models/yolo_stamp_detector.pt",
    output_dir: Optional[str] = None,
    conf_threshold: float = 0.8,
    recursive: bool = False,
    dpi: int = 200,
) -> Dict:
    """
    Удобная функция для детекции печатей в папке.
    
    Args:
        input_folder: Путь к папке с изображениями и PDF
        model_path: Путь к модели YOLO
        output_dir: Директория для сохранения результатов
        conf_threshold: Порог уверенности
        recursive: Рекурсивный поиск
        dpi: DPI для конвертации PDF
        
    Returns:
        Dict со статистикой обработки
    """
    detector = StampDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
    )
    return detector.detect_folder(
        input_folder=input_folder,
        output_dir=output_dir,
        recursive=recursive,
        dpi=dpi,
    )


def remove_stamps_from_image(
    image_path: str,
    model_path: str = "/Users/elinacertova/PycharmProjects/documents_preprocessing/src/models/yolo_stamp_detector.pt",
    output_dir: Optional[str] = None,
    conf_threshold: float = 0.8,
    dpi: int = 200,
) -> Dict:
    """
    Детектирует и затирает печати на одном изображении/PDF.
    
    Args:
        image_path: Путь к изображению или PDF
        model_path: Путь к модели YOLO
        output_dir: Директория для сохранения результатов
        conf_threshold: Порог уверенности
        dpi: DPI для конвертации PDF
        
    Returns:
        Dict с результатами детекции и путями к очищенным файлам
    """
    detector = StampDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        remove_stamps=True,  # Включаем затирание
    )
    return detector.detect_single_image(
        image_path=image_path,
        output_dir=output_dir,
        visualize=False,
        dpi=dpi,
    )


def remove_stamps_from_folder(
    input_folder: str,
    model_path: str = "/Users/elinacertova/PycharmProjects/documents_preprocessing/src/models/yolo_stamp_detector.pt",
    output_dir: Optional[str] = None,
    conf_threshold: float = 0.8,
    recursive: bool = False,
    dpi: int = 200,
) -> Dict:
    """
    Детектирует и затирает печати во всех изображениях/PDF в папке.
    
    Args:
        input_folder: Путь к папке с изображениями и PDF
        model_path: Путь к модели YOLO
        output_dir: Директория для сохранения результатов
        conf_threshold: Порог уверенности
        recursive: Рекурсивный поиск
        dpi: DPI для конвертации PDF
        
    Returns:
        Dict со статистикой обработки
    """
    detector = StampDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        remove_stamps=True,  # Включаем затирание
    )
    return detector.detect_folder(
        input_folder=input_folder,
        output_dir=output_dir,
        recursive=recursive,
        dpi=dpi,
    )


if __name__ == "__main__":
    # Пример использования
    import sys
    
    if len(sys.argv) < 2:
        print("Использование:")
        print("  Одно изображение: python stamp_detector.py <путь_к_изображению>")
        print("  Папка: python stamp_detector.py <путь_к_папке> --folder")
    else:
        path = sys.argv[1]
        
        if "--folder" in sys.argv or os.path.isdir(path):
            # Обработка папки
            result = detect_stamps_folder(path)
        else:
            # Обработка одного изображения
            result = detect_stamps_single(path)

