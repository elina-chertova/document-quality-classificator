"""
Раннер для правки ориентации под углами 90/180/270.
"""

import os
from src.methods.improver.rotate.rotate_right import RightAngleRotation
from src.methods.improver.rotate.rotate_right_simple import SimpleRightAngleRotation
from .config import PipelineConfig


def rotate_right(root_dir: str = "/Users/elinacertova/Downloads/documents_dataset"):
    cfg = PipelineConfig()
    if root_dir:
        cfg.paths.root_dir = root_dir
    input_dir = cfg.paths.split_folder
    output_dir = cfg.paths.rotated_folder
    failed_dir = cfg.paths.failed_rotate_folder
    
    # Пытаемся использовать оптимизированный PaddleOCR
    try:
        print("[INFO] Инициализируем оптимизированный PaddleOCR...")
        rotator = RightAngleRotation(input_dir, output_dir, failed_dir)
        if rotator.orientation_classifier is not None:
            print("[INFO] Используем DocImgOrientationClassification для определения ориентации")
            rotator.process_all()
        else:
            raise Exception("DocImgOrientationClassification не инициализирован")
    except Exception as e:
        print(f"[WARNING] PaddleOCR не работает: {e}")
        print("[INFO] Переключаемся на упрощенную версию без OCR")
        rotator = SimpleRightAngleRotation(input_dir, output_dir, failed_dir)
        rotator.process_all()


if __name__ == "__main__":
    pass


