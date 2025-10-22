"""
Раннер для правки ориентации под углами 90/180/270.
"""

from src.methods.improver.rotate.rotate_right import RightAngleRotation


def rotate_right(input_dir: str, output_dir: str, failed_dir: str):
    print("[INFO] Инициализируем PaddleOCR...")
    rotator = RightAngleRotation(input_dir, output_dir, failed_dir)
    if rotator.orientation_classifier is not None:
        print("[INFO] Используем DocImgOrientationClassification для определения ориентации")
        rotator.process_all()
    else:
        raise Exception("DocImgOrientationClassification не инициализирован")



