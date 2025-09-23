"""
Конфигурация конвейера обработки документов.
"""

from dataclasses import dataclass, field
import os

from src.methods.improver.document_lightener import LightenParams


@dataclass
class ClassifierConfig:
    dpi: int = 200
    brightness_threshold: float = 80.0
    dark_pixels_threshold: float = 30.0
    contrast_threshold: float = 15.0
    very_dark_pixels_threshold: float = 15.0
    copy_to_dirs: bool = True
    max_workers: int = 4


@dataclass
class PathsConfig:
    root_dir: str = "/Users/elinacertova/Downloads/documents_dataset"
    # Полный исходный датасет (все файлы до разбиения)
    @property
    def full_dataset_folder(self) -> str:
        return os.path.join(self.root_dir, "all")
    # Результаты по шагам пайплайна
    @property
    def split_folder(self) -> str:
        return os.path.join(self.root_dir, "results", "all_splitted")
    @property
    def rotated_folder(self) -> str:
        return os.path.join(self.root_dir, "results", "rotated")
    @property
    def failed_rotate_folder(self) -> str:
        return os.path.join(self.root_dir, "results", "failed")
    @property
    def deskewed_folder(self) -> str:
        return os.path.join(self.root_dir, "results", "deskewed")
    @property
    def failed_deskew_folder(self) -> str:
        return os.path.join(self.root_dir, "results", "failed_deskew")
    @property
    def input_folder(self) -> str:
        # вход для объединённого процессора (после deskew)
        return self.deskewed_folder
    @property
    def output_folder(self) -> str:
        # итоговый вывод объединённого процессора
        return os.path.join(self.root_dir, "results", "processed")
    @property
    def dark_folder(self) -> str:
        return os.path.join(self.root_dir, "results", "dark")

    @property
    def lines_cleaned_folder(self) -> str:
        return os.path.join(self.root_dir, "results", "no_lines")

    @property
    def no_lines_ok_folder(self) -> str:
        return os.path.join(self.root_dir, "results", "no_lines_ok")

    # OCR
    @property
    def ocr_output_folder(self) -> str:
        return os.path.join(self.root_dir, "results", "ocr")
    @property
    def ocr_csv_path(self) -> str:
        return os.path.join(self.ocr_output_folder, "ocr_results.csv")

    # Примеры качества и путь к обучающему CSV
    @property
    def example_quality_base(self) -> str:
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "example", "quality")

    @property
    def training_csv_path(self) -> str:
        # CSV в корне репозитория, как раньше
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        return os.path.join(project_root, "classification_analysis.csv")

    @property
    def trained_model_path(self) -> str:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        return os.path.join(project_root, "final_quality_classifier_model.pkl")


@dataclass
class LightenerConfig:
    dpi: int = 200
    params: LightenParams = field(
        default_factory=lambda: LightenParams(
            target_long_side=2200,
            bg_kernel_frac=0.06,
            clahe_clip=2.0,
            denoise_h=6,
            sharpen_amount=1.4,
            edge_low=50,
            edge_high=150,
            keep_color=False,
        )
    )


@dataclass
class PipelineConfig:
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    lightener: LightenerConfig = field(default_factory=LightenerConfig)


