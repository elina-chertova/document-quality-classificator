from PIL import Image
from src.methods.classificator.classificator_easyocr import PDFQualityAssessorEasyOCR


class PDFQualityAssessorEasyOCROptimized(PDFQualityAssessorEasyOCR):
    def _estimate_skew_deg(self, image: Image.Image) -> float:
        return 0.0
    
    def _core_content_fraction(self, image: Image.Image) -> float:
        return 0.5


def get_optimized_assessor(
    dpi: int = 200,
    device: str = "cuda",
    max_workers: int = 4,
) -> PDFQualityAssessorEasyOCROptimized:
    return PDFQualityAssessorEasyOCROptimized(
        dpi=dpi,
        copy_to_dirs=False,
        max_workers=max_workers,
        device=device,
    )
