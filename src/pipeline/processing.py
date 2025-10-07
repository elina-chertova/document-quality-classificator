import os
from typing import List, Dict, Any

from src.methods.classificator.dark_document_classifier import DarkDocumentClassifier
from src.methods.improver.document_lightener import LightenParams, PDFDocumentLightener


def classify_documents(input_folder: str, dark_folder: str, classifier: DarkDocumentClassifier):
    results = classifier.process_folder(
        input_folder=input_folder,
        output_folder=dark_folder,
        normal_subdir="normal_brightness",
        dark_subdir="dark_documents",
    )
    dark_docs = [r for r in results if getattr(r, "is_dark", False) and not getattr(r, "error", False)]
    normal_docs = [r for r in results if not getattr(r, "is_dark", False) and not getattr(r, "error", False)]
    error_docs = [r for r in results if getattr(r, "error", False)]
    return results, dark_docs, normal_docs, error_docs


def copy_normal_documents(normal_docs: List[Any], output_folder: str) -> int:
    import shutil

    copied = 0
    for doc in normal_docs:
        src_path = doc.pdf_path
        dst_path = os.path.join(output_folder, os.path.basename(src_path))
        try:
            shutil.copy2(src_path, dst_path)
            copied += 1
        except Exception:
            pass
    return copied


def lighten_dark_documents(
    dark_folder: str,
    output_folder: str,
    lightener: PDFDocumentLightener,
    lighten_params: LightenParams,
    passes: int = 2,
) -> Dict[str, Any]:
    dark_input_folder = os.path.join(dark_folder, "dark_documents")
    if not (os.path.exists(dark_input_folder) and os.listdir(dark_input_folder)):
        return {"processed": 0, "success": 0, "failed": 0, "errors": []}

    results = lightener.process_dark_folder(
        input_folder=dark_input_folder,
        output_folder=output_folder,
        lighten_params=lighten_params,
        passes=passes,
    )
    return results








