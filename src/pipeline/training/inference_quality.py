import os
import math
import joblib
import numpy as np
from typing import Dict, List, Optional
from PIL import Image

try:
    import fitz
    USE_PYMUPDF = True
except ImportError:
    from pdf2image import convert_from_path
    USE_PYMUPDF = False

from src.methods.classificator.classificator_easyocr import (
    PDFQualityAssessorEasyOCR,
    PDFQualityResult,
)


def load_trained_model(model_path: str) -> Dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    obj = joblib.load(model_path)
    if not isinstance(obj, dict) or 'model' not in obj or 'features' not in obj:
        raise ValueError("Неверный формат файла модели: ожидаются ключи 'model' и 'features'")
    return obj


def extract_features(res: PDFQualityResult) -> Dict[str, float]:
    feats: Dict[str, float] = {
        'median_ocr_conf': float(res.median_ocr_conf),
        'mean_ocr_conf': float(res.mean_ocr_conf),
        'pct80': float(res.pct80),
        'avg_blur': float(res.avg_blur),
        'words_count': int(res.words_count),
        'text_density': float(res.text_density),
        'roi_frac': float(res.roi_frac),
        'core_frac': float(res.core_frac),
        'is_table_like': 1.0 if bool(res.is_table_like) else 0.0,
        'avg_skew_deg': float(res.avg_skew_deg),
        'bbox_area_text_frac': float(res.bbox_area_text_frac),
        'conf_iqr': float(res.conf_iqr),
        'line_height_med': float(res.line_height_med),
        'line_height_var': float(res.line_height_var),
        'line_spacing_med': float(res.line_spacing_med),
        'line_spacing_var': float(res.line_spacing_var),
        'text_blocks_count': int(res.text_blocks_count),
        'avg_block_width': float(res.avg_block_width),
        'avg_block_height': float(res.avg_block_height),
    }

    feats['conf_range'] = feats['mean_ocr_conf'] - feats['median_ocr_conf']
    feats['blur_per_word'] = feats['avg_blur'] / (feats['words_count'] + 1.0)
    feats['density_per_conf'] = feats['text_density'] * feats['median_ocr_conf']
    feats['pct80_squared'] = feats['pct80'] ** 2
    feats['conf_log'] = math.log1p(feats['median_ocr_conf'])
    feats['words_log'] = math.log1p(feats['words_count'])
    feats['bbox_area_log'] = math.log1p(feats['bbox_area_text_frac'] * 1000.0)
    feats['line_height_cv'] = feats['line_height_var'] / (feats['line_height_med'] + 1.0)
    feats['line_spacing_cv'] = feats['line_spacing_var'] / (feats['line_spacing_med'] + 1.0)

    return feats


def _vectorize_features(feature_dict: Dict[str, float], feature_order: List[str]) -> np.ndarray:
    row = [feature_dict.get(name, 0.0) for name in feature_order]
    return np.asarray(row, dtype=float).reshape(1, -1)


def _assess_single_page(
    assessor: PDFQualityAssessorEasyOCR,
    page_image: Image.Image,
) -> PDFQualityResult:
    roi_img, roi_frac = assessor._crop_roi(page_image)
    blur = assessor._blur_score(roi_img)
    density = assessor._text_density(roi_img)
    core_frac = assessor._core_content_fraction(roi_img)
    skew_deg = assessor._estimate_skew_deg(roi_img)
    is_table = assessor._is_table_like(roi_img)
    
    ocr_ready = assessor._prep_for_ocr(roi_img)
    mconf, meanconf, p80, words = assessor._ocr_metrics_easyocr(ocr_ready)
    
    return PDFQualityResult(
        pdf_path="",
        category="",
        reason="",
        avg_blur=blur,
        median_ocr_conf=mconf,
        mean_ocr_conf=meanconf,
        pct80=p80,
        words_count=words,
        text_density=density,
        roi_frac=roi_frac,
        core_frac=core_frac,
        is_table_like=is_table,
        avg_skew_deg=skew_deg,
    )


def predict_quality_for_pdf(
    pdf_path: str,
    model_path: str,
    dpi: int = 400,
    device: Optional[str] = None,
) -> Dict[str, str | float | dict]:
    obj = load_trained_model(model_path)
    model = obj['model']
    feature_order: List[str] = obj['features']

    assessor = PDFQualityAssessorEasyOCR(
        dpi=dpi,
        copy_to_dirs=False,
        max_workers=1,
        device=device,
    )
    res = assessor.assess_pdf(pdf_path)
    feats = extract_features(res)
    X = _vectorize_features(feats, feature_order)
    pred = model.predict(X)[0]

    proba = None
    try:
        if hasattr(model, 'predict_proba'):
            proba_vec = model.predict_proba(X)[0]
            classes_ = model.classes_ if hasattr(model, 'classes_') else None
            if classes_ is not None:
                proba = {str(c): float(p) for c, p in zip(classes_, proba_vec)}
    except Exception:
        proba = None

    return {
        'file': os.path.basename(pdf_path),
        'predicted': str(pred),
        'reason': res.reason or '',
        'proba': proba,
    }


def predict_quality_for_pdf_pages(
    pdf_path: str,
    model_path: str,
    dpi: int = 400,
    device: Optional[str] = None,
) -> List[Dict[str, str | float | dict | int]]:
    obj = load_trained_model(model_path)
    model = obj['model']
    feature_order: List[str] = obj['features']

    assessor = PDFQualityAssessorEasyOCR(
        dpi=dpi,
        copy_to_dirs=False,
        max_workers=1,
        device=device,
    )

    Image.MAX_IMAGE_PIXELS = None

    if USE_PYMUPDF:
        doc = fitz.open(pdf_path)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pages = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append(img)
        doc.close()
    else:
        pages = convert_from_path(pdf_path, dpi=dpi)

    if not pages:
        raise RuntimeError("PDF has 0 pages")

    results = []
    for page_num, page_image in enumerate(pages, start=1):
        res = _assess_single_page(assessor, page_image)
        feats = extract_features(res)
        X = _vectorize_features(feats, feature_order)
        pred = model.predict(X)[0]

        proba = None
        try:
            if hasattr(model, 'predict_proba'):
                proba_vec = model.predict_proba(X)[0]
                classes_ = model.classes_ if hasattr(model, 'classes_') else None
                if classes_ is not None:
                    proba = {str(c): float(p) for c, p in zip(classes_, proba_vec)}
        except Exception:
            proba = None

        results.append({
            'file': os.path.basename(pdf_path),
            'page': page_num,
            'predicted': str(pred),
            'proba': proba,
            'median_ocr_conf': res.median_ocr_conf,
            'avg_blur': res.avg_blur,
            'words_count': res.words_count,
        })

    return results


def predict_quality_for_folder(
    input_folder: str,
    model_path: str,
    dpi: int = 400,
    device: Optional[str] = None,
) -> List[Dict[str, str | float | dict]]:
    input_folder = os.path.abspath(input_folder)
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Папка не найдена: {input_folder}")

    files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    files.sort()

    results: List[Dict[str, str | float | dict]] = []
    for idx, fname in enumerate(files, start=1):
        pdf_path = os.path.join(input_folder, fname)
        r = predict_quality_for_pdf(pdf_path, model_path=model_path, dpi=dpi, device=device)
        print(f"[{idx}/{len(files)}] {fname} → {r['predicted'].upper()} (why={r['reason']})")
        results.append(r)
    return results


# if __name__ == "__main__":
#     results = predict_quality_for_pdf_pages(
#         pdf_path="/Users/elinacertova/Downloads/Договор_купли_продажи_недвижимого_имущества_пример_2025_для_двух.pdf",
#         model_path="/Users/elinacertova/PycharmProjects/document-quality-classificator/src/pipeline/training/final_quality_classifier_model.pkl",
#         dpi=400,
#         device=None,
#     )
#
#     print(f"Файл: {results[0]['file']}")
#     print(f"Всего страниц: {len(results)}\n")
#
#     for r in results:
#         print(f"Страница {r['page']}: {r['predicted'].upper()}")
#         print(f"  Вероятности: {r['proba']}")
#         print(f"  OCR confidence: {r['median_ocr_conf']:.1f}, blur: {r['avg_blur']:.1f}, words: {r['words_count']}")
#         print()
    #
    # results = predict_quality_for_folder(
    #     input_folder="/path/to/pdfs",
    #     model_path="final_quality_classifier_model.pkl",
    #     dpi=400,
    #     device=None,
    # )
    # print(f"\nОбработано документов: {len(results)}")

