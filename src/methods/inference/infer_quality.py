
import os
import math
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from src.methods.classificator.classificator_extended import (
    ExtendedPDFQualityAssessor,
    ExtendedPDFQualityResult,
)
from src.pipeline.config import PipelineConfig


def load_trained_model(model_path: str | None = None) -> Dict:
    cfg = PipelineConfig()
    path = model_path or cfg.paths.trained_model_path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Модель не найдена: {path}")
    obj = joblib.load(path)
    if not isinstance(obj, dict) or 'model' not in obj or 'features' not in obj:
        raise ValueError("Неверный формат файла модели: ожидаются ключи 'model' и 'features'")
    return obj


def extract_features(res: ExtendedPDFQualityResult) -> Dict[str, float]:
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


def predict_quality_for_pdf(pdf_path: str, model_path: str | None = None) -> Dict[str, str | float]:
    obj = load_trained_model(model_path)
    model = obj['model']
    feature_order: List[str] = obj['features']

    assessor = ExtendedPDFQualityAssessor(
        dpi=200,
        copy_to_dirs=False,
        max_workers=1,
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


def predict_quality_for_folder(input_folder: str, model_path: str | None = None) -> List[Dict[str, str | float | dict]]:
    input_folder = os.path.abspath(input_folder)
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Папка не найдена: {input_folder}")

    files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    files.sort()

    results: List[Dict[str, str | float | dict]] = []
    for idx, fname in enumerate(files, start=1):
        pdf_path = os.path.join(input_folder, fname)
        r = predict_quality_for_pdf(pdf_path, model_path=model_path)
        print(f"[{idx}/{len(files)}] {fname} → {r['predicted'].upper()} (why={r['reason']})")
        results.append(r)
    return results



