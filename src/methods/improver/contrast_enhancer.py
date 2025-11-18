import cv2
import numpy as np


def enhance_text_contrast(
    img_bgr: np.ndarray,
    target_l_mean: float = 185.0,
    max_light_boost: float = 1.35,
    saturation_boost: float = 1.12,
    value_boost: float = 1.05,
):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_clahe = clahe.apply(L)
    avg_l = float(np.mean(L_clahe))

    gain = min(target_l_mean / max(avg_l, 1.0), max_light_boost) if avg_l > 0 else 1.0
    L_boosted = np.clip(L_clahe * gain, 0, 255).astype(np.uint8)

    lab_enhanced = cv2.merge([L_boosted, a, b])
    enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * value_boost, 0, 255)
    enhanced_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    modified = not np.array_equal(enhanced_bgr, img_bgr)
    return enhanced_bgr, modified, avg_l

