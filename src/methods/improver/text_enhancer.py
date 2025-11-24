import cv2
import numpy as np


def enhance_text_regions(
    img_bgr: np.ndarray,
    clahe_clip: float = 3.0,
    text_darkening: float = 0.35,
    sharpen_amount: float = 1.2,
):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    L_eq = clahe.apply(L)
    avg_l = float(np.mean(L_eq))

    binary = cv2.adaptiveThreshold(
        L_eq,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    text_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    text_mask_f = cv2.GaussianBlur(text_mask, (0, 0), sigmaX=1.2)
    text_mask_norm = (text_mask_f.astype(np.float32) / 255.0)[..., None]

    darkened = L_eq.astype(np.float32) * (1.0 - text_darkening * text_mask_norm.squeeze())
    darkened = np.clip(darkened, 0, 255).astype(np.uint8)

    blurred = cv2.GaussianBlur(darkened, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(darkened, 1.0 + sharpen_amount, blurred, -sharpen_amount, 0)

    L_final = np.where(text_mask > 0, sharpened, L_eq).astype(np.uint8)
    lab_enhanced = cv2.merge([L_final, a, b])
    enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    modified = not np.array_equal(enhanced_bgr, img_bgr)
    return enhanced_bgr, modified, avg_l

