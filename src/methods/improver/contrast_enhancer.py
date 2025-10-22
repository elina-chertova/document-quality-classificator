import cv2
import numpy as np


def enhance_text_contrast(img_bgr, brightness_thresh=120, darkness_boost=1.5):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    inv = 255 - gray
    _, mask = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dark_pixels = gray[mask > 0]
    if len(dark_pixels) == 0:
        return img_bgr, False, 0.0

    avg_dark = np.mean(dark_pixels)

    if avg_dark > brightness_thresh:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        enhanced = np.clip(enhanced / darkness_boost, 0, 255).astype(np.uint8)

        img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = enhanced
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR), True, avg_dark
    else:
        return img_bgr, False, avg_dark

