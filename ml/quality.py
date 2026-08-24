from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def image_quality(image: Image.Image) -> dict:
    """Cheap diagnostics for the real-world lighting/background research question."""
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    sat = float(hsv[:, :, 1].mean())
    h, w = gray.shape
    warnings: list[str] = []
    if blur < 40:
        warnings.append("Image looks blurry; fine lesion texture may be lost.")
    if brightness < 40:
        warnings.append("Image is very dark; lighting may hide spots.")
    if brightness > 220:
        warnings.append("Image is very bright; overexposure may wash out symptoms.")
    if min(h, w) < 160:
        warnings.append("Resolution is low for reliable identification.")
    return {
        "width": int(w),
        "height": int(h),
        "blur_var": round(blur, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "mean_saturation": round(sat, 2),
        "warnings": warnings,
    }
