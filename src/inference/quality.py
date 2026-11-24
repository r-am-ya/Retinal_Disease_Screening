"""Input-quality safeguards for screening images."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class QualityResult:
    passed: bool
    blur_variance: float
    brightness: float
    contrast: float
    fundus_fraction: float
    reasons: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def check_image_quality(image: Image.Image | np.ndarray) -> QualityResult:
    """Check sharpness, exposure, contrast, and a bright circular retinal field."""
    array = np.asarray(image.convert("RGB")) if isinstance(image, Image.Image) else image
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    mask = gray > max(12, int(np.percentile(gray, 15)))
    fraction = float(mask.mean())
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circular = False
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0.0
        circular = area > 0.15 * gray.size and circularity > 0.30
    reasons: list[str] = []
    if blur < 35:
        reasons.append("Image appears blurred (low Laplacian variance).")
    if not 35 <= brightness <= 220:
        reasons.append("Brightness is outside the acceptable range.")
    if contrast < 18:
        reasons.append("Image contrast is too low.")
    if not circular or not 0.15 <= fraction <= 0.95:
        reasons.append("Image does not resemble a bright circular fundus field on a dark background.")
    return QualityResult(not reasons, blur, brightness, contrast, fraction, reasons)
