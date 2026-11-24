"""Shared colour fundus preprocessing used for training and inference."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop_fundus(image: np.ndarray) -> np.ndarray:
    """Crop the bright circular retinal field, preserving original RGB colour."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    threshold = max(8, int(np.percentile(gray, 12)))
    mask = (gray > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    contour = max(contours, key=cv2.contourArea)
    x_coord, y_coord, width, height = cv2.boundingRect(contour)
    if width * height < 0.1 * image.shape[0] * image.shape[1]:
        return image
    return image[y_coord : y_coord + height, x_coord : x_coord + width]


def enhance_colour(image: np.ndarray, clahe: bool = True, green_channel: bool = False) -> np.ndarray:
    """Apply optional local contrast enhancement without discarding colour."""
    if not clahe:
        return image
    equaliser = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if green_channel:
        enhanced = image.copy()
        enhanced[:, :, 1] = equaliser.apply(enhanced[:, :, 1])
        return enhanced
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = equaliser.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_image(
    image: Image.Image | np.ndarray | Path,
    image_size: int = 224,
    clahe: bool = True,
    green_channel: bool = False,
    normalise: bool = False,
) -> np.ndarray:
    """Crop, enhance, and resize an RGB image, optionally ImageNet-normalising it."""
    if isinstance(image, Path):
        with Image.open(image) as opened:
            array = np.asarray(opened.convert("RGB"))
    elif isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = image
    array = crop_fundus(array)
    array = enhance_colour(array, clahe=clahe, green_channel=green_channel)
    array = cv2.resize(array, (image_size, image_size), interpolation=cv2.INTER_AREA)
    if not normalise:
        return array
    values = array.astype(np.float32) / 255.0
    return (values - IMAGENET_MEAN) / IMAGENET_STD

