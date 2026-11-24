"""Shared, deterministic dataset discovery and image-hash helpers."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = [
    "central_serous_chorioretinopathy",
    "diabetic_retinopathy",
    "disc_edema",
    "glaucoma",
    "healthy",
    "macular_scar",
    "myopia",
    "retinal_detachment",
    "retinitis_pigmentosa",
]


def normalise_class_name(name: str) -> str:
    """Map archive folder labels to stable slugs."""
    lowered = name.lower()
    aliases = {
        "central serous chorioretinopathy": "central_serous_chorioretinopathy",
        "diabetic retinopathy": "diabetic_retinopathy",
        "disc edema": "disc_edema",
        "glaucoma": "glaucoma",
        "healthy": "healthy",
        "macular scar": "macular_scar",
        "myopia": "myopia",
        "pterygium": "pterygium",
        "retinal detachment": "retinal_detachment",
        "retinitis pigmentosa": "retinitis_pigmentosa",
    }
    for phrase, slug in aliases.items():
        if phrase in lowered:
            return slug
    return re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")


def discover_images(root: Path) -> dict[str, list[Path]]:
    """Discover archive images by their nearest recognised class directory."""
    images: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        labels = [normalise_class_name(part) for part in path.relative_to(root).parts[:-1]]
        label = next((value for value in reversed(labels) if value in {*CLASS_NAMES, "pterygium"}), None)
        if label:
            images[label].append(path)
    return dict(images)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phash(path: Path, transform: str = "identity") -> np.uint64:
    """64-bit DCT perceptual hash; transforms make matching flip/rotation robust."""
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"))
    if transform == "flip_lr":
        gray = np.fliplr(gray)
    elif transform == "flip_ud":
        gray = np.flipud(gray)
    elif transform == "rotate_180":
        gray = np.rot90(gray, 2)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coefficients = cv2.dct(resized)[:8, :8]
    median = np.median(coefficients.flatten()[1:])
    bits = (coefficients > median).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return np.uint64(value)


def hamming(left: np.uint64, right: np.uint64) -> int:
    return int(int(left) ^ int(right)).bit_count()

