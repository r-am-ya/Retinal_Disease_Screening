"""Vercel serverless endpoint: POST raw JPEG/PNG bytes -> class probabilities.

Preprocessing is copied from src/data/preprocessing.py so training and serving match.
"""

from __future__ import annotations

import io
import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = Path(__file__).parent
IMAGE_SIZE = 192  # must equal data.image_size in configs/config.yaml
USE_CLAHE = True  # must equal data.clahe in configs/config.yaml
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_BYTES = 4_000_000

NAMES = json.loads((HERE / "class_names.json").read_text(encoding="utf-8"))
_session: ort.InferenceSession | None = None


def get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        _session = ort.InferenceSession(str(HERE / "model.onnx"), providers=["CPUExecutionProvider"])
    return _session


def crop_fundus(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    threshold = max(8, int(np.percentile(gray, 12)))
    mask = (gray > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if w * h < 0.1 * image.shape[0] * image.shape[1]:
        return image
    return image[y : y + h, x : x + w]


def enhance_colour(image: np.ndarray) -> np.ndarray:
    if not USE_CLAHE:
        return image
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = eq.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess(array: np.ndarray) -> np.ndarray:
    """RGB uint8 HxWx3 -> float32 1x3xSxS, identical to training."""
    array = enhance_colour(crop_fundus(array))
    array = cv2.resize(array, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    values = (array.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None], dtype=np.float32)


def quality(array: np.ndarray) -> dict:
    """Simple heuristic checks (not clinical grading)."""
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    issues = []
    if blur < 20:
        issues.append("image looks blurry")
    if brightness < 25:
        issues.append("image is very dark")
    if brightness > 200:
        issues.append("image is overexposed")
    return {
        "passed": not issues,
        "blur": round(blur, 1),
        "brightness": round(brightness, 1),
        "note": "Acceptable for screening." if not issues else "Check image: " + ", ".join(issues) + ".",
    }


def softmax(values: np.ndarray) -> np.ndarray:
    e = np.exp(values - values.max())
    return e / e.sum()


class handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._send(200, {"status": "ok", "classes": NAMES, "image_size": IMAGE_SIZE})

    def do_POST(self) -> None:
        try:
            size = int(self.headers.get("Content-Length", 0))
            if size <= 0 or size > MAX_BYTES:
                return self._send(413, {"error": "Image missing or larger than 4 MB."})
            data = self.rfile.read(size)
            array = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
            logits = get_session().run(None, {"input": preprocess(array)})[0][0]
            probs = softmax(logits.astype(np.float64))
            self._send(200, {
                "classes": NAMES,
                "probabilities": [float(p) for p in probs],
                "quality": quality(array),
            })
        except Exception as exc:  # return a readable error to the page
            self._send(400, {"error": f"Could not analyse image: {exc}"})