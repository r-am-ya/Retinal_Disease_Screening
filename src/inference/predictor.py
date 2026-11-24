"""Singleton ONNX/PyTorch screening predictor; never stores uploads by default."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.preprocessing import preprocess_image
from src.inference.quality import check_image_quality

ROOT = Path(__file__).resolve().parents[2]
HIGH_RISK = {"retinal_detachment", "glaucoma", "diabetic_retinopathy"}


class Predictor:
    def __init__(self, model_path: Path = ROOT / "models" / "model.onnx") -> None:
        self.class_names = json.loads((ROOT / "models" / "class_names.json").read_text())
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image: Image.Image, top_k: int = 3) -> dict[str, object]:
        quality = check_image_quality(image)
        processed = preprocess_image(image, normalise=True)
        batch = processed.transpose(2, 0, 1)[None].astype(np.float32)
        logits = self.session.run(None, {self.input_name: batch})[0][0]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        order = np.argsort(probabilities)[::-1][:top_k]
        predictions = [{"class": self.class_names[int(index)], "probability": float(probabilities[index])} for index in order]
        best = predictions[0]
        refer = (not quality.passed or best["probability"] < 0.60 or (best["class"] in HIGH_RISK and best["probability"] >= 0.35))
        return {"disclaimer": "Screening aid only. Not a medical diagnosis. Consult a qualified ophthalmologist.", "predictions": predictions, "quality": quality.to_dict(), "refer_to_ophthalmologist": refer, "preprocessed": processed}


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()
