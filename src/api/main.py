"""FastAPI prediction service."""

from __future__ import annotations

from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.inference.predictor import get_predictor

app = FastAPI(title="Retinal Screening Aid")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "disclaimer": "Screening aid only. Not a medical diagnosis. Consult a qualified ophthalmologist."}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, object]:
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(415, "Only JPEG and PNG images are supported.")
    content = await image.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image exceeds 10 MB limit.")
    try:
        uploaded = Image.open(BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as error:
        raise HTTPException(422, "Unreadable image.") from error
    result = get_predictor().predict(uploaded)
    result.pop("preprocessed", None)
    return result
