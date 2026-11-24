# RetinaScope

**RetinaScope** is a research-facing retinal fundus image screening dashboard. Upload a JPEG or PNG fundus image to receive a nine-class model output, image-quality checks, and a transparent view of the model's held-out evaluation results.

> **Live app:** [retina-research-screening-aid.vercel.app](https://retina-research-screening-aid.vercel.app/)

> **Research use only.** This project is not a medical device, does not diagnose disease, and must not be used to make clinical decisions. Any result requires review by a qualified ophthalmologist.

## What the app provides

- Fundus-image upload with browser-side resizing and drag-and-drop support.
- Nine-class ONNX inference served by a Vercel Python function.
- Ranked class probabilities and simple image-quality checks for blur and exposure.
- A conservative clinical-review safeguard for low confidence, high-risk output, or poor image quality.
- Research dashboard with held-out accuracy, macro/weighted F1, per-class precision/recall/F1/AUC, training curves, data-split counts, and confusion matrix.
- Responsive interface designed for desktop and mobile review.

## Classes

The classifier outputs one of the following labels:

1. Central serous chorioretinopathy
2. Diabetic retinopathy
3. Disc edema
4. Glaucoma
5. Healthy
6. Macular scar
7. Myopia
8. Retinal detachment
9. Retinitis pigmentosa

## Evaluation snapshot

The figures displayed in the app are calculated once on a held-out test split of **796 images**.

| Metric | Result |
| --- | ---: |
| Accuracy | 67.96% |
| Macro F1 | 62.18% |
| Weighted F1 | 68.55% |
| Best validation macro F1 | 63.54% |

The dashboard includes class-level values and support counts. Some classes have small test samples, so their individual results should be interpreted cautiously.

## Model and preprocessing

- **Architecture:** MobileNetV3 Large (ImageNet-pretrained during training)
- **Input:** 192 × 192 RGB fundus image
- **Serving format:** ONNX Runtime on CPU
- **Preprocessing:** fundus border crop, CLAHE colour enhancement, resize, and ImageNet normalization
- **Training approach:** class-weighted cross-entropy with label smoothing and mild augmentation
- **Checkpoint selection:** best validation macro F1

The inference preprocessing in `api/predict.py` intentionally mirrors the training preprocessing.

## Project structure

```text
api/                 Vercel serverless prediction endpoint and ONNX model
public/              Static dashboard interface and evaluation data
src/data/            Dataset inspection, preprocessing, and split tooling
src/models/          Training, ONNX export, evaluation, and Grad-CAM utilities
configs/             Training and preprocessing configuration
reports/             Evaluation artefacts and data/split documentation
```

## Run locally

### Dashboard only

```bash
npm install
npm run dev
```

This serves the static UI at `http://localhost:3000`. The `/api/predict` route is provided by Vercel, so use Vercel's local development tooling when testing uploads locally.

### Python environment

For inference dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

For training and evaluation, install the dependencies in `requirements-train.txt` and place the prepared data splits under `data/splits/`.

## Deploy to Vercel

The repository includes `vercel.json` and `pyproject.toml` for the Python function entrypoint and inference dependencies.

```bash
npx vercel --prod
```

Vercel deploys the static interface from `public/` and the prediction function from `api/predict.py`.

## Important limitations

- The dataset is not patient-level; patient-level leakage cannot be ruled out.
- Validation and test data are original images, but external validation has not been performed.
- Data originate from two hospitals in one country and may not generalize to different populations or devices.
- Some conditions have small held-out test sets.
- Model confidence is not clinical certainty, and a low-confidence result is not evidence that disease is absent.

## Data acknowledgement

The dashboard documents the source as Sharmin et al. (2024), *Data in Brief* 57, 110979. See [`reports/data_report.md`](reports/data_report.md) for the repository's dataset notes and split documentation.
