# RetinaScope — Retinal Disease Screening Dashboard

[![Live application](https://img.shields.io/badge/Live%20application-Open%20RetinaScope-087D78?style=for-the-badge)](https://retina-research-screening-aid.vercel.app/)

**RetinaScope** is a research dashboard for nine-class retinal fundus image classification. It combines a browser-based upload experience, a serverless ONNX inference service, image-quality safeguards, and a complete held-out-test evaluation view in one interface.

## Live application

### [https://retina-research-screening-aid.vercel.app/](https://retina-research-screening-aid.vercel.app/)

> **Research and educational use only — not a medical device.** RetinaScope does not diagnose disease and must not be used for clinical decisions. Every output requires review by a qualified ophthalmologist.

---

## What RetinaScope does

1. A researcher uploads a JPEG or PNG retinal fundus image.
2. The browser resizes it before transfer, then sends it to the Vercel prediction endpoint.
3. The endpoint applies the same core preprocessing used for training.
4. The deployed model returns probabilities for nine retinal classes.
5. The dashboard shows the top result, the next two differential signals, image-quality checks, and an explicit clinical-review safeguard.
6. The same page exposes the held-out test results: accuracy, F1 scores, learning curves, class-level metrics, class balance, and a confusion matrix.

The application does **not** retain uploaded images.

---

## System architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                            Researcher browser                           │
│  Upload JPEG / PNG → preview → browser-side resize → results dashboard │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │ POST /api/predict
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Vercel Python serverless function                  │
│  1. Decode RGB image                 4. ImageNet normalization           │
│  2. Fundus-border crop               5. MobileNetV3-Large ONNX Runtime   │
│  3. CLAHE colour enhancement         6. Softmax + quality checks         │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │ JSON probabilities + quality result
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Research evidence interface                     │
│  Top signal • confidence • safeguards • F1/accuracy • AUC • curves • CM │
└─────────────────────────────────────────────────────────────────────────┘
```

The static UI and `metrics.json` are served from `public/`; the prediction function and the exported ONNX weights are served from `api/`.

---

## Models used and why

Three model families were considered during development. They did **not** form an ensemble: only the final MobileNetV3 model is deployed.

| Model | Role in the project | Training outcome | Deployment status |
| --- | --- | ---: | --- |
| **Baseline CNN** | A compact three-convolution-block reference model used to establish a basic performance floor and validate the training pipeline. | Best observed validation macro F1: **0.247** | Not deployed |
| **EfficientNet-B0** | An early ImageNet-pretrained transfer-learning candidate used to explore a stronger convolutional backbone. | Best observed validation macro F1: **0.173** in its short experimental run | Not deployed |
| **MobileNetV3 Large 100** | Final selected classifier. It offered the strongest validation result among the completed candidates and an efficient serverless inference footprint. | Best validation macro F1: **0.635** | **Deployed** |

### Final model: MobileNetV3 Large 100

The production model is an ImageNet-pretrained **MobileNetV3 Large 100** backbone with its classification head adapted to nine output classes. It was trained in two phases:

1. **Head training:** the pretrained backbone was frozen and the classification head was optimized for 2 epochs.
2. **Fine-tuning:** the full network was unfrozen and optimized for 6 additional epochs.

The checkpoint with the best validation macro F1 was exported from PyTorch to **ONNX**. The serverless endpoint runs the ONNX artifact with **ONNX Runtime on CPU**. The exported model is approximately 16.8 MB and was numerically compared with PyTorch output during export (maximum absolute difference below `1e-3`).

### Training design

- **Input size:** 192 × 192 RGB
- **Optimizer:** AdamW with cosine annealing during fine-tuning
- **Loss:** class-weighted cross-entropy with label smoothing
- **Augmentation:** mild training-only image augmentation
- **Model selection:** best validation macro F1
- **Split seed:** 42
- **Leakage precaution:** exact perceptual-hash duplicate clusters were kept in one partition

---

## Preprocessing and inference

The deployed API deliberately uses training-compatible image preparation:

```text
Fundus image
  → RGB conversion
  → fundus-border crop
  → CLAHE contrast enhancement in LAB colour space
  → resize to 192 × 192
  → ImageNet mean / standard-deviation normalization
  → MobileNetV3 Large ONNX inference
  → softmax class probabilities
```

The endpoint also computes non-diagnostic quality heuristics:

- blur, using variance of the Laplacian;
- very dark image detection;
- overexposure detection.

These checks do not grade image quality clinically. They exist to prompt cautious interpretation when the uploaded image is unsuitable for screening.

---

## Predicted classes

| # | Class |
| ---: | --- |
| 1 | Central serous chorioretinopathy |
| 2 | Diabetic retinopathy |
| 3 | Disc edema |
| 4 | Glaucoma |
| 5 | Healthy |
| 6 | Macular scar |
| 7 | Myopia |
| 8 | Retinal detachment |
| 9 | Retinitis pigmentosa |

---

## Held-out test evaluation

The final model was evaluated once on a held-out test split of **796 original images**. The test split was not used for checkpoint selection or training.

| Metric | Score |
| --- | ---: |
| Accuracy | **67.96%** |
| Macro F1 | **62.18%** |
| Weighted F1 | **68.55%** |
| Test images | **796** |

### Data split

| Partition | Images | Purpose |
| --- | ---: | --- |
| Train | 3,722 | Parameter learning and training-only augmentation |
| Validation | 800 | Model selection and early-stopping signal |
| Test | 796 | Final, held-out evaluation |
| **Total** | **5,318** | Original images |

### Per-class held-out results

| Class | Precision | Recall | F1 | One-vs-rest AUC | Test images |
| --- | ---: | ---: | ---: | ---: | ---: |
| Central serous chorioretinopathy | 0.286 | 0.400 | 0.333 | 0.849 | 15 |
| Diabetic retinopathy | 0.960 | 0.739 | 0.835 | 0.961 | 226 |
| Disc edema | 0.419 | 0.947 | 0.581 | 0.991 | 19 |
| Glaucoma | 0.739 | 0.490 | 0.589 | 0.879 | 202 |
| Healthy | 0.675 | 0.745 | 0.708 | 0.911 | 153 |
| Macular scar | 0.568 | 0.636 | 0.600 | 0.879 | 66 |
| Myopia | 0.543 | 0.760 | 0.633 | 0.942 | 75 |
| Retinal detachment | 0.442 | 1.000 | 0.613 | 0.997 | 19 |
| Retinitis pigmentosa | 0.576 | 0.905 | 0.704 | 0.993 | 21 |

Classes with fewer than 30 test images should be treated as indicative rather than conclusive. The live dashboard presents these metrics alongside the confusion matrix and optimization curves.

---

## Repository layout

```text
api/          Vercel endpoint, class labels, and final ONNX model
public/       Website UI, dashboard JavaScript, styles, and metrics data
src/          Reproducible data, training, evaluation, export, and Grad-CAM code
configs/      Model and preprocessing configuration
reports/      Generated evaluation figures and training/split artefacts
```

Large source-image archives and PyTorch checkpoint files are intentionally excluded from Git.

---

## Run locally

### Static dashboard

```bash
npm run dev
```

The dashboard opens at `http://localhost:3000`. For the local `/api/predict` serverless route, use Vercel's local development tooling.

### Python environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Deploy to Vercel

```bash
npx vercel --prod
```

`vercel.json` configures the static output and bundles the ONNX model with the prediction function. `pyproject.toml` declares the Vercel Python entrypoint and its runtime dependencies.

---

## Limitations and responsible use

- This is a research prototype, not a clinical decision-support device.
- Patient identifiers were unavailable; patient-level leakage cannot be fully excluded.
- The dataset comes from two hospitals in one country, so performance may not transfer to other devices, populations, or care settings.
- No external validation study has been completed.
- Several classes have small held-out test samples.
- A high model score is not proof of disease; a low score is not evidence that disease is absent.
- Uploaded images must be fundus photographs. Inputs from other modalities may produce meaningless predictions.

## Data acknowledgement

The project documents the image source as Sharmin et al. (2024), *Data in Brief* 57, 110979. The external-eye pterygium class was excluded. Augmented image files were not used for validation or testing because they could not be reliably linked to their original sources without risking data leakage.
