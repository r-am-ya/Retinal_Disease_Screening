"""Evaluate a checkpoint ONCE on the TEST split and write the dashboard artefacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.data.dataset import FundusDataset, load_or_create_class_names
from src.models.build_model import build_model

IMG_EXT = {".jpg", ".jpeg", ".png"}


def count_splits(split_dir: Path) -> dict[str, dict[str, int]]:
    """Count images per class for train/val/test straight from the folders."""
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        counts[split] = {}
        for cls in sorted(d for d in (split_dir / split).iterdir() if d.is_dir()):
            counts[split][cls.name] = sum(1 for f in cls.rglob("*") if f.suffix.lower() in IMG_EXT)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="models/best.pt")
    parser.add_argument("--history", default=None, help="training log CSV (default: derived from architecture)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    split_dir = Path(cfg["data"]["split_dir"])
    names = load_or_create_class_names(split_dir, Path("models/class_names.json"))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    arch = ckpt["architecture"]
    model = build_model(arch, len(names), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = FundusDataset(split_dir / "test", names, cfg["data"]["image_size"],
                       cfg["data"]["clahe"], cfg["data"]["green_channel"])
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    truth: list[int] = []
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, labels in loader:
            probs.append(torch.softmax(model(inputs), dim=1).numpy())
            truth.extend(labels.tolist())
    prob = np.concatenate(probs)
    y_true = np.array(truth)
    y_pred = prob.argmax(1)
    labels_idx = list(range(len(names)))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, zero_division=0)
    aucs: list[float | None] = []
    for i in labels_idx:
        try:
            aucs.append(float(roc_auc_score((y_true == i).astype(int), prob[:, i])))
        except ValueError:
            aucs.append(None)
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

    hist_path = Path(args.history or f"reports/training_log_{arch}.csv")
    history = []
    with hist_path.open(encoding="utf-8") as src:
        for row in csv.DictReader(src):
            history.append({k: (row[k] if k == "phase" else float(row[k])) for k in row})

    dashboard = {
        "model": {"architecture": arch, "image_size": cfg["data"]["image_size"],
                  "best_val_f1": float(ckpt.get("val_f1", 0.0))},
        "metrics": {
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "test_samples": int(len(y_true)),
        },
        "per_class": [
            {"class": n, "precision": float(precision[i]), "recall": float(recall[i]),
             "f1": float(f1[i]), "auc": aucs[i], "support": int(support[i])}
            for i, n in enumerate(names)
        ],
        "confusion_matrix": {"labels": names, "counts": cm.tolist()},
        "split_counts": count_splits(split_dir),
        "history": history,
    }
    Path("public").mkdir(exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    text = json.dumps(dashboard, indent=2)
    Path("public/metrics.json").write_text(text, encoding="utf-8")
    Path("reports/test_metrics.json").write_text(text, encoding="utf-8")

    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, mat, title, fmt in ((axes[0], cm, "Counts", "d"), (axes[1], norm, "Row-normalised (recall)", ".2f")):
        ax.imshow(mat, cmap="Blues")
        ax.set_xticks(labels_idx, names, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(labels_idx, names, fontsize=7)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{arch}: {title}")
        for r in labels_idx:
            for c in labels_idx:
                ax.text(c, r, format(mat[r, c], fmt), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(f"reports/figures/confusion_matrix_{arch}.png", dpi=150)
    plt.close(fig)

    print(json.dumps(dashboard["metrics"], indent=2))
    for row in dashboard["per_class"]:
        print(f"{row['class']:36s} P={row['precision']:.2f} R={row['recall']:.2f} F1={row['f1']:.2f} n={row['support']}")


if __name__ == "__main__":
    main()