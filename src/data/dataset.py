from __future__ import annotations
import json
from pathlib import Path
import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset
from src.data.preprocessing import IMAGENET_MEAN, IMAGENET_STD, preprocess_image

def get_train_transforms() -> A.Compose:
    return A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5), A.Affine(scale=(0.95, 1.05), translate_percent=(-0.03, 0.03), rotate=(-15, 15), p=0.5), A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4)])

class FundusDataset(Dataset):
    def __init__(self, split_dir: Path, class_names: list[str], image_size: int = 224, clahe: bool = True, green_channel: bool = False, augment: bool = False, max_per_class: int | None = None) -> None:
        self.samples: list[tuple[Path, int]] = []
        self.image_size, self.clahe, self.green_channel = image_size, clahe, green_channel
        self.transforms = get_train_transforms() if augment else None
        for idx, cls_name in enumerate(class_names):
            cls_dir = split_dir / cls_name
            if not cls_dir.exists(): continue
            files = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
            if max_per_class: files = files[:max_per_class]
            self.samples.extend((f, idx) for f in files)
    def __len__(self) -> int: return len(self.samples)
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        img = preprocess_image(path, image_size=self.image_size, clahe=self.clahe, green_channel=self.green_channel, normalise=False)
        if self.transforms is not None: img = self.transforms(image=img)['image']
        values = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(values.transpose(2, 0, 1)).float(), label

def load_or_create_class_names(split_dir: Path, out_path: Path) -> list[str]:
    if out_path.exists(): return json.loads(out_path.read_text(encoding='utf-8'))
    names = sorted(d.name for d in (split_dir / 'train').iterdir() if d.is_dir())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(names, indent=2), encoding='utf-8')
    return names
