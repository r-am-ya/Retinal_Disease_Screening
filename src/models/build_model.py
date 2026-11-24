"""Model factory for the baseline CNN and timm backbones."""

from __future__ import annotations

import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, classes))

    def forward(self, values: object) -> object:
        return self.head(self.features(values))


def build_model(architecture: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if architecture == "baseline_cnn":
        return BaselineCNN(num_classes)
    import timm

    return timm.create_model(architecture, pretrained=pretrained, num_classes=num_classes)
