"""Optional Grad-CAM overlay with a safe no-dependency fallback."""

from __future__ import annotations

import numpy as np


def make_gradcam(model: object, tensor: object, rgb_image: np.ndarray) -> np.ndarray | None:
    """Return an RGB overlay, or None when PyTorch Grad-CAM is unavailable."""
    try:
        import cv2
        import torch.nn as nn
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        if not layers:
            return None
        with GradCAM(model=model, target_layers=[layers[-1]]) as cam:
            heatmap = cam(input_tensor=tensor)[0]
        return show_cam_on_image(rgb_image.astype(np.float32) / 255, heatmap, use_rgb=True)
    except (ImportError, AttributeError, RuntimeError):
        return None
