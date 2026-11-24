from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml

from src.models.build_model import build_model


def main() -> None:
    cfg = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    size = int(cfg["data"]["image_size"])
    ckpt_path, onnx_path = Path("models/best.pt"), Path("models/model.onnx")
    class_names = json.loads(Path("models/class_names.json").read_text(encoding="utf-8"))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["architecture"], num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(model, dummy, str(onnx_path), input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    test_in = torch.randn(8, 3, size, size)
    with torch.no_grad():
        pt_out = model(test_in).numpy()
    ort_out = sess.run(None, {"input": test_in.numpy()})[0]
    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    print(f"Exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB), input {size}x{size} | max diff {max_diff:.2e}")
    assert max_diff < 1e-3


if __name__ == "__main__":
    main()