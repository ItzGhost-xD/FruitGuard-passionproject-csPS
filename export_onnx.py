from __future__ import annotations

import json
from pathlib import Path

import torch

from ml.config import IMAGE_SIZE
from ml.models import build_model

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "ml" / "checkpoints" / "mobilenet_v3_best.pt"
ONNX_PATH = ROOT / "ml" / "checkpoints" / "mobilenet_v3_best.onnx"
META_PATH = ROOT / "ml" / "checkpoints" / "mobilenet_v3_best.json"


def main():
    print(f"Loading checkpoint: {CHECKPOINT}")

    ckpt = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model_name = ckpt["model_name"]
    class_ids = ckpt["class_ids"]

    model = build_model(
        model_name,
        num_classes=len(class_ids),
        pretrained=False,
    )

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    print(f"Exporting ONNX model to: {ONNX_PATH}")

    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    metadata = {
        "model_name": model_name,
        "class_ids": class_ids,
        "image_size": IMAGE_SIZE,
    }

    META_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print("Export complete.")
    print(f"ONNX:     {ONNX_PATH}")
    print(f"Metadata: {META_PATH}")
    print(f"Classes:  {len(class_ids)}")


if __name__ == "__main__":
    main()