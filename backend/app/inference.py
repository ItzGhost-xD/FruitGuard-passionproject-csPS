from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.config import CHECKPOINTS, IMAGE_SIZE  # noqa: E402
from ml.models import build_model  # noqa: E402
from ml.taxonomy import by_id, class_ids  # noqa: E402

DISCLAIMER = (
    "FruitGuard is an educational identification aid, not a diagnosis or treatment "
    "service. Do not use these results to choose pesticides or other treatments. "
    "Uncertain or high-stakes cases should be reviewed by an agricultural expert "
    "or local extension service."
)

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _pick_checkpoint() -> Path | None:
    preferred = CHECKPOINTS / "mobilenet_v3_best.pt"
    if preferred.exists():
        return preferred
    found = sorted(CHECKPOINTS.glob("*_best.pt"))
    return found[-1] if found else None


@lru_cache(maxsize=1)
def load_runtime():
    path = _pick_checkpoint()
    if path is None:
        return None
    device = torch.device("cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], num_classes=len(ckpt["class_ids"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {"model": model, "class_ids": ckpt["class_ids"], "name": ckpt["model_name"], "path": str(path)}


@torch.no_grad()
def predict_image(image: Image.Image, top_k: int = 3) -> dict:
    runtime = load_runtime()
    if runtime is None:
        return {
            "model_available": False,
            "model_name": None,
            "top_k": [],
            "uncertain": True,
        }

    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0)
    logits = runtime["model"](tensor)
    probs = torch.softmax(logits, dim=1)[0]
    values, indices = torch.topk(probs, k=min(top_k, probs.numel()))
    items = []
    for conf, idx in zip(values.tolist(), indices.tolist()):
        cid = runtime["class_ids"][idx]
        meta = by_id(cid)
        items.append(
            {
                "id": cid,
                "fruit": meta["fruit"],
                "label": meta["label"],
                "status": meta["status"],
                "confidence": round(float(conf), 4),
                "summary": meta["summary"],
                "look_for": meta["look_for"],
            }
        )
    top = items[0]["confidence"] if items else 0.0
    return {
        "model_available": True,
        "model_name": runtime["name"],
        "top_k": items,
        "uncertain": top < 0.55 or (len(items) > 1 and items[0]["confidence"] - items[1]["confidence"] < 0.12),
    }
