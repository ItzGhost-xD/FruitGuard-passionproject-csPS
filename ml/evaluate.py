from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from ml.config import CHECKPOINTS, OOD_DIR, RAW_DIR, RESULTS
from ml.dataset import MappedImageFolder, eval_transforms, make_loaders
from ml.models import build_model
from ml.taxonomy import class_ids


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def metrics_block(y_true, y_pred, labels: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(labels))), zero_division=0
    )
    report = classification_report(y_true, y_pred, labels=list(range(len(labels))), target_names=labels, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))).tolist()
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return {
        "accuracy": acc,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            labels[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(labels))
        },
        "confusion_matrix": cm,
        "sklearn_report": report,
    }


def load_checkpoint(path: Path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], num_classes=len(ckpt["class_ids"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def evaluate_ood(model, device, labels: list[str], ood_dir: Path) -> dict | None:
    if not ood_dir.exists():
        return None

    try:
        ds = MappedImageFolder(ood_dir, transform=eval_transforms())
    except Exception:
        return None
    if len(ds) == 0:
        return None
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    y_true, y_pred = predict(model, loader, device)
    block = metrics_block(y_true, y_pred, labels)
    block["n_images"] = int(len(ds))
    block["note"] = (
        "External real-world PlantDoc images. The gap between controlled "
        "PlantVillage performance and OOD performance is a central result "
        "of this research."
    )
    return block


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a FruitGuard checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=RAW_DIR)
    parser.add_argument("--ood", type=Path, default=OOD_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        candidates = sorted(CHECKPOINTS.glob("*_best.pt"))
        if not candidates:
            raise FileNotFoundError("No checkpoint found. Train a model first.")
        ckpt_path = candidates[-1]

    model, ckpt = load_checkpoint(ckpt_path, device)
    labels = ckpt.get("class_ids", class_ids())
    _, _, test_loader, split_info, _, _ = make_loaders(args.data, batch_size=64)
    y_true, y_pred = predict(model, test_loader, device)

    payload = {
        "checkpoint": str(ckpt_path),
        "model_name": ckpt.get("model_name"),
        "val_acc_at_save": ckpt.get("val_acc"),
        "split": split_info,
        "held_out_test": metrics_block(y_true, y_pred, labels),
        "ood_real": evaluate_ood(model, device, labels, args.ood),
        "class_ids": labels,
        "limitations": [
            "PlantVillage images are relatively clean compared with orchard phone photos.",
            "Similar foliar symptoms can confuse classes (e.g. early vs late blight).",
            "Predictions are identification hypotheses, not agronomic prescriptions.",
        ],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{ckpt.get('model_name', 'model')}_eval.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("checkpoint", "model_name", "held_out_test") if k != "held_out_test"}, indent=2))
    print("test accuracy:", payload["held_out_test"]["accuracy"])
    print("wrote", out)


if __name__ == "__main__":
    main()
