from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from ml.config import CHECKPOINTS, RAW_DIR, RESULTS, SEED
from ml.dataset import make_loaders
from ml.models import build_model
from ml.taxonomy import class_ids


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(train)
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += labels.size(0)
    return total_loss / max(seen, 1), correct / max(seen, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a FruitGuard image classifier.")
    parser.add_argument("--model", default="mobilenet_v3", help="scratch_cnn | mobilenet_v3 | efficientnet_b0 | resnet18")
    parser.add_argument("--data", type=Path, default=RAW_DIR)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    set_seed()
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _, split_info, _, _ = make_loaders(args.data, args.batch_size)
    print(json.dumps(split_info, indent=2))

    model = build_model(args.model, num_classes=len(class_ids()), pretrained=not args.no_pretrained)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best_val = -1.0
    ckpt_path = CHECKPOINTS / f"{args.model}_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, False)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        print(f"epoch {epoch}: train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "model_name": args.model,
                    "state_dict": model.state_dict(),
                    "class_ids": class_ids(),
                    "val_acc": val_acc,
                    "split": split_info,
                },
                ckpt_path,
            )

    history_path = RESULTS / f"{args.model}_history.json"
    history_path.write_text(json.dumps({"split": split_info, "history": history}, indent=2), encoding="utf-8")
    print(f"saved {ckpt_path} (best val_acc={best_val:.3f})")


if __name__ == "__main__":
    main()
