from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from ml.config import BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS, SEED, TEST_FRACTION, VAL_FRACTION
from ml.taxonomy import class_ids, folder_to_id


def train_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def eval_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class MappedImageFolder(Dataset):
    def __init__(self, root: Path, transform):
        allowed = folder_to_id()
        inner = ImageFolder(root=str(root), transform=transform)
        id_list = class_ids()
        self.samples: list[tuple[str, int]] = []
        self.paths: list[str] = []
        remap = {}
        for folder_name, idx in inner.class_to_idx.items():
            if folder_name in allowed:
                remap[idx] = id_list.index(allowed[folder_name])
        for path, raw_idx in inner.samples:
            if raw_idx in remap:
                self.samples.append((path, remap[raw_idx]))
                self.paths.append(path)
        self.transform = transform
        self.loader = inner.loader

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = self.loader(path)
        if self.transform:
            image = self.transform(image)
        return image, label


def stratified_split(labels: list[int], seed: int = SEED) -> tuple[list[int], list[int], list[int]]:
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        by_class.setdefault(label, []).append(i)
    train, val, test = [], [], []
    for indices in by_class.values():
        rng.shuffle(indices)
        n = len(indices)
        n_test = max(1, int(n * TEST_FRACTION)) if n >= 10 else max(1, n // 6 or 0)
        n_val = max(1, int(n * VAL_FRACTION)) if n >= 10 else max(1, n // 6 or 0)
        if n_test + n_val >= n:
            n_test = max(1, n // 5)
            n_val = max(1, n // 5)
            if n_test + n_val >= n:
                n_test, n_val = 0, 0
        test.extend(indices[:n_test])
        val.extend(indices[n_test : n_test + n_val])
        train.extend(indices[n_test + n_val :])
    return train, val, test


def class_counts(labels: list[int]) -> dict[str, int]:
    ids = class_ids()
    counts = Counter(labels)
    return {ids[i]: counts.get(i, 0) for i in range(len(ids))}


def make_loaders(data_root: Path, batch_size: int = BATCH_SIZE):
    probe = MappedImageFolder(data_root, transform=None)
    if len(probe) == 0:
        raise FileNotFoundError(
            f"No in-scope images found under {data_root}. "
            "Expected PlantVillage-style folders named in taxonomy.json."
        )
    labels = [label for _, label in probe.samples]
    train_idx, val_idx, test_idx = stratified_split(labels)

    train_ds = MappedImageFolder(data_root, transform=train_transforms())
    eval_ds = MappedImageFolder(data_root, transform=eval_transforms())

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        Subset(eval_ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        Subset(eval_ds, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    split_info = {
        "n_total": len(probe),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "class_counts": class_counts(labels),
    }
    return train_loader, val_loader, test_loader, split_info, test_idx, probe.paths
