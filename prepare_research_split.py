from pathlib import Path
import csv
import json
import random
from collections import defaultdict, Counter

SEED = 42

# Change only if your folders are somewhere else
PLANTVILLAGE_ROOT = Path(r"..\PlantVillage-official")
COLOR_ROOT = PLANTVILLAGE_ROOT / "raw" / "color"
LEAF_MAP_PATH = PLANTVILLAGE_ROOT / "leaf-map.json"

OUTPUT_DIR = Path("ml/data/splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "fruitguard_split_manifest.csv"
LEAF_MAP_ALIASES = {
    "Apple_Frogeye Spot": "Apple___Black_rot",
}
TARGET_CLASSES = {
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy",
}

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
UNGROUPED_CLASSES = {
    "Grape___healthy",
}
def add_ungrouped_classes(rows):
    rng = random.Random(SEED)

    for class_name in UNGROUPED_CLASSES:
        class_dir = COLOR_ROOT / class_name

        images = sorted(
            [
                p for p in class_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )

        rng.shuffle(images)

        n = len(images)
        n_test = round(n * TEST_FRACTION)
        n_val = round(n * VAL_FRACTION)

        test_images = images[:n_test]
        val_images = images[n_test:n_test + n_val]
        train_images = images[n_test + n_val:]

        for split, split_images in [
            ("train", train_images),
            ("val", val_images),
            ("test", test_images),
        ]:
            for image_path in split_images:
                rows.append(
                    {
                        "split": split,
                        "class_name": class_name,
                        "leaf_id": "",
                        "image_id": image_path.stem,
                        "path": str(image_path.resolve()),
                    }
                )

def parse_leaf_map():
    with open(LEAF_MAP_PATH, encoding="utf-8") as f:
        leaf_map = json.load(f)

    grouped = []

    for leaf_id, entries in leaf_map.items():
        valid_entries = []

        for entry in entries:
            if ":::" not in entry:
                continue

            class_name, image_id = entry.rsplit(":::", 1)

            class_name = LEAF_MAP_ALIASES.get(class_name, class_name)

            if class_name not in TARGET_CLASSES:
                continue

            valid_entries.append(
                {
                    "leaf_id": leaf_id,
                    "class_name": class_name,
                    "image_id": image_id,
                }
            )

        if valid_entries:
            grouped.append((leaf_id, valid_entries))

    return grouped


def find_image(class_name, image_id):
    class_dir = COLOR_ROOT / class_name

    if not class_dir.exists():
        return None

    image_id_clean = image_id.replace(".0", "")

    candidates = list(class_dir.glob(f"*{image_id_clean}*"))

    if not candidates:
        return None

    return candidates[0]


def split_groups(groups):
    rng = random.Random(SEED)

    by_class = defaultdict(list)

    for leaf_id, entries in groups:
        # In normal PlantVillage grouping, one leaf group should map to one class.
        classes = {x["class_name"] for x in entries}

        if len(classes) != 1:
            continue

        class_name = next(iter(classes))
        by_class[class_name].append((leaf_id, entries))

    split_assignment = {}

    for class_name, class_groups in by_class.items():
        rng.shuffle(class_groups)

        n = len(class_groups)

        n_test = max(1, round(n * TEST_FRACTION))
        n_val = max(1, round(n * VAL_FRACTION))

        if n_test + n_val >= n:
            n_test = max(1, n // 5)
            n_val = max(1, n // 5)

        test_groups = class_groups[:n_test]
        val_groups = class_groups[n_test:n_test + n_val]
        train_groups = class_groups[n_test + n_val:]

        for leaf_id, entries in train_groups:
            split_assignment[leaf_id] = "train"

        for leaf_id, entries in val_groups:
            split_assignment[leaf_id] = "val"

        for leaf_id, entries in test_groups:
            split_assignment[leaf_id] = "test"

    return split_assignment


def main():
    print("Reading official PlantVillage leaf grouping...")
    groups = parse_leaf_map()

    print(f"Relevant leaf groups found: {len(groups)}")

    assignments = split_groups(groups)

    rows = []
    missing = []

    for leaf_id, entries in groups:
        split = assignments.get(leaf_id)

        if not split:
            continue

        for item in entries:
            image_path = find_image(
                item["class_name"],
                item["image_id"],
            )

            if image_path is None:
                missing.append(item)
                continue

            rows.append(
                {
                    "split": split,
                    "class_name": item["class_name"],
                    "leaf_id": leaf_id,
                    "image_id": item["image_id"],
                    "path": str(image_path.resolve()),
                }
            )
    add_ungrouped_classes(rows)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "class_name",
                "leaf_id",
                "image_id",
                "path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Split summary")
    print("=" * 50)

    split_counts = Counter(row["split"] for row in rows)

    for split in ["train", "val", "test"]:
        print(f"{split}: {split_counts[split]} images")

    print()
    print("Per-class counts")
    print("=" * 50)

    for class_name in sorted(TARGET_CLASSES):
        counts = Counter(
            row["split"]
            for row in rows
            if row["class_name"] == class_name
        )

        total = sum(counts.values())

        print(
            f"{class_name}: "
            f"train={counts['train']} "
            f"val={counts['val']} "
            f"test={counts['test']} "
            f"total={total}"
        )

    print()
    print(f"Total images in manifest: {len(rows)}")
    print(f"Missing image matches: {len(missing)}")
    if missing:
        print("\nMissing entries:")
    for item in missing:
        print(item)
    print(f"Manifest saved to: {OUTPUT_CSV.resolve()}")

    # Leakage check
    split_leaf_ids = defaultdict(set)

    for row in rows:
        if row["leaf_id"]:
            split_leaf_ids[row["split"]].add(row["leaf_id"])

    train_val_overlap = split_leaf_ids["train"] & split_leaf_ids["val"]
    train_test_overlap = split_leaf_ids["train"] & split_leaf_ids["test"]
    val_test_overlap = split_leaf_ids["val"] & split_leaf_ids["test"]

    print()
    print("Leakage check")
    print("=" * 50)
    print("train/val overlapping leaf groups:", len(train_val_overlap))
    print("train/test overlapping leaf groups:", len(train_test_overlap))
    print("val/test overlapping leaf groups:", len(val_test_overlap))


if __name__ == "__main__":
    main()