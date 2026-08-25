from pathlib import Path
import csv
import random
import requests
from urllib.parse import quote

SEED = 42
MAX_PER_CLASS = 10

OUTPUT_ROOT = Path("ml/data/ood_real")
API_ROOT = "https://api.github.com/repos/pratikkayal/PlantDoc-Dataset/contents/train"

CLASS_MAP = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Apple leaf": "Apple___healthy",

    "grape leaf black rot": "Grape___Black_rot",
    "grape leaf": "Grape___healthy",

    "Peach leaf": "Peach___healthy",

    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato leaf": "Tomato___healthy",
}

rng = random.Random(SEED)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

manifest_rows = []

print("Downloading PlantDoc OOD images...")
print("=" * 60)

for source_class, target_class in CLASS_MAP.items():
    url = f"{API_ROOT}/{quote(source_class, safe='')}"

    r = requests.get(url, timeout=30)

    if r.status_code != 200:
        print(f"SKIPPED: {source_class} ({r.status_code})")
        continue

    entries = r.json()

    images = [
        item for item in entries
        if item.get("type") == "file"
        and item["name"].lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    rng.shuffle(images)
    selected = images[:MAX_PER_CLASS]

    target_dir = OUTPUT_ROOT / target_class
    target_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0

    for i, item in enumerate(selected, start=1):
        download_url = item.get("download_url")

        if not download_url:
            continue

        try:
            img = requests.get(download_url, timeout=30)
            img.raise_for_status()
        except Exception as e:
            print("Failed:", item["name"], e)
            continue

        filename = f"plantdoc_{i:03d}.jpg"
        dst = target_dir / filename

        dst.write_bytes(img.content)

        manifest_rows.append({
            "filename": str(dst.relative_to(OUTPUT_ROOT)),
            "class_name": target_class,
            "source": "PlantDoc",
            "source_original_class": source_class,
            "source_file": item["name"],
            "source_url": download_url,
            "source_type": "public_dataset",
            "notes": "external real-world OOD image",
        })

        downloaded += 1

    print(f"{target_class}: {downloaded}")

manifest_path = OUTPUT_ROOT / "manifest.csv"

with manifest_path.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "class_name",
            "source",
            "source_original_class",
            "source_file",
            "source_url",
            "source_type",
            "notes",
        ],
    )

    writer.writeheader()
    writer.writerows(manifest_rows)

print()
print("=" * 60)
print("Total OOD images:", len(manifest_rows))
print("Manifest:", manifest_path.resolve())