from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "ml"
DATA_DIR = ML_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OOD_DIR = DATA_DIR / "ood_phone"
CHECKPOINTS = ML_DIR / "checkpoints"
RESULTS = ML_DIR / "results"
TAXONOMY_PATH = ML_DIR / "taxonomy.json"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # Windows-safe default
SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

MODELS = ("baseline_svm", "scratch_cnn", "mobilenet_v3", "efficientnet_b0", "resnet18")
