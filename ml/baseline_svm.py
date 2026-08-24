from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ml.config import IMAGE_SIZE, RAW_DIR, RESULTS, SEED
from ml.dataset import MappedImageFolder, stratified_split
from ml.taxonomy import class_ids


def color_histogram(path: str, bins: int = 16) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        return np.zeros(bins * 3, dtype=np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    hists = []
    for channel in range(3):
        hist = cv2.calcHist([image], [channel], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hists.append(hist)
    return np.concatenate(hists).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Traditional OpenCV histogram + SVM baseline.")
    parser.add_argument("--data", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    ds = MappedImageFolder(args.data, transform=None)
    labels = [label for _, label in ds.samples]
    train_idx, _, test_idx = stratified_split(labels)
    x = np.stack([color_histogram(ds.paths[i]) for i in range(len(ds))])
    y = np.array(labels)

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", LinearSVC(max_iter=4000, dual=False, random_state=SEED)),
        ]
    )
    clf.fit(x[train_idx], y[train_idx])
    pred = clf.predict(x[test_idx])
    truth = y[test_idx]
    names = class_ids()
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, pred, labels=list(range(len(names))), zero_division=0
    )
    payload = {
        "model_name": "baseline_svm",
        "accuracy": float((pred == truth).mean()),
        "macro_f1": float(np.mean(f1)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "confusion_matrix": confusion_matrix(truth, pred, labels=list(range(len(names)))).tolist(),
        "report": classification_report(truth, pred, target_names=names, zero_division=0, output_dict=True),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "note": "Color-histogram SVM is the traditional computer-vision baseline, not a deep model.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "baseline_svm_eval.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("SVM accuracy:", payload["accuracy"])
    print("wrote", out)


if __name__ == "__main__":
    main()
