from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from ml.config import CHECKPOINTS, IMAGE_SIZE
from ml.taxonomy import by_id

DISCLAIMER = (
    "FruitGuard is an educational identification aid, not a diagnosis or treatment "
    "service. Do not use these results to choose pesticides or other treatments. "
    "Uncertain or high-stakes cases should be reviewed by an agricultural expert "
    "or local extension service."
)

ONNX_PATH = CHECKPOINTS / "mobilenet_v3_best.onnx"
META_PATH = CHECKPOINTS / "mobilenet_v3_best.json"

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
).reshape(1, 1, 3)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
).reshape(1, 1, 3)


def _prepare_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD

    # HWC -> CHW
    array = np.transpose(array, (2, 0, 1))

    # Add batch dimension
    array = np.expand_dims(array, axis=0)

    return np.ascontiguousarray(array, dtype=np.float32)


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / np.sum(exp)


@lru_cache(maxsize=1)
def load_runtime():
    if not ONNX_PATH.exists() or not META_PATH.exists():
        return None

    metadata = json.loads(
        META_PATH.read_text(encoding="utf-8")
    )

    options = ort.SessionOptions()

    # Keep memory use low for Render's small instance.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False

    session = ort.InferenceSession(
        str(ONNX_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    return {
        "session": session,
        "class_ids": metadata["class_ids"],
        "name": metadata["model_name"],
        "path": str(ONNX_PATH),
    }


def predict_image(image: Image.Image, top_k: int = 3) -> dict:
    runtime = load_runtime()

    if runtime is None:
        return {
            "model_available": False,
            "model_name": None,
            "top_k": [],
            "uncertain": True,
        }

    tensor = _prepare_image(image)

    session = runtime["session"]
    input_name = session.get_inputs()[0].name

    outputs = session.run(
        None,
        {input_name: tensor},
    )

    logits = outputs[0][0]
    probs = _softmax(logits)

    indices = np.argsort(probs)[::-1][:top_k]

    items = []

    for idx in indices:
        confidence = float(probs[idx])

        cid = runtime["class_ids"][int(idx)]
        meta = by_id(cid)

        items.append(
            {
                "id": cid,
                "fruit": meta["fruit"],
                "label": meta["label"],
                "status": meta["status"],
                "confidence": round(confidence, 4),
                "summary": meta["summary"],
                "look_for": meta["look_for"],
            }
        )

    top = items[0]["confidence"] if items else 0.0

    uncertain = (
        top < 0.55
        or (
            len(items) > 1
            and items[0]["confidence"] - items[1]["confidence"] < 0.12
        )
    )

    return {
        "model_available": True,
        "model_name": runtime["name"],
        "top_k": items,
        "uncertain": uncertain,
    }