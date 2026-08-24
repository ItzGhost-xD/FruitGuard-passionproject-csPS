from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.inference import DISCLAIMER, load_runtime, predict_image
from backend.app.schemas import PredictResponse
from ml.quality import image_quality
from ml.taxonomy import load_taxonomy

RESULTS = ROOT / "ml" / "results"

app = FastAPI(
    title="FruitGuard",
    description="Computer vision API for fruit disease and pest identification research.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    runtime = load_runtime()
    return {
        "ok": True,
        "model_available": runtime is not None,
        "model_name": None if runtime is None else runtime["name"],
    }


@app.get("/api/taxonomy")
def taxonomy():
    return load_taxonomy()


@app.get("/api/research")
def research():
    latest = RESULTS / "latest.json"
    evals = sorted(RESULTS.glob("*_eval.json"))
    payloads = []
    for path in evals:
        try:
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    placeholder = {}
    if latest.exists():
        placeholder = json.loads(latest.read_text(encoding="utf-8"))
    return {
        "research_question": load_taxonomy()["research_question"],
        "evaluations": payloads,
        "placeholder": placeholder,
        "experiment_plan": [
            "Traditional baseline: HSV color histograms + linear SVM",
            "From-scratch CNN trained only on the in-scope leaf/fruit images",
            "Transfer learning: MobileNetV3-Small, EfficientNet-B0, ResNet-18",
            "Metrics: accuracy, precision, recall, F1, confusion matrix",
            "Stress test: phone photos in ml/data/ood_phone that never enter training",
        ],
    }


@app.post("/api/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    try:
        image = Image.open(file.file)
        image.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read that image.") from exc

    quality = image_quality(image)
    pred = predict_image(image)
    advice = (
        "This case looks uncertain. Treat the ranking as a hypothesis and confirm with an agricultural expert."
        if pred["uncertain"]
        else "The model is relatively confident, but field confirmation is still recommended."
    )
    if not pred["model_available"]:
        advice = (
            "No trained checkpoint is loaded yet. Image quality was still analyzed. "
            "Train a model with `python -m ml.train --model mobilenet_v3` after adding PlantVillage images."
        )
    return PredictResponse(
        model_available=pred["model_available"],
        model_name=pred["model_name"],
        disclaimer=DISCLAIMER,
        quality=quality,
        top_k=pred["top_k"],
        uncertain=pred["uncertain"],
        advice=advice,
    )
