# FruitGuard

Computer vision for fruit disease and pest identification.

**Research question:** How accurately can a computer-vision model identify common fruit diseases and pests from user-submitted images under real-world lighting and background conditions?

This is a scoped research project, not a 100-class plant encyclopedia. The first cut covers **apple, grape, peach, and tomato** (14 classes: healthy tissue plus 2–3 common conditions each). The website identifies a likely class, shows confidence, and gives short educational notes. It does **not** recommend pesticides or treatments.

## Four parts

1. **Dataset** — PlantVillage-style leaf folders in `ml/data/raw`, plus a smaller phone-photo set in `ml/data/ood_phone`.
2. **Model** — HSV histogram + SVM baseline, a from-scratch CNN, and transfer learning (MobileNetV3, EfficientNet-B0, ResNet-18).
3. **Website** — React frontend + FastAPI backend. Upload a photo, inspect quality (blur/lighting), then see ranked hypotheses.
4. **Evaluation** — accuracy, precision, recall, F1, confusion matrix, and a held-out phone-photo stress test.

## Run locally

Use Python 3.11+ from the project root.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Until you train a checkpoint, uploads still receive an image-quality report; class scores appear after `ml/checkpoints/*_best.pt` exists.

## Train and evaluate

Put images in the folder names listed in `ml/data/README.txt` and `ml/taxonomy.json`.

```bash
python -m ml.train --model mobilenet_v3 --epochs 8
python -m ml.train --model scratch_cnn --epochs 12 --no-pretrained
python -m ml.baseline_svm
python -m ml.evaluate --checkpoint ml/checkpoints/mobilenet_v3_best.pt
```

Metrics land in `ml/results/` and show up on the Research page.

## Safety

Predictions are identification hypotheses for education and research. Uncertain cases should be checked by an agricultural expert. Do not use this system to choose chemical controls.

See `docs/RESEARCH.md` for dataset logging, splits, and limitation notes you will want in a write-up.
