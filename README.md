# FruitGuard

**Computer vision for plant disease identification — and a research project about what happens when high model accuracy meets the real world.**

[**Live App**](https://fruitguard-zeta.vercel.app)

[**Live API**](https://fruitguard-api.onrender.com)

[**Repository**](https://github.com/ItzGhost-xD/FruitGuard-passionproject-csPS)

---

## About the Project

FruitGuard is a computer vision research project built around a simple question:

> **How accurately do computer-vision models trained on controlled PlantVillage images identify common crop leaf diseases when evaluated on real-world images with varying lighting, backgrounds, framing, and image quality?**

The project began as an attempt to build a plant disease classifier, but the main goal became understanding whether a model's impressive test accuracy actually translates into reliable performance outside its training environment.

FruitGuard focuses on **apple, grape, peach, and tomato**, covering **14 healthy and diseased classes**.

The web application allows a user to upload an image and receive:

- Ranked disease predictions
- Confidence scores
- Image-quality information
- Symptoms associated with predicted classes
- An uncertainty warning when appropriate

FruitGuard is an **educational and research tool**, not a professional agricultural diagnosis or treatment service.

---

## The Research

The models were trained using selected classes from the **PlantVillage** dataset.

Controlled evaluation used a frozen train/validation/test split designed to reduce related-image leakage where metadata allowed it.

Two neural-network approaches were compared:

### MobileNetV3-Small
A transfer-learning model initialized with pretrained features.

### Scratch CNN
A convolutional neural network trained from scratch on the FruitGuard dataset.

After controlled testing, both models were evaluated on a separate **real-world out-of-distribution (OOD)** dataset created from **PlantDoc**.

PlantDoc images were never used for training.

---

## Results

| Model | PlantVillage Test Accuracy | PlantDoc OOD Accuracy | Generalization Drop |
|---|---:|---:|---:|
| **MobileNetV3-Small** | **100.00%** | **18.95%** | **81.05 percentage points** |
| **Scratch CNN** | **82.06%** | **14.74%** | **67.32 percentage points** |

The OOD evaluation contained **95 PlantDoc images across 10 compatible FruitGuard classes**.

### Main Finding

The most important result of FruitGuard was not the 100% controlled accuracy.

It was the dramatic decline when the same models were exposed to less standardized real-world images.

MobileNetV3 dropped from:

**100.00% → 18.95%**

while the scratch CNN dropped from:

**82.06% → 14.74%**

This demonstrates a major **domain-generalization gap**.

A model can perform extremely well on images similar to its training distribution while struggling with changes in:

- Backgrounds
- Lighting
- Framing
- Resolution
- Leaf orientation
- Disease severity
- Occlusion
- Overall image quality

Strong held-out accuracy alone therefore does not guarantee real-world reliability.

---

## Supported Classes

FruitGuard currently contains 14 classes across four crops.

### Apple
- Apple Scab
- Black Rot
- Cedar Apple Rust
- Healthy

### Grape
- Black Rot
- Esca / Black Measles
- Leaf Blight
- Healthy

### Peach
- Bacterial Spot
- Healthy

### Tomato
- Early Blight
- Late Blight
- Leaf Mold
- Healthy

---

## Datasets

### PlantVillage
Used for model training, validation, and controlled testing.

Source:  
https://github.com/spMohanty/PlantVillage-Dataset

### PlantDoc
Used only for external real-world OOD evaluation.

Source:  
https://github.com/pratikkayal/PlantDoc-Dataset

The PlantDoc evaluation dataset contains **95 sampled images** from classes that could be reasonably mapped to FruitGuard's taxonomy.

Classes without sufficiently reliable equivalents were not force-mapped.

---

## Dataset Split

The final PlantVillage research split contains:

| Split | Images |
|---|---:|
| Train | 9,630 |
| Validation | 2,063 |
| Test | 2,062 |
| **Total** | **13,755** |

Random seed:

```text
42
````

Where PlantVillage leaf-group metadata was available, related images were kept within the same split to reduce leakage.

See [`docs/RESEARCH.md`](docs/RESEARCH.md) for the complete methodology and experiment log.

---

## Web Application

FruitGuard includes a complete frontend and backend system.

### Frontend

* React
* Vite
* Deployed with Vercel

### Backend

* Python
* FastAPI
* PyTorch
* Pillow
* Deployed with Render

### Machine Learning

* PyTorch
* torchvision
* MobileNetV3-Small
* Custom CNN
* scikit-learn for evaluation metrics

---

## Live Deployment

### Website

[https://fruitguard-zeta.vercel.app](https://fruitguard-zeta.vercel.app)

### API

[https://fruitguard-api.onrender.com](https://fruitguard-api.onrender.com)

### API Health Check

[https://fruitguard-api.onrender.com/api/health](https://fruitguard-api.onrender.com/api/health)

Expected response:

```json
{
  "ok": true,
  "model_available": true,
  "model_name": "mobilenet_v3"
}
```

---

## Running FruitGuard Locally

Clone the repository:

```bash
git clone https://github.com/ItzGhost-xD/FruitGuard-passionproject-csPS.git
cd FruitGuard-passionproject-csPS
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the backend dependencies:

```powershell
pip install -r backend\requirements.txt
```

Start the FastAPI backend:

```powershell
python -m uvicorn backend.app.main:app --reload --port 8000
```

Then open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

The frontend will normally run at:

```text
http://localhost:5173
```

The backend will run at:

```text
http://127.0.0.1:8000
```

---

## Model Evaluation

Evaluate MobileNetV3:

```powershell
python -m ml.evaluate --checkpoint ml/checkpoints/mobilenet_v3_best.pt --ood ml/data/ood_real
```

Evaluate the scratch CNN:

```powershell
python -m ml.evaluate --checkpoint ml/checkpoints/scratch_cnn_best.pt --ood ml/data/ood_real
```

Evaluation outputs are stored in:

```text
ml/results/
```

---

## Project Structure

```text
FruitGuard/
│
├── backend/
│   └── app/
│       ├── inference.py
│       ├── main.py
│       └── schemas.py
│
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── Home.jsx
│       │   ├── Identify.jsx
│       │   ├── Research.jsx
│       │   └── About.jsx
│       └── App.jsx
│
├── ml/
│   ├── checkpoints/
│   ├── data/
│   ├── results/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── models.py
│   ├── taxonomy.json
│   ├── taxonomy.py
│   └── train.py
│
├── docs/
│   └── RESEARCH.md
│
├── prepare_plantdoc_ood.py
├── prepare_research_split.py
└── README.md
```

---

## Limitations

FruitGuard should not be treated as a production agricultural diagnostic system.

Important limitations include:

* The OOD evaluation contains only 95 images.
* Only 10 of the 14 classes had sufficiently compatible PlantDoc counterparts.
* PlantVillage contains relatively standardized imagery.
* Real-world conditions can differ significantly from the training distribution.
* High model confidence does not necessarily mean a prediction is correct.
* Further testing on independently collected field photographs would be required before considering real-world deployment for serious agricultural decisions.

The poor OOD performance is therefore not hidden as a failure of the project — it is one of the project's central research findings.

---

## Safety

FruitGuard provides **identification hypotheses for education and research**.

It does not recommend pesticides, chemicals, treatments, or agricultural interventions.

Uncertain or important cases should be reviewed by a qualified agricultural professional or local agricultural extension service.

---

## Why FruitGuard?

FruitGuard was built to explore more than just:

> *“Can I train a model that gets a high accuracy score?”*

The more important question became:

> *“Can that accuracy actually be trusted when the environment changes?”*

The answer from this experiment was clear: **not necessarily**.

FruitGuard demonstrates why evaluating machine-learning systems beyond their original dataset matters, and why impressive benchmark performance should not automatically be interpreted as real-world reliability.

---

**Built as a computer science and machine-learning research project.**

**The Developer @ FruitGuard**
