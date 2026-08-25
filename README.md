# FruitGuard

**Computer vision for plant disease identification — and a research project about what happens when high model accuracy meets the real world.**

**Live App:** https://fruitguard-zeta.vercel.app  
**Live API:** https://fruitguard-api.onrender.com  
**Repository:** https://github.com/ItzGhost-xD/FruitGuard-passionproject-csPS

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