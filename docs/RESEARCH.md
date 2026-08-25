# FruitGuard — Final Research Report

> **Research integrity note:** FruitGuard reports the results that were actually obtained. Poor real-world performance is treated as a research finding rather than hidden or replaced with more favorable results.

---

## Developer's Note

FruitGuard started with what seemed like a straightforward idea: build a computer-vision model capable of identifying diseases from images of plant leaves.

But while working on the project, a more interesting question appeared.

A machine-learning model can achieve an impressive accuracy score on a test dataset, but what happens when the images stop looking like the dataset?

I did not want FruitGuard to become another project where a model reaches a high percentage, the number gets displayed in a README, and the project is considered finished.

I wanted to know whether that number could actually be trusted.

That became the real purpose of FruitGuard.

The project compares computer-vision models trained on the controlled PlantVillage dataset and then evaluates the same frozen models on external PlantDoc images containing more realistic backgrounds, lighting, framing, orientations, and image conditions.

The difference was substantial.

The strongest model achieved 100% accuracy on the controlled PlantVillage test set, but only approximately 19% accuracy on the real-world OOD evaluation.

Instead of treating that result as a failure, I consider it the most important part of the project.

FruitGuard taught me that developing machine learning systems is not only about making models perform well. It is also about understanding where they fail, why they fail, and whether the environment used to evaluate them reflects the environment in which they might actually be used.

This project therefore became less about creating the "best plant disease classifier" and more about investigating the gap between benchmark accuracy and real-world reliability.

**Yours truly,**  
**The Developer @ FruitGuard**

---

# 1. Research Question

## Primary Research Question

**How accurately do computer-vision models trained on controlled PlantVillage images identify common crop leaf diseases when evaluated on real-world images with varying lighting, backgrounds, framing, and image quality?**

## Secondary Research Question

**How large is the generalization gap between performance on controlled PlantVillage images and performance on external real-world out-of-distribution images?**

---

# 2. Hypothesis

The project hypothesized that:

1. Transfer-learning models would perform strongly on held-out PlantVillage images.
2. A pretrained MobileNetV3 model would outperform a CNN trained from scratch.
3. Both models would experience a substantial decline when evaluated on images outside the PlantVillage distribution.
4. High controlled-domain accuracy would not necessarily indicate strong real-world reliability.

---

# 3. Project Scope

FruitGuard contains **14 classes across four crops**.

| Crop | Classes |
|---|---|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Grape | Black Rot, Esca / Black Measles, Leaf Blight, Healthy |
| Peach | Bacterial Spot, Healthy |
| Tomato | Early Blight, Late Blight, Leaf Mold, Healthy |

**Total classes: 14**

The final research scope focuses on **leaf-disease classification**.

Pest identification was excluded from the final experiment.

---

# 4. Datasets

## 4.1 PlantVillage

The official PlantVillage dataset was used as the controlled training and in-distribution evaluation dataset.

Source:

https://github.com/spMohanty/PlantVillage-Dataset

Only the fourteen FruitGuard classes were retained.

The final research dataset contained:

| Split | Images |
|---|---:|
| Training | 9,630 |
| Validation | 2,063 |
| Test | 2,062 |
| **Total** | **13,755** |

Random seed:

```text
42
````

The approximate intended split was:

* 70% training
* 15% validation
* 15% test

---

## 4.2 Leakage-Aware Splitting

A normal random image-level split can accidentally place photographs of the same physical leaf into both training and testing sets.

This could artificially increase evaluation performance.

Where PlantVillage leaf-group metadata was available, images originating from the same physical leaf were therefore kept within the same split.

### Exception — Grape Healthy

Leaf-group metadata was unavailable for:

```text
Grape___healthy
```

That class was split reproducibly at image level using seed 42.

This limitation is documented because related-image leakage cannot be ruled out for that class.

---

## 4.3 Metadata Exclusions

Four PlantVillage metadata entries belonging to:

```text
Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
```

could not be matched to corresponding image files.

These four entries were excluded.

No replacement or invented images were added.

---

# 5. Training Data Augmentation

Random augmentation was applied **only to training images**.

Training transformations included:

* resizing
* random cropping
* horizontal flipping
* small rotations
* brightness adjustment
* contrast adjustment
* saturation adjustment
* minor hue variation

Validation and test images used deterministic resizing and normalization without random augmentation.

---

# 6. Models

The final experimental comparison focused on two neural-network approaches.

## 6.1 MobileNetV3-Small

MobileNetV3-Small was selected as the transfer-learning model.

Reasons included:

* pretrained ImageNet features
* relatively small checkpoint size
* efficient inference
* suitability for web deployment
* strong expected performance compared with training a network from scratch

The final trained checkpoint is:

```text
ml/checkpoints/mobilenet_v3_best.pt
```

---

## 6.2 Scratch CNN

A custom convolutional neural network was trained without pretrained ImageNet features.

Its purpose was to provide a direct comparison showing how well a neural network trained solely from the FruitGuard training images performed relative to transfer learning.

The final checkpoint is:

```text
ml/checkpoints/scratch_cnn_best.pt
```

---

# 7. Controlled-Domain Training Results

## MobileNetV3-Small

Training epochs:

```text
15
```

Best validation accuracy:

```text
1.000
```

Held-out PlantVillage test accuracy:

```text
1.000
```

Therefore:

**PlantVillage Test Accuracy: 100.00%**

---

## Scratch CNN

Training epochs:

```text
8
```

Best validation accuracy:

```text
0.829
```

Final training accuracy:

```text
0.903
```

Held-out PlantVillage test accuracy:

```text
0.8205625606207565
```

Therefore:

**PlantVillage Test Accuracy: 82.06%**

---

# 8. Controlled-Domain Comparison

| Model                 | Validation Accuracy | PlantVillage Test Accuracy |
| --------------------- | ------------------: | -------------------------: |
| **MobileNetV3-Small** |         **100.00%** |                **100.00%** |
| **Scratch CNN**       |          **82.90%** |                 **82.06%** |

MobileNetV3-Small substantially outperformed the CNN trained from scratch.

This suggests that pretrained visual features provided a major advantage when learning the fourteen FruitGuard classes.

However, this was only the first half of the experiment.

---

# 9. Out-of-Distribution Evaluation

The primary research question required evaluating the trained models on images outside the PlantVillage distribution.

For this purpose, an external dataset was created using images from **PlantDoc**.

Source:

[https://github.com/pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset)

PlantDoc contains less standardized plant-disease imagery with environmental differences that may include:

* natural backgrounds
* clutter
* variable lighting
* different framing
* different resolutions
* different leaf orientations
* occlusion
* varying disease severity
* different image sources

PlantDoc images were **never used for FruitGuard training**.

---

# 10. PlantDoc OOD Dataset

The final PlantDoc OOD dataset contained:

```text
95 images
```

Only classes with sufficiently compatible mappings between PlantDoc and FruitGuard were included.

## Included Classes

* Apple___Apple_scab
* Apple___Cedar_apple_rust
* Apple___healthy
* Grape___Black_rot
* Grape___healthy
* Peach___healthy
* Tomato___Early_blight
* Tomato___Late_blight
* Tomato___Leaf_Mold
* Tomato___healthy

## Classes Without Reliable OOD Equivalents

The following classes were not force-mapped:

* Apple___Black_rot
* Grape___Esca_(Black_Measles)
* Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
* Peach___Bacterial_spot

The OOD experiment therefore covers **10 of the 14 FruitGuard classes**.

This limitation is intentionally reported rather than assigning questionable labels merely to create a complete fourteen-class dataset.

---

# 11. OOD Results

The same trained checkpoints used for the PlantVillage test evaluation were evaluated on the PlantDoc dataset **without retraining**.

| Model                 | PlantVillage Accuracy | PlantDoc OOD Accuracy |         Generalization Drop |
| --------------------- | --------------------: | --------------------: | --------------------------: |
| **MobileNetV3-Small** |           **100.00%** |            **18.95%** | **81.05 percentage points** |
| **Scratch CNN**       |            **82.06%** |            **14.74%** | **67.32 percentage points** |

---

# 12. Generalization Gap

The generalization gap was calculated as:

```text
PlantVillage test accuracy - PlantDoc OOD accuracy
```

## MobileNetV3-Small

```text
100.00% - 18.95% = 81.05 percentage points
```

## Scratch CNN

```text
82.06% - 14.74% = 67.32 percentage points
```

Both models therefore experienced severe performance degradation outside the controlled PlantVillage distribution.

---

# 13. Main Finding

The most important result of the experiment is the contrast between controlled and external performance.

MobileNetV3-Small achieved:

```text
100.00% controlled accuracy
18.95% OOD accuracy
```

The scratch CNN achieved:

```text
82.06% controlled accuracy
14.74% OOD accuracy
```

MobileNetV3 remained the stronger model on both datasets.

However, its perfect PlantVillage test accuracy did **not** translate into reliable classification of external real-world images.

This strongly supports the project's central hypothesis:

> **High in-distribution benchmark accuracy is not sufficient evidence of real-world robustness.**

---

# 14. Selected Class-Level Findings

MobileNetV3 performed differently across OOD classes.

Examples include:

### Tomato Early Blight

Recall:

```text
88.89%
```

The model correctly identified a relatively large proportion of the available Tomato Early Blight OOD examples.

### Grape Black Rot

Recall:

```text
57.14%
```

Performance was substantially better than several other classes but remained far below the controlled-domain result.

### Several Other Classes

Some classes achieved zero recall on the OOD dataset.

This shows that the effects of domain shift were not uniform across diseases.

Certain visual features transferred better than others.

---

# 15. Confidence and Reliability

One especially important observation from the project is that model confidence should not automatically be interpreted as correctness.

A neural network can assign high probability to an incorrect prediction when presented with images outside its training distribution.

For this reason, the deployed FruitGuard application:

* displays ranked predictions rather than claiming a definitive diagnosis
* displays model confidence
* checks basic image-quality characteristics
* warns users when predictions are uncertain
* includes an educational-use disclaimer
* does not recommend pesticides or treatments

---

# 16. Interpretation

The experiment suggests that the models learned highly effective representations for the PlantVillage image distribution.

However, PlantVillage images are relatively standardized compared with many photographs a real user might upload.

The large performance decline on PlantDoc indicates sensitivity to domain shift.

Possible contributing factors include:

* background differences
* lighting differences
* framing
* scale
* leaf orientation
* image compression
* camera characteristics
* disease severity
* visual clutter
* occlusion
* symptom variation

It is not possible from this experiment alone to determine exactly how much each factor contributes.

However, the overall result clearly demonstrates that dataset distribution has a major effect on model performance.

---

# 17. Did the Hypothesis Hold?

## Hypothesis 1

> Transfer learning would perform strongly on controlled PlantVillage images.

**Supported.**

MobileNetV3 achieved 100% PlantVillage test accuracy.

---

## Hypothesis 2

> MobileNetV3 would outperform a CNN trained from scratch.

**Supported.**

Controlled test results:

```text
MobileNetV3: 100.00%
Scratch CNN: 82.06%
```

OOD results:

```text
MobileNetV3: 18.95%
Scratch CNN: 14.74%
```

---

## Hypothesis 3

> Both models would lose accuracy on real-world OOD images.

**Strongly supported.**

Both models experienced large declines.

---

## Hypothesis 4

> High controlled accuracy would not necessarily indicate real-world reliability.

**Strongly supported.**

The 81.05 percentage-point drop experienced by MobileNetV3 is the strongest evidence produced by the experiment.

---

# 18. Model Selected for Deployment

MobileNetV3-Small was selected as the model used by the FruitGuard web application.

Reasons:

* highest controlled-domain accuracy
* highest OOD accuracy of the two tested models
* lightweight architecture
* relatively small checkpoint
* suitable inference speed
* practical for deployment

The selection does **not** imply that MobileNetV3 is sufficiently accurate for professional agricultural diagnosis.

It is simply the strongest model evaluated within the project.

---

# 19. Web Application

FruitGuard was developed into a complete web application.

## Frontend

Technologies:

* React
* Vite
* JavaScript

Deployment:

[https://fruitguard-zeta.vercel.app](https://fruitguard-zeta.vercel.app)

---

## Backend

Technologies:

* Python
* FastAPI
* PyTorch
* torchvision
* Pillow

Deployment:

[https://fruitguard-api.onrender.com](https://fruitguard-api.onrender.com)

Health endpoint:

[https://fruitguard-api.onrender.com/api/health](https://fruitguard-api.onrender.com/api/health)

The production API loads:

```text
mobilenet_v3_best.pt
```

---

# 20. Prediction Pipeline

A user:

1. selects an image
2. uploads it through the React frontend
3. sends the image to the FastAPI backend
4. the backend preprocesses the image
5. MobileNetV3 generates logits
6. softmax converts logits into probabilities
7. the top three classes are returned
8. FruitGuard displays the predictions and confidence values

The response also includes:

* crop
* disease label
* description
* symptoms to look for
* image-quality information
* uncertainty status
* safety disclaimer

---

# 21. Evaluation Metrics

The evaluation pipeline records:

* accuracy
* precision
* recall
* F1
* per-class metrics
* confusion matrix
* macro averages
* weighted averages

One important limitation is that the OOD dataset contains only ten supported classes.

Therefore, macro metrics calculated over all fourteen FruitGuard labels include classes with zero OOD support.

For this reason, the final report emphasizes:

* overall OOD accuracy
* generalization gap
* supported per-class behavior

rather than presenting the fourteen-class OOD macro-F1 as the primary finding.

---

# 22. Limitations

FruitGuard has several important limitations.

## Dataset Size

The external OOD evaluation contains only:

```text
95 images
```

This is sufficient to demonstrate a substantial distribution shift but not enough to estimate universal real-world accuracy.

---

## Partial Class Coverage

Only 10 of the 14 classes had sufficiently compatible PlantDoc classes.

The external benchmark is therefore incomplete.

---

## PlantVillage Characteristics

PlantVillage imagery is relatively clean and standardized.

Even leakage-aware splitting cannot make the held-out PlantVillage test set equivalent to arbitrary field photographs.

---

## Grape Healthy Split

`Grape___healthy` lacked leaf-group metadata and therefore required image-level splitting.

Potential related-image similarity cannot be completely excluded for this class.

---

## Dataset-to-Dataset Differences

PlantDoc is itself a specific dataset.

Performance on PlantDoc should not be interpreted as the exact expected accuracy on:

* every phone camera
* every farm
* every climate
* every disease stage
* every crop variety
* every real-world environment

---

## Classification Is Not Diagnosis

Visible symptoms may be caused by:

* multiple diseases
* nutrient deficiencies
* environmental stress
* insect damage
* physical damage
* image artifacts

FruitGuard cannot establish a definitive agricultural diagnosis from an image alone.

---

# 23. Research Integrity

Several decisions were made specifically to protect the integrity of the experiment.

### No invented OOD results

When the OOD dataset was initially empty, no score was reported.

### No training on PlantDoc before evaluation

PlantDoc remained external to model training.

### No force-mapping unsupported classes

Classes without sufficiently reliable equivalents were excluded.

### Frozen PlantVillage split

The same controlled test split was used for comparing the final models.

### Poor results were retained

The low OOD scores were not discarded, hidden, or replaced.

They became the central finding of the research.

---

# 24. Conclusion

FruitGuard began as a plant disease classification project but ultimately became an investigation into machine-learning generalization.

MobileNetV3-Small achieved a perfect:

```text
100.00%
```

accuracy on the controlled PlantVillage test set.

A CNN trained from scratch achieved:

```text
82.06%
```

However, on 95 external PlantDoc images, performance fell to:

```text
MobileNetV3-Small: 18.95%
Scratch CNN: 14.74%
```

These results demonstrate a severe generalization gap between controlled and real-world image distributions.

The experiment therefore answers the original research question clearly:

> **Computer-vision models trained on controlled PlantVillage imagery can perform extremely well on similarly distributed held-out images while remaining unreliable when evaluated on substantially different real-world imagery.**

The findings demonstrate why benchmark accuracy should not be treated as proof that an AI system will perform equally well after deployment.

For FruitGuard, the biggest lesson was not how to reach 100% accuracy.

It was learning to ask what that 100% actually meant.

---

# 25. Future Work

Possible future extensions include:

* collecting a larger independently photographed field dataset
* balanced OOD coverage for all fourteen classes
* confidence calibration
* explicit out-of-distribution detection
* unsupported-image rejection
* background segmentation
* disease localization
* Grad-CAM explainability
* domain adaptation
* training with more diverse field imagery
* additional crops
* additional diseases
* mobile-device inference
* comparison with vision transformers
* cross-dataset training experiments

One particularly valuable future experiment would be to retrain or fine-tune the model using a mixture of PlantVillage and diverse real-world images, then evaluate it against a **third dataset that remains completely unseen**.

This would test whether greater training diversity actually reduces the generalization gap.

---

# 26. Final Results Summary

| Item                           | Result                      |
| ------------------------------ | --------------------------- |
| Target crops                   | Apple, Grape, Peach, Tomato |
| Total classes                  | 14                          |
| PlantVillage images used       | 13,755                      |
| Training images                | 9,630                       |
| Validation images              | 2,063                       |
| Test images                    | 2,062                       |
| OOD dataset                    | PlantDoc                    |
| OOD images                     | 95                          |
| OOD classes covered            | 10 / 14                     |
| Random seed                    | 42                          |
| MobileNet test accuracy        | 100.00%                     |
| MobileNet OOD accuracy         | 18.95%                      |
| MobileNet generalization gap   | 81.05 pp                    |
| Scratch CNN test accuracy      | 82.06%                      |
| Scratch CNN OOD accuracy       | 14.74%                      |
| Scratch CNN generalization gap | 67.32 pp                    |
| Final deployed model           | MobileNetV3-Small           |

---

# 27. Repository and Deployment

## Repository

[https://github.com/ItzGhost-xD/FruitGuard-passionproject-csPS](https://github.com/ItzGhost-xD/FruitGuard-passionproject-csPS)

## Live FruitGuard

[https://fruitguard-zeta.vercel.app](https://fruitguard-zeta.vercel.app)

## API

[https://fruitguard-api.onrender.com](https://fruitguard-api.onrender.com)

## Research Files

Final evaluation outputs are stored under:

```text
ml/results/
```

The frozen research split is stored at:

```text
ml/data/splits/fruitguard_split_manifest.csv
```

The OOD provenance manifest is stored at:

```text
ml/data/ood_real/manifest.csv
```

---

## Final Note From the Developer

FruitGuard did not end with the result I would have expected when I first started building it.

At the beginning, seeing a model reach 100% accuracy felt like success.

By the end, the 18.95% result taught me more.

It showed me why experiments need to challenge the assumptions behind their own results.

A number looks impressive only when we understand what was required to produce it, what data it represents, and where it stops being reliable.

That is what I wanted FruitGuard to demonstrate.

Not that AI can magically identify every disease from a photograph, but that responsible development requires us to test where our models stop working too.

**Yours truly,**
**The Developer @ FruitGuard**

````