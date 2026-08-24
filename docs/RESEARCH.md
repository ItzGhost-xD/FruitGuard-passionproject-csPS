# FruitGuard Research Log

> **Research integrity note:** Empty results are better than invented results. Placeholder experiments are clearly separated from final research results.

## Research Question

**Primary question**

How accurately do computer-vision models trained on controlled PlantVillage images identify common crop leaf diseases when evaluated on real-world images with varying lighting, backgrounds, framing, and image quality?

**Secondary question**

How large is the generalization gap between performance on controlled PlantVillage images and performance on real-world out-of-distribution (OOD) images?

## Hypothesis

Transfer-learning models are expected to perform strongly on held-out PlantVillage images but experience a measurable reduction in accuracy and macro-F1 when evaluated on natural-scene images.

It is also expected that pretrained convolutional neural networks will generalize better than a scratch CNN and traditional HSV-histogram baseline.

## Scope

| Crop   | Conditions                                            |
| ------ | ----------------------------------------------------- |
| Apple  | scab, black rot, cedar apple rust, healthy            |
| Grape  | black rot, esca (black measles), leaf blight, healthy |
| Peach  | bacterial spot, healthy                               |
| Tomato | early blight, late blight, leaf mold, healthy         |

**Total classes: 14**

The project focuses on leaf-disease classification. Pest identification is currently outside the defined scope.

## Dataset

### Primary Training Dataset — PlantVillage

PlantVillage will be used as the controlled-domain training and evaluation dataset.

Record before final experiments:

* Dataset source:
* Exact dataset version:
* Source URL:
* License:
* Download date:
* Total downloaded images:
* Total images retained after class filtering:
* Excluded/corrupted images:
* Dataset checksum/version identifier, if available:

Only the fourteen classes defined in the project scope will be retained.

### Dataset Splitting

Target split:

* Training: 70%
* Validation: 15%
* Test: 15%
* Random seed: 42

The split must be stratified by class.

Where leaf-group metadata is available, images originating from the same leaf should remain within the same split to reduce the risk of data leakage.

The final split counts will be recorded after the real dataset is prepared.

| Split      | Images | Percentage |
| ---------- | -----: | ---------: |
| Train      |    TBD |        70% |
| Validation |    TBD |        15% |
| Test       |    TBD |        15% |

### Training Augmentation

Augmentation is applied **only to training images**.

Planned transformations:

* resized/random crop
* horizontal flip
* small rotations
* moderate brightness/contrast/color jitter

Validation and test images will not receive random augmentation.

## Out-of-Distribution Evaluation

The main research question requires testing beyond the controlled PlantVillage distribution.

### OOD-A — Natural-Scene Public Dataset

A compatible subset of a natural-scene plant-disease dataset such as PlantDoc will be used where labels can be reliably mapped to FruitGuard's fourteen classes.

The model will **not be retrained on the OOD evaluation images** before this evaluation.

Record:

* dataset/version:
* license:
* matching classes:
* images retained:
* excluded classes:
* label-mapping decisions:

### OOD-B — FruitGuard Phone-Photo Set

A small independent collection of phone photographs will be used as an additional real-world demonstration set.

These photographs may include:

* indoor and outdoor lighting
* shadows/backlighting
* plain and cluttered backgrounds
* different distances
* partial leaves
* multiple leaves
* mild blur
* varying camera angles

Metadata should be recorded where possible.

Suggested metadata fields:

`filename, true_class, lighting, background, blur, framing, device, notes`

Incomplete class coverage will be reported honestly. The phone-photo dataset will not be presented as a balanced benchmark unless adequate samples exist for every class.

## Models

| Approach                   | Reason for Comparison                                       |
| -------------------------- | ----------------------------------------------------------- |
| HSV histogram + linear SVM | Traditional computer-vision baseline                        |
| Scratch CNN                | Measures performance without pretrained ImageNet features   |
| MobileNetV3-Small          | Lightweight transfer-learning model suitable for deployment |
| EfficientNet-B0            | Efficient higher-capacity transfer-learning comparison      |
| ResNet-18                  | Standard residual-network baseline                          |

## Experimental Controls

To ensure fair comparison:

* Models use the same fourteen target classes.
* Neural models use the same train/validation/test partitions.
* The same test images are used for every model.
* Random seed is recorded.
* Test images are never used during training or hyperparameter selection.
* OOD images are kept separate from model training during the primary experiment.
* Neural-model input size is standardized where practical.
* Training settings and deviations are recorded.

## Training Strategy

For pretrained neural networks:

1. Initialize with pretrained ImageNet weights.
2. Replace the final classification layer with a fourteen-class head.
3. Train the classifier head while the feature extractor is initially frozen.
4. Fine-tune pretrained layers using a lower learning rate.
5. Monitor validation loss and macro-F1.
6. Use early stopping to reduce unnecessary training and overfitting.
7. Save the checkpoint with the best validation performance.

Training settings to record:

| Setting                   | Value |
| ------------------------- | ----- |
| Random seed               | 42    |
| Input resolution          | TBD   |
| Batch size                | TBD   |
| Optimizer                 | TBD   |
| Initial learning rate     | TBD   |
| Fine-tuning learning rate | TBD   |
| Maximum epochs            | TBD   |
| Early-stopping patience   | TBD   |
| Loss function             | TBD   |
| Class weighting           | TBD   |

## Evaluation Metrics

### Classification Performance

Report:

* Accuracy
* Per-class precision
* Per-class recall
* Per-class F1
* Macro-F1
* Confusion matrix

Because the dataset is class-imbalanced, macro-F1 will be considered alongside overall accuracy.

### Cross-Domain Performance

For each model report:

* PlantVillage test accuracy
* PlantVillage macro-F1
* OOD accuracy
* OOD macro-F1
* Generalization gap

**Generalization gap:**

`PlantVillage accuracy - OOD accuracy`

A larger gap indicates poorer transfer from controlled images to natural-scene images.

### Deployment Metrics

Where practical also record:

* model parameter count
* checkpoint/file size
* average inference time

This allows the final deployed model to be selected using both predictive performance and practical efficiency.

## Results

### Controlled-Domain Results

| Model             | Test Accuracy | Macro-F1 | Parameters | Model Size | Inference Time |
| ----------------- | ------------: | -------: | ---------: | ---------: | -------------: |
| HSV + SVM         |           TBD |      TBD |        N/A |        TBD |            TBD |
| Scratch CNN       |           TBD |      TBD |        TBD |        TBD |            TBD |
| MobileNetV3-Small |           TBD |      TBD |        TBD |        TBD |            TBD |
| EfficientNet-B0   |           TBD |      TBD |        TBD |        TBD |            TBD |
| ResNet-18         |           TBD |      TBD |        TBD |        TBD |            TBD |

### Cross-Domain Results

| Model             | PlantVillage Accuracy | OOD Accuracy | Accuracy Gap | PlantVillage Macro-F1 | OOD Macro-F1 |
| ----------------- | --------------------: | -----------: | -----------: | --------------------: | -----------: |
| HSV + SVM         |                   TBD |          TBD |          TBD |                   TBD |          TBD |
| Scratch CNN       |                   TBD |          TBD |          TBD |                   TBD |          TBD |
| MobileNetV3-Small |                   TBD |          TBD |          TBD |                   TBD |          TBD |
| EfficientNet-B0   |                   TBD |          TBD |          TBD |                   TBD |          TBD |
| ResNet-18         |                   TBD |          TBD |          TBD |                   TBD |          TBD |

## Failure Analysis

Incorrect predictions will be reviewed rather than reporting metrics alone.

### Disease Ambiguity

* visually similar diseases
* early vs late blight
* overlapping symptom patterns

### Domain Shift

* cluttered backgrounds
* different lighting
* shadows/backlighting
* unusual camera angles
* leaf occupying only part of the image

### Image Quality

* blur
* compression
* low resolution
* poor focus

### Biological Ambiguity

* mixed symptoms
* nutritional deficiencies
* damaged leaves
* different disease severity

### Confidence Errors

Special attention will be given to **high-confidence incorrect predictions**, since these represent an important reliability problem for a user-facing classifier.

Example failure-case fields:

| Image | True Class | Predicted Class | Confidence | Environment | Notes |
| ----- | ---------- | --------------- | ---------: | ----------- | ----- |
| TBD   | TBD        | TBD             |        TBD | TBD         | TBD   |

## Current Status — Pipeline Validation

**IMPORTANT:** Current numerical results are generated using placeholder sample images and are **not research findings**.

Current placeholder dataset:

* 140 images total
* 10 images per class
* 14 classes

Current placeholder split:

* Training: 112 images — 8/class
* Validation: 14 images — 1/class
* Test: 14 images — 1/class

These images exist only to validate:

* dataset loading
* training
* checkpoint creation
* inference
* evaluation
* frontend/backend integration

Current placeholder accuracy values must not be interpreted as evidence of FruitGuard's real-world performance.

## Next Research Milestones

1. Obtain and record the real PlantVillage dataset.
2. Filter the dataset to the fourteen FruitGuard classes.
3. Verify image counts and corrupted files.
4. Generate leakage-aware train/validation/test splits.
5. Freeze the final test split.
6. Train the traditional baseline.
7. Train the scratch CNN.
8. Train MobileNetV3-Small.
9. Train EfficientNet-B0.
10. Train ResNet-18.
11. Evaluate every model on the identical PlantVillage test set.
12. Prepare the compatible natural-scene OOD dataset.
13. Evaluate the frozen models on OOD data.
14. Collect the independent FruitGuard phone-photo subset.
15. Perform failure-case analysis.
16. Compare predictive performance with deployment efficiency.
17. Select the model used by the FruitGuard web application.
18. Write the final findings and limitations.

## Limitations

PlantVillage images are collected under relatively controlled conditions, so high in-distribution accuracy does not guarantee equivalent performance on arbitrary phone photographs.

The OOD datasets may be smaller and more imbalanced than the primary dataset.

Not every real-world disease condition or severity level can be represented.

Image classification alone cannot establish a definitive agricultural diagnosis.

FruitGuard is a research and educational computer-vision project and does not provide treatment or pesticide recommendations.

## Future Work

Possible extensions include:

* unsupported-image rejection
* uncertainty calibration
* disease localization
* larger field-image datasets
* broader crop coverage
* additional disease categories
* improved domain adaptation
* mobile-device deployment

## Experiment Log

### 2026-08-24 — Controlled PlantVillage evaluation

A leakage-aware research split was created from the official PlantVillage dataset.

Final split:
- Training: 9,630 images
- Validation: 2,063 images
- Test: 2,062 images
- Total: 13,755 images
- Classes: 14

Where official leaf-group metadata was available, images from the same physical leaf were kept in the same split to reduce leakage.

One exception was `Grape___healthy`, for which leaf-group metadata was unavailable. This class was split reproducibly at image level using seed 42.

Four metadata entries for `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` could not be matched to files and were excluded.

### MobileNetV3-Small

Training:
- Epochs: 15
- Best validation accuracy: 1.000
- Best checkpoint: `ml/checkpoints/mobilenet_v3_best.pt`

Held-out PlantVillage test accuracy:
- 1.000

Interpretation:
MobileNetV3 achieved perfect accuracy on the controlled PlantVillage test set. This result should not be interpreted as proof of real-world reliability because PlantVillage images are relatively clean and standardized.

### Scratch CNN

Training:
- Epochs: 8
- Best validation accuracy: 0.829
- Final training accuracy: 0.903

Held-out PlantVillage test accuracy:
- 0.8206

Interpretation:
The pretrained MobileNetV3 substantially outperformed the CNN trained from scratch on the same controlled dataset, suggesting that transfer learning provided a major advantage.

### Current controlled-data comparison

| Model | Validation Accuracy | Test Accuracy |
|---|---:|---:|
| MobileNetV3-Small | 1.000 | 1.000 |
| Scratch CNN | 0.829 | 0.821 |

### Next experiment — real-world OOD evaluation

The next stage will evaluate both models on real-world images that differ from PlantVillage in:
- lighting
- background
- framing
- camera quality
- orientation
- occlusion

The goal is to measure the generalization gap between controlled laboratory-style images and realistic user-submitted images.

### 2026-08-24 — OOD evaluation setup

The OOD evaluation pipeline was added and tested with both trained models.

The first attempt returned no OOD metrics because the real-world OOD dataset had not yet been populated with usable images. No OOD result was recorded from this attempt.

Next step:
Collect genuine real-world images for the 14 target classes and rerun the same checkpoints without retraining.

python -c "import json; d=json.load(open('ml/results/mobilenet_v3_eval.json')); print(json.dumps(d.get('ood_real'), indent=2))"